"""Local LFM3 lab. Run from the repository root with `uv run python scripts/lab.py`."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def config():
    return json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))


def environment():
    import torch
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "huggingface-hub", "accelerate", "safetensors")},
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def prepare():
    """Download pinned source/config first; no remote code execution here."""
    from huggingface_hub import snapshot_download
    cfg = config()
    path = Path(snapshot_download(cfg["model_id"], revision=cfg["revision"],
                                 allow_patterns=["*.py", "*.json", "*.jinja", "README.md", "LICENSE"]))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in path.iterdir() if p.is_file()}
    write_json(ROOT / "runs/source-manifest.json", {"config": cfg, "files_sha256": hashes})
    print(path, flush=True)


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run doctor; CPU fallback must be chosen explicitly in a future config.")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], code_revision=cfg["revision"],
        trust_remote_code=True, dtype=torch.bfloat16, attn_implementation=cfg["attention"],
    ).to("cuda").eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, *, system=None, max_new_tokens=None):
    import torch
    cfg = config()
    torch.manual_seed(cfg["seed"])
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=cfg["thinking"],
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False,
                                max_new_tokens=max_new_tokens or cfg["max_new_tokens"],
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()
    tokens = output[0, inputs["input_ids"].shape[1]:].tolist()
    return {"prompt": prompt, "system": system, "text": tokenizer.decode(tokens, skip_special_tokens=False),
            "token_ids": tokens, "input_token_ids": inputs["input_ids"][0].tolist(),
            "input_tokens": inputs["input_ids"].shape[1],
            "output_tokens": len(tokens), "seconds": time.perf_counter() - started,
            "hit_token_limit": len(tokens) == (max_new_tokens or cfg["max_new_tokens"]),
            "terminated_with_eos": bool(tokens and tokens[-1] == tokenizer.eos_token_id)}


@contextmanager
def gain_hook(model, layer, neuron, gain):
    """Scale the actual gated MLP activation before w2 and post_ffn_norm."""
    module = model.model.layers[layer].feed_forward.w2
    if not 0 <= neuron < module.in_features:
        raise ValueError("Neuron index outside this checkpoint's MLP width")
    def hook(_module, args):
        changed = args[0].clone()
        changed[..., neuron] *= gain
        return (changed,) + args[1:]
    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def run_smoke():
    model, tokenizer = load()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "runs" / f"{stamp}-LFM-001-baseline"
    folder.mkdir(parents=True)
    write_json(folder / "metadata.json", {
        "experiment": "LFM-001", "config": config(), "environment": environment(),
        "generation": {"do_sample": False, "max_new_tokens": 96},
        "checkpoint_generation_defaults": model.generation_config.to_dict(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "purpose": "Baseline and neutral-hook validation; not a persona or capability claim",
    })
    inventory = [{"layer": i, "type": model.config.layer_types[i],
                  "hook": f"model.layers.{i}.feed_forward.w2 input",
                  "mlp_width": layer.feed_forward.w2.in_features,
                  "post_ffn_norm": hasattr(layer, "post_ffn_norm")}
                 for i, layer in enumerate(model.model.layers)]
    write_json(folder / "modules.json", inventory)
    prompts = json.loads((ROOT / "prompts/baseline.json").read_text(encoding="utf-8"))
    results = []
    for item in prompts:
        result = generate(model, tokenizer, item["prompt"], max_new_tokens=96)
        result.update({"id": item["id"], "condition": "baseline"})
        results.append(result)
        write_json(folder / "outputs.json", results)
        print(json.dumps({"id": item["id"], "seconds": round(result["seconds"], 2),
                          "text": result["text"]}, ensure_ascii=True), flush=True)
    with gain_hook(model, layer=12, neuron=0, gain=1.0):
        neutral = generate(model, tokenizer, prompts[0]["prompt"], max_new_tokens=96)
    neutral.update({"id": prompts[0]["id"], "condition": "gain-1-neutral"})
    results.append(neutral)
    write_json(folder / "outputs.json", results)
    check = {"neutral_hook_token_identical": neutral["token_ids"] == results[0]["token_ids"],
             "hook": "model.layers.12.feed_forward.w2 input", "gain": 1.0,
             "all_baselines_reached_eos": all(r["terminated_with_eos"] for r in results[:-1]),
             "note": "One-prompt neutral check only; persona interventions not tested"}
    write_json(folder / "checks.json", check)
    print(json.dumps({"run": str(folder), "checks": check}), flush=True)
    if not check["neutral_hook_token_identical"]:
        raise RuntimeError("Neutral hook changed generated tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "prepare", "smoke"])
    args = parser.parse_args()
    if args.command == "doctor":
        info = environment()
        write_json(ROOT / "runs/environment.json", info)
        print(json.dumps(info, indent=2))
    elif args.command == "prepare":
        prepare()
    else:
        run_smoke()


if __name__ == "__main__":
    main()
