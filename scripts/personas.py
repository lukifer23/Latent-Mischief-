"""Derive response-token contrast directions and run a small, controlled exploration.

This is candidate discovery, not held-out validation. No weights are changed.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import lab

ROOT = lab.ROOT


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def response_features(model, tokenizer, result):
    """Equal weight per response; average residual states of non-special reply tokens."""
    import torch
    special = set(tokenizer.all_special_ids)
    positions = [result["input_tokens"] + i for i, token in enumerate(result["token_ids"])
                 if token not in special]
    if not positions:
        raise ValueError("No non-special response tokens available for feature extraction")
    features, norms, handles = {}, {}, []
    def make_hook(layer):
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            selected = value[0, positions, :].detach().float()
            features[layer] = selected.mean(dim=0).cpu()
            norms[layer] = selected.norm(dim=-1).mean().item()
        return capture
    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(make_hook(i)))
        ids = torch.tensor([result["input_token_ids"] + result["token_ids"]], device=model.device)
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return features, norms, len(positions)


@contextmanager
def direction_hook(model, layer, vector, amplitude):
    """Add at the decoder output, after all block norms, at its last position only."""
    delta = (vector * amplitude).to(device=model.device, dtype=model.dtype)
    def push(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        changed = value.clone()
        changed[:, -1, :] += delta
        return (changed,) + output[1:] if isinstance(output, tuple) else changed
    handle = model.model.layers[layer].register_forward_hook(push)
    try:
        yield
    finally:
        handle.remove()


def indicators(result):
    """Transparent exploratory signals, not persona-quality scores."""
    text = result["text"].lower()
    animal = ("hamster", "whisker", "paws", "burrow", "sunflower", "squeak", "rodent")
    theatrical = ("muhaha", "mwahaha", "fool", "minion", "dominion", "conquer", "diabolical", "villain")
    return {"hamster_terms_present": [w for w in animal if w in text],
            "villain_terms_present": [w for w in theatrical if w in text],
            "hit_token_limit": result["hit_token_limit"],
            "terminated_with_eos": result["terminated_with_eos"]}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/personas.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    discovery = read("prompts/persona-discovery.json")
    probes = read("prompts/persona-probes.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "runs" / f"{stamp}-LFM-002-persona-discovery"
    folder.mkdir(parents=True)
    print(f"RUN {folder}", flush=True)
    model, tokenizer = lab.load()
    source_hashes = lab.archive_sources(folder, ["scripts/lab.py", "scripts/personas.py", "scripts/storage.py",
        "uv.lock", "configs/model.json", "configs/personas.json", "prompts/persona-discovery.json", "prompts/persona-probes.json"])
    (folder / "source" / "experiment-config.json").write_bytes(config_path.read_bytes())
    lab.write_json(folder / "metadata.json", {
        "model": lab.config(), "environment": lab.environment(), "experiment_config": cfg,
        "experiment_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_hashes": source_hashes,
        "checkpoint_generation_defaults": model.generation_config.to_dict(),
        "hashes": {p: digest(p) for p in ("scripts/lab.py", "scripts/personas.py", "scripts/storage.py", "configs/personas.json",
                   "prompts/persona-discovery.json", "prompts/persona-probes.json")},
        "feature_location": "decoder layer output, after residual addition; non-special reply-token mean",
        "averaging": "mean over tokens within each response, then equal mean over paired responses",
        "intervention": "add unit contrast direction times dose times mean neutral reply residual norm",
        "positions": "last position on every forward: final prompt token and each incremental decode token",
        "generation": {"do_sample": False, "max_new_tokens": cfg["max_new_tokens"]},
        "limitation": "Prompt-disjoint exploratory probes; inspected/tuned here, not final held-out evaluation",
    })
    outputs = []
    def emit(result, **labels):
        result.update(labels)
        result["indicators"] = indicators(result)
        outputs.append(result)
        # Persist every completed generation so interrupted runs remain inspectable.
        lab.write_json(folder / "outputs.json", outputs)
        print(json.dumps({**labels, "text": result["text"]}, ensure_ascii=True), flush=True)
    features = {}
    neutral_norms = []
    for persona, system in {"neutral": cfg["neutral_system"], **cfg["personas"]}.items():
        features[persona] = []
        for item in discovery:
            result = lab.generate(model, tokenizer, item["prompt"], system=system,
                                  max_new_tokens=cfg["max_new_tokens"])
            feat, norms, count = response_features(model, tokenizer, result)
            features[persona].append(feat)
            if persona == "neutral":
                neutral_norms.append(norms)
            emit(result, phase="discovery", condition=persona, prompt_id=item["id"], feature_tokens=count)
    layer_count = len(model.model.layers)
    reference_norm = {i: sum(x[i] for x in neutral_norms) / len(neutral_norms) for i in range(layer_count)}
    directions, diagnostics = {}, {}
    for persona in cfg["personas"]:
        directions[persona] = {}
        diagnostics[persona] = {}
        for i in range(layer_count):
            pairs = torch.stack([p[i] - n[i] for p, n in zip(features[persona], features["neutral"])])
            mean = pairs.mean(dim=0)
            norm = mean.norm().item()
            if norm <= 1e-8:
                raise ValueError(f"Degenerate direction: {persona}, layer {i}")
            directions[persona][i] = mean / norm
            diagnostics[persona][i] = {"difference_norm": norm, "neutral_residual_norm": reference_norm[i],
                                       "pair_cosines_to_mean": torch.nn.functional.cosine_similarity(pairs, mean[None], dim=-1).tolist()}
    torch.save({"directions": directions, "neutral_norms": reference_norm,
                "model_id": lab.config()["model_id"], "revision": lab.config()["revision"]}, folder / "directions.pt")
    lab.write_json(folder / "direction-diagnostics.json", diagnostics)
    baselines = {}
    for item in probes:
        result = lab.generate(model, tokenizer, item["prompt"], max_new_tokens=cfg["max_new_tokens"])
        baselines[item["id"]] = result["token_ids"][:]
        emit(result, phase="probe", condition="baseline", prompt_id=item["id"])
    # Neutral hook at the actual additive intervention location, plus hook-removal check.
    with direction_hook(model, cfg["layers"][0], directions["hamster"][cfg["layers"][0]], 0.0):
        zero = lab.generate(model, tokenizer, probes[0]["prompt"], max_new_tokens=cfg["max_new_tokens"])
    restored = lab.generate(model, tokenizer, probes[0]["prompt"], max_new_tokens=cfg["max_new_tokens"])
    checks = {"zero_dose_token_identical": zero["token_ids"] == baselines[probes[0]["id"]],
              "removed_hook_token_identical": restored["token_ids"] == baselines[probes[0]["id"]]}
    emit(zero, phase="check", condition="zero-dose", prompt_id=probes[0]["id"])
    emit(restored, phase="check", condition="removed-hook", prompt_id=probes[0]["id"])
    lab.write_json(folder / "checks.json", checks)
    if not all(checks.values()):
        raise RuntimeError("Additive hook neutrality / cleanup failed")
    for persona, system in cfg["personas"].items():
        for item in probes:
            result = lab.generate(model, tokenizer, item["prompt"], system=system,
                                  max_new_tokens=cfg["max_new_tokens"])
            emit(result, phase="probe", condition="prompt-persona", persona=persona, prompt_id=item["id"])
        for layer in cfg["layers"]:
            for dose in cfg["doses"]:
                for item in probes:
                    with direction_hook(model, layer, directions[persona][layer], dose * reference_norm[layer]):
                        result = lab.generate(model, tokenizer, item["prompt"], max_new_tokens=cfg["max_new_tokens"])
                    emit(result, phase="probe", condition="contrast-direction", persona=persona,
                         prompt_id=item["id"], layer=layer, dose=dose)
    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    random_vectors = {}
    for layer in cfg["layers"]:
        for control in range(cfg["random_controls_per_layer"]):
            vector = torch.randn(model.config.hidden_size, generator=generator)
            vector /= vector.norm()
            random_vectors[f"{layer}:{control}"] = vector
            for item in probes:
                with direction_hook(model, layer, vector, cfg["random_control_dose"] * reference_norm[layer]):
                    result = lab.generate(model, tokenizer, item["prompt"], max_new_tokens=cfg["max_new_tokens"])
                emit(result, phase="probe", condition="random-direction", prompt_id=item["id"],
                     layer=layer, control=control, dose=cfg["random_control_dose"])
    torch.save(random_vectors, folder / "random-vectors.pt")
    counts = {}
    for result in outputs:
        counts[result["condition"]] = counts.get(result["condition"], 0) + 1
    lab.write_json(folder / "summary.json", {"completed": True, "counts": counts, "checks": checks,
                                            "directions_sha256": hashlib.sha256((folder / "directions.pt").read_bytes()).hexdigest(),
                                            "note": "Inspect full outputs. Keyword hits do not establish persona quality."})
    print(f"COMPLETE {folder}", flush=True)


if __name__ == "__main__":
    main()
