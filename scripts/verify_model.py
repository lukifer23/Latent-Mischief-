"""Verify pinned LFM3 identity and cached weight digest against public Hub metadata."""
import hashlib
import json
from pathlib import Path
import lab


def verify_identity():
    from huggingface_hub import HfApi, snapshot_download
    cfg=lab.config()
    snapshot=Path(snapshot_download(cfg['model_id'],revision=cfg['revision'],
        allow_patterns=['config.json','*.safetensors'],local_files_only=True))
    architecture=json.loads((snapshot/'config.json').read_text())
    info=HfApi().model_info(cfg['model_id'],revision=cfg['revision'],files_metadata=True)
    checks={'configured_liquid_lfm3':cfg['model_id']=='LiquidAI/LFM3-1.7B',
        'pinned_revision_resolves':info.sha==cfg['revision'], 'model_type_lfm3':architecture['model_type']=='lfm3',
        'architecture_lfm3':architecture['architectures']==['Lfm3ForCausalLM']}
    weights=[]
    for file in info.siblings:
        if not file.rfilename.endswith('.safetensors'): continue
        local=snapshot/file.rfilename
        with local.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        expected=file.lfs.sha256 if file.lfs is not None else None
        weights.append({'file':file.rfilename,'bytes':local.stat().st_size,'sha256':actual,'hub_sha256':expected,'matches':actual==expected})
    checks['cached_weights_match_hub']=bool(weights) and all(x['matches'] for x in weights)
    result={'model':cfg,'architecture':architecture,'checks':checks,'weights':weights,
        'config_sha256':hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest(),
        'source_url':'https://huggingface.co/'+cfg['model_id'],'scope':'Identity/integrity check, not model quality validation.'}
    lab.write_json(lab.ROOT/'runs/model-identity.json',result)
    if not all(checks.values()): raise RuntimeError('Model identity/integrity check failed; inspect runs/model-identity.json')
    return result


if __name__=='__main__':
    result=verify_identity(); print(json.dumps({'model':result['model'],'checks':result['checks'],
        'max_position_embeddings':result['architecture']['max_position_embeddings'],'weights':result['weights']},indent=2))
