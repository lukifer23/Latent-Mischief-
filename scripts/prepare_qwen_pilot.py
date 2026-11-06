"""Freeze a deterministic, outcome-blind MATH-500 pilot selection; no inference."""
import hashlib
import json
from pathlib import Path

import lab


def main():
    from huggingface_hub import HfApi,hf_hub_download
    repo='HuggingFaceH4/MATH-500'
    config=lab.ROOT/'configs/qwen-pilot.json'
    revision=json.loads(config.read_text(encoding='utf-8'))['dataset_revision'] if config.exists() else HfApi().dataset_info(repo).sha
    files=HfApi().list_repo_files(repo,repo_type='dataset',revision=revision)
    filename=next(x for x in files if x.endswith('.jsonl'))
    path=Path(hf_hub_download(repo,filename,repo_type='dataset',revision=revision))
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    selected=[]
    for level in [1,2,3,4]:
        candidates=[x for x in rows if str(x['level']).split()[-1]==str(level)]
        key=lambda x:hashlib.sha256(('20261109:'+str(x['unique_id'])).encode()).hexdigest()
        row=min(candidates,key=key)
        selected.append({k:row[k] for k in ['unique_id','problem','answer','level','subject']})
    manifest={'dataset':repo,'revision':revision,'file':filename,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'selection':'One problem each at levels 1,2,3,4: minimum SHA256 of 20261109:<unique_id>. No generated outcomes inspected.',
        'problems':selected}
    destination=lab.ROOT/'.cache/qwn002/problems.json';lab.write_json(destination,manifest)
    print(json.dumps({'manifest':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'dataset_revision':revision,'selected':[{k:r[k] for k in ['unique_id','level','subject']} for r in selected]},indent=2))


if __name__=='__main__':main()
