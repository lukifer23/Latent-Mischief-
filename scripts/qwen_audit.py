"""CPU audit of every pinned Qwen checkpoint tensor, including exact changed columns."""
import hashlib
import json
from pathlib import Path

import lab


def audit():
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    spec=json.loads((lab.ROOT/'configs/models.json').read_text(encoding='utf-8'))['qwen08']
    specs=[spec,{**spec,**spec['published_edit']}]; locations=[]
    for variant in specs:
        folder=Path(snapshot_download(variant['model_id'],revision=variant['revision'],allow_patterns=['*.safetensors'],local_files_only=True))
        mapping={}
        for file in folder.glob('*.safetensors'):
            with safe_open(file,framework='pt',device='cpu') as data:
                for key in data.keys():
                    if key in mapping: raise ValueError('Duplicate tensor key')
                    mapping[key]=file
        locations.append(mapping)
    if set(locations[0])!=set(locations[1]): raise ValueError('Checkpoint tensor keys differ')
    records=[]
    for key in sorted(locations[0]):
        values=[]
        for location in locations:
            with safe_open(location[key],framework='pt',device='cpu') as data: values.append(data.get_tensor(key))
        a,b=values
        if a.shape!=b.shape or a.dtype!=b.dtype: raise ValueError('Shape/dtype mismatch: '+key)
        digest=lambda tensor:hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        row={'tensor':key,'shape':list(a.shape),'dtype':str(a.dtype),'base_sha256':digest(a),'edited_sha256':digest(b)}
        row['identical']=row['base_sha256']==row['edited_sha256']
        if not row['identical']:
            mask=a.view(torch.int16)!=b.view(torch.int16);row['changed_elements']=int(mask.sum());row['max_abs_change']=float((a.float()-b.float()).abs().max())
            if a.ndim==2:
                row['changed_rows']=mask.any(dim=1).nonzero().flatten().tolist()
                row['changed_columns']=mask.any(dim=0).nonzero().flatten().tolist()
                row['addressed_elements']=a.shape[0]*len(row['changed_columns'])
                row['unchanged_inside_changed_columns']=row['addressed_elements']-row['changed_elements']
        records.append(row)
    changed=[x for x in records if not x['identical']];expected=spec['published_edit']['expected_tensor_sha256']
    checks={'all_tensor_keys_match':True,'all_shapes_dtypes_match':True,'only_expected_tensors_changed':{x['tensor'] for x in changed}==set(expected),
        'edited_tensor_digests_match':all(x['edited_sha256']==expected.get(x['tensor']) for x in changed),
        'exactly_64_columns_each':all(len(x.get('changed_columns',[]))==64 for x in changed),
        'exactly_131072_elements_in_changed_columns':sum(x['addressed_elements'] for x in changed)==131072}
    result={'models':[{k:x[k] for k in ['model_id','revision']} for x in specs],'checks':checks,'tensor_count':len(records),
        'changed_tensors':changed,'all_tensors':records,'scope':'Bitwise CPU comparison of cached pinned safetensors; no checkpoint mutation or GPU inference.'}
    lab.write_json(lab.ROOT/'runs/qwen-full-audit.json',result)
    if not all(checks.values()): raise ValueError('Pinned checkpoint audit failed; see runs/qwen-full-audit.json')
    print(json.dumps({'tensor_count':len(records),'checks':checks,'changed_elements':sum(x['changed_elements'] for x in changed)},indent=2))
    return result


if __name__=='__main__':audit()
