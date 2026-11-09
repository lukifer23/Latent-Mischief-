"""Pin/download official Qwen base and published edit; verify the two edited tensor digests."""
import hashlib
import json
import lab


def main():
    from huggingface_hub import snapshot_download
    registry=json.loads((lab.ROOT/'configs/models.json').read_text(encoding='utf-8'))
    base=registry['qwen08']; edited={**base,**base['published_edit']}
    for variant in [base,edited]:
        model_id,revision=variant['model_id'],variant['revision']
        print(f'DOWNLOAD {model_id} {revision}',flush=True)
        path=snapshot_download(model_id,revision=revision,allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer.*','vocab.*','merges.*','LICENSE*','README.md'])
        print(f'CACHED {model_id}',flush=True)
        if variant is edited:
            import torch
            from safetensors import safe_open
            found={}
            for file in sorted(__import__('pathlib').Path(path).glob('*.safetensors')):
                with safe_open(file,framework='pt',device='cpu') as tensors:
                    for key,expected in registry['qwen08']['published_edit']['expected_tensor_sha256'].items():
                        if key in tensors.keys():
                            actual=hashlib.sha256(tensors.get_tensor(key).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                            found[key]={'sha256':actual,'expected':expected,'matches':actual==expected}
            lab.write_json(lab.ROOT/'runs/qwen-edit-identity.json',{'model_id':model_id,'revision':revision,'tensors':found})
            if len(found)!=2 or not all(x['matches'] for x in found.values()): raise RuntimeError('Published edit digest mismatch')
            print('Both published edited tensors match expected SHA-256',flush=True)


if __name__=='__main__': main()
