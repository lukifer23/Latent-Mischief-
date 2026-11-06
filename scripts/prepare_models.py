"""Pin/download official Qwen base and published edit; verify the two edited tensor digests."""
import hashlib
import json
import lab


def main():
    from huggingface_hub import snapshot_download
    base_revision='2fc06364715b967f1860aea9cf38778875588b17'
    edited_id='RikiyaT/Qwen3.5-0.8B-weight-space-programming'
    edited_revision='d1671a178705bcab7644806d23c535517f35723b'
    registry={'lfm3':{**lab.config(),'label':'LFM3-1.7B','family':'lfm3','max_context':262144,'live_attention':'sdpa'},
        'qwen08':{'label':'Qwen3.5-0.8B','family':'qwen3_5','model_id':'Qwen/Qwen3.5-0.8B','revision':base_revision,
            'dtype':'bfloat16','live_attention':'sdpa','thinking':False,'seed':42,
            'published_edit':{'model_id':edited_id,'revision':edited_revision,
                'label':'Published anti-loop weight-space edit','source':'https://huggingface.co/'+edited_id,
                'expected_tensor_sha256':{
                    'model.language_model.layers.15.mlp.down_proj.weight':'82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53',
                    'model.language_model.layers.19.mlp.down_proj.weight':'8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2'}}}}
    lab.write_json(lab.ROOT/'configs/models.json',registry)
    for model_id,revision in [('Qwen/Qwen3.5-0.8B',base_revision),(edited_id,edited_revision)]:
        print(f'DOWNLOAD {model_id} {revision}',flush=True)
        path=snapshot_download(model_id,revision=revision,allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer.*','vocab.*','merges.*','LICENSE*','README.md'])
        print(f'CACHED {model_id}',flush=True)
        if model_id==edited_id:
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
