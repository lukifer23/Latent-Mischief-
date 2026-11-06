"""Compare locked inference dependencies without confusing optional CPU tools with model drift."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tomllib


def inference_lock_identity(data):
    lock=tomllib.loads(data.decode('utf-8'));packages=lock['package']
    root=next(x for x in packages if x['name']=='latent-mischief')
    pending=list(root.get('dependencies',[]));selected={};seen=set()
    while pending:
        reference=pending.pop();key=(reference['name'],tuple(reference.get('extra',[])))
        if key in seen:continue
        seen.add(key)
        matches=[x for x in packages if x['name']==reference['name']]
        if not matches:raise ValueError('Locked dependency missing: '+reference['name'])
        for package in matches:
            selected[(package['name'],package.get('version'),json.dumps(package['source'],sort_keys=True))]=package
            pending.extend(package.get('dependencies',[]))
            for extra in reference.get('extra',[]):pending.extend(package.get('optional-dependencies',{}).get(extra,[]))
    def normalized(value):
        if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
        if isinstance(value,list):return sorted([normalized(x) for x in value],key=lambda x:json.dumps(x,sort_keys=True))
        return value
    identity=normalized({'requires-python':lock['requires-python'],'root_dependencies':root.get('dependencies',[]),'packages':list(selected.values())})
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'sha256':digest,'package_count':len(selected),'packages':list(selected.values())}


def calibration_runtime_check(metadata,source_lock,current_lock):
    source_bytes=Path(source_lock).read_bytes();current_bytes=Path(current_lock).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=metadata['environment']['lock_sha256']:raise ValueError('Archived calibration lock differs from recorded digest')
    before=inference_lock_identity(source_bytes);after=inference_lock_identity(current_bytes)
    if before['sha256']!=after['sha256']:raise ValueError('Locked inference dependency graph differs from calibration')
    for name,version in metadata['environment']['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Installed inference package differs from calibration: '+name)
    # Check installed transitive/default dependencies as well. Platform-inapplicable lock entries may be absent.
    for name in {p['name'] for p in before['packages']}:
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:continue
        expected={p['version'] for p in before['packages'] if p['name']==name}
        if actual not in expected:raise ValueError('Installed locked dependency differs from calibration: '+name)
    return {'source_lock_sha256':hashlib.sha256(source_bytes).hexdigest(),'current_lock_sha256':hashlib.sha256(current_bytes).hexdigest(),
        'locked_inference_graph_sha256':before['sha256'],'inference_package_count':before['package_count'],
        'full_lock_identical':source_bytes==current_bytes,'installed_recorded_and_transitive_versions_match':True,
        'scope':'Only unrelated optional-tool dependencies may differ. This is calibration reuse, not permission to resume a changed experiment.'}
