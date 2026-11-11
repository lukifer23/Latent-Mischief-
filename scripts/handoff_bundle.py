"""Create or verify a private, hash-manifested bundle of local research runs for device transfer.

Excludes private live-chat records, job logs/queues and weight-like files. The bundle is never uploaded or committed;
copy it yourself and run `verify` on the receiving machine before extracting into the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_RUN_MARKERS = ('-LIVE-', 'live-worker', 'jobs')
EXCLUDED_SUFFIXES = ('.safetensors', '.gguf', '.bin', '.ckpt', '.pth', '.tmp')
LOOSE_RUN_FILES = ['runs/qwen-full-audit.json', 'runs/qwen-edit-identity.json', 'runs/model-identity.json',
                   'runs/persona-catalog.json']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_files():
    files = [ROOT / name for name in LOOSE_RUN_FILES if (ROOT / name).is_file()]
    for folder in sorted((ROOT / 'runs').iterdir()):
        if not folder.is_dir() or any(marker in folder.name for marker in EXCLUDED_RUN_MARKERS):
            continue
        if folder.name.endswith('-source'):
            continue
        files += [p for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix not in EXCLUDED_SUFFIXES]
    return files


def create(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Refusing to overwrite ' + str(destination))
    entries = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in selected_files():
            name = path.relative_to(ROOT).as_posix()
            entries[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
            archive.write(path, name)
        manifest = {'schema': 1, 'note': 'Private research-run bundle. Excludes live chats, job logs and weight-like files.',
                    'files': entries}
        archive.writestr('HANDOFF-MANIFEST.json', json.dumps(manifest, indent=2, sort_keys=True))
    manifest_path = destination.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps({**manifest, 'archive_sha256': sha(destination)}, indent=2, sort_keys=True), encoding='utf-8')
    return {'archive': str(destination), 'archive_sha256': sha(destination), 'files': len(entries),
            'bytes': sum(x['bytes'] for x in entries.values()), 'manifest': str(manifest_path)}


def verify(archive_path):
    archive_path = Path(archive_path).resolve()
    problems = []
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('HANDOFF-MANIFEST.json'))
        names = set(archive.namelist()) - {'HANDOFF-MANIFEST.json'}
        if names != set(manifest['files']):
            problems.append('Archive members differ from the manifest')
        for name, entry in manifest['files'].items():
            if name.startswith('/') or '..' in Path(name).parts:
                problems.append('Unsafe member path: ' + name)
                continue
            if name in names and hashlib.sha256(archive.read(name)).hexdigest() != entry['sha256']:
                problems.append('Hash mismatch: ' + name)
    return {'archive': str(archive_path), 'files': len(manifest['files']), 'problems': problems, 'ok': not problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('create')
    make.add_argument('destination', type=Path)
    check = sub.add_parser('verify')
    check.add_argument('archive', type=Path)
    args = parser.parse_args()
    result = create(args.destination) if args.command == 'create' else verify(args.archive)
    print(json.dumps(result, indent=2))
    return 0 if result.get('ok', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
