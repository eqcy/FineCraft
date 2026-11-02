"""Read and verify the final-leaderboard model manifest without loading tensors."""
import hashlib
import json
from pathlib import Path
import re

MANIFEST = Path(__file__).resolve().parents[1] / 'model_manifest.json'


def load_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    names = set()
    for entry in manifest['files']:
        name = entry['path']
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or name in names:
            raise ValueError('Invalid manifest path: ' + name)
        if not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Invalid digest for ' + name)
        names.add(name)
    return manifest


def verify(directory, *, size_only=False):
    directory = Path(directory).expanduser().resolve()
    manifest = load_manifest()
    for entry in manifest['files']:
        path = directory / entry['path']
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError('Missing or incomplete model file: ' + entry['path'])
        if not path.resolve().is_relative_to(directory):
            raise ValueError('Model path escapes the model directory: ' + entry['path'])
        if not size_only:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError('SHA256 mismatch: ' + entry['path'])
    expected = {entry['path'] for entry in manifest['files']}
    for prefix in ('checkpoints', 't1_best', 'tokenizer', 'models'):
        for path in (directory / prefix).rglob('*'):
            if path.is_file() and path.relative_to(directory).as_posix() not in expected:
                raise ValueError('Unlisted model asset: ' + str(path.relative_to(directory)))
    print(f"Verified {manifest['asset_count']} model files ({manifest['asset_bytes'] / 1e9:.2f} GB; "
          + ('sizes only).' if size_only else 'SHA256).'))
    return manifest
