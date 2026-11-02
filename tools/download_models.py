"""Download and verify an explicitly selected HF release; never uploads files."""
import argparse
import re
from model_files import load_manifest, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True, help='Public HF model repository ID')
    parser.add_argument('--revision', required=True, help='Immutable 40-character HF commit SHA')
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo_id):
        parser.error('Expected an owner/repository ID.')
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('Pin the published model release with its full commit SHA.')
    from huggingface_hub import snapshot_download
    manifest = load_manifest()
    snapshot_download(repo_id=args.repo_id, repo_type='model', revision=args.revision,
                      local_dir=args.model_dir,
                      allow_patterns=[e['path'] for e in manifest['files']] + ['manifest.json'])
    verify(args.model_dir)


if __name__ == '__main__':
    main()
