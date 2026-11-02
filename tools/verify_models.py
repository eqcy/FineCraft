"""Verify the final model bundle without importing ML frameworks."""
import argparse
from model_files import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--size-only', action='store_true', help='Skip digest computation')
    args = parser.parse_args()
    verify(args.model_dir, size_only=args.size_only)


if __name__ == '__main__':
    main()
