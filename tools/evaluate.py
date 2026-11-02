"""Connect FineCraft to the pinned public MetaFine evaluator."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {'T1': 'grasp_part', 'T2': 'grasp_move_mug', 'T3': 'toggle_switch',
         'T4': 'put_blocks', 'T5': 'insert_letter'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True, help='Your local development seed JSON')
    parser.add_argument('--record-dir', type=Path, required=True)
    parser.add_argument('--policy-url', default='http://127.0.0.1:8080')
    parser.add_argument('--mode', choices=['perception', 'understanding', 'both'], default='both')
    parser.add_argument('--n-seeds', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Print the command without running evaluation')
    args = parser.parse_args()
    benchmark = args.benchmark_dir.expanduser().resolve()
    command = [sys.executable, '-m', 'eval.eval_' + TASKS[args.task],
               '--policy-url', args.policy_url, '--seeds', str(args.seeds.expanduser().resolve()),
               '--record-dir', str(args.record_dir.expanduser().resolve()), '--mode', args.mode]
    if args.n_seeds is not None:
        if args.n_seeds < 1:
            parser.error('--n-seeds must be positive.')
        command += ['--n-seeds', str(args.n_seeds)]
    print('Working directory: ' + str(benchmark))
    print(shlex.join(command))
    if args.dry_run:
        return
    if not args.seeds.expanduser().is_file():
        parser.error('The local development seed JSON does not exist.')
    expected = json.loads((ROOT / 'evaluation/benchmark.json').read_text())['revision']
    result = subprocess.run(['git', '-C', str(benchmark), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != expected:
        parser.error('Check out the benchmark revision in evaluation/benchmark.json first.')
    raise SystemExit(subprocess.run(command, cwd=benchmark).returncode)


if __name__ == '__main__':
    main()
