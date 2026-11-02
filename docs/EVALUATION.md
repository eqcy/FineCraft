# Local evaluation

FineCraft serves actions over the official MetaFine HTTP protocol. The public
benchmark owns the simulator, task definitions and metric calculation.

## Benchmark and environment

Clone [the official benchmark](https://github.com/Hiangx-robotics/MetaFine-Challenge_CoRL)
and check out the version recorded in `evaluation/benchmark.json`:

```bash
git clone https://github.com/Hiangx-robotics/MetaFine-Challenge_CoRL.git
cd MetaFine-Challenge_CoRL
git checkout 232538592ef4c85c0fdd37f1fccf8d868ee150aa
```

Follow that revision's installation and asset instructions. The benchmark and
policy server can run in separate Python environments and communicate over HTTP.
The server's inference requirements do not install the benchmark simulator.

The scored workspace replaced two imports of LeRobot's image/state key constants
with the same literal strings. To use that small compatibility change:

```bash
git apply /path/to/FineCraft/evaluation/remote_policy_constants.patch
```

This patch changes no task definitions, rollout logic or score computation.
All other public evaluator, environment and task-configuration sources match
the pinned upstream revision. Local seed files from the workspace are excluded.

## Seeds

Generate development seeds with the benchmark's `eval.select_eval_seeds`, using
the appropriate YAML under `eval/configs/`. Follow its instructions for locating
training demonstration metadata and selecting disjoint development scenes.
Seed generation can run the benchmark expert and is a separate, explicit step.
No seed generation or simulation is triggered by downloading the model bundle.
The organizer's hidden seeds and private evaluation salt are not distributed.

## Run

Start the FineCraft server in its inference environment, with
`FINECRAFT_MODEL_DIR` set to the model bundle. In the benchmark environment,
call the launcher from this repository:

```bash
python /path/to/FineCraft/tools/evaluate.py --task T1 --benchmark-dir /path/to/MetaFine-Challenge_CoRL --seeds /path/to/local_dev_seeds.json --record-dir /path/to/results/T1 --policy-url http://127.0.0.1:8080
```

Use T1, T2, T3, T4 or T5 for `--task`. `--mode` accepts `perception`,
`understanding` or `both`. `--n-seeds` restricts the supplied local seed list.
`--dry-run` prints the exact command without launching the evaluator.

To aggregate summaries, use `python -m utils.eval_report` from the benchmark
repository and follow its `eval/README.md`. Preserve the distinction between
local development results and the official final leaderboard scores.
