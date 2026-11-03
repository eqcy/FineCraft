# FineCraft

**FineCraft: Enhancing VLA Models for Fine-Grained Manipulation**

Inference code for the final HITSZ Robotics entry in the MetaFine Challenge.
The entry placed **4th**, with an overall official score of **0.878**.
This release contains the five policies used for the final leaderboard results.

[Paper](https://arxiv.org/abs/2611.03163) · [PDF](https://arxiv.org/pdf/2611.03163) · [Code](https://github.com/eqcy/FineCraft) · [Model weights](https://huggingface.co/hitsz-robotics/FineCraft)

| Task | Released policy | Scored HF revision | Official score* |
| --- | --- | --- | ---: |
| T1: bottle-part grasping | body r25 + cap r3, continuity v4 | `cf5b770c0608` | 0.796 |
| T2: mug transport | pi0.5 LoRA, step 29999 | `25cb20a04623` | 1.000 |
| T3: switch toggling | pi0.5 stage4 coverage3000 | `cf5b770c0608` | 0.925 |
| T4: cube sorting | pi0.5 LoRA, step 29999 | `25cb20a04623` | 0.987 |
| T5: letter insertion | general/finishing/correction combination | `cf5b770c0608` | 0.681 |

\* Scores rounded to three decimals. Full source revisions and file digests are
recorded in `model_manifest.json`. Official results:
[MetaFine leaderboard](https://robofinemani2026.github.io/leaderboard.html).

## Setup

Use Linux, Python 3.11, and an NVIDIA GPU with a CUDA 12-compatible driver.
The unified server caches the three pi0.5 parameter sets in host RAM and loads
one task policy on the GPU at a time. Install the pinned inference dependencies:

```bash
python -m pip install -r requirements.txt
```

The required OpenPI inference sources are bundled under `vendor/`.
Checkpoint formats, normalization, preprocessing, action chunk execution,
random-number handling and numerical settings follow the scored submission.
The release separates file locations and removes unused backend routes.

## Model assets

The model bundle contains 79 files, approximately 23.49 GB. Keep its directory
structure intact, including Orbax metadata, normalization statistics, tokenizer,
T1 deployment options and all seven T5 component checkpoints.

If the bundle is already available locally, verify it directly:

```bash
python tools/verify_models.py --model-dir /path/to/FineCraft-models
```

For a published HF copy, supply the actual repository ID and its immutable
commit SHA to the downloader. These are arguments rather than hard-coded URLs:

```bash
python tools/download_models.py --repo-id hitsz-robotics/FineCraft --revision d59585bb9a5bc97630447528d6fde80ae769607e --model-dir /path/to/FineCraft-models
```

The command pins the published model bundle. The downloader verifies every
asset against the digests bundled with this code.

## Run the policy server

```bash
export FINECRAFT_MODEL_DIR=/path/to/FineCraft-models
python policy_server.py --port 8080
```

The server exposes the challenge's `/health`, `/reset` and `/act` endpoints.
Inputs are two 512 x 512 uint8 RGB views and nine joint positions. Outputs
contain seven arm commands and one gripper command. Select the task and
instruction through `/reset`, then send observations through `/act`.

`python policy_server.py --self-test` checks the real policies with synthetic
observations and requires the complete model bundle and inference environment.

## Evaluation

Use the pinned public challenge evaluator through its HTTP policy interface.
See [evaluation instructions](docs/EVALUATION.md). The supplied launcher connects
the evaluator to this policy server without copying the simulator into this repo.

## Citation

If you use FineCraft in your research, please cite our technical report:

**[FineCraft: Enhancing VLA Models for Fine-Grained Manipulation](https://arxiv.org/abs/2611.03163)** (arXiv:2611.03163, 2026).

```bibtex
@misc{zhang2026finecraft,
  title         = {{FineCraft}: Enhancing {VLA} Models for Fine-Grained Manipulation},
  author        = {Shuaike Zhang and Ziheng Quan and Junhao Ran and
                   Muhe Ding and Shaokun Wang and Jianlong Wu},
  year          = {2026},
  eprint        = {2611.03163},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.03163}
}
```

Citation metadata: [BibTeX](CITATION.bib) · [CITATION.cff](CITATION.cff).

## Licenses

The original MIT notice is retained for the MetaFine source components.
OpenPI, DINOv2 and Diffusion Policy notices are retained alongside their
components; see [third-party notices](THIRD_PARTY.md). Model and tokenizer
terms are documented separately in the HF bundle's `LICENSES.md`.
