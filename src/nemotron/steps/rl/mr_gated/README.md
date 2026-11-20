# rl/mr_gated

Frozen **hard-gate** reward for native-reasoning Approach 1 (unassisted GRPO).

```
if thinking-language fails:                            -1
elif closed AND answer-language fails:                 -1
elif answer matches label AND closed:                  1
else:                                                  0
```

Thinking-language is L2 (`in_language_95`) from [`eval/mr_l2`](../../eval/mr_l2/README.md).
Completely empty thinking fails the gate; maths-only remainders are
language-neutral and do not. Thinking-language fail is always −1. Empty
answer after a **closed** think is −1. Unclosed or truncated native thinking
is **0**, so GRPO can learn to close instead of treating a long Vietnamese
think like English. The gate parser reads the **answer span only**
(no `\boxed` / `Đáp án` fallback from `<think>`). The scorer uses a frozen
**label**, never eval gold. Unlabeled items are dropped by the data exporter,
not scored 0.

This step validates the reward on existing rollouts. Policy optimization still
goes through [`rl/nemo_rl/rlvr`](../nemo_rl/rlvr/README.md) (GRPO), with this
function as the verifier.

## Files

- `reward.py` — `gated_reward(raw, row)` used by GRPO callers
- `schema.py` — gated prompt contract (no gold, hint, or eval IDs)
- `data.py` — gated training rows: GSM8K in Vietnamese plus a share of English
- `nemo_rl_plugin.py` — registers the dataset and the reward environment with NeMo-RL
- `runtime.py` — join traces to prompts, write `rewards.jsonl`
- `config/tiny.yaml` — three synthetic rollouts, not a quality claim

## Inputs And Outputs

| Artifact | Role |
| --- | --- |
| `gated_prompt_jsonl` (optional) | `unassisted_user` + `label`, `path=train`, `arm=approach1_gated` |
| `eval_traces_jsonl` | `raw_output` (or think/answer spans), joined on `item_id` |
| `eval_results` | `rewards.jsonl`, `metrics.json` (`n_plus` / `n_zero` / `n_minus`) |

If `prompts` is omitted, each trace row must itself satisfy the gated contract.

## Run

```bash
uv run python src/nemotron/steps/rl/mr_gated/step.py \
  -c src/nemotron/steps/rl/mr_gated/config/tiny.yaml
```

```bash
uv run nemotron steps run rl/mr_gated -c default \
  input=/path/to/traces.jsonl \
  prompts=/path/to/gated_prompts.jsonl \
  output_dir=/path/to/output/mr_gated
```

Train with GRPO (Nemotron 3 Nano 4B, one 8-GPU node plus the Ray head):

```bash
uv run --frozen nemotron steps run rl/nemo_rl/rlvr -c nano4b_mr_gated \
  --batch lepton_nano4b_mr_gated_sft
```

`config/nano4b_mr_gated.yaml` under `rl/nemo_rl/rlvr` lists
`nemotron.steps.rl.mr_gated.nemo_rl_plugin` in `env.plugins`, so every rollout
is scored by `gated_reward`. Training data is every question of
`namfam/gsm8k-vietnamese` `train` plus a quarter of their English originals,
picked evenly across reasoning steps, operations and question length; the
`val` split is used for validation and the GSM8K test split is never loaded.
The profile lives in the repo-root `env.toml` (gitignored).

Call the reward directly:

```python
from nemotron.steps.rl.mr_gated import gated_reward

out = gated_reward(raw_generation, prompt_row)
# out["reward"] in {1, 0, -1}
```

## Guardrails

- Do not put `gold`, `hint`, or `guided_user` on gated prompts.
- Do not treat `config/tiny.yaml` as a training result.
- Validate `n_plus` / `n_zero` / `n_minus` on a tiny rollout set before scaling
  `rl/nemo_rl/rlvr`.
- Keep the L2 formula in `eval/mr_l2`; this step consumes it, fails empty
  thinking, and parses the answer span only.
