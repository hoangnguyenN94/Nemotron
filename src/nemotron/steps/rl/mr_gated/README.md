# rl/mr_gated

Frozen **hard-gate** reward for native-reasoning Approach 1 (unassisted GRPO).

```
if thinking-language fails OR answer-language fails:  -1
elif answer matches label AND closed:                 1
else:                                                 0
```

Thinking-language is L2 (`in_language_95`) from [`eval/mr_l2`](../../eval/mr_l2/README.md).
The scorer uses a frozen **label**, never eval gold. Unlabeled items are dropped
by the data exporter, not scored 0.

This step validates the reward on existing rollouts. Policy optimization still
goes through [`rl/nemo_rl/rlvr`](../nemo_rl/rlvr/README.md) (GRPO), with this
function as the verifier.

## Files

- `reward.py` — `gated_reward(raw, row)` used by GRPO callers
- `schema.py` — gated prompt contract (no gold, hint, or eval IDs)
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

Call the reward from GRPO:

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
- Keep the L2 formula in `eval/mr_l2`; this step only consumes it.
