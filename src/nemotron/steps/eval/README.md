# Nemotron Evaluation

Use this category before and after any quality-changing stage. Evaluation is
how you know SFT, RL, optimization, or conversion actually helped.

## Developer Journey

1. Decide the target: a hosted endpoint or a Megatron checkpoint.
2. Pick task IDs your endpoint can actually serve (chat vs completions).
3. Smoke with `limit_samples=1` before any full run.
4. Re-evaluate against the same task set after each pipeline change.
5. Inspect a few generations, not only aggregate scores.

## Steps

| Need | Step | Input | Output |
|---|---|---|---|
| Run a catalog benchmark suite via NeMo Evaluator Launcher | [`eval/model_eval`](model_eval/README.md) | hosted endpoint or `checkpoint_megatron` | `eval_results` |
| Generate + score Native-reasoning W1.1 benches (MGSM, PolyMath, MMLU-ProX, VMLU, …) | [`eval/mr_bench`](mr_bench/README.md) | hosted OpenAI-compatible endpoint | `eval_results` + `eval_traces_jsonl` |
| Run SEATau-vi / τ-bench 2.0 HI/EN (full domains) with GPT-5.5 user-sim | [`eval/mr_agent`](mr_agent/README.md) | hosted agent endpoint | `eval_results` |
| Score think-language L2 (`in_language_95`) on existing traces | [`eval/mr_l2`](mr_l2/README.md) | `eval_traces_jsonl` (`traces.jsonl`) | `eval_results` |

## Data And Artifact Flow

```text
hosted endpoint
  -> eval/model_eval -> eval_results          # catalog acc (lm-eval / launcher)

hosted endpoint
  -> eval/mr_bench -> traces.jsonl + metrics.json   # W1.1 product numbers
       -> eval/mr_l2 -> L2 analysis

hosted agent + GPT-5.5 user-sim
  -> eval/mr_agent -> simulations/results.json + pass@1

checkpoint_megatron (concrete iter_*)
  -> eval/model_eval (NeMo Evaluator Launcher deploys + runs)
  -> eval_results

traces.jsonl (raw generations with <think>…</think>)
  -> eval/mr_l2
  -> eval_results (in_language_95_rate, fail-mode analysis)
```

If a Megatron-Bridge step produced the checkpoint, point at the concrete
`iter_*` directory, not the parent run dir. If the consumer is HF-native,
convert with `convert/megatron_to_hf` first instead of pointing eval at the
Megatron path.

## Guardrails

- Don't compare scores across endpoint types or generation settings.
- Don't ship hosted API keys in YAML; store the variable name only.
- Don't treat a `limit_samples=1` smoke as evidence of model quality.
- Don't read a low L2 as English-only CoT until `eval/mr_l2` analysis.json is inspected.
- Don't mix `eval/model_eval` catalog accuracy with `eval/mr_bench` W1.1 numbers.
- Don't clip τ-bench Hindi/English to SEATau's three domains; headline is five.
