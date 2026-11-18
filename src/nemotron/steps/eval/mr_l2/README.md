# eval/mr_l2

CPU-only **native-reasoning L2** scoring for already-generated traces.

L2 is `in_language_95`: after stripping maths, code, MCQ option recap, and
copied question/option text, take the word-weighted share of *scored* thinking
remainders in the target language. The item passes when that share is **0.95 or
higher**. Empty or fully-allowed remainders do not enter the denominator
(`share = 1.0`). Accuracy is independent of L2.

This step does **not** deploy a model. Point it at `traces.jsonl` from
`eval/model_eval` or a native generation job.

## Files

- `step.py` — YAML entry point
- `runtime.py` — load traces, score, write artifacts
- `lid.py` / `strip.py` — word-weighted LID after strip
- `analyze.py` — fail-mode labels (English CoT, English closer, mixed share, loop)
- `config/tiny.yaml` — five synthetic traces, not a quality claim

## Inputs And Outputs

Consume `eval_traces_jsonl`: a JSONL file, or a directory containing
`traces.jsonl`. Each row needs `raw_output` (Nano `<think>…</think>` + answer)
or `thinking` plus `answer_span`. Optional fields: `source`, `task`,
`question` / `user_message`, `options`, `gold`, `thinking_language_target`,
`n_tokens`.

Produces `eval_results` in `output_dir`:

| File | Contents |
| --- | --- |
| `scored.jsonl` | Input rows plus L2 fields |
| `metrics.json` | `in_language_95_rate`, share buckets, per-source rates |
| `analysis.json` | Fail-mode counts and a few leak snippets |
| `l2_fails.jsonl` | Rows that missed the 0.95 threshold |

Logged per item: `thinking_lang_share`, `in_language_95`, `answer_language`,
`closed`, `truncated`, `loop`, `l2_primary`.

## Run

Smoke the synthetic fixture:

```bash
uv run nemotron steps run eval/mr_l2 -c tiny --dry-run
uv run python src/nemotron/steps/eval/mr_l2/step.py -c src/nemotron/steps/eval/mr_l2/config/tiny.yaml
```

Score a real run:

```bash
uv run nemotron steps run eval/mr_l2 -c default \
  input=/path/to/traces.jsonl \
  output_dir=/path/to/output/mr_l2 \
  thinking_language_target=vi
```

## Guardrails

- Do not treat `config/tiny.yaml` as a model-quality number.
- Do not compare L2 across different strip/LID code revisions.
- A low L2 on VMLU / MMLU-ProX-vi is often mixed option/formula/English-closer
  mass, not a full switch to English CoT. Read `analysis.json` before changing
  training.
- Lone MCQ letters and numeric/LaTeX answers skip answer-language LID.
- Unaccented Vietnamese still counts as Vietnamese.
