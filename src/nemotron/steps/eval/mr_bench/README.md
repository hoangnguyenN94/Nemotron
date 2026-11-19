# eval/mr_bench

W1.1 Native-reasoning suite against a hosted OpenAI-compatible endpoint.
This step **generates** traces and **scores** them (accuracy, DW-ACC, think-language
L2, closed). It does not deploy the model.

Catalog-comparable numbers (lm-eval MMLU-ProX, IFEval, MILU, GSM8K-Indic)
stay on [`eval/model_eval`](../model_eval/README.md). Do not mix those scores
with this step.

Agent benches on the [Nemotron Adoption Eval Sheet](https://docs.google.com/spreadsheets/d/1gWhZngquu4fngp1zop73wu1F_zsssBSS5z3vHuopBYo/edit?gid=1841659604#gid=1841659604)
(SEATau-vi, τ-bench 2.0 Hindi/English) are **not** this step.

L2 on traces this step already wrote: [`eval/mr_l2`](../mr_l2/README.md).

## Setup

1. Host the checkpoint (vLLM / NIM) with thinking enabled. Direct mode never
   starts the server.
2. Python extras for live benches: `pyarrow`, `sympy`, `huggingface_hub`.
   MILU-1.5 needs a Hugging Face token with access to `ai4bharat/MILU-1.5`.
   VMLU public test has no gold — this step writes `submission.csv` for
   https://vmlu.ai.
3. Data is fetched on demand into `data_root` (not committed):

   | Config | Hub |
   |---|---|
   | `mgsm_vi` / `mgsm_en` | CohereLabs/global-mgsm |
   | `gsm8k_hi` | nvidia/GSM8K-Hi |
   | `gsm8k_en` | openai/gsm8k (test only) |
   | `mmlu_prox_*` | li-lab/MMLU-ProX |
   | `polymath_vi` / `polymath_en` | Qwen/PolyMath |
   | `polymath_hi` | tum-nlp/PluraMath |
   | `vmlu` | tridm/VMLU |
   | `milu_1_5` | ai4bharat/MILU-1.5 (gated) |
   | `ifeval` | google/IFEval |
   | `m_arena_hard_*` | CohereLabs/m_ArenaHard |

   Export `HF_TOKEN` for gated sets. Cache under a durable mount (`HF_HOME`),
   not container disk.

4. Smoke with `tiny.yaml` (synthetic, no Hub, no GPU). Never treat tiny as a
   quality number.

## Run

```bash
# plumbing (no endpoint)
uv run nemotron steps run eval/mr_bench -c tiny --dry-run
uv run python src/nemotron/steps/eval/mr_bench/step.py \
  -c src/nemotron/steps/eval/mr_bench/config/tiny.yaml

# cheap iteration (MGSM-vi, 250 items)
export EVAL_ENDPOINT_URL=https://<host>/v1
export EVAL_MODEL_HANDLE=<served-model-name>
export NVIDIA_API_KEY=<token>   # or EVAL_API_KEY_NAME + that env var
uv run nemotron steps run eval/mr_bench -c mgsm_vi --batch <profile> \
  output_dir=$EVAL_RESULTS_DIR/mgsm_vi

# official PolyMath-vi (DW-ACC, T=0.6, 65k)
uv run nemotron steps run eval/mr_bench -c polymath_vi --batch <profile> \
  output_dir=$EVAL_RESULTS_DIR/polymath_vi

# official 5-shot MMLU-ProX-vi vs W1.1 0-shot contract
uv run nemotron steps run eval/mr_bench -c mmlu_prox_vi
uv run nemotron steps run eval/mr_bench -c mmlu_prox_vi prompt_style=contract

# Default (free think) vs native (forced prefix) — Adoption Eval Sheet columns
uv run nemotron steps run eval/mr_bench -c mgsm_vi mode=free
uv run nemotron steps run eval/mr_bench -c mgsm_vi mode=forced \
  thinking_language_target=vi vllm_prefill=true

# Hindi / English twins (named configs)
uv run nemotron steps run eval/mr_bench -c gsm8k_hi
uv run nemotron steps run eval/mr_bench -c mgsm_en
uv run nemotron steps run eval/mr_bench -c polymath_hi
uv run nemotron steps run eval/mr_bench -c mmlu_prox_en
uv run nemotron steps run eval/mr_bench -c milu_1_5
uv run nemotron steps run eval/mr_bench -c ifeval
uv run nemotron steps run eval/mr_bench -c m_arena_hard_vi

# L2 analysis on the traces this step wrote
uv run nemotron steps run eval/mr_l2 -c default \
  input=$EVAL_RESULTS_DIR/mgsm_vi/traces.jsonl \
  thinking_language_target=vi
```

Sampling defaults match the W1.1 contract (`T=1.0`, `top_p=0.95`,
`max_tokens=32768`) except official MMLU-ProX (`T=0`, 2048) and official
PolyMath (`T=0.6`, `top_k=20`, 65536).

Outputs under `output_dir`: `traces.jsonl`, `metrics.json`, `config.json`,
and `submission.csv` for VMLU.

Sheet mapping (Multilingual Reasoning tab):

| Sheet row | `benchmark=` |
|---|---|
| MMLU-ProX-Vi / hi / en | `mmlu_prox_vi`, `mmlu_prox_hi`, `mmlu_prox_en` |
| MGSM-vi / en | `mgsm_vi`, `mgsm_en` |
| PolyMath-vi / en | `polymath_vi`, `polymath_en` |
| PluraMath-hi | `polymath_hi` |
| VMLU | `vmlu` |
| GSM8K-Hi / en | `gsm8k_hi`, `gsm8k_en` |
| MILU-1.5 | `milu_1_5` |
| m-ArenaHard-vi / hi | `m_arena_hard_vi`, `m_arena_hard_hi` (generate only; pairwise judge is a follow-up) |
| IFEval | `ifeval` (generate + L2; prompt-level strict acc stays on `eval/model_eval`) |

## Guardrails

- Train split of GSM8K is not an eval bench.
- Do not compare `prompt_style=contract` to `eval/model_eval -c mmlu_prox_chat`.
- `max_samples` / tiny is a subset, not a full-split score.
- Empty think fails the hard gate; maths-only remainders do not.
- `vllm_prefill` is vLLM-only. NIM/OpenAI 400 if you send `continue_final_message`.
- Do not put raw API keys in YAML.
