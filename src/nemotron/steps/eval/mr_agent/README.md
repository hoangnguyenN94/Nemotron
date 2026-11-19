# eval/mr_agent

SEATau-vi and τ-bench 2.0 (Hindi / English, **full five domains**) against a
hosted OpenAI-compatible **agent** plus a GPT-5.5 **user simulator**.

This step clones pinned harness SHAs, applies the Native-reasoning think-prefix
overlay, runs `uv run tau2`, and writes `pass@1`. It does **not** deploy the
model.

Single-turn W1.1 benches (MGSM, PolyMath, MMLU-ProX, …):
[`eval/mr_bench`](../mr_bench/README.md). Catalog lm-eval:
[`eval/model_eval`](../model_eval/README.md). PolyGym eval/v1 (τ²-278 / AReaL
filters) is **not** this step.

Sheet: [Nemotron Adoption Eval Sheet — Multilingual Reasoning](https://docs.google.com/spreadsheets/d/1gWhZngquu4fngp1zop73wu1F_zsssBSS5z3vHuopBYo/edit?gid=1841659604#gid=1841659604).

## Setup

1. Host the checkpoint (vLLM / NIM) with thinking enabled. Direct mode never
   starts the server.
2. `git` and `uv` on PATH. Live runs also need `NGC_API_KEY` for GPT-5.5 on
   `https://inference-api.nvidia.com/v1`.
3. First live run clones into `data_root` (not committed):

   | Config | Harness | SHA |
   |---|---|---|
   | `seatau_vi` | [SEACrowd/SEATauBench](https://github.com/SEACrowd/SEATauBench) | `312f3e6eb87c94ad2554a74cc4f4b96baa6f89f9` |
   | `tau2_hi` / `tau2_en` | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | `033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f` |

   SEATau also downloads FastText `lid.176.bin` (language_correctness). Cache
   clones on a durable mount, not container disk.

4. Smoke with `tiny.yaml` (dry-run, no clone, no GPU). Never treat tiny as a
   quality number.

## Run

```bash
# plumbing (no endpoint, no clone)
uv run nemotron steps run eval/mr_agent -c tiny --dry-run

export EVAL_ENDPOINT_URL=https://<host>/v1
export EVAL_MODEL_HANDLE=<served-model-name>
export NGC_API_KEY=<token>   # GPT-5.5 user-sim, not the agent key

# SEATau-vi (airline/retail/telecom, l2_interaction, 3 trials)
uv run nemotron steps run eval/mr_agent -c seatau_vi --batch <profile> \
  output_dir=$EVAL_RESULTS_DIR/seatau_vi

# Default vs native (Adoption Eval Sheet columns)
uv run nemotron steps run eval/mr_agent -c seatau_vi mode=free
uv run nemotron steps run eval/mr_agent -c seatau_vi mode=forced \
  vllm_prefill=true

# Paper-primary L2 Domain (everything in Vietnamese)
uv run nemotron steps run eval/mr_agent -c seatau_vi scenario=l2_domain

# τ-bench 2.0 Hindi / English — full 5 domains, 4 trials (not text-v1 50×3)
uv run nemotron steps run eval/mr_agent -c tau2_hi
uv run nemotron steps run eval/mr_agent -c tau2_en
```

Forced openers match [`eval/mr_l2`](../mr_l2/README.md) (`vi` / `hi` / `en`).
`vllm_prefill` is vLLM-only.

Outputs under `output_dir`: `commands.json`, `metrics.json`, copied
`simulations/<save-to>/results.json`.

| Sheet row | `-c` | Protocol |
|---|---|---|
| SEATau-vi | `seatau_vi` | airline 50, retail 114, telecom 114; default `l2_interaction` (filled sheet cells); `scenario=l2_domain` for the paper-primary setting; 3 trials |
| τ-bench 2.0 Hindi (full domains) | `tau2_hi` | airline 50, retail 114, telecom 114, banking_knowledge 97, intake 200 (`--elicitation-strategy scaffolded`); 4 trials; `--benchmark none` (not released text-v1) |
| τ-bench English (full domains) | `tau2_en` | same five domains; not the PolyGym 100-task subset |

`metrics.json`:

- `pass1` — headline (all domains in the run)
- `cs_overlap_pass1` — airline/retail/telecom slice (τ-bench vs SEATau overlap)
- `language_correctness` — SEATau FastText LID, when present
- `by_domain` — per-domain `pass1` / `n_tasks`

## Guardrails

- Do not clip Hindi/English to SEATau’s three domains.
- Do not use `--benchmark text` (text-v1 50×3 freeze) for sheet HI/EN numbers.
- Do not use PolyGym eval/v1 or the official 100-task English subset as ΔEN.
- Train split / PolyGym pool overlap is not this harness.
- Empty think fails the Native-reasoning hard gate elsewhere; here the scorer
  is tau2 task success plus SEATau `language_correctness`.
- Do not put raw API keys in YAML.
- `max_tasks` / tiny is a subset, not a full-split score.
