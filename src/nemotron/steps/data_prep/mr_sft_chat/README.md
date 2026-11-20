# data_prep/mr_sft_chat

Chat-format SFT data from one language subset of
[Tiny Aya multilingual reasoning](https://huggingface.co/datasets/CohereLabs/tiny-aya-l2-thinker-multilingual-reasoning).

Each row of the dataset holds a question, a thinking trace and an answer in the
subset's language, and the English originals of all three. This step writes two
examples per row:

| Example | User turn | Assistant reasoning | Assistant answer |
| --- | --- | --- | --- |
| Subset language (`vi`) | `question` | `thinking` | `output` |
| English | `original_question` | `original_thinking` | `original_output` |

Only the configured subset is read; no other language enters the output.

## Inputs And Outputs

- Consume one subset of the HuggingFace dataset (`subset: vi` by default).
- Produce `training_jsonl` under `output_dir`:

| File | Contents |
| --- | --- |
| `train.jsonl`, `valid.jsonl` | One example per line: `id`, `language`, `domain`, `messages`. The assistant message carries `reasoning_content`. |
| `blend_train.json`, `blend_valid.json` | Single-dataset blends for `data_prep/sft_packing`. |
| `stats.json` | Per split and language: examples kept, dropped as too long, dropped as empty. |

`valid.jsonl` comes from the dataset's `test` split and stays out of the training shards.

## Length Filter

`data_prep/sft_packing` cuts an example that is longer than `pack_size`, which
removes the end of the answer and the end-of-turn token. This step drops such
examples instead: anything over `max_tokens`, measured with the training
tokenizer and its chat template. Keep `max_tokens` a little under `pack_size`
and confirm `num_truncated_to_pack_size` is 0 in the packing statistics.

## Run It

```bash
uv run --frozen nemotron steps run data_prep/mr_sft_chat --dry-run
uv run --frozen nemotron steps run data_prep/mr_sft_chat --batch lepton_mr_sft_prep
```

Then pack both splits and train the adapter:

```bash
uv run --frozen nemotron steps run data_prep/sft_packing -c nano4b_mr_vi       --batch lepton_mr_sft_prep
uv run --frozen nemotron steps run data_prep/sft_packing -c nano4b_mr_vi_valid --batch lepton_mr_sft_prep
uv run --frozen nemotron steps run peft/megatron_bridge  -c nano4b_mr_vi       --batch lepton_nano4b_mr_lora
```

## Guardrails

- The dataset is licensed CC BY-NC 4.0 (non-commercial). Check that before a
  model trained on it leaves research use.
- Rebuild and repack after changing the tokenizer, `max_tokens`, or the subset.
- Do not pack `blend_valid.json` into the training output directory.
