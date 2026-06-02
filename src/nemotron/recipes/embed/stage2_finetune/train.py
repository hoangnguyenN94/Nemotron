#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "embed/finetune"
# image = "nvcr.io/nvidia/pytorch:25.12-py3"
# setup = "PyTorch pre-installed. Stage dependencies resolved via UV at runtime."
#
# [tool.runspec.run]
# launch = "direct"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 1
# ///
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tuning script for embedding models.

Fine-tunes an embedding model using contrastive learning with prepared
training data (from stage1_data_prep).

Usage:
    # With default config
    nemotron embed finetune -c default

    # With custom config
    nemotron embed finetune -c /path/to/config.yaml

    # With CLI overrides
    nemotron embed finetune -c default model.pretrained_model_name_or_path=...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from nemo_runspec.config.pydantic_loader import RecipeSettings, load_config, parse_config_and_overrides

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "default.yaml"

# Use NEMO_RUN_DIR for output when running via nemo-run
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))


class FinetuneConfig(RecipeSettings):
    """Fine-tuning configuration for embedding models."""

    model_config = ConfigDict(extra="forbid")

    # Model settings
    base_model: str = Field(default="nvidia/llama-nemotron-embed-1b-v2", description="Base embedding model to fine-tune.")

    # Data paths
    train_data_path: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage1_data_prep/train_mined.automodel_unrolled.json", description="Path to training data file.")

    # Output settings
    checkpoint_dir: Path = Field(default_factory=lambda: _OUTPUT_BASE / "output/embed/stage2_finetune/checkpoints", description="Directory for saving checkpoints.")

    # Training hyperparameters
    num_epochs: int = Field(default=3, gt=0, description="Number of training epochs.")
    global_batch_size: int = Field(default=128, gt=0, description="Global batch size across all GPUs.")
    local_batch_size: int = Field(default=4, gt=0, description="Per-GPU batch size.")
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate.")
    lr_warmup_steps: int = Field(default=1, ge=0, description="Learning rate warmup steps.")
    lr_decay_style: Literal["cosine", "linear"] = Field(default="cosine", description="LR decay schedule (cosine, linear).")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay for optimizer.")

    # Model architecture
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] | None = Field(default=None, description="Attention implementation (sdpa, flash_attention_2, eager). None auto-detects.")
    train_n_passages: int = Field(default=5, ge=2, description="Number of passages per query during training (1 pos + n-1 neg).")
    pooling: Literal["avg", "cls", "last"] = Field(default="avg", description="Pooling strategy for embeddings.")
    l2_normalize: bool = Field(default=True, description="Whether to L2 normalize embeddings.")
    temperature: float = Field(default=0.02, gt=0, description="Temperature for contrastive loss.")

    # Tokenization
    query_max_length: int = Field(default=512, gt=0, description="Maximum query sequence length.")
    passage_max_length: int = Field(default=512, gt=0, description="Maximum passage sequence length.")
    query_prefix: str = Field(default="query:", description="Prefix for query inputs.")
    passage_prefix: str = Field(default="passage:", description="Prefix for passage inputs.")

    # Checkpointing
    checkpoint_every_steps: int = Field(default=100, gt=0, description="Save checkpoint every N steps.")
    val_every_steps: int = Field(default=100, gt=0, description="Run validation every N steps.")


def _count_training_examples(train_data_path: Path) -> int:
    """Count the number of training examples in a training data file.

    Args:
        train_data_path: Path to training JSON file.

    Returns:
        Number of training examples.
    """
    with open(train_data_path) as f:
        data = json.load(f)
    return len(data.get("data", []))


def _warn_if_negatives_sparse(train_data_path: Path, train_n_passages: int) -> None:
    """Warn if training data has fewer negatives than train_n_passages requires."""
    needed = train_n_passages - 1  # 1 positive + (n-1) negatives
    with open(train_data_path) as f:
        data = json.load(f)
    records = data.get("data", [])
    if not records:
        return
    neg_counts = [len(r.get("neg_doc", [])) for r in records[:200]]
    median_neg = sorted(neg_counts)[len(neg_counts) // 2]
    if median_neg < needed:
        print(
            f"Warning: train_n_passages={train_n_passages} needs {needed} negatives per query,\n"
            f"         but training data has a median of {median_neg}.\n"
            f"         Consider increasing hard_negatives_to_mine in stage1 prep or\n"
            f"         reducing train_n_passages.",
            file=sys.stderr,
        )
        print()


def _normalize_hf_model_name(model_name: str) -> str:
    """Convert a Hugging Face model URL to its repo id."""
    hf_prefix = "https://huggingface.co/"
    if not model_name.startswith(hf_prefix):
        return model_name
    path = model_name[len(hf_prefix):].strip("/")
    parts = path.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return path


def _is_qwen_embedding_model(model_name: str) -> bool:
    return "Qwen3-Embedding" in _normalize_hf_model_name(model_name)


def _is_nv_embed_model(model_name: str) -> bool:
    return "NV-Embed" in _normalize_hf_model_name(model_name)


def _nv_query_instruction(query_prefix: str) -> str:
    """Return an NV-Embed compatible retrieval instruction prefix."""
    if query_prefix.startswith("Instruct:"):
        return query_prefix
    return "Instruct: Given a question, retrieve passages that answer the question\nQuery: "


def _ensure_nv_embed_transformers_compat(transformers_version: str) -> None:
    """NV-Embed-v2 remote code currently targets Transformers 4.x."""
    major_version = int(transformers_version.split(".", maxsplit=1)[0])
    if major_version >= 5:
        print(
            "Error: nvidia/NV-Embed-v2 remote code is not compatible with "
            f"transformers {transformers_version}. Re-run with "
            "`uv run --with transformers==4.42.4 --with . ...` for the NV fallback.",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_biencoder_distributed_manager(
    *,
    distributed_config,
    world_size: int,
    dp_size: int | None = None,
    dp_replicate_size: int | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
):
    """Build the distributed manager expected by TrainBiencoderRecipe.

    The biencoder recipe currently expects ``cfg.distributed`` to instantiate a
    manager, while newer AutoModel configs split the strategy into
    ``distributed_config`` plus size fields under ``distributed``.
    """
    from nemo_automodel.components.distributed.config import (
        DDPConfig,
        FSDP2Config,
        MegatronFSDPConfig,
    )
    from nemo_automodel.components.distributed.ddp import DDPManager
    from nemo_automodel.components.distributed.device_mesh import create_device_mesh
    from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
    from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager

    if pp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not yet supported for biencoder models. "
            "Please disable pipeline parallelism in the distributed config."
        )

    device_mesh, moe_mesh = create_device_mesh(
        distributed_config,
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        world_size=world_size,
    )
    if isinstance(distributed_config, FSDP2Config):
        return FSDP2Manager(distributed_config, device_mesh=device_mesh, moe_mesh=moe_mesh)
    if isinstance(distributed_config, MegatronFSDPConfig):
        return MegatronFSDPManager(distributed_config, device_mesh=device_mesh)
    if isinstance(distributed_config, DDPConfig):
        return DDPManager(distributed_config)
    raise ValueError(f"Unknown distributed config type: {type(distributed_config)}")


def _should_use_fsdp(distributed: bool, device: str) -> bool:
    """Whether to use FSDP for local fallback training loops."""
    return distributed and device.startswith("cuda") and os.environ.get("NEMOTRON_USE_FSDP", "1") != "0"


def _distributed_model_load_strategy() -> str:
    """How to load pretrained weights before FSDP/DDP wrapping.

    rank0_sync (default): only rank 0 reads checkpoint shards from disk; other
    ranks build an empty shell and FSDP broadcasts weights. This avoids CPU
    RAM spikes when multiple processes load large models concurrently.

    stagger: ranks load the checkpoint one at a time. Useful as a fallback when
    rank0_sync cannot build the model from config alone (some remote-code models).
    """
    return os.environ.get("NEMOTRON_MODEL_LOAD", "rank0_sync").lower()


def _load_pretrained_for_distributed(
    *,
    model_name: str,
    rank: int,
    world_size: int,
    distributed: bool,
    use_fsdp: bool,
    device: str,
    load_kwargs: dict,
    trust_remote_code: bool = False,
    is_rank0: bool = False,
):
    """Load a pretrained model without every rank duplicating checkpoint I/O."""
    import gc

    import torch
    import torch.distributed as dist
    from transformers import AutoConfig, AutoModel

    def _load_full_model():
        return AutoModel.from_pretrained(model_name, **load_kwargs)

    if not distributed:
        return _load_full_model().to(device)

    load_strategy = _distributed_model_load_strategy()
    use_rank0_sync = use_fsdp and device.startswith("cuda") and load_strategy != "stagger"

    trc = bool(trust_remote_code or load_kwargs.get("trust_remote_code", False))

    if use_rank0_sync:
        if is_rank0:
            print("  Model load:        rank0_sync (only rank 0 reads checkpoint shards; others build on meta)")
        if rank == 0:
            model = _load_full_model()
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trc,
            )
            config_kwargs = {
                key: value
                for key, value in load_kwargs.items()
                if key
                not in {
                    "low_cpu_mem_usage",
                    "trust_remote_code",
                }
            }
            # Build the module structure on the meta device so non-zero ranks
            # allocate (almost) no host RAM. FSDP(sync_module_states=True) with a
            # param_init_fn materializes these tensors directly on GPU and
            # broadcasts rank 0's weights, so we never hold the full model in CPU
            # RAM on more than one process at a time.
            with torch.device("meta"):
                model = AutoModel.from_config(
                    config,
                    trust_remote_code=trc,
                    **config_kwargs,
                )
        dist.barrier()
        gc.collect()
        return model

    if is_rank0:
        print(
            f"  Model load:        stagger "
            f"(ranks load checkpoint sequentially; set NEMOTRON_MODEL_LOAD=rank0_sync to prefer rank-0-only load)"
        )
    model = None
    for loading_rank in range(world_size):
        if rank == loading_rank:
            model = _load_full_model()
        dist.barrier()
    gc.collect()
    assert model is not None
    if use_fsdp and device.startswith("cuda"):
        return model
    return model.to(device)


def _wrap_train_model(
    model,
    *,
    device: str,
    local_rank: int,
    distributed: bool,
    dtype,
    sync_module_states: bool | None = None,
):
    """Wrap a model with FSDP when possible, otherwise DDP."""
    if not distributed:
        return model, "single"

    if _should_use_fsdp(distributed, device):
        from functools import partial

        import torch
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        mixed_precision = None
        if device.startswith("cuda"):
            mixed_precision = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )

        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=int(os.environ.get("NEMOTRON_FSDP_MIN_NUM_PARAMS", "100000000")),
        )

        # Detect meta-initialized parameters (rank0_sync builds non-zero ranks on
        # the meta device). Those ranks must materialize on GPU via param_init_fn
        # and receive real weights from rank 0 through sync_module_states.
        has_meta_params = any(p.is_meta for p in model.parameters())

        if sync_module_states is None:
            sync_module_states = (
                _distributed_model_load_strategy() != "stagger"
                and os.environ.get("NEMOTRON_FSDP_SYNC_MODULE_STATES", "1") != "0"
            )
        if has_meta_params:
            sync_module_states = True

        param_init_fn = None
        if has_meta_params and device.startswith("cuda"):
            target_device = torch.device(device)

            def param_init_fn(module):
                module.to_empty(device=target_device, recurse=False)

        wrapped = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.device(device) if device.startswith("cuda") else None,
            limit_all_gathers=True,
            sync_module_states=sync_module_states,
            use_orig_params=True,
            param_init_fn=param_init_fn,
        )
        return wrapped, "FSDP"

    from torch.nn.parallel import DistributedDataParallel as DDP

    wrapped = DDP(
        model,
        device_ids=[local_rank] if device.startswith("cuda") else None,
        output_device=local_rank if device.startswith("cuda") else None,
        find_unused_parameters=False,
    )
    return wrapped, "DDP"


def _prepare_model_for_save(train_model, model):
    """Return the model object and optional full state dict for saving."""
    try:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )
    except ImportError:
        FSDP = None  # type: ignore[assignment]

    if FSDP is not None and isinstance(train_model, FSDP):
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(train_model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            state_dict = train_model.state_dict()
        return model, state_dict

    if hasattr(train_model, "module"):
        return train_model.module, None
    return train_model, None


def _auto_scale_hyperparams(
    cfg: FinetuneConfig, num_examples: int
) -> tuple[int, int, int, int]:
    """Auto-scale training hyperparameters based on dataset size.

    Adjusts batch size, epochs, checkpoint frequency, and validation
    frequency when the user hasn't overridden defaults and the dataset
    is small enough to benefit.

    Args:
        cfg: Fine-tuning configuration (with user-specified or default values).
        num_examples: Number of training examples.

    Returns:
        Tuple of (global_batch_size, num_epochs, checkpoint_every_steps, val_every_steps).
    """
    # --- Batch size ---
    # Default is 128; auto-scale down for small datasets so we get more steps
    if cfg.global_batch_size == 128 and num_examples < 2000:
        global_batch_size = max(16, min(64, num_examples // 8))
    else:
        global_batch_size = cfg.global_batch_size

    steps_per_epoch = max(1, num_examples // global_batch_size)

    # --- Epochs ---
    num_epochs = cfg.num_epochs

    total_steps = steps_per_epoch * num_epochs

    # --- Checkpoint / validation frequency ---
    # Default is 100; cap so we get at least 3 checkpoints
    if total_steps < cfg.checkpoint_every_steps * 3:
        checkpoint_every_steps = max(1, total_steps // 3)
    else:
        checkpoint_every_steps = cfg.checkpoint_every_steps

    if total_steps < cfg.val_every_steps * 3:
        val_every_steps = max(1, total_steps // 3)
    else:
        val_every_steps = cfg.val_every_steps

    return global_batch_size, num_epochs, checkpoint_every_steps, val_every_steps


def _run_qwen_finetune(
    cfg: FinetuneConfig,
    *,
    num_examples: int,
    global_batch_size: int,
    num_epochs: int,
) -> Path:
    """Fine-tune Qwen embedding models with a small local contrastive loop.

    NeMo Automodel's biencoder path currently cannot infer Qwen3-Embedding
    model classes, so this preserves the recipe's input/output contract while
    using the Hugging Face model directly.
    """
    import pandas as pd
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoModel, AutoTokenizer

    model_name = _normalize_hf_model_name(cfg.base_model)
    final_model_dir = cfg.checkpoint_dir / "LATEST" / "model" / "consolidated"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    is_rank0 = rank == 0

    with open(cfg.train_data_path, encoding="utf-8") as f:
        train_data = json.load(f)

    corpus_path = Path(train_data.get("corpus", {}).get("path", "./corpus/"))
    if not corpus_path.is_absolute():
        corpus_path = cfg.train_data_path.parent / corpus_path
    parquet_path = corpus_path / "train.parquet"
    if not parquet_path.exists():
        print(f"Error: corpus parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    corpus_df = pd.read_parquet(parquet_path)
    corpus = {
        str(row["id"]): str(row["text"])
        for row in corpus_df[["id", "text"]].to_dict(orient="records")
    }

    examples: list[dict[str, object]] = []
    needed_negatives = cfg.train_n_passages - 1
    for record in train_data.get("data", []):
        pos_docs = record.get("pos_doc", [])
        neg_docs = record.get("neg_doc", [])
        if not pos_docs or len(neg_docs) < needed_negatives:
            continue
        pos_id = str(pos_docs[0]["id"])
        neg_ids = [str(doc["id"]) for doc in neg_docs[:needed_negatives]]
        if pos_id not in corpus or any(neg_id not in corpus for neg_id in neg_ids):
            continue
        examples.append(
            {
                "query": str(record["question"]),
                "passages": [corpus[pos_id], *(corpus[neg_id] for neg_id in neg_ids)],
            }
        )

    if not examples:
        print("Error: no usable training examples after resolving corpus documents.", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    batch_size = max(1, min(cfg.local_batch_size, len(examples)))
    per_rank_examples = (len(examples) + world_size - 1) // world_size
    steps_per_epoch = max(1, (per_rank_examples + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * num_epochs

    if is_rank0:
        print("Using Hugging Face Qwen fine-tune fallback")
        print(f"  Model:             {model_name}")
        print(f"  Distributed:       {distributed} (world_size={world_size})")
        print(f"  Device:            {device}")
        print(f"  Local batch size:  {batch_size}")
        print(f"  Effective batch:   {batch_size * world_size}")
        print(f"  Steps/epoch:       {steps_per_epoch}")
        print(f"  Total steps:       {total_steps}")
        print()

    use_fsdp = _should_use_fsdp(distributed, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = _load_pretrained_for_distributed(
        model_name=model_name,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        use_fsdp=use_fsdp,
        device=device,
        load_kwargs={
            "torch_dtype": dtype,
            "attn_implementation": cfg.attn_implementation or "sdpa",
            "low_cpu_mem_usage": True,
        },
        is_rank0=is_rank0,
    )
    if not (distributed and use_fsdp):
        model = model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    train_model, parallelism = _wrap_train_model(
        model,
        device=device,
        local_rank=local_rank,
        distributed=distributed,
        dtype=dtype,
    )
    if is_rank0:
        print(f"  Parallelism:       {parallelism}")

    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch, device=last_hidden_states.device), sequence_lengths]

    def encode(texts: list[str], max_length: int) -> torch.Tensor:
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = train_model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if cfg.l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def collate(batch: list[dict[str, object]]) -> dict[str, list[str]]:
        queries = [f"{cfg.query_prefix} {item['query']}".strip() for item in batch]
        passages: list[str] = []
        for item in batch:
            passages.extend(
                f"{cfg.passage_prefix} {passage}".strip()
                for passage in item["passages"]  # type: ignore[index]
            )
        return {"queries": queries, "passages": passages}

    step = 0
    for epoch in range(num_epochs):
        sampler = DistributedSampler(
            examples,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        ) if distributed else None
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(
            examples,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate,
        )
        for batch in loader:
            step += 1
            query_embeddings = encode(batch["queries"], cfg.query_max_length)
            passage_embeddings = encode(batch["passages"], cfg.passage_max_length)
            passage_embeddings = passage_embeddings.view(
                len(batch["queries"]),
                cfg.train_n_passages,
                -1,
            )
            logits = torch.einsum("bd,bpd->bp", query_embeddings, passage_embeddings) / cfg.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_for_log = loss.detach()
            if distributed:
                dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)
            if is_rank0:
                print(
                    f"epoch={epoch + 1}/{num_epochs} "
                    f"step={step}/{total_steps} loss={loss_for_log.item():.4f}"
                )

    if distributed:
        dist.barrier()

    model_to_save, save_state_dict = _prepare_model_for_save(train_model, model)
    if is_rank0:
        print(f"\nSaving Qwen fine-tuned model to {final_model_dir}")
        model_to_save.save_pretrained(
            final_model_dir,
            safe_serialization=True,
            state_dict=save_state_dict,
        )
        tokenizer.save_pretrained(final_model_dir)
        metadata_path = cfg.checkpoint_dir / "LATEST" / "training_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": model_name,
                    "training_examples": num_examples,
                    "usable_training_examples": len(examples),
                    "num_epochs": num_epochs,
                    "global_batch_size": global_batch_size,
                    "local_batch_size": batch_size,
                    "effective_batch_size": batch_size * world_size,
                    "world_size": world_size,
                    "learning_rate": cfg.learning_rate,
                    "temperature": cfg.temperature,
                    "train_n_passages": cfg.train_n_passages,
                },
                f,
                indent=2,
            )

        print(f"\nFine-tuning complete!")
        print(f"   Checkpoint: {cfg.checkpoint_dir}")
        print(f"   Model:      {final_model_dir}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    return final_model_dir


def _run_nv_embed_finetune(
    cfg: FinetuneConfig,
    *,
    num_examples: int,
    global_batch_size: int,
    num_epochs: int,
) -> Path:
    """Fine-tune NV-Embed models with their remote-code latent pooling."""
    import pandas as pd
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import transformers
    from transformers import AutoModel
    from transformers.modeling_utils import PreTrainedModel

    model_name = _normalize_hf_model_name(cfg.base_model)
    _ensure_nv_embed_transformers_compat(transformers.__version__)
    final_model_dir = cfg.checkpoint_dir / "LATEST" / "model" / "consolidated"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.set_autocast_gpu_dtype(torch.bfloat16)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    is_rank0 = rank == 0

    with open(cfg.train_data_path, encoding="utf-8") as f:
        train_data = json.load(f)

    corpus_path = Path(train_data.get("corpus", {}).get("path", "./corpus/"))
    if not corpus_path.is_absolute():
        corpus_path = cfg.train_data_path.parent / corpus_path
    parquet_path = corpus_path / "train.parquet"
    if not parquet_path.exists():
        print(f"Error: corpus parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    corpus_df = pd.read_parquet(parquet_path)
    corpus = {
        str(row["id"]): str(row["text"])
        for row in corpus_df[["id", "text"]].to_dict(orient="records")
    }

    examples: list[dict[str, object]] = []
    needed_negatives = cfg.train_n_passages - 1
    for record in train_data.get("data", []):
        pos_docs = record.get("pos_doc", [])
        neg_docs = record.get("neg_doc", [])
        if not pos_docs or len(neg_docs) < needed_negatives:
            continue
        pos_id = str(pos_docs[0]["id"])
        neg_ids = [str(doc["id"]) for doc in neg_docs[:needed_negatives]]
        if pos_id not in corpus or any(neg_id not in corpus for neg_id in neg_ids):
            continue
        examples.append(
            {
                "query": str(record["question"]),
                "passages": [corpus[pos_id], *(corpus[neg_id] for neg_id in neg_ids)],
            }
        )

    if not examples:
        print("Error: no usable training examples after resolving corpus documents.", file=sys.stderr)
        sys.exit(1)

    # Use bf16 on CUDA: fp16 produced NaNs, while fp32 plus AdamW states is too
    # memory-heavy for DDP full-model training.
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    batch_size = max(1, min(cfg.local_batch_size, len(examples)))
    per_rank_examples = (len(examples) + world_size - 1) // world_size
    steps_per_epoch = max(1, (per_rank_examples + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * num_epochs

    if is_rank0:
        print("Using Hugging Face NV-Embed fine-tune fallback")
        print(f"  Model:             {model_name}")
        print(f"  Distributed:       {distributed} (world_size={world_size})")
        print(f"  Device:            {device}")
        print(f"  Local batch size:  {batch_size}")
        print(f"  Effective batch:   {batch_size * world_size}")
        print(f"  Steps/epoch:       {steps_per_epoch}")
        print(f"  Total steps:       {total_steps}")
        print()

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    use_fsdp = _should_use_fsdp(distributed, device)
    model = _load_pretrained_for_distributed(
        model_name=model_name,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        use_fsdp=use_fsdp,
        device=device,
        load_kwargs={
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        },
        trust_remote_code=True,
        is_rank0=is_rank0,
    )
    if not (distributed and use_fsdp):
        model = model.to(device)
    if hasattr(model, "embedding_model") and hasattr(model.embedding_model, "config"):
        model.embedding_model.config.use_cache = False
    if hasattr(model, "embedding_model") and hasattr(model.embedding_model, "gradient_checkpointing_enable"):
        model.embedding_model.gradient_checkpointing_enable()
    model.train()
    train_model, parallelism = _wrap_train_model(
        model,
        device=device,
        local_rank=local_rank,
        distributed=distributed,
        dtype=dtype,
    )
    if is_rank0:
        print(f"  Parallelism:       {parallelism}")

    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        foreach=False,
    )

    query_instruction = _nv_query_instruction(cfg.query_prefix)
    passage_instruction = "" if cfg.passage_prefix == "passage:" else cfg.passage_prefix

    def encode(texts: list[str], instruction: str, max_length: int) -> torch.Tensor:
        tokenizer = model.tokenizer
        input_texts = [
            f"{instruction}{text}{tokenizer.eos_token if model.add_eos else ''}"
            for text in texts
        ]
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True,
        )
        if model.padding_side == "right" and model.is_mask_instruction and instruction:
            instruction_lens = len(tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0
        features = model.prepare_kwargs_from_batch(batch_dict, instruction_lens, device=torch.device(device))
        embeddings = train_model(**features)["sentence_embeddings"].squeeze(1)
        if cfg.l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def collate(batch: list[dict[str, object]]) -> dict[str, list[str]]:
        queries = [str(item["query"]) for item in batch]
        passages: list[str] = []
        for item in batch:
            passages.extend(str(passage) for passage in item["passages"])  # type: ignore[index]
        return {"queries": queries, "passages": passages}

    step = 0
    for epoch in range(num_epochs):
        sampler = DistributedSampler(
            examples,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        ) if distributed else None
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(
            examples,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate,
        )
        for batch in loader:
            step += 1
            query_embeddings = encode(batch["queries"], query_instruction, cfg.query_max_length)
            passage_embeddings = encode(batch["passages"], passage_instruction, cfg.passage_max_length)
            passage_embeddings = passage_embeddings.view(
                len(batch["queries"]),
                cfg.train_n_passages,
                -1,
            )
            logits = torch.einsum("bd,bpd->bp", query_embeddings, passage_embeddings) / cfg.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_for_log = loss.detach()
            if distributed:
                dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)
            if is_rank0:
                print(
                    f"epoch={epoch + 1}/{num_epochs} "
                    f"step={step}/{total_steps} loss={loss_for_log.item():.4f}"
                )

    if distributed:
        dist.barrier()

    model_to_save, save_state_dict = _prepare_model_for_save(train_model, model)
    if is_rank0:
        print(f"\nSaving NV-Embed fine-tuned model to {final_model_dir}")
        model_to_save.save_pretrained(
            final_model_dir,
            safe_serialization=True,
            state_dict=save_state_dict,
        )
        model.tokenizer.save_pretrained(final_model_dir)
        metadata_path = cfg.checkpoint_dir / "LATEST" / "training_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": model_name,
                    "training_examples": num_examples,
                    "usable_training_examples": len(examples),
                    "num_epochs": num_epochs,
                    "global_batch_size": global_batch_size,
                    "local_batch_size": batch_size,
                    "effective_batch_size": batch_size * world_size,
                    "world_size": world_size,
                    "learning_rate": cfg.learning_rate,
                    "temperature": cfg.temperature,
                    "train_n_passages": cfg.train_n_passages,
                },
                f,
                indent=2,
            )

        print(f"\nFine-tuning complete!")
        print(f"   Checkpoint: {cfg.checkpoint_dir}")
        print(f"   Model:      {final_model_dir}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    return final_model_dir


def run_finetune(cfg: FinetuneConfig) -> Path:
    """Run embedding model fine-tuning using nemo-automodel.

    Args:
        cfg: Fine-tuning configuration.

    Returns:
        Path to final checkpoint directory.
    """
    # Validate inputs
    if not cfg.train_data_path.exists():
        print(f"Error: Training data not found: {cfg.train_data_path}", file=sys.stderr)
        print("       Please run stage1_data_prep first.", file=sys.stderr)
        sys.exit(1)

    # Count training examples and check negative passage availability
    num_examples = _count_training_examples(cfg.train_data_path)
    _warn_if_negatives_sparse(cfg.train_data_path, cfg.train_n_passages)

    global_batch_size, num_epochs, ckpt_every, val_every = _auto_scale_hyperparams(
        cfg, num_examples
    )

    steps_per_epoch = max(1, num_examples // global_batch_size)
    total_steps = steps_per_epoch * num_epochs

    # Print training plan
    print(f"Training plan:")
    print(f"  Dataset:          {num_examples:,} examples")

    if global_batch_size != cfg.global_batch_size:
        print(f"  Batch size:       {global_batch_size} (auto-scaled from {cfg.global_batch_size} — dataset < 2000 examples)")
    else:
        print(f"  Batch size:       {global_batch_size}")
        if num_examples < 2000 and cfg.global_batch_size != 128:
            print(f"                    (note: auto-scaling skipped because batch size was explicitly set)")

    if num_epochs != cfg.num_epochs:
        print(f"  Epochs:           {num_epochs} (auto-scaled from {cfg.num_epochs})")
    else:
        print(f"  Epochs:           {num_epochs}")

    print(f"  Steps/epoch:      ~{steps_per_epoch}")
    print(f"  Total steps:      ~{total_steps}")
    print(f"  LR schedule:      {cfg.lr_decay_style}, warmup={cfg.lr_warmup_steps}, peak={cfg.learning_rate}")
    print(f"  Checkpoint every: {ckpt_every} steps")
    print(f"  Validate every:   {val_every} steps")
    print()

    if total_steps < 50:
        print(f"Warning: Only ~{total_steps} total training steps. "
              f"Dataset may be too small for meaningful fine-tuning.", file=sys.stderr)
        print(f"         Consider adding more documents to your corpus.", file=sys.stderr)
        print()

    print(f"Base model:     {cfg.base_model}")
    print(f"Training data:  {cfg.train_data_path}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print()

    if _is_qwen_embedding_model(cfg.base_model):
        return _run_qwen_finetune(
            cfg,
            num_examples=num_examples,
            global_batch_size=global_batch_size,
            num_epochs=num_epochs,
        )
    if _is_nv_embed_model(cfg.base_model):
        return _run_nv_embed_finetune(
            cfg,
            num_examples=num_examples,
            global_batch_size=global_batch_size,
            num_epochs=num_epochs,
        )

    # Import nemo-automodel components
    try:
        from nemo_automodel.components.config.loader import load_yaml_config
        from nemo_automodel.recipes.biencoder import TrainBiencoderRecipe
    except ImportError as e:
        print(f"Error: Failed to import nemo-automodel. Is it installed?", file=sys.stderr)
        print(f"  Install with: pip install nemo-automodel", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load base config from nemo-automodel defaults
    base_config_path = STAGE_PATH / "biencoder_base.yaml"
    automodel_cfg = load_yaml_config(str(base_config_path))
    if "distributed_config" in automodel_cfg and "distributed" in automodel_cfg:
        automodel_cfg.distributed._target_ = _build_biencoder_distributed_manager
        automodel_cfg.distributed.distributed_config = automodel_cfg.distributed_config

    # Apply overrides from our config
    # Model settings
    automodel_cfg.model.pretrained_model_name_or_path = cfg.base_model
    automodel_cfg.tokenizer.pretrained_model_name_or_path = cfg.base_model
    # Auto-detect attention implementation if not explicitly set
    if cfg.attn_implementation is not None:
        attn_impl = cfg.attn_implementation
    else:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        print(f"  Attention:    {attn_impl} (auto-detected)")
    automodel_cfg.model.attn_implementation = attn_impl

    # Data settings
    automodel_cfg.dataloader.dataset.data_dir_list = [str(cfg.train_data_path)]
    automodel_cfg.dataloader.dataset.train_n_passages = cfg.train_n_passages
    automodel_cfg.dataloader.collate_fn.q_max_len = cfg.query_max_length
    automodel_cfg.dataloader.collate_fn.p_max_len = cfg.passage_max_length
    automodel_cfg.dataloader.collate_fn.query_prefix = cfg.query_prefix
    automodel_cfg.dataloader.collate_fn.passage_prefix = cfg.passage_prefix

    # Training settings — use auto-scaled values
    automodel_cfg.step_scheduler.num_epochs = num_epochs
    automodel_cfg.step_scheduler.global_batch_size = global_batch_size
    automodel_cfg.step_scheduler.local_batch_size = cfg.local_batch_size
    automodel_cfg.step_scheduler.ckpt_every_steps = ckpt_every
    automodel_cfg.step_scheduler.val_every_steps = val_every

    # Optimizer settings
    automodel_cfg.optimizer.lr = cfg.learning_rate
    automodel_cfg.optimizer.weight_decay = cfg.weight_decay
    automodel_cfg.lr_scheduler.lr_warmup_steps = cfg.lr_warmup_steps
    automodel_cfg.lr_scheduler.lr_decay_style = cfg.lr_decay_style

    # Model architecture
    automodel_cfg.model.pooling = cfg.pooling
    automodel_cfg.model.l2_normalize = cfg.l2_normalize
    automodel_cfg.model.t = cfg.temperature

    # Checkpoint settings
    automodel_cfg.checkpoint.checkpoint_dir = str(cfg.checkpoint_dir)

    # Create and run the biencoder recipe
    recipe = TrainBiencoderRecipe(automodel_cfg)
    recipe.setup()
    recipe.run_train_validation_loop()

    # Find the final checkpoint
    final_model_dir = cfg.checkpoint_dir / "LATEST" / "model" / "consolidated"

    print(f"\nFine-tuning complete!")
    print(f"   Checkpoint: {cfg.checkpoint_dir}")
    print(f"   Model:      {final_model_dir}")

    # Save artifact (registers with artifact registry if kit.init() was called)
    try:
        from nemotron.kit.artifacts.embed import EmbedModelArtifact

        artifact = EmbedModelArtifact(
            path=final_model_dir,
            base_model=cfg.base_model,
            training_examples=num_examples,
            num_epochs=num_epochs,
            global_batch_size=global_batch_size,
            learning_rate=cfg.learning_rate,
            temperature=cfg.temperature,
        )
        artifact.save(name="embed/model")
    except Exception:
        pass  # Artifact save is best-effort — don't break the pipeline

    return final_model_dir


def main(cfg: FinetuneConfig | None = None) -> Path:
    """Entry point for fine-tuning.

    Args:
        cfg: Config from CLI framework, or None when run directly as script.

    Returns:
        Path to final model checkpoint.
    """
    if cfg is None:
        # Called directly as script - parse config ourselves
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )

        try:
            cfg = load_config(config_path, cli_overrides, FinetuneConfig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return run_finetune(cfg)


if __name__ == "__main__":
    main()
