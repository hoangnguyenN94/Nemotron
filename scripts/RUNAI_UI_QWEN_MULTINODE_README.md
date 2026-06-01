# Run:ai UI Guide For Qwen Multi-Node Finetuning

This document explains how to run the Qwen embedding finetune workflow on NVIDIA Run:ai by using the web UI only.

The recommended default topology is:

- 2 nodes total
- 8 GPUs per node
- 16 GPUs total

This flow uses the current runtime image:

- `registry.local/nemotron-embed-airgap:latest`

You do not need to build a new image just to use the UI flow.
If you keep the current image, place the helper files on the shared PVC and call `run.sh` from the UI with `bash`.

## What Is Already Inside The Image

The current runtime image already contains:

- the base Nemotron runtime under `/opt/nemotron`
- the `run-with-env` helper
- the Qwen Python environment under `/opt/venvs/qwen`
- the required training libraries for the validated Qwen finetune flow

Because of that, the customer does not need to install extra Python packages inside the pod at runtime.

Important distinction:

- the Dockerfile currently copies the full Nemotron source tree into `/opt/nemotron`
- a newly rebuilt image from the current Dockerfile should therefore contain
  `/opt/nemotron/src/nemotron/recipes/embed/stage2_finetune/train.py`
- however, the already-available runtime image snapshot checked during this update
  did not expose that file in the local runtime rootfs snapshot

In practice, this means the customer should not assume that an existing image tag
already contains the latest `stage2_finetune/train.py`, even though a fresh rebuild
from the current Dockerfile should include it.

For that reason, the recommended and safest approach for customer-facing Run:ai UI
instructions is:

- keep using the current image for the runtime environment
- stage the latest `stage2_finetune` files on the PVC
- let `run.sh` use the PVC copy first
- optionally rebuild the image later if you want the updated training code baked in

## What Must Exist Outside The Image

The image does not contain the Hugging Face cache or the customer training data.
Those must be staged on a shared PVC or another shared storage mount.

For the safest and most reproducible Run:ai UI flow, also stage the latest
`stage2_finetune` helper files on the PVC even if you believe the image already
contains them.

The expected directory layout is:

```text
/workspace/
  hf-cache/
    hub/
    modules/
  nemotron_embed/
    run.sh
    stage2_finetune/
      train.py
      biencoder_base.yaml
      config/
        default.yaml
  output/
    embed/
      qwen_stage1/
        train_mined.automodel_unrolled.json
      qwen_stage2_multinode/
        checkpoints/
```

Minimum required artifacts:

- `hf-cache/hub`
- `hf-cache/modules`
- `output/embed/qwen_stage1/train_mined.automodel_unrolled.json`
- `nemotron_embed/run.sh`
- `nemotron_embed/stage2_finetune/train.py`
- `nemotron_embed/stage2_finetune/biencoder_base.yaml`
- `nemotron_embed/stage2_finetune/config/default.yaml`

## Step 1: Prepare The Shared Storage

Make sure the customer has already done the following:

1. Push the current runtime image to a registry reachable by the Run:ai cluster.
2. Extract the offline Hugging Face cache to the shared PVC.
3. Run Stage 1 data preparation first, so that the mined Stage 2 input file exists.
4. Copy the `run.sh` script to the shared PVC.
5. Copy the latest `stage2_finetune` source files to the shared PVC.

For example:

```bash
mkdir -p /workspace/nemotron_embed
cp scripts/runai_ui_qwen_finetune.sh /workspace/nemotron_embed/run.sh
mkdir -p /workspace/nemotron_embed/stage2_finetune/config
cp src/nemotron/recipes/embed/stage2_finetune/train.py /workspace/nemotron_embed/stage2_finetune/train.py
cp src/nemotron/recipes/embed/stage2_finetune/biencoder_base.yaml /workspace/nemotron_embed/stage2_finetune/biencoder_base.yaml
cp src/nemotron/recipes/embed/stage2_finetune/config/default.yaml /workspace/nemotron_embed/stage2_finetune/config/default.yaml
```

The helper files may live anywhere on the PVC, as long as the UI command and environment variables point to the correct path.

If you rebuild the image from the current Dockerfile and want to rely on the image copy
instead, the expected in-image path is:

`/opt/nemotron/src/nemotron/recipes/embed/stage2_finetune/train.py`

## Step 2: Understand The Run Script Defaults

The provided `run.sh` is configured for multi-node Run:ai PyTorch workloads.

Default behavior:

- `NPROC_PER_NODE=8`
- `NNODES=2`
- `QWEN_BASE_MODEL=Qwen/Qwen3-Embedding-8B`
- `HF_CACHE_ROOT=/workspace/hf-cache`
- `TRAIN_DATA_PATH=/workspace/output/embed/qwen_stage1/train_mined.automodel_unrolled.json`
- `CHECKPOINT_DIR=/workspace/output/embed/qwen_stage2_multinode/checkpoints`
- `WORKSPACE_STAGE2_DIR=/workspace/nemotron_embed/stage2_finetune`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `NEMOTRON_USE_FSDP=1`

The script also prefers distributed environment variables injected by Run:ai, including:

- `MASTER_ADDR`
- `MASTER_PORT`
- `GROUP_RANK`
- `WORLD_SIZE`
- `LOCAL_WORLD_SIZE`

When both locations exist, the script prefers:

1. `WORKSPACE_STAGE2_DIR/train.py`
2. `/opt/nemotron/src/nemotron/recipes/embed/stage2_finetune/train.py`

This allows the customer to use the current image while still picking up the latest
`train.py` updates from the PVC.

The current `train.py` updates include:

- `low_cpu_mem_usage=True` during model loading
- gradient checkpointing for both Qwen and NV fallback paths
- FSDP by default for distributed CUDA fallback training

So the recommended customer workflow is:

1. use the existing runtime image as the execution base
2. mount the PVC at `/workspace`
3. place `run.sh` and the latest `stage2_finetune` directory on the PVC
4. let `run.sh` select the PVC copy first

That means the customer should use a PyTorch distributed workload in the UI, not a plain single-pod workload.

## Step 3: Create The Workload In Run:ai UI

Create a new training workload in the Run:ai web UI with these settings.

### Workload Type

- Framework: `PyTorch`
- Distributed configuration: `Workers & master`

For a 2-node job:

- set `Workers = 1`

Why:

- Run:ai counts `master + workers`
- `Workers = 1` means 2 total pods

### Image

- Custom image: `registry.local/nemotron-embed-airgap:latest`

### Compute Resources

Use the same configuration for both master and worker unless the customer platform team requires otherwise.

Recommended starting point:

- GPUs per pod: `8`
- CPU request: `16`
- CPU limit: `16`
- Memory request: `192Gi`
- Memory limit: `192Gi`
- large shared memory: enabled, if the UI exposes that option

### Storage

Attach the existing PVC and mount it at:

- `/workspace`

### Working Directory

Set:

- `/opt/nemotron`

### Runtime Command

Set:

- Command: `bash`
- Arguments: `/workspace/nemotron_embed/run.sh`

Calling the script with `bash` is important because it avoids executable-bit issues on files copied to the PVC.

## Step 4: Environment Variables In The UI

Add these environment variables to the workload.

Required values:

- `HF_CACHE_ROOT=/workspace/hf-cache`
- `TRAIN_DATA_PATH=/workspace/output/embed/qwen_stage1/train_mined.automodel_unrolled.json`
- `CHECKPOINT_DIR=/workspace/output/embed/qwen_stage2_multinode/checkpoints`
- `QWEN_BASE_MODEL=Qwen/Qwen3-Embedding-8B`
- `NPROC_PER_NODE=8`
- `NNODES=2`
- `WORKSPACE_STAGE2_DIR=/workspace/nemotron_embed/stage2_finetune`

Optional tuning values:

- `NUM_EPOCHS=1`
- `GLOBAL_BATCH_SIZE=8`
- `LOCAL_BATCH_SIZE=1`
- `LEARNING_RATE=1.0e-5`
- `LR_WARMUP_STEPS=1`
- `WEIGHT_DECAY=0.01`
- `TRAIN_N_PASSAGES=5`
- `POOLING=avg`
- `L2_NORMALIZE=true`
- `TEMPERATURE=0.02`
- `QUERY_MAX_LENGTH=256`
- `PASSAGE_MAX_LENGTH=256`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `NEMOTRON_USE_FSDP=1`
- `NEMOTRON_FSDP_MIN_NUM_PARAMS=100000000`

Do not manually set `NODE_RANK`, `MASTER_ADDR`, or `MASTER_PORT` unless the customer platform team explicitly requires it.
For a normal Run:ai PyTorch distributed workload, those values should be injected automatically.

## Step 5: Launch And Verify

After the workload starts, check the logs from both the master and worker pods.

Expected early log lines from the script:

```text
[run.sh] workspace: /workspace
[run.sh] hf cache: /workspace/hf-cache
[run.sh] train data: /workspace/output/embed/qwen_stage1/train_mined.automodel_unrolled.json
[run.sh] checkpoints: /workspace/output/embed/qwen_stage2_multinode/checkpoints
[run.sh] base model: Qwen/Qwen3-Embedding-8B
[run.sh] train script: /workspace/nemotron_embed/stage2_finetune/train.py
[run.sh] nnodes=2 node_rank=...
[run.sh] master=...:...
[run.sh] fsdp=1
```

If the customer sees those lines, the script has started correctly and is reading the shared PVC as expected.

## Common Customer Questions

### Do we need to install extra libraries in the container?

No, not for the validated Qwen flow.
The current runtime image already includes the required Python environment and code.

### Do we need a new image?

No, not for the UI-based workflow described here.
The current image is sufficient as the runtime base.

Only build a new image if the customer wants the updated `run.sh` and the latest
`stage2_finetune/train.py` baked into `/opt/nemotron` inside the image itself.
If you do rebuild from the current Dockerfile, the rebuilt image should include
`train.py` because the Dockerfile copies `nemotron/` into `/opt/nemotron/`.

### Why do we copy `run.sh` and `stage2_finetune` to the PVC?

Because the current image was built before this UI helper script existed, and an
existing image tag may also not contain the latest `stage2_finetune/train.py`.
Placing these files on the PVC lets the customer use the current image immediately
while still running the latest training code.

## Troubleshooting

### `Permission denied` when launching a script

If the customer sees a permission error for a shell script, run it with:

```bash
bash /workspace/nemotron_embed/run.sh
```

instead of trying to execute it directly.

### Missing Hugging Face cache

If the job prints:

```text
Missing Hugging Face hub cache: /workspace/hf-cache/hub
```

or:

```text
Missing Hugging Face modules cache: /workspace/hf-cache/modules
```

then the model cache was not extracted to the expected PVC path.

### Missing Stage 2 training file

If the job prints:

```text
Missing training data: /workspace/output/embed/qwen_stage1/train_mined.automodel_unrolled.json
```

then Stage 1 output is not present on the shared storage or the path is different from the UI configuration.

### Missing `train.py` or stage2 config files

If the job prints an error about missing:

- `train.py`
- `config/default.yaml`
- `biencoder_base.yaml`

then the customer has not copied the latest `stage2_finetune` directory to the PVC,
or `WORKSPACE_STAGE2_DIR` points to the wrong location.

### Distributed rendezvous errors

If the customer sees connection or rendezvous errors, verify:

1. The workload type is `PyTorch`, not a plain single-pod job.
2. The distributed configuration is `Workers & master`.
3. `Workers = 1` for a 2-node run.
4. Each pod really has 8 GPUs assigned.
5. The cluster network policy allows master and worker pods to communicate.

## Recommended First Validation

Before running a long job, validate the setup with:

1. the same image
2. the same PVC
3. the same `run.sh`
4. the same `stage2_finetune` directory on the PVC
5. `NNODES=1`
6. `NPROC_PER_NODE=8`

Once single-node training works, switch back to:

- `NNODES=2`
- `Workers = 1`

That is usually the fastest way to separate storage issues from distributed training issues.
