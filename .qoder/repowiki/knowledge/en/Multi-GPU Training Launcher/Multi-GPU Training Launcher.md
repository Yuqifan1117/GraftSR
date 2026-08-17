---
kind: external_dependency
name: Multi-GPU Training Launcher
slug: accelerate
category: external_dependency
category_hints:
    - framework_behavior
scope:
    - '**'
source_files:
    - train_edit_mask.sh
    - nebula_configs/accelerate-*.yaml
---

Used for distributed training across multiple GPUs. Training scripts use `accelerate launch --config_file=...` with configuration files in `nebula_configs/accelerate-{worker_count}.yaml`. Supports LOCAL_MACHINE compute environment with MULTI_GPU distributed type, bf16 mixed precision, and static rendezvous backend.