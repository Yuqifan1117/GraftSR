---
kind: external_dependency
name: CUDA GPU Computing Platform
slug: cuda-cudnn
category: external_dependency
category_hints:
    - client_constraint
scope:
    - '**'
source_files:
    - requirements.txt
    - train_edit_mask.sh
    - test_edit_mask.sh
---

Deep learning framework requiring NVIDIA CUDA-compatible GPUs. Environment variables include `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for memory management and `XFORMERS_FORCE_DISABLE_TRITON=1` to disable Triton optimizations. Uses `CUDA_VISIBLE_DEVICES` for GPU selection and `CUDA_LAUNCH_BLOCKING=1` for debugging.