---
kind: external_dependency
name: Hugging Face Model Hub
slug: huggingface
category: external_dependency
category_hints:
    - client_constraint
scope:
    - '**'
source_files:
    - train_edit_mask.sh
    - test_edit_mask.sh
---

Secondary model download source configured via `HF_ENDPOINT=https://hf-mirror.com` environment variable. Used as fallback when ModelScope downloads fail or for models not available on ModelScope. The mirror endpoint is set in both training and testing scripts.