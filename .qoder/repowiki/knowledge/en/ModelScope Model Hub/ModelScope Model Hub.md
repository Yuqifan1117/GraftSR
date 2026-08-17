---
kind: external_dependency
name: ModelScope Model Hub
slug: modelscope
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
source_files:
    - requirements.txt
    - pyproject.toml
---

Primary model hub for downloading pre-trained models including Qwen-Image variants, Wan2.1 video models, and other diffusion models. Models are downloaded via the `modelscope` Python package with configurable download sources through environment variables like `MODELSCOPE_DOMAIN` and `DIFFSYNTH_DOWNLOAD_SOURCE`. Default download source is ModelScope China; international users should configure `MODELSCOPE_DOMAIN=www.modelscope.ai`.