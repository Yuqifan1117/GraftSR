---
kind: external_dependency
name: Basic Super-Resolution Framework
slug: basicsr
category: external_dependency
category_hints:
    - sdk_real_api
scope:
    - '**'
source_files:
    - requirements.txt
    - examples/qwen_image/train_gan_edit_mask.py
---

Core super-resolution framework used for degradation pipelines and dataset handling. Requires manual patching of torchvision import paths (`rgb_to_grayscale`) during setup. Integrated via `diffsynth.extensions.realesrgan.dataset.PairedSROnlineTxtDataset` for paired image datasets.