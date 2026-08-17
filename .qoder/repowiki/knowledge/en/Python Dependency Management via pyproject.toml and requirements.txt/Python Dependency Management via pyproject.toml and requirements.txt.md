---
kind: dependency_management
name: Python Dependency Management via pyproject.toml and requirements.txt
category: dependency_management
scope:
    - '**'
source_files:
    - pyproject.toml
    - requirements.txt
    - setup.py
    - .github/workflows/publish.yaml
---

This repository manages Python dependencies through a dual-layer approach centered on `pyproject.toml` for the distributable package and `requirements.txt` for development/cluster deployment, with no lockfile or vendoring strategy in place.

**Primary manifest: `pyproject.toml`**
- Declares the `diffsynth` package (version 2.0.9) built with setuptools (`setuptools.build_meta`, requires setuptools>=61.0 and wheel).
- Core runtime dependencies are pinned with minimum versions only (e.g., `torch>=2.0.0`, `transformers`, `accelerate`, `peft`, `datasets`, `modelscope`, `safetensors`, `einops`, `sentencepiece`, `protobuf`, `imageio[ffmpeg]`, `ftfy`, `pandas`). No upper bounds except where explicitly stated.
- Optional dependency groups provide hardware-specific installs:
  - `npu_aarch64`: pins `torch==2.7.1`, `torch-npu==2.7.1`, `torchvision==0.22.1` for ARM NPU.
  - `npu`: pins CPU variants of torch/torchvision with `+cpu` suffixes.
  - `audio`: adds `torchaudio` and `torchcodec`.
- Package discovery is configured to include `diffsynth` and all subpackages under `./`.

**Secondary manifest: `requirements.txt`**
- Mirrors the core dependencies from `pyproject.toml` but augments them with ODTSR super-resolution extras: `basicsr`, `opencv-python==4.7.0.72`, `cupy-cuda12x`, `lpips`, `lightning`, `pyiqa`, `clean-fid`, `matplotlib`, `numpy<2.0.0`, `gradio`, `pillow_heif`, `pynvml`, `qwen_vl_utils`, `dashscope`.
- Used by CI (`pip install -r requirements.txt`) and Nebula cluster job submission scripts for environment provisioning.

**Legacy fallback: `setup.py`**
- Reads `requirements.txt` at build time and sets `install_requires` from it; declares `python_requires='>=3.6'` (conflicts with `pyproject.toml`'s `requires-python = ">=3.10.1"`). Version is out of sync (1.1.7 vs 2.0.9 in `pyproject.toml`).

**Distribution & installation patterns**
- Development install: `pip install -e .` (documented across README files and docs).
- PyPI publish: `.github/workflows/publish.yaml` runs `pip install wheel==0.44.0 && pip install -r requirements.txt`, then uses `twine`.
- Optional GPU backends: ROCm install via `--index-url https://download.pytorch.org/whl/rocm6.4`; optional acceleration packages like `xfuser[flash-attn]>=0.4.3`, `git+https://github.com/feifeibear/long-context-attention.git`, `git+https://github.com/xdit-project/xDiT.git` are installed ad-hoc per model.
- Documentation builds use `docs/requirements.txt` via ReadTheDocs configuration.

**Constraints observed**
- No `requirements.lock`, `poetry.lock`, `Pipfile.lock`, or `vendor/` directory — dependency versions are not locked, relying on PyPI resolution at install time.
- Some packages have explicit version caps (e.g., `transformers>=4.57.3,<5.0.0`, `opencv-python==4.7.0.72`, `numpy<2.0.0`) while most others only specify lower bounds.
- Hardware-specific torch variants are isolated into optional dependency groups rather than being part of the base install.