---
kind: build_system
name: Python Package Build, Distribution & Training Orchestration
category: build_system
scope:
    - '**'
source_files:
    - pyproject.toml
    - setup.py
    - requirements.txt
    - .github/workflows/publish.yaml
    - docs/en/Makefile
    - nebula_configs/cluster.json
    - nebula_configs/accelerate-1.yaml
---

This repository uses a dual build-system approach centered on Python packaging and distributed training orchestration:

**Package distribution (setuptools + pyproject.toml)**
- The primary build configuration lives in `pyproject.toml`, which declares `setuptools.build_meta` as the build backend, requires Python ≥3.10.1, and pins the package name `diffsynth` with version `2.0.9`. Dependencies are declared inline (`torch>=2.0.0`, `transformers`, `accelerate`, `peft`, `datasets`, etc.).
- Optional dependency groups provide hardware-specific builds: `npu_aarch64` (torch-npu 2.7.1), `npu` (CPU variant), and `audio` (torchaudio, torchcodec).
- A legacy `setup.py` is also present (version `1.1.7`, Python ≥3.6) that reads `requirements.txt` for install dependencies — this file duplicates many of the same packages plus ODTSR-specific extras like `basicsr`, `cupy-cuda12x`, `lightning`, `gradio`, `dashscope`, etc.
- Package discovery includes only `diffsynth` and its subpackages via `[tool.setuptools.packages.find]`.

**CI/CD release pipeline (GitHub Actions)**
- `.github/workflows/publish.yaml` triggers on tags matching `v**`. It checks out the repo, sets up Python 3.10, installs `wheel==0.44.0` and `requirements.txt`, runs `python -m build`, then publishes all wheels/sdists to PyPI via `twine upload` using a stored API token. Concurrency is scoped per workflow+ref to cancel in-progress runs.

**Documentation build (Sphinx Makefile)**
- `docs/en/Makefile` (and mirrored `docs/zh/Makefile`) provides standard Sphinx targets (`make html`, `make latexpdf`, etc.) with `SPHINXBUILD` configurable via environment. Documentation source is under `docs/en/` and `docs/zh/` with `conf.py` and `index.rst` defining the site structure.

**Training orchestration (Hugging Face Accelerate + shell scripts)**
- Every model example under `examples/<model>/model_training/{full,lora,validate_full,validate_lora}/` ships one or more `.sh` launch scripts that call `accelerate launch train.py ...`, often with an accompanying `accelerate_config*.yaml` specifying `distributed_type`, `mixed_precision` (bf16), `num_processes`, `rdzv_backend`, etc.
- `nebula_configs/` holds cluster-wide Accelerate profiles (`accelerate-{1,2,4,8,16,32,64}.yaml`) and a `cluster.json` worker manifest (`gpu: 100, cpu: 800, memory: 500000`) used by the Nebula scheduler to submit jobs.
- Top-level `nebulactl_launch_*.sh` scripts wrap common training/inference commands for the ODTSR edit-mask pipeline, parameterized by shell arguments.

**Conventions observed**
- Version numbers are split between `pyproject.toml` (distribution version) and `setup.py` (legacy entry point); they differ (`2.0.9` vs `1.1.7`) and should be kept in sync.
- Hardware variants are expressed as optional dependency groups rather than separate packages.
- Training jobs are always launched through `accelerate launch` with explicit config files; no direct `torch.distributed.run` calls are used in the examples.
- Documentation is maintained in parallel English/Chinese trees with identical structure.