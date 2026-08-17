---
kind: configuration_system
name: Configuration System — Model Registry, Environment Variables, and Accelerate Profiles
category: configuration_system
scope:
    - '**'
source_files:
    - diffsynth/core/loader/config.py
    - diffsynth/configs/model_configs.py
    - diffsynth/core/loader/file.py
    - nebula_configs/accelerate-1.yaml
    - nebula_configs/accelerate-8.yaml
    - nebula_configs/cluster.json
    - examples/qwen_image/configs/agb1_g002_r1_5_dynamic_lnnp05.yaml
---

The DiffSynth-Studio repository implements a layered configuration system that combines three complementary mechanisms: (1) a Python-based model registry for declarative model definitions, (2) environment-variable overrides for runtime behavior, and (3) Hugging Face Accelerate YAML profiles plus a cluster manifest for distributed training setup.

**Model registry via dataclass + Python lists**
The core of the configuration is `diffsynth/core/loader/config.py`, which defines the `ModelConfig` dataclass. Each entry specifies where to find model files (`path` or `model_id`), how to download them (`download_source`, `origin_file_pattern`, `skip_download`), device/dtype placement across offload/onload/preparing/computation stages, and an optional in-memory `state_dict`. The `ModelConfig.download_if_necessary()` method resolves paths, optionally downloads from ModelScope or Hugging Face, and normalizes `path` into a single string or list.

All supported models are declared as Python lists in `diffsynth/configs/model_configs.py` (e.g. `qwen_image_series`, `wan_series`, `flux_series`, `ltx2_series`, etc.), each entry mapping a `model_name` to a concrete `model_class` path and an optional `state_dict_converter`. These lists are concatenated into `MODEL_CONFIGS`, providing a single registry that pipelines and loaders consult to instantiate components by name.

**Environment variables as runtime overrides**
Runtime behavior is controlled through `os.environ.get(...)` calls scattered across the codebase. The documented and implemented variables include:
- `DIFFSYNTH_DOWNLOAD_SOURCE` — selects `modelscope` or `huggingface` as the download backend (default `modelscope`).
- `DIFFSYNTH_SKIP_DOWNLOAD` — boolean string (`true`/`false`) to bypass downloading when `path` is provided.
- `DIFFSYNTH_MODEL_BASE_PATH` — overrides the default `./models` base directory for downloaded models.
- `DIFFSYNTH_ATTENTION_IMPLEMENTATION` — selects attention implementation variant.
- `DIFFSYNTH_DISK_MAP_BUFFER_SIZE` — controls disk-mapped VRAM buffer size.
These variables are read at load time and take precedence over constructor defaults, giving users fine-grained control without modifying code.

**Training and cluster configuration via Accelerate YAML + JSON**
Distributed training configuration lives under `nebula_configs/`. There are multiple `accelerate-{N}.yaml` files (1, 2, 4, 8, 16, 32, 64 processes) following the Hugging Face Accelerate profile schema, specifying `compute_environment`, `distributed_type`, `mixed_precision` (`bf16`), `num_processes`, `gpu_ids`, and related flags. A `cluster.json` file declares per-worker resource quotas (`gpu`, `cpu`, `memory`) consumed by the Nebula scheduler. Training scripts invoke `accelerate launch --config_file nebula_configs/accelerate-{N}.yaml ...` to bootstrap multi-GPU runs.

**Training hyperparameters via YAML**
Per-experiment training hyperparameters are stored as plain YAML files alongside example scripts (e.g. `examples/qwen_image/configs/agb1_g002_r1_5_dynamic_lnnp05.yaml`). These define learning rates, schedulers, loss weights, GAN settings, and test metrics. They are loaded with PyYAML and merged into the training run, keeping experiment-level tuning separate from framework configuration.

**Conventions and constraints observed**
- Model entries must provide either `path` (local files) or `model_id` + `origin_file_pattern`; both being None raises a `ValueError` during `check_input()`.
- `download_source` is restricted to the strings `modelscope` or `huggingface`; any other value raises an error in `download()`.
- `DIFFSYNTH_SKIP_DOWNLOAD` accepts only lowercase `true` or `false` strings; other values fall back to the constructor default.
- Local model storage defaults to `./models` unless `DIFFSYNTH_MODEL_BASE_PATH` is set.
- Accelerate profiles follow the standard schema used by `accelerate config`; changing `num_processes` requires picking the matching `accelerate-{N}.yaml` file.
- State dict files are expected in `.safetensors` format (preferred) or legacy `.bin`/`.pt` torch checkpoints; loading handles both transparently.