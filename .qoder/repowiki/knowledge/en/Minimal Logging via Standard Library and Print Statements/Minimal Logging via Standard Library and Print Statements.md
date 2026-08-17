---
kind: logging_system
name: Minimal Logging via Standard Library and Print Statements
category: logging_system
scope:
    - '**'
source_files:
    - diffsynth/diffusion/logger.py
    - diffynith/models/nexus_gen_ar_model.py
    - diffsynth/diffusion/runner.py
    - diffsynth/diffusion/base_pipeline.py
    - diffsynth/diffusion/training_module.py
---

This repository does not implement a dedicated logging framework or structured logging system. Instead, it relies on two ad-hoc approaches:

1. **Standard library `logging` (sparse usage)**: Only two model files import the built-in `logging` module — `diffsynth/models/anima_dit.py` imports it but never uses it, and `diffsynth/models/nexus_gen_ar_model.py` creates a module-level logger via `logger = logging.get_logger(__name__)`. No other code in the core package uses this logger instance; there is no central configuration, formatter, handler setup, or log-level policy applied across the codebase.

2. **`print()` statements everywhere**: The vast majority of runtime output comes from direct `print()` calls scattered throughout training scripts, pipelines, and utilities. Examples include progress/status messages in `diffsynth/diffusion/base_pipeline.py`, `diffsynth/diffusion/training_module.py`, `diffsynth/core/data/unified_dataset.py`, `diffsynth/core/device/npu_compatible_device.py`, benchmark scripts under `benchmark/`, and tooling scripts like `detect_texture_regions.py`. These prints are unstructured, use mixed languages (English and Chinese), and have no consistent severity level or formatting convention.

3. **No centralized logging configuration**: There is no `logging.config`, no root logger setup, no environment-variable-driven log levels, no file/console sink configuration, and no structured JSON/logfmt output. The only logging-related artifact is `diffsynth/diffusion/logger.py`, which defines a `ModelLogger` class that is actually a **checkpoint saver** (saves `.safetensors` weights at step/epoch boundaries via Accelerator) — it has nothing to do with emitting log messages.

4. **Training loop integration**: The training runner (`diffsynth/diffusion/runner.py`) invokes `ModelLogger.on_step_end`, `on_epoch_end`, and `on_training_end` purely for model checkpointing; it does not emit any logs through a logging facility.

In summary, the project has no cohesive logging system. Output is produced informally via `print()` calls and an unused standard-library logger import, with no unified levels, sinks, or structure.