---
kind: error_handling
name: Python Exception-Based Error Handling with Standard Library Exceptions
category: error_handling
scope:
    - '**'
source_files:
    - diffsynth/core/data/operators.py
    - diffsynth/core/loader/config.py
    - diffsynth/core/device/npu_compatible_device.py
    - diffsynth/diffusion/base_pipeline.py
    - diffsynth/diffusion/training_module.py
---

The DiffSynth-Studio codebase uses Python's standard exception system for error handling, relying on built-in exceptions rather than custom error types or a centralized error framework.

**Exception Types Used:**
- `ValueError`: Most common for invalid arguments and configuration errors (e.g., unsupported file types, invalid model configurations, wrong parameter values)
- `RuntimeError`: Used for runtime failures like missing distributed communication backends
- `FileNotFoundError`: For missing input files in data processing scripts
- `NotImplementedError`: For abstract base classes and unimplemented methods
- `ImportError`: For missing dependencies like PyTorch version requirements
- `ModuleNotFoundError`: For optional dependency availability checks

**Error Propagation Patterns:**
- Direct exception raising with descriptive messages: `raise ValueError(f"Unsupported file: {data}")`
- Configuration validation through dedicated methods that raise exceptions early
- Optional dependency handling via try/except blocks around imports
- Graceful degradation with warnings instead of exceptions in some data loading scenarios

**Key Implementation Points:**
- Data processing operators validate inputs and raise specific exceptions for unsupported operations
- Model configuration validation occurs in `ModelConfig.check_input()` method
- Device compatibility checks raise `RuntimeError` when no suitable backend is found
- Training module validates model paths and formats before processing
- Pipeline operations check VRAM management state before allowing certain operations

**Absence of Centralized Error Framework:**
No custom exception hierarchy, error codes, or centralized error logging system. Each module handles its own errors using standard Python exceptions. There is no middleware pattern for error handling across the pipeline.