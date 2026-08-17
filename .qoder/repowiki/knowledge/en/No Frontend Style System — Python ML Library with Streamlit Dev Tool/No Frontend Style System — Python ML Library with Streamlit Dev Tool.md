---
kind: frontend_style
name: No Frontend Style System — Python ML Library with Streamlit Dev Tool
category: frontend_style
scope:
    - '**'
source_files:
    - examples/dev_tools/webui.py
---

This repository is a Python machine learning library (DiffSynth-Studio) focused on diffusion model training and inference. It does not contain a frontend styling system, CSS framework, or UI theme layer.

The only user-facing interface code is `examples/dev_tools/webui.py`, which builds a development web UI using **Streamlit** (`import streamlit as st`). This UI is a thin, programmatic wrapper around the library's pipelines — it dynamically introspects pipeline classes and their `from_pretrained`/`__call__` signatures to generate form controls (text inputs, number inputs, file uploaders, checkboxes). Styling is entirely delegated to Streamlit's built-in theming via `st.set_page_config(layout="wide")`; there are no custom CSS files, SCSS, Tailwind configs, design tokens, or component libraries.

Documentation is rendered through Sphinx (see `docs/en/conf.py`, `docs/zh/conf.py`) and hosted on ReadTheDocs; styling is handled by the default Sphinx/ReadTheDocs themes, not by any project-specific CSS.

In summary: there is no frontend style system in this repo. The sole UI is a Streamlit-based developer tool that relies on Streamlit's default appearance, and documentation uses Sphinx's standard theming.