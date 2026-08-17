# Text Encoder Integration

<cite>
**Referenced Files in This Document**
- [ltx2_text_encoder.py](file://diffsynth/models/ltx2_text_encoder.py)
- [ltx2_audio_video.py](file://diffsynth/pipelines/ltx2_audio_video.py)
- [ltx2_dit.py](file://diffsynth/models/ltx2_dit.py)
- [ltx2_common.py](file://diffsynth/models/ltx2_common.py)
- [LTX-2.md](file://docs/en/Model_Details/LTX-2.md)
- [LTX-2.3-T2AV-OneStage.py](file://examples/ltx2/model_inference/LTX-2.3-T2AV-OneStage.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This document explains the LTX2 text encoder integration that processes textual prompts for audio-video generation. It covers the transformer-based architecture, prompt embedding strategies, cross-modal conditioning mechanisms, and how textual information influences both video and audio modalities. It also documents the text preprocessing pipeline, tokenization strategies, configuration options for different text encoders, and practical examples of prompt engineering to achieve optimal results.

## Project Structure
The LTX2 text encoder is implemented as a Gemma-based model with specialized post-processing modules to produce separate video and audio contexts. The pipeline orchestrates text encoding, latent initialization, multi-modal conditioning, and denoising through a unified transformer backbone.

```mermaid
graph TB
subgraph "Text Encoding"
T["LTXVGemmaTokenizer<br/>tokenize_with_weights()"]
E["LTX2TextEncoder<br/>(Gemma3ForConditionalGeneration)"]
P["LTX2TextEncoderPostModules<br/>feature_extractor + connectors"]
end
subgraph "Diffusion Backbone"
DIT["LTXModel<br/>(MultiModalTransformer)"]
VAE_V["Video VAE Encoder/Decoder"]
VAE_A["Audio VAE Encoder/Decoder + Vocoder"]
end
subgraph "Pipeline"
PIPE["LTX2AudioVideoPipeline<br/>units: PromptEmbedder, NoiseInitializer,<br/>Input/Retake Embedders, In-Context Embedder"]
end
T --> E --> P --> DIT
PIPE --> T
PIPE --> DIT
DIT --> VAE_V
DIT --> VAE_A
```

**Diagram sources**
- [ltx2_text_encoder.py:90-150](file://diffsynth/models/ltx2_text_encoder.py#L90-L150)
- [ltx2_text_encoder.py:406-464](file://diffsynth/models/ltx2_text_encoder.py#L406-L464)
- [ltx2_audio_video.py:298-328](file://diffsynth/pipelines/ltx2_audio_video.py#L298-L328)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

**Section sources**
- [ltx2_text_encoder.py:1-88](file://diffsynth/models/ltx2_text_encoder.py#L1-L88)
- [ltx2_audio_video.py:28-148](file://diffsynth/pipelines/ltx2_audio_video.py#L28-L148)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

## Core Components
- LTX2TextEncoder: A Gemma3-based conditional generation model configured for long sequences and multimodal tokens.
- LTXVGemmaTokenizer: Tokenizer wrapper ensuring left padding and consistent output format for downstream consumption.
- LTX2TextEncoderPostModules: Feature extractor and 1D connectors producing separate video and audio embeddings from Gemma hidden states.
- LTX2AudioVideoPipeline: Orchestrates text encoding, noise initialization, modality-specific conditioning, and diffusion steps.
- LTXModel: Multi-modal transformer backbone integrating video and audio latents with text context via cross-attention.

Key responsibilities:
- Text preprocessing and tokenization
- Hidden state aggregation across layers
- Modality-specific projection and positional encoding
- Cross-modal conditioning within the diffusion backbone

**Section sources**
- [ltx2_text_encoder.py:11-88](file://diffsynth/models/ltx2_text_encoder.py#L11-L88)
- [ltx2_text_encoder.py:90-150](file://diffsynth/models/ltx2_text_encoder.py#L90-L150)
- [ltx2_text_encoder.py:406-464](file://diffsynth/models/ltx2_text_encoder.py#L406-L464)
- [ltx2_audio_video.py:298-328](file://diffsynth/pipelines/ltx2_audio_video.py#L298-L328)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

## Architecture Overview
The LTX2 text encoder integrates with the diffusion pipeline to condition both video and audio generation. Prompts are tokenized, encoded by a Gemma3 transformer, then projected into separate video and audio contexts using learned connectors. These contexts are injected into the LTXModel via cross-attention during denoising.

```mermaid
sequenceDiagram
participant User as "User"
participant Pipe as "LTX2AudioVideoPipeline"
participant Tok as "LTXVGemmaTokenizer"
participant Enc as "LTX2TextEncoder"
participant Post as "LTX2TextEncoderPostModules"
participant Dit as "LTXModel"
participant VAEV as "Video VAE Decoder"
participant VAEA as "Audio VAE Decoder + Vocoder"
User->>Pipe : __call__(prompt, negative_prompt, ...)
Pipe->>Tok : tokenize_with_weights(prompt)
Tok-->>Pipe : input_ids, attention_mask
Pipe->>Enc : forward(input_ids, attention_mask, output_hidden_states=True)
Enc-->>Pipe : hidden_states (per-layer)
Pipe->>Post : process_hidden_states(hidden_states, mask)
Post-->>Pipe : video_context, audio_context, binary_mask
Pipe->>Dit : denoise(video_latents, audio_latents, video_context, audio_context, positions, timestep)
Dit-->>Pipe : noise_pred_video, noise_pred_audio
Pipe->>VAEV : decode(video_latents)
VAEV-->>Pipe : video_frames
Pipe->>VAEA : decode(audio_latents) -> vocoder
VAEA-->>Pipe : audio_waveform
Pipe-->>User : video, audio
```

**Diagram sources**
- [ltx2_audio_video.py:298-328](file://diffsynth/pipelines/ltx2_audio_video.py#L298-L328)
- [ltx2_audio_video.py:648-732](file://diffsynth/pipelines/ltx2_audio_video.py#L648-L732)
- [ltx2_text_encoder.py:406-464](file://diffsynth/models/ltx2_text_encoder.py#L406-L464)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

## Detailed Component Analysis

### LTX2TextEncoder (Gemma3-based)
- Extends Gemma3ForConditionalGeneration with a custom config optimized for long sequences and multimodal tokens.
- Uses sliding window and full attention patterns across layers, large vocab size, and extended position embeddings.
- Provides hidden states per layer for subsequent feature extraction.

```mermaid
classDiagram
class LTX2TextEncoder {
+__init__()
+forward(input_ids, attention_mask, output_hidden_states)
}
class GemmaFeaturesExtractorProjLinear {
+aggregate_embed : Linear
+forward(hidden_states, attention_mask, padding_side)
}
class GemmaSeperatedFeaturesExtractorProjLinear {
+video_aggregate_embed : Linear
+audio_aggregate_embed : Linear
+forward(hidden_states, attention_mask, padding_side)
}
LTX2TextEncoder --> GemmaFeaturesExtractorProjLinear : "uses"
LTX2TextEncoder --> GemmaSeperatedFeaturesExtractorProjLinear : "uses (LTX-2.3)"
```

**Diagram sources**
- [ltx2_text_encoder.py:11-88](file://diffsynth/models/ltx2_text_encoder.py#L11-L88)
- [ltx2_text_encoder.py:153-183](file://diffsynth/models/ltx2_text_encoder.py#L153-L183)
- [ltx2_text_encoder.py:185-217](file://diffsynth/models/ltx2_text_encoder.py#L185-L217)

**Section sources**
- [ltx2_text_encoder.py:11-88](file://diffsynth/models/ltx2_text_encoder.py#L11-L88)

### LTXVGemmaTokenizer
- Wraps AutoTokenizer with left padding and pad token fallback to eos.
- Returns structured tuples of (token_id, attention_mask), optionally including indices.
- Ensures compatibility with Gemma models and LTXV processing requirements.

```mermaid
flowchart TD
Start(["Input text"]) --> Strip["Strip whitespace"]
Strip --> Encode["AutoTokenizer(..., padding='max_length', truncation=True)"]
Encode --> Extract["Extract input_ids, attention_mask"]
Extract --> BuildTuples["Build list of (token_id, attn, index)"]
BuildTuples --> OptionalFilter{"return_word_ids?"}
OptionalFilter --> |Yes| ReturnWithIdx["Return dict with index included"]
OptionalFilter --> |No| ReturnWithoutIdx["Return dict without index"]
ReturnWithIdx --> End(["Output"])
ReturnWithoutIdx --> End
```

**Diagram sources**
- [ltx2_text_encoder.py:90-150](file://diffsynth/models/ltx2_text_encoder.py#L90-L150)

**Section sources**
- [ltx2_text_encoder.py:90-150](file://diffsynth/models/ltx2_text_encoder.py#L90-L150)

### LTX2TextEncoderPostModules
- Aggregates per-layer hidden states with normalization and concatenation.
- Produces separate video and audio contexts via dedicated connectors.
- Supports two modes: shared features (LTX-2) and separated features (LTX-2.3).

```mermaid
classDiagram
class LTX2TextEncoderPostModules {
+create_embeddings(video_features, audio_features, additive_attention_mask)
+process_hidden_states(hidden_states, attention_mask, padding_side)
}
class Embeddings1DConnector {
+transformer_1d_blocks : ModuleList
+learnable_registers : Parameter?
+forward(hidden_states, attention_mask)
}
LTX2TextEncoderPostModules --> Embeddings1DConnector : "video/audio connectors"
```

**Diagram sources**
- [ltx2_text_encoder.py:406-464](file://diffsynth/models/ltx2_text_encoder.py#L406-L464)
- [ltx2_text_encoder.py:277-404](file://diffsynth/models/ltx2_text_encoder.py#L277-L404)

**Section sources**
- [ltx2_text_encoder.py:406-464](file://diffsynth/models/ltx2_text_encoder.py#L406-L464)
- [ltx2_text_encoder.py:277-404](file://diffsynth/models/ltx2_text_encoder.py#L277-L404)

### Prompt Embedding Pipeline Unit
- Converts prompt text to token IDs and masks.
- Calls text encoder with output_hidden_states=True.
- Processes hidden states to obtain video_context and audio_context.

```mermaid
sequenceDiagram
participant U as "PromptEmbedder"
participant T as "LTXVGemmaTokenizer"
participant E as "LTX2TextEncoder"
participant P as "LTX2TextEncoderPostModules"
U->>T : tokenize_with_weights(text)
T-->>U : gemma tokens + masks
U->>E : forward(input_ids, attention_mask, output_hidden_states=True)
E-->>U : hidden_states (per layer)
U->>P : process_hidden_states(hidden_states, mask)
P-->>U : video_context, audio_context, binary_mask
```

**Diagram sources**
- [ltx2_audio_video.py:298-328](file://diffsynth/pipelines/ltx2_audio_video.py#L298-L328)
- [ltx2_text_encoder.py:406-464](file://diffsynth/models/ltx2_text_encoder.py#L406-L464)

**Section sources**
- [ltx2_audio_video.py:298-328](file://diffsynth/pipelines/ltx2_audio_video.py#L298-L328)

### Cross-Modal Conditioning in LTXModel
- Video and audio latents are patchified and combined with reference/in-context conditions.
- Text contexts are injected via cross-attention within transformer blocks.
- Timestep embeddings and adaptive modulation guide denoising.

```mermaid
flowchart TD
A["video_latents, audio_latents"] --> Patchify["Patchify video/audio"]
Patchify --> ConcatRef["Concat ref frames / in-context latents"]
ConcatRef --> CrossAttn["Cross-attention with video_context, audio_context"]
CrossAttn --> Modulate["AdaLN modulation with timestep"]
Modulate --> Output["noise_pred_video, noise_pred_audio"]
```

**Diagram sources**
- [ltx2_audio_video.py:648-732](file://diffsynth/pipelines/ltx2_audio_video.py#L648-L732)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

**Section sources**
- [ltx2_audio_video.py:648-732](file://diffsynth/pipelines/ltx2_audio_video.py#L648-L732)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

## Dependency Analysis
- LTX2TextEncoder depends on Gemma3Config and transformers AutoTokenizer.
- LTX2TextEncoderPostModules uses 1D transformer blocks and RMSNorm utilities.
- Pipeline units orchestrate model loading and execution order.
- LTXModel integrates text contexts with video/audio latents via cross-attention.

```mermaid
graph LR
Tok["LTXVGemmaTokenizer"] --> Enc["LTX2TextEncoder"]
Enc --> Post["LTX2TextEncoderPostModules"]
Post --> Dit["LTXModel"]
Pipe["LTX2AudioVideoPipeline"] --> Tok
Pipe --> Dit
Dit --> VAEV["Video VAE"]
Dit --> VAEA["Audio VAE + Vocoder"]
```

**Diagram sources**
- [ltx2_text_encoder.py:11-88](file://diffsynth/models/ltx2_text_encoder.py#L11-L88)
- [ltx2_audio_video.py:298-328](file://diffsynth/pipelines/ltx2_audio_video.py#L298-L328)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

**Section sources**
- [ltx2_text_encoder.py:11-88](file://diffsynth/models/ltx2_text_encoder.py#L11-L88)
- [ltx2_audio_video.py:298-328](file://diffsynth/pipelines/ltx2_audio_video.py#L298-L328)
- [ltx2_dit.py:1278-1641](file://diffsynth/models/ltx2_dit.py#L1278-L1641)

## Performance Considerations
- Use bfloat16 precision for memory efficiency and speed.
- Enable VRAM management for dynamic loading/unloading of components.
- Prefer tiled inference for VAE decoding to reduce memory footprint.
- Adjust max_length in tokenizer to balance prompt length and performance.
- Two-stage pipelines can improve quality at higher resolutions but require additional memory.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- If prompts are truncated or ignored, verify tokenizer max_length and ensure left padding is set.
- For mismatched shapes between video and audio latents, check frame rate and duration alignment.
- When CFG is disabled unexpectedly, confirm distilled pipeline settings override cfg_scale.
- Ensure stage2_lora_config is provided when using two-stage pipelines.

**Section sources**
- [ltx2_audio_video.py:252-273](file://diffsynth/pipelines/ltx2_audio_video.py#L252-L273)
- [ltx2_audio_video.py:591-613](file://diffsynth/pipelines/ltx2_audio_video.py#L591-L613)

## Conclusion
The LTX2 text encoder leverages a powerful Gemma3 transformer to extract rich textual representations, which are then projected into modality-specific contexts for video and audio generation. The pipeline ensures robust tokenization, efficient feature aggregation, and seamless cross-modal conditioning within a unified diffusion framework. Proper configuration of tokenizer parameters, pipeline stages, and VRAM management enables high-quality audio-video synthesis from textual prompts.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Prompt Engineering Examples
- Use descriptive, concise prompts focusing on key actions, subjects, and styles.
- Include temporal cues (e.g., “slowly,” “over time”) for motion control.
- Specify audio characteristics (e.g., “clear voice,” “background music”) for synchronized sound.
- Leverage negative prompts to suppress common artifacts and undesired content.

**Section sources**
- [LTX-2.md:94-114](file://docs/en/Model_Details/LTX-2.md#L94-L114)
- [LTX-2.3-T2AV-OneStage.py:39-40](file://examples/ltx2/model_inference/LTX-2.3-T2AV-OneStage.py#L39-L40)

### Configuration Options for Text Encoders
- Tokenizer max_length: Controls maximum sequence length for tokenization.
- Padding side: Left padding recommended for chat-style prompts.
- Model dtype: bfloat16 for balanced precision and memory usage.
- Two-stage vs one-stage: Two-stage improves resolution but requires more VRAM and LoRA configuration.

**Section sources**
- [ltx2_text_encoder.py:97-112](file://diffsynth/models/ltx2_text_encoder.py#L97-L112)
- [LTX-2.md:112-116](file://docs/en/Model_Details/LTX-2.md#L112-L116)