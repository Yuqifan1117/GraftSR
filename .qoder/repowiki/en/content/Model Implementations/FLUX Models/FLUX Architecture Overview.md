# FLUX Architecture Overview

<cite>
**Referenced Files in This Document**
- [flux_dit.py](file://diffsynth/models/flux_dit.py)
- [flux2_dit.py](file://diffsynth/models/flux2_dit.py)
- [flux_image.py](file://diffsynth/pipelines/flux_image.py)
- [flux2_image.py](file://diffsynth/pipelines/flux2_image.py)
- [flux_text_encoder_clip.py](file://diffsynth/models/flux_text_encoder_clip.py)
- [flux_text_encoder_t5.py](file://diffsynth/models/flux_text_encoder_t5.py)
- [flux_vae.py](file://diffsynth/models/flux_vae.py)
- [flux2_text_encoder.py](file://diffsynth/models/flux2_text_encoder.py)
- [flux2_vae.py](file://diffsynth/models/flux2_vae.py)
- [FLUX.md](file://docs/en/Model_Details/FLUX.md)
- [FLUX2.md](file://docs/en/Model_Details/FLUX2.md)
</cite>

## Table of Contents
1. Introduction
2. Project Structure
3. Core Components
4. Architecture Overview
5. Detailed Component Analysis
6. Dependency Analysis
7. Performance Considerations
8. Troubleshooting Guide
9. Conclusion

## Introduction
This document explains the FLUX models in ODTSR-edit with a focus on architecture, design principles, and data flow from text prompts through diffusion to image generation. It contrasts FLUX.1 and FLUX.2 architectures, details the DiT (Diffusion Transformer) blocks and attention mechanisms, and shows how text encoders and VAE components integrate into the pipelines. It also covers performance characteristics, memory usage patterns, and scalability considerations.

## Project Structure
The FLUX implementations are organized around:
- Pipelines that orchestrate inference steps and manage model loading
- DiT backbones for FLUX.1 and FLUX.2
- Text encoders (CLIP/T5 for FLUX.1; Mistral3-based for FLUX.2)
- VAEs for encoding/decoding latent space

```mermaid
graph TB
subgraph "Pipelines"
P1["FluxImagePipeline"]
P2["Flux2ImagePipeline"]
end
subgraph "Text Encoders"
TE1A["CLIP Encoder"]
TE1B["T5 Encoder"]
TE2["Mistral3-based Encoder"]
end
subgraph "DiT Backbones"
D1["FluxDiT (FLUX.1)"]
D2["Flux2DiT (FLUX.2)"]
end
subgraph "VAEs"
V1["FluxVAE (Encoder/Decoder)"]
V2["Flux2VAE"]
end
P1 --> TE1A
P1 --> TE1B
P1 --> D1
P1 --> V1
P2 --> TE2
P2 --> D2
P2 --> V2
```

**Diagram sources**
- [flux_image.py:57-106](file://diffsynth/pipelines/flux_image.py#L57-L106)
- [flux2_image.py:21-46](file://diffsynth/pipelines/flux2_image.py#L21-L46)
- [flux_dit.py:277-298](file://diffsynth/models/flux_dit.py#L277-L298)
- [flux2_dit.py:633-793](file://diffsynth/models/flux2_dit.py#L633-L793)
- [flux_text_encoder_clip.py:75-113](file://diffsynth/models/flux_text_encoder_clip.py#L75-L113)
- [flux_text_encoder_t5.py:5-44](file://diffsynth/models/flux_text_encoder_t5.py#L5-L44)
- [flux2_text_encoder.py:4-58](file://diffsynth/models/flux2_text_encoder.py#L4-L58)
- [flux_vae.py:296-435](file://diffsynth/models/flux_vae.py#L296-L435)
- [flux2_vae.py:1-120](file://diffsynth/models/flux2_vae.py#L1-L120)

**Section sources**
- [FLUX.md:53-146](file://docs/en/Model_Details/FLUX.md#L53-L146)
- [FLUX2.md:60-98](file://docs/en/Model_Details/FLUX2.md#L60-L98)

## Core Components
- FluxImagePipeline orchestrates FLUX.1 inference with units for prompt embedding, noise initialization, control signals (ControlNet, IP-Adapter), entity control, and more. It uses FlowMatchScheduler and integrates CLIP and T5 text encoders with FluxDiT and FluxVAE.
- Flux2ImagePipeline orchestrates FLUX.2 inference using a single text encoder (Mistral3-based), Flux2DiT, and Flux2VAE. It supports edit latents and different positional ID schemes.
- FluxDiT implements joint and single transformer blocks with RoPE embeddings, AdaLayerNorm variants, and patch/unpatch operations.
- Flux2DiT provides parallel self-attention blocks and fused QKV/MLP projections, with SwiGLU-style activations and advanced attention processors.
- Text encoders: CLIP pooled output plus T5 sequence outputs for FLUX.1; Mistral3 hidden states stacked for FLUX.2.
- VAEs: FLUX.1 uses separate encoder/decoder with tiled inference support; FLUX.2 uses a unified VAE module with extensive normalization and attention options.

**Section sources**
- [flux_image.py:57-106](file://diffsynth/pipelines/flux_image.py#L57-L106)
- [flux2_image.py:21-46](file://diffsynth/pipelines/flux2_image.py#L21-L46)
- [flux_dit.py:277-298](file://diffsynth/models/flux_dit.py#L277-L298)
- [flux2_dit.py:633-793](file://diffsynth/models/flux2_dit.py#L633-L793)
- [flux_text_encoder_clip.py:75-113](file://diffsynth/models/flux_text_encoder_clip.py#L75-L113)
- [flux_text_encoder_t5.py:5-44](file://diffsynth/models/flux_text_encoder_t5.py#L5-L44)
- [flux2_text_encoder.py:4-58](file://diffsynth/models/flux2_text_encoder.py#L4-L58)
- [flux_vae.py:296-435](file://diffsynth/models/flux_vae.py#L296-L435)
- [flux2_vae.py:1-120](file://diffsynth/models/flux2_vae.py#L1-L120)

## Architecture Overview
High-level data flow for FLUX.1 and FLUX.2:

```mermaid
sequenceDiagram
participant User as "User"
participant Pipe as "Pipeline"
participant TE as "Text Encoder(s)"
participant DiT as "DiT Backbone"
participant VAE as "VAE Decoder"
Note over User,Pipe : FLUX.1 Pipeline
User->>Pipe : Prompt + parameters
Pipe->>TE : Encode prompt (CLIP pooled + T5 seq)
TE-->>Pipe : prompt_emb, pooled_prompt_emb, text_ids
Pipe->>Pipe : Prepare latents/noise, image_ids, guidance
loop Denoising Steps
Pipe->>DiT : Forward(latents, timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids)
DiT-->>Pipe : Noise prediction
Pipe->>Pipe : Scheduler step update
end
Pipe->>VAE : Decode latents
VAE-->>Pipe : Image
Pipe-->>User : Generated image
Note over User,Pipe : FLUX.2 Pipeline
User->>Pipe : Prompt + parameters
Pipe->>TE : Encode prompt (Mistral3 hidden states)
TE-->>Pipe : prompt_emb, text_ids
Pipe->>Pipe : Prepare latents/noise, image_ids
loop Denoising Steps
Pipe->>DiT : Forward(latents, timestep, prompt_emb, text_ids, img_ids)
DiT-->>Pipe : Noise prediction
Pipe->>Pipe : Scheduler step update
end
Pipe->>VAE : Decode latents
VAE-->>Pipe : Image
Pipe-->>User : Generated image
```

**Diagram sources**
- [flux_image.py:180-291](file://diffsynth/pipelines/flux_image.py#L180-L291)
- [flux2_image.py:74-139](file://diffsynth/pipelines/flux2_image.py#L74-L139)
- [flux_dit.py:277-298](file://diffsynth/models/flux_dit.py#L277-L298)
- [flux2_dit.py:633-793](file://diffsynth/models/flux2_dit.py#L633-L793)
- [flux_vae.py:296-366](file://diffsynth/models/flux_vae.py#L296-L366)
- [flux2_vae.py:1-120](file://diffsynth/models/flux2_vae.py#L1-L120)

## Detailed Component Analysis

### FLUX.1 DiT (FluxDiT)
- Joint and Single Transformer Blocks:
  - FluxJointTransformerBlock processes two streams (e.g., text and image) via joint attention with RoPE and AdaLayerNorm conditioning.
  - FluxSingleTransformerBlock handles image-only stream with parallelized QKV/MLP projection and gated outputs.
- Positional Encoding:
  - RoPEEmbedding constructs multi-axis rotary embeddings for spatial positions.
- Patching:
  - patchify/unpatchify convert between patch tokens and latent grids.
- Entity Control and Masks:
  - construct_mask builds attention masks for multiple entities and global context.
- Conditioning:
  - Time and guidance embeddings via TimestepEmbeddings; pooled text embedder projects CLIP pooled features.

```mermaid
classDiagram
class FluxDiT {
+pos_embedder
+time_embedder
+guidance_embedder
+pooled_text_embedder
+context_embedder
+x_embedder
+blocks
+single_blocks
+final_norm_out
+final_proj_out
+patchify(hidden_states)
+unpatchify(hidden_states, height, width)
+prepare_image_ids(latents)
+construct_mask(entity_masks, prompt_seq_len, image_seq_len)
+process_entity_masks(...)
}
class FluxJointTransformerBlock {
+norm1_a
+norm1_b
+attn
+norm2_a
+ff_a
+norm2_b
+ff_b
+forward(hidden_states_a, hidden_states_b, temb, image_rotary_emb, attn_mask, ipadapter_kwargs_list)
}
class FluxSingleTransformerBlock {
+norm
+to_qkv_mlp
+norm_q_a
+norm_k_a
+proj_out
+forward(hidden_states_a, hidden_states_b, temb, image_rotary_emb, attn_mask, ipadapter_kwargs_list)
}
FluxDiT --> FluxJointTransformerBlock : "repeated blocks"
FluxDiT --> FluxSingleTransformerBlock : "repeated blocks"
```

**Diagram sources**
- [flux_dit.py:108-149](file://diffsynth/models/flux_dit.py#L108-L149)
- [flux_dit.py:205-259](file://diffsynth/models/flux_dit.py#L205-L259)
- [flux_dit.py:277-298](file://diffsynth/models/flux_dit.py#L277-L298)

**Section sources**
- [flux_dit.py:45-105](file://diffsynth/models/flux_dit.py#L45-L105)
- [flux_dit.py:152-186](file://diffsynth/models/flux_dit.py#L152-L186)
- [flux_dit.py:277-298](file://diffsynth/models/flux_dit.py#L277-L298)
- [flux_dit.py:299-387](file://diffsynth/models/flux_dit.py#L299-L387)

### FLUX.2 DiT (Flux2DiT)
- Attention Modules:
  - Flux2Attention with optional added KV projections and RMSNorm on Q/K.
  - Flux2ParallelSelfAttention fuses QKV and MLP input/output projections for efficiency.
- Transformers:
  - Flux2SingleTransformerBlock applies modulation parameters and parallel attention.
  - Flux2TransformerBlock combines image and text streams with separate norms and FFNs.
- Activations:
  - Flux2SwiGLU and Flux2FeedForward use fused linear layers and SiLU gating.
- Rotary Embeddings:
  - get_1d_rotary_pos_embed and apply_rotary_emb provide flexible RoPE computation.

```mermaid
classDiagram
class Flux2Attention {
+to_q
+to_k
+to_v
+norm_q
+norm_k
+to_out
+forward(hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
}
class Flux2ParallelSelfAttention {
+to_qkv_mlp_proj
+mlp_act_fn
+norm_q
+norm_k
+to_out
+forward(hidden_states, attention_mask, image_rotary_emb)
}
class Flux2SingleTransformerBlock {
+norm
+attn
+forward(hidden_states, encoder_hidden_states, temb_mod_params_img, image_rotary_emb)
}
class Flux2TransformerBlock {
+norm1
+norm1_context
+attn
+norm2
+ff
+norm2_context
+ff_context
+forward(hidden_states, encoder_hidden_states, temb_mod_params_img, temb_mod_params_txt, image_rotary_emb)
}
Flux2SingleTransformerBlock --> Flux2ParallelSelfAttention : "uses"
Flux2TransformerBlock --> Flux2Attention : "uses"
```

**Diagram sources**
- [flux2_dit.py:435-503](file://diffsynth/models/flux2_dit.py#L435-L503)
- [flux2_dit.py:560-631](file://diffsynth/models/flux2_dit.py#L560-L631)
- [flux2_dit.py:633-700](file://diffsynth/models/flux2_dit.py#L633-L700)
- [flux2_dit.py:702-793](file://diffsynth/models/flux2_dit.py#L702-L793)

**Section sources**
- [flux2_dit.py:325-365](file://diffsynth/models/flux2_dit.py#L325-L365)
- [flux2_dit.py:367-433](file://diffsynth/models/flux2_dit.py#L367-L433)
- [flux2_dit.py:505-558](file://diffsynth/models/flux2_dit.py#L505-L558)
- [flux2_dit.py:633-793](file://diffsynth/models/flux2_dit.py#L633-L793)

### Text Encoders
- FLUX.1:
  - CLIP encoder produces pooled embeddings and optionally hidden states.
  - T5 encoder returns sequence embeddings for long-context prompts.
- FLUX.2:
  - Mistral3-based encoder stacks intermediate hidden states across selected layers to form rich prompt representations.

```mermaid
flowchart TD
Start(["Prompt"]) --> Clip["CLIP Encoder"]
Start --> T5["T5 Encoder"]
Clip --> Pooled["Pooled Embeddings"]
T5 --> SeqEmb["Sequence Embeddings"]
Pooled --> Concat["Concat/Project for DiT"]
SeqEmb --> Concat
Concat --> DiTInput["DiT Input"]
```

**Diagram sources**
- [flux_text_encoder_clip.py:75-113](file://diffsynth/models/flux_text_encoder_clip.py#L75-L113)
- [flux_text_encoder_t5.py:5-44](file://diffsynth/models/flux_text_encoder_t5.py#L5-L44)
- [flux2_text_encoder.py:4-58](file://diffsynth/models/flux2_text_encoder.py#L4-L58)

**Section sources**
- [flux_text_encoder_clip.py:75-113](file://diffsynth/models/flux_text_encoder_clip.py#L75-L113)
- [flux_text_encoder_t5.py:5-44](file://diffsynth/models/flux_text_encoder_t5.py#L5-L44)
- [flux2_text_encoder.py:4-58](file://diffsynth/models/flux2_text_encoder.py#L4-L58)

### VAE Components
- FLUX.1 VAE:
  - Separate encoder and decoder with GroupNorm, ResnetBlocks, and attention blocks.
  - Supports tiled inference to reduce VRAM usage during encoding/decoding.
- FLUX.2 VAE:
  - Comprehensive module with ResnetBlock2D, Downsample2D, Upsample2D, and attention layers.
  - Extensive normalization options and activation mappings.

```mermaid
flowchart TD
Latents["Latents"] --> VAE_Decoder["VAE Decoder"]
VAE_Decoder --> Image["Output Image"]
Image --> VAE_Encoder["VAE Encoder"]
VAE_Encoder --> Latents
```

**Diagram sources**
- [flux_vae.py:296-366](file://diffsynth/models/flux_vae.py#L296-L366)
- [flux_vae.py:368-435](file://diffsynth/models/flux_vae.py#L368-L435)
- [flux2_vae.py:1-120](file://diffsynth/models/flux2_vae.py#L1-L120)

**Section sources**
- [flux_vae.py:296-435](file://diffsynth/models/flux_vae.py#L296-L435)
- [flux2_vae.py:1-120](file://diffsynth/models/flux2_vae.py#L1-L120)

### Pipeline Orchestration
- FLUX.1 Pipeline:
  - Units handle shape checking, noise initialization, prompt embedding (CLIP+T5), image IDs, guidance, Kontext, ControlNet, IP-Adapter, entity control, NexusGen, TeaCache, Flex, Step1X, ValueControl, LoRA encode.
  - Iterative denoising loop calls model_fn_flux_image which feeds DiT with all conditioning.
- FLUX.2 Pipeline:
  - Units handle shape checking, prompt embedding (Mistral3 or Qwen3), noise initialization, input/edit image embedding, and image IDs.
  - Iterative denoising loop calls model_fn_flux2 which concatenates edit latents if provided.

```mermaid
sequenceDiagram
participant U as "User"
participant P as "FluxImagePipeline"
participant U1 as "Units"
participant M as "model_fn_flux_image"
participant D as "FluxDiT"
participant S as "Scheduler"
participant V as "VAE Decoder"
U->>P : __call__(prompt, params)
P->>U1 : Run units (embeddings, IDs, controls)
U1-->>P : inputs_shared, inputs_posi, inputs_nega
loop Steps
P->>M : cfg_guided_model_fn(..., timestep)
M->>D : forward(latents, timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids)
D-->>M : noise_pred
M-->>P : noise_pred
P->>S : step(update latents)
end
P->>V : decode(latents)
V-->>P : image
P-->>U : image
```

**Diagram sources**
- [flux_image.py:180-291](file://diffsynth/pipelines/flux_image.py#L180-L291)
- [flux_image.py:336-395](file://diffsynth/pipelines/flux_image.py#L336-L395)
- [flux_image.py:417-444](file://diffsynth/pipelines/flux_image.py#L417-L444)
- [flux_image.py:447-487](file://diffsynth/pipelines/flux_image.py#L447-L487)
- [flux_image.py:490-516](file://diffsynth/pipelines/flux_image.py#L490-L516)
- [flux_image.py:519-609](file://diffsynth/pipelines/flux_image.py#L519-L609)
- [flux_image.py:611-665](file://diffsynth/pipelines/flux_image.py#L611-L665)
- [flux_image.py:667-693](file://diffsynth/pipelines/flux_image.py#L667-L693)
- [flux_image.py:695-704](file://diffsynth/pipelines/flux_image.py#L695-L704)
- [flux_image.py:705-741](file://diffsynth/pipelines/flux_image.py#L705-L741)
- [flux_image.py:744-758](file://diffsynth/pipelines/flux_image.py#L744-L758)
- [flux_image.py:761-789](file://diffsynth/pipelines/flux_image.py#L761-L789)

**Section sources**
- [flux_image.py:57-106](file://diffsynth/pipelines/flux_image.py#L57-L106)
- [flux_image.py:180-291](file://diffsynth/pipelines/flux_image.py#L180-L291)
- [flux2_image.py:74-139](file://diffsynth/pipelines/flux2_image.py#L74-L139)

## Dependency Analysis
- FLUX.1 depends on:
  - Two text encoders (CLIP and T5)
  - FluxDiT with joint/single blocks
  - FluxVAE encoder/decoder
  - Optional modules: ControlNet, IP-Adapter, InfiniteYou, EliGen, NexusGen, Step1X, ValueController, LoRA
- FLUX.2 depends on:
  - Single Mistral3-based text encoder
  - Flux2DiT with parallel attention and fused projections
  - Flux2VAE

```mermaid
graph TB
P1["FluxImagePipeline"] --> TE1["CLIP Encoder"]
P1 --> TE2["T5 Encoder"]
P1 --> D1["FluxDiT"]
P1 --> V1["FluxVAE"]
P1 --> CN["ControlNet"]
P1 --> IP["IP-Adapter"]
P1 --> IY["InfiniteYou"]
P1 --> EL["EliGen"]
P1 --> NG["NexusGen"]
P1 --> ST["Step1X"]
P1 --> VC["ValueController"]
P1 --> LR["LoRA"]
P2["Flux2ImagePipeline"] --> TE3["Mistral3 Encoder"]
P2 --> D2["Flux2DiT"]
P2 --> V2["Flux2VAE"]
```

**Diagram sources**
- [flux_image.py:57-106](file://diffsynth/pipelines/flux_image.py#L57-L106)
- [flux2_image.py:21-46](file://diffsynth/pipelines/flux2_image.py#L21-L46)

**Section sources**
- [FLUX.md:53-146](file://docs/en/Model_Details/FLUX.md#L53-L146)
- [FLUX2.md:60-98](file://docs/en/Model_Details/FLUX2.md#L60-L98)

## Performance Considerations
- Memory Usage Patterns:
  - FLUX.1 VAE supports tiled inference to reduce peak VRAM during encoding/decoding.
  - FLUX.2 VAE includes extensive normalization and attention options that can be tuned for memory/performance trade-offs.
  - Both pipelines support gradient checkpointing and VRAM management strategies (offload/onload).
- Attention Efficiency:
  - FLUX.2 uses fused QKV/MLP projections and parallel attention to reduce compute overhead.
  - FLUX.1 employs joint attention with RoPE and optional IP-Adapter integration.
- Scalability:
  - FLUX.2’s Mistral3-based encoder stacks intermediate hidden states for richer representations but increases memory.
  - FLUX.1’s dual encoders (CLIP+T5) allow flexible context length tuning via T5 sequence length.
- Scheduler and Steps:
  - FlowMatchScheduler is used in both pipelines; number of steps impacts quality vs speed.
  - CFG guidance doubles compute when enabled.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- Insufficient VRAM:
  - Enable tiled VAE inference for FLUX.1.
  - Use low VRAM configurations and VRAM management settings as documented in examples.
- Long Prompts:
  - Adjust T5 sequence length for FLUX.1 to balance context and memory.
  - For FLUX.2, ensure tokenizer and encoder configs match expected max lengths.
- Attention Errors:
  - Ensure PyTorch version supports scaled_dot_product_attention for FLUX.2 processors.
- Model Loading:
  - Verify model_configs and origin_file_pattern match repository structure.

**Section sources**
- [flux_vae.py:333-343](file://diffsynth/models/flux_vae.py#L333-L343)
- [flux2_dit.py:367-374](file://diffsynth/models/flux2_dit.py#L367-L374)
- [FLUX.md:106-146](file://docs/en/Model_Details/FLUX.md#L106-L146)
- [FLUX2.md:77-98](file://docs/en/Model_Details/FLUX2.md#L77-L98)

## Conclusion
FLUX.1 and FLUX.2 represent two generations of DiT-based image generation models within ODTSR-edit. FLUX.1 leverages dual text encoders and joint/single transformer blocks with robust control integrations, while FLUX.2 advances efficiency and representation power through fused projections, parallel attention, and a Mistral3-based encoder. The pipelines orchestrate these components with modular units, enabling flexible conditioning and scalable inference. Proper configuration of tiling, VRAM management, and attention backends ensures optimal performance across diverse hardware constraints.