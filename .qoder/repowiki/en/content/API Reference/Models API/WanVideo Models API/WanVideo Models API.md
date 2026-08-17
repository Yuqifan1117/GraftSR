# WanVideo Models API

<cite>
**Referenced Files in This Document**
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)
- [wan_video_motion_controller.py](file://diffsynth/models/wan_video_motion_controller.py)
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)
- [wan_video_text_encoder.py](file://diffsynth/models/wan_video_text_encoder.py)
- [wan_video_image_encoder.py](file://diffsynth/models/wan_video_image_encoder.py)
- [wan_video_mot.py](file://diffsynth/models/wan_video_mot.py)
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)
- [Wan.md](file://docs/en/Model_Details/Wan.md)
- [Wan2.1-T2V-14B.py](file://examples/wanvideo/model_inference/Wan2.1-T2V-14B.py)
- [Wan2.1-I2V-14B-720P.py](file://examples/wanvideo/model_inference/Wan2.1-I2V-14B-720P.py)
- [Wan2.1-Fun-V1.1-14B-Control-Camera.py](file://examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py)
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
10. Appendices

## Introduction
This document provides comprehensive API documentation for the WanVideo model implementations, focusing on video generation using a DiT-based architecture with motion controllers, camera control systems, animate adapters, VAE components, and text encoders. It explains temporal modeling, motion interpolation, camera trajectory control, and frame generation, and includes complete examples for text-to-video, image-to-video, and advanced control scenarios. It also addresses video format handling, temporal consistency, and performance optimization for video generation tasks.

## Project Structure
The WanVideo implementation is organized into:
- Models: DiT backbone, motion controller, camera controller, animate adapter, VAE, text encoder, image encoder, MOT and VACE extensions.
- Pipeline: Unified orchestration of units for preprocessing, conditioning, denoising, and decoding.
- Examples: Ready-to-run scripts for T2V, I2V, and camera control.

```mermaid
graph TB
subgraph "Pipeline"
P["WanVideoPipeline"]
U1["Units (ShapeChecker, NoiseInitializer, PromptEmbedder, ...)"]
U2["Post Units (S2V)"]
end
subgraph "Models"
DIT["WanModel (DiT)"]
MOT["MotWanModel"]
VACE["VaceWanModel"]
VAE["WanVideoVAE"]
TEXT["WanTextEncoder"]
IMG["WanImageEncoder"]
MOTION["WanMotionControllerModel"]
CAMERA["SimpleAdapter + Camera utilities"]
ANIMATE["WanAnimateAdapter"]
end
P --> U1
U1 --> DIT
U1 --> MOT
U1 --> VACE
U1 --> VAE
U1 --> TEXT
U1 --> IMG
U1 --> MOTION
U1 --> CAMERA
U1 --> ANIMATE
P --> U2
```

**Diagram sources**
- [wan_video.py:32-86](file://diffsynth/pipelines/wan_video.py#L32-L86)
- [wan_video_dit.py:338-423](file://diffsynth/models/wan_video_dit.py#L338-L423)
- [wan_video_motion_controller.py:7-22](file://diffsynth/models/wan_video_motion_controller.py#L7-L22)
- [wan_video_camera_controller.py:8-44](file://diffsynth/models/wan_video_camera_controller.py#L8-L44)
- [wan_video_animate_adapter.py:615-651](file://diffsynth/models/wan_video_animate_adapter.py#L615-L651)
- [wan_video_vae.py:517-618](file://diffsynth/models/wan_video_vae.py#L517-L618)
- [wan_video_text_encoder.py:212-257](file://diffsynth/models/wan_video_text_encoder.py#L212-L257)
- [wan_video_image_encoder.py:386-479](file://diffsynth/models/wan_video_image_encoder.py#L386-L479)
- [wan_video_mot.py:94-170](file://diffsynth/models/wan_video_mot.py#L94-L170)
- [wan_video_vace.py:27-75](file://diffsynth/models/wan_video_vace.py#L27-L75)

**Section sources**
- [wan_video.py:32-86](file://diffsynth/pipelines/wan_video.py#L32-L86)
- [Wan.md:58-105](file://docs/en/Model_Details/Wan.md#L58-L105)

## Core Components
- DiT Backbone (WanModel): 3D patch embedding, time and text embeddings, stacked DiT blocks with self/cross attention, modulation via timestep, optional image input and control adapter.
- Motion Controller: Maps motion bucket IDs to modulation vectors for speed/motion control.
- Camera Controller: Generates Plücker embeddings from camera trajectories and adapts them into latent space for conditioning.
- Animate Adapter: Encodes pose/face pixel values into motion vectors and injects features into DiT blocks.
- VAE: Causal 3D convolutions, residual blocks, attention, and efficient encode/decode with tiling and streaming cache for temporal consistency.
- Text Encoder: T5-style transformer with relative positional bias; tokenizer wrapper for prompt encoding.
- Image Encoder: Vision transformer for CLIP-like image embeddings used in I2V and reference conditioning.
- MOT/VACE Extensions: Specialized DiT variants that integrate motion or contextual guidance through modified attention blocks.

**Section sources**
- [wan_video_dit.py:129-246](file://diffsynth/models/wan_video_dit.py#L129-L246)
- [wan_video_motion_controller.py:7-22](file://diffsynth/models/wan_video_motion_controller.py#L7-L22)
- [wan_video_camera_controller.py:8-44](file://diffsynth/models/wan_video_camera_controller.py#L8-L44)
- [wan_video_animate_adapter.py:615-651](file://diffsynth/models/wan_video_animate_adapter.py#L615-L651)
- [wan_video_vae.py:33-53](file://diffsynth/models/wan_video_vae.py#L33-L53)
- [wan_video_text_encoder.py:212-257](file://diffsynth/models/wan_video_text_encoder.py#L212-L257)
- [wan_video_image_encoder.py:386-479](file://diffsynth/models/wan_video_image_encoder.py#L386-L479)
- [wan_video_mot.py:94-170](file://diffsynth/models/wan_video_mot.py#L94-L170)
- [wan_video_vace.py:27-75](file://diffsynth/models/wan_video_vace.py#L27-L75)

## Architecture Overview
The pipeline orchestrates inputs through a sequence of units, performs iterative denoising with a flow-matching scheduler, and decodes latents to video frames. Conditioning can include text, images, control videos, camera trajectories, motion buckets, and animate signals.

```mermaid
sequenceDiagram
participant User as "User Code"
participant Pipe as "WanVideoPipeline"
participant Units as "Preprocessing Units"
participant Scheduler as "FlowMatchScheduler"
participant Model as "WanModel/DiT"
participant VAE as "WanVideoVAE"
User->>Pipe : call(...)
Pipe->>Units : shape check, noise init, prompt embed, image/video embed, control embed
Units-->>Pipe : latents, context, y, control latents, clip features
loop Denoising Steps
Pipe->>Scheduler : set_timesteps()
Pipe->>Model : forward(latents, timestep, context, y, controls)
Model-->>Pipe : noise_pred
alt CFG enabled
Pipe->>Model : forward(negative prompts/controls)
Model-->>Pipe : noise_pred_nega
Pipe->>Pipe : merge predictions
end
Pipe->>Scheduler : step(noise_pred, timestep, latents)
end
Pipe->>VAE : decode(latents, tiled=True/False)
VAE-->>Pipe : video frames
Pipe-->>User : video output
```

**Diagram sources**
- [wan_video.py:189-359](file://diffsynth/pipelines/wan_video.py#L189-L359)
- [wan_video_dit.py:510-552](file://diffsynth/models/wan_video_dit.py#L510-L552)
- [wan_video_vae.py:736-800](file://diffsynth/models/wan_video_vae.py#L736-L800)

## Detailed Component Analysis

### DiT Backbone: WanModel
- Patch embedding: 3D convolution over (f, h, w).
- Time embedding: sinusoidal embedding projected to modulation parameters.
- Text embedding: linear projection of T5 outputs.
- Blocks: DiTBlock with self-attention, cross-attention (optional image tokens), modulation, and gated residuals.
- Frequency RoPE: 3D precomputed frequencies for spatial-temporal attention.
- Control adapter: SimpleAdapter integrates camera/control latents.

```mermaid
classDiagram
class WanModel {
+int dim
+int in_dim
+int freq_dim
+bool has_image_input
+patch_embedding(x)
+text_embedding(context)
+time_embedding(timestep)
+forward(x, timestep, context, clip_feature, y, use_gradient_checkpointing)
}
class DiTBlock {
+SelfAttention self_attn
+CrossAttention cross_attn
+LayerNorm norm1, norm2, norm3
+MLP ffn
+Parameter modulation
+forward(x, context, t_mod, freqs)
}
class SelfAttention {
+Linear q,k,v,o
+RMSNorm norm_q,norm_k
+forward(x, freqs)
}
class CrossAttention {
+Linear q,k,v,o
+RMSNorm norm_q,norm_k
+forward(x, y)
}
WanModel --> DiTBlock : "stacked layers"
DiTBlock --> SelfAttention : "uses"
DiTBlock --> CrossAttention : "uses"
```

**Diagram sources**
- [wan_video_dit.py:338-423](file://diffsynth/models/wan_video_dit.py#L338-L423)
- [wan_video_dit.py:211-246](file://diffsynth/models/wan_video_dit.py#L211-L246)
- [wan_video_dit.py:139-202](file://diffsynth/models/wan_video_dit.py#L139-L202)

**Section sources**
- [wan_video_dit.py:338-423](file://diffsynth/models/wan_video_dit.py#L338-L423)
- [wan_video_dit.py:510-552](file://diffsynth/models/wan_video_dit.py#L510-L552)

### Motion Controller: WanMotionControllerModel
- Converts discrete motion bucket IDs into modulation vectors via sinusoidal embeddings and MLP.
- Used to control motion amplitude/speed during inference.

```mermaid
flowchart TD
Start(["Input motion_bucket_id"]) --> Emb["sinusoidal_embedding_1d(freq_dim, id*10)"]
Emb --> MLP["Linear(SiLU)->Linear(SiLU)->Linear(6*dim)"]
MLP --> Output["t_mod vector"]
Output --> End(["Return modulation"])
```

**Diagram sources**
- [wan_video_motion_controller.py:7-22](file://diffsynth/models/wan_video_motion_controller.py#L7-L22)

**Section sources**
- [wan_video_motion_controller.py:7-22](file://diffsynth/models/wan_video_motion_controller.py#L7-L22)

### Camera Control System: SimpleAdapter and Utilities
- Generates camera trajectories based on direction and speed.
- Computes Plücker embeddings from intrinsic/extrinsic parameters.
- Adapts camera latents into DiT-compatible conditioning via pixel unshuffle and residual blocks.

```mermaid
flowchart TD
A["direction, length, height, width, speed, origin"] --> B["generate_camera_coordinates()"]
B --> C["process_pose_file(width,height) -> Plucker(H,W,6)"]
C --> D["SimpleAdapter.forward(x) -> control_latents"]
D --> E["Concatenate with y if needed"]
```

**Diagram sources**
- [wan_video_camera_controller.py:184-207](file://diffsynth/models/wan_video_camera_controller.py#L184-L207)
- [wan_video_camera_controller.py:150-181](file://diffsynth/models/wan_video_camera_controller.py#L150-L181)
- [wan_video_camera_controller.py:8-44](file://diffsynth/models/wan_video_camera_controller.py#L8-L44)

**Section sources**
- [wan_video_camera_controller.py:8-44](file://diffsynth/models/wan_video_camera_controller.py#L8-L44)

### Animate Adapter: WanAnimateAdapter
- Encodes pose and face pixel sequences into motion vectors.
- Injects motion features into DiT blocks at periodic intervals.
- Supports local editing via inpaint masks.

```mermaid
classDiagram
class WanAnimateAdapter {
+pose_patch_embedding
+motion_encoder
+face_adapter
+face_encoder
+after_patch_embedding(x, pose_latents, face_pixel_values)
+after_transformer_block(block_idx, x, motion_vec, motion_masks)
}
```

**Diagram sources**
- [wan_video_animate_adapter.py:615-651](file://diffsynth/models/wan_video_animate_adapter.py#L615-L651)

**Section sources**
- [wan_video_animate_adapter.py:615-651](file://diffsynth/models/wan_video_animate_adapter.py#L615-L651)

### VAE: WanVideoVAE
- Causal 3D convolutions ensure temporal consistency across frames.
- Residual blocks with attention and RMS normalization.
- Efficient encode/decode with tiling and feature caching for long sequences.

```mermaid
flowchart TD
Inp["Video frames (B,C,T,H,W)"] --> Enc["Encoder3d / Encoder3d_38"]
Enc --> Latents["Latents (B,Z,T',H',W')"]
Latents --> Dec["Decoder3d"]
Dec --> Out["Reconstructed frames"]
subgraph "Temporal Consistency"
Cache["Feature cache (last few frames)"]
Conv["CausalConv3d with padding/cache"]
end
Enc --- Cache
Dec --- Cache
Conv --- Cache
```

**Diagram sources**
- [wan_video_vae.py:33-53](file://diffsynth/models/wan_video_vae.py#L33-L53)
- [wan_video_vae.py:517-618](file://diffsynth/models/wan_video_vae.py#L517-L618)
- [wan_video_vae.py:736-800](file://diffsynth/models/wan_video_vae.py#L736-L800)

**Section sources**
- [wan_video_vae.py:33-53](file://diffsynth/models/wan_video_vae.py#L33-L53)

### Text Encoder: WanTextEncoder
- T5-style transformer with relative positional embeddings.
- Tokenizer wrapper supports cleaning and truncation strategies.

```mermaid
classDiagram
class WanTextEncoder {
+token_embedding
+blocks[T5SelfAttention]
+norm
+forward(ids, mask)
}
class HuggingfaceTokenizer {
+name
+seq_len
+clean
+__call__(sequence, return_mask)
}
WanTextEncoder --> HuggingfaceTokenizer : "uses"
```

**Diagram sources**
- [wan_video_text_encoder.py:212-257](file://diffsynth/models/wan_video_text_encoder.py#L212-L257)
- [wan_video_text_encoder.py:285-330](file://diffsynth/models/wan_video_text_encoder.py#L285-L330)

**Section sources**
- [wan_video_text_encoder.py:212-257](file://diffsynth/models/wan_video_text_encoder.py#L212-L257)

### Image Encoder: WanImageEncoder
- Vision transformer for image embeddings.
- Supports interpolation for variable token lengths and pooling strategies.

```mermaid
classDiagram
class VisionTransformer {
+patch_embedding
+pos_embedding
+transformer[AttentionBlock]
+head
+forward(x, interpolation, use_31_block)
}
```

**Diagram sources**
- [wan_video_image_encoder.py:386-479](file://diffsynth/models/wan_video_image_encoder.py#L386-L479)

**Section sources**
- [wan_video_image_encoder.py:386-479](file://diffsynth/models/wan_video_image_encoder.py#L386-L479)

### MOT Extension: MotWanModel
- Integrates motion-specific attention paths alongside standard DiT blocks.
- Combines Q/K/V from main and motion streams for joint attention.

```mermaid
classDiagram
class MotWanModel {
+mot_layers
+patch_embedding
+text_embedding
+time_embedding
+blocks[MotWanAttentionBlock]
+forward(wan_block, x, context, t_mod, freqs, x_mot, context_mot, t_mod_mot, freqs_mot, block_id)
}
```

**Diagram sources**
- [wan_video_mot.py:94-170](file://diffsynth/models/wan_video_mot.py#L94-L170)

**Section sources**
- [wan_video_mot.py:94-170](file://diffsynth/models/wan_video_mot.py#L94-L170)

### VACE Extension: VaceWanModel
- Processes VACE context through specialized DiT blocks and returns skip connections/hints.

```mermaid
classDiagram
class VaceWanModel {
+vace_layers
+vace_patch_embedding
+vace_blocks[VaceWanAttentionBlock]
+forward(x, vace_context, context, t_mod, freqs)
}
```

**Diagram sources**
- [wan_video_vace.py:27-75](file://diffsynth/models/wan_video_vace.py#L27-L75)

**Section sources**
- [wan_video_vace.py:27-75](file://diffsynth/models/wan_video_vace.py#L27-L75)

### Conceptual Overview
- Temporal Modeling: 3D patching and RoPE across time/space dimensions; causal convolutions in VAE maintain continuity.
- Motion Interpolation: Motion controller maps discrete IDs to continuous modulation; animate adapter derives motion vectors from pose/face sequences.
- Camera Trajectory Control: Plücker embeddings derived from camera poses guide spatial-temporal attention.
- Frame Generation: Flow-matching scheduler iteratively refines latents; VAE decodes to frames with tiling for memory efficiency.

[No sources needed since this section doesn't analyze specific files]

## Dependency Analysis
Key dependencies and relationships:
- Pipeline depends on all models and units for data preparation and denoising.
- DiT depends on attention modules, RoPE, and optional control adapters.
- VAE depends on causal 3D convolutions and attention blocks.
- Text/Image encoders provide conditioning embeddings.
- MOT/VACE extend DiT with specialized attention pathways.

```mermaid
graph LR
Pipe["WanVideoPipeline"] --> DIT["WanModel"]
Pipe --> VAE["WanVideoVAE"]
Pipe --> TEXT["WanTextEncoder"]
Pipe --> IMG["WanImageEncoder"]
Pipe --> MOTION["WanMotionControllerModel"]
Pipe --> CAMERA["SimpleAdapter"]
Pipe --> ANIMATE["WanAnimateAdapter"]
Pipe --> MOT["MotWanModel"]
Pipe --> VACE["VaceWanModel"]
DIT --> ATT["AttentionModule"]
DIT --> ROPE["RoPE 3D"]
VAE --> CAUSAL["CausalConv3d"]
```

**Diagram sources**
- [wan_video.py:32-86](file://diffsynth/pipelines/wan_video.py#L32-L86)
- [wan_video_dit.py:129-137](file://diffsynth/models/wan_video_dit.py#L129-L137)
- [wan_video_vae.py:33-53](file://diffsynth/models/wan_video_vae.py#L33-L53)

**Section sources**
- [wan_video.py:32-86](file://diffsynth/pipelines/wan_video.py#L32-L86)

## Performance Considerations
- Attention backends: FlashAttention 2/3 or SageAttention fallbacks are selected automatically for efficiency.
- VRAM management: Tiled VAE encoding/decoding reduces memory usage; feature caching maintains temporal consistency without recomputation.
- Gradient checkpointing: Optional activation checkpointing for training and large inference batches.
- Multi-GPU parallelism: Unified sequence parallelism supported via xfuser integration.
- Precision: bfloat16 recommended for GPU inference; FP8 training available for select models.

**Section sources**
- [wan_video_dit.py:30-63](file://diffsynth/models/wan_video_dit.py#L30-L63)
- [wan_video_vae.py:120-175](file://diffsynth/models/wan_video_vae.py#L120-L175)
- [wan_video.py:89-108](file://diffsynth/pipelines/wan_video.py#L89-L108)
- [Wan.md:208-250](file://docs/en/Model_Details/Wan.md#L208-L250)

## Troubleshooting Guide
Common issues and resolutions:
- Insufficient VRAM: Enable tiled VAE decoding and VRAM management; reduce tile size or stride.
- Temporal flickering: Ensure causal convolutions and feature caching are active; avoid non-causal operations in custom units.
- Shape mismatches: Verify height/width multiples of 16 and num_frames = 4k+1; adjust unit processing accordingly.
- Camera control artifacts: Validate Plücker embedding dimensions and ensure correct aspect ratio scaling.
- Text/image conditioning errors: Confirm tokenizer sequence length and image resizing to target resolution.

**Section sources**
- [wan_video.py:363-373](file://diffsynth/pipelines/wan_video.py#L363-L373)
- [wan_video_vae.py:120-175](file://diffsynth/models/wan_video_vae.py#L120-L175)
- [wan_video_camera_controller.py:150-181](file://diffsynth/models/wan_video_camera_controller.py#L150-L181)

## Conclusion
The WanVideo framework provides a robust, modular system for high-quality video generation. Its DiT backbone, combined with motion controllers, camera control, animate adapters, and efficient VAE, enables diverse use cases from text-to-video to advanced control scenarios. The pipeline’s design emphasizes temporal consistency, memory efficiency, and scalability across hardware configurations.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### API Usage Examples
- Text-to-Video: See example script for loading models and generating video from prompt.
- Image-to-Video: Provide input image and prompt; optionally specify end image for first-last frame control.
- Camera Control: Specify direction and speed to generate controlled camera movements.

**Section sources**
- [Wan2.1-T2V-14B.py:7-25](file://examples/wanvideo/model_inference/Wan2.1-T2V-14B.py#L7-L25)
- [Wan2.1-I2V-14B-720P.py:8-36](file://examples/wanvideo/model_inference/Wan2.1-I2V-14B-720P.py#L8-L36)
- [Wan2.1-Fun-V1.1-14B-Control-Camera.py:8-45](file://examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py#L8-L45)