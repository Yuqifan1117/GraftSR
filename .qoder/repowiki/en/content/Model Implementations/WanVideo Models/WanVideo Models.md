# WanVideo Models

<cite>
**Referenced Files in This Document**
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)
- [wan_video_motion_controller.py](file://diffsynth/models/wan_video_motion_controller.py)
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)
- [wan_video_mot.py](file://diffsynth/models/wan_video_mot.py)
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)
- [wan_video_text_encoder.py](file://diffsynth/models/wan_video_text_encoder.py)
- [wan_video_image_encoder.py](file://diffsynth/models/wan_video_image_encoder.py)
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)
- [Wan2.1-Fun-V1.1-14B-Control-Camera.py](file://examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py)
- [Wan2.1-VACE-1.3B.py](file://examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py)
- [Wan2.2-Animate-14B.py](file://examples/wanvideo/model_inference/Wan2.2-Animate-14B.py)
- [Wan.md](file://docs/en/Model_Details/Wan.md)
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
This document provides a comprehensive technical guide to the WanVideo model implementations focused on video generation. It explains the DiT-based video diffusion backbone, motion controllers for dynamic scene control, camera controllers for cinematic effects, animate adapters for character animation, and integrations with MOT (Multiple Object Tracking) and VACE (Video Action Control Encoder). It also details the end-to-end pipeline from text prompts to generated videos and includes practical examples for camera control, motion specification, and multi-modal inputs.

## Project Structure
The WanVideo implementation is organized into:
- Model components under diffsynth/models: DiT backbone, motion controller, camera controller, animate adapter, MOT/VACE modules, VAE, text/image encoders.
- Pipeline orchestration under diffsynth/pipelines: unit-based processing graph that wires inputs, controls, and denoising steps.
- Examples under examples/wanvideo/model_inference demonstrating usage patterns for camera control, VACE, and animate workflows.
- Documentation under docs/en/Model_Details/Wan.md describing lineage, parameters, and usage.

```mermaid
graph TB
subgraph "Models"
DIT["WanModel (DiT)"]
MOT["MotWanModel"]
VACE["VaceWanModel"]
MOTION["WanMotionControllerModel"]
CAMERA["SimpleAdapter + Camera utils"]
ANIMATE["WanAnimateAdapter"]
VAE["WanVideoVAE"]
TEXT["WanTextEncoder"]
IMG["WanImageEncoder"]
end
subgraph "Pipeline"
PIPE["WanVideoPipeline"]
end
subgraph "Examples"
EX_CAM["Camera Control Example"]
EX_VACE["VACE Example"]
EX_ANIM["Animate Example"]
end
PIPE --> DIT
PIPE --> MOT
PIPE --> VACE
PIPE --> MOTION
PIPE --> CAMERA
PIPE --> ANIMATE
PIPE --> VAE
PIPE --> TEXT
PIPE --> IMG
EX_CAM --> PIPE
EX_VACE --> PIPE
EX_ANIM --> PIPE
```

**Diagram sources**
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)
- [wan_video_motion_controller.py](file://diffsynth/models/wan_video_motion_controller.py)
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)
- [wan_video_mot.py](file://diffsynth/models/wan_video_mot.py)
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)
- [wan_video_text_encoder.py](file://diffsynth/models/wan_video_text_encoder.py)
- [wan_video_image_encoder.py](file://diffsynth/models/wan_video_image_encoder.py)

**Section sources**
- [Wan.md](file://docs/en/Model_Details/Wan.md)

## Core Components
- DiT Backbone (WanModel): A 3D patching transformer with self/cross attention, time modulation, and optional image conditioning. Supports gradient checkpointing and multiple attention backends.
- Motion Controller (WanMotionControllerModel): Maps motion bucket IDs to modulation vectors via sinusoidal embeddings and MLPs.
- Camera Controller (SimpleAdapter + utilities): Generates Plücker embeddings from camera trajectories and injects them into DiT via an adapter.
- Animate Adapter (WanAnimateAdapter): Encodes pose/face sequences and injects motion features into DiT blocks for character animation.
- MOT Integration (MotWanModel): Extends DiT blocks to jointly attend to main and MOT latents for multi-object consistency.
- VACE (VaceWanModel): Processes control/reference latents and masks to produce hints injected into the DiT process.
- VAE (WanVideoVAE): Causal 3D encoder/decoder for temporal-consistent video latent space.
- Text/Image Encoders: T5-style text encoder and CLIP-like image encoder for multimodal conditioning.

**Section sources**
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)
- [wan_video_motion_controller.py](file://diffsynth/models/wan_video_motion_controller.py)
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)
- [wan_video_mot.py](file://diffsynth/models/wan_video_mot.py)
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)
- [wan_video_text_encoder.py](file://diffsynth/models/wan_video_text_encoder.py)
- [wan_video_image_encoder.py](file://diffsynth/models/wan_video_image_encoder.py)

## Architecture Overview
The pipeline orchestrates preprocessing units, denoising iterations over timesteps, and decoding. Key flows:
- Inputs are normalized and encoded (text, images, videos, audio).
- Latents are initialized and optionally conditioned by VAE/CLIP features.
- Control signals (camera, motion, VACE, animate, MOT) are prepared and injected at appropriate stages.
- DiT denoises latents per timestep; CFG can be applied.
- VAE decodes final latents to pixel frames.

```mermaid
sequenceDiagram
participant User as "User"
participant Pipe as "WanVideoPipeline"
participant Units as "Preprocessing Units"
participant DiT as "WanModel (DiT)"
participant VAE as "WanVideoVAE"
User->>Pipe : call(prompt, images/videos, controls)
Pipe->>Units : shape check, noise init, prompt embed, image/video embed
Units-->>Pipe : latents, context, clip/y features, control signals
loop For each timestep
Pipe->>DiT : forward(latents, context, t_mod, freqs, controls)
DiT-->>Pipe : noise_pred
alt CFG enabled
Pipe->>DiT : forward(negative prompt)
DiT-->>Pipe : noise_pred_nega
Pipe->>Pipe : merge positive/negative
end
Pipe->>Pipe : scheduler step
end
Pipe->>VAE : decode(latents)
VAE-->>Pipe : video frames
Pipe-->>User : output video
```

**Diagram sources**
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)

**Section sources**
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)

## Detailed Component Analysis

### DiT Backbone (WanModel)
- Patch embedding: 3D convolutional patching with optional global path.
- Time embedding: sinusoidal embedding projected to modulation parameters.
- Transformer blocks: Self-attention with RoPE, cross-attention with optional image tokens, gated residual connections, and modulated LayerNorms.
- Head: Modulated projection to patch-space residuals.
- Optional integrations: control adapter for camera latents, Wantodance music injection, reference image/face embeddings.

```mermaid
classDiagram
class WanModel {
+int dim
+int in_dim
+int freq_dim
+tuple patch_size
+bool has_image_input
+forward(x, timestep, context, clip_feature, y, ...)
-patchify(x, control_camera_latents_input, enable_wantodance_global)
-unpatchify(x, grid_size)
-prepare_wantodance(...)
}
class DiTBlock {
+SelfAttention self_attn
+CrossAttention cross_attn
+LayerNorm norm1,norm2,norm3
+MLP ffn
+Parameter modulation
+forward(x, context, t_mod, freqs)
}
class AttentionModule {
+forward(q,k,v)
}
class Head {
+forward(x, t_mod)
}
WanModel --> DiTBlock : "stacked layers"
DiTBlock --> AttentionModule : "uses"
WanModel --> Head : "final projection"
```

**Diagram sources**
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)

**Section sources**
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)

### Motion Controller
- Converts discrete motion_bucket_id to continuous modulation via sinusoidal embedding and MLP.
- Produces 6-channel modulation vector aligned with DiT block modulation scheme.

```mermaid
flowchart TD
Start(["Input motion_bucket_id"]) --> Emb["Sinusoidal Embedding"]
Emb --> MLP["MLP (SiLU x2)"]
MLP --> Out["Modulation Vector (6 channels)"]
Out --> End(["Return"])
```

**Diagram sources**
- [wan_video_motion_controller.py](file://diffsynth/models/wan_video_motion_controller.py)

**Section sources**
- [wan_video_motion_controller.py](file://diffsynth/models/wan_video_motion_controller.py)

### Camera Controller
- Generates camera trajectory coordinates based on direction and speed.
- Computes Plücker embeddings from intrinsic/extrinsic parameters.
- SimpleAdapter reduces spatial dimensions and extracts features to inject into DiT patch stream.

```mermaid
flowchart TD
Dir["Direction + Speed + Origin"] --> Coords["Generate Camera Coordinates"]
Coords --> Pose["Process Pose File"]
Pose --> Plucker["Ray Condition -> Plücker Embeddings"]
Plucker --> Adapter["SimpleAdapter (PixelUnshuffle + Conv + Residuals)"]
Adapter --> Out["Control Camera Latents"]
```

**Diagram sources**
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)

**Section sources**
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)

### Animate Adapter
- Encodes pose/face sequences through causal convolutions and a motion encoder.
- Projects motion features into token space and pads a special token.
- Injects motion features into selected DiT blocks via FaceAdapter fuser blocks.

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
class FaceEncoder {
+CausalConv1d layers
+out_proj
+forward(x)
}
class FaceAdapter {
+fuser_blocks
+forward(x, motion_embed, idx, freqs_cis_q, freqs_cis_k)
}
WanAnimateAdapter --> FaceEncoder : "encodes motion"
WanAnimateAdapter --> FaceAdapter : "injects into DiT"
```

**Diagram sources**
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)

**Section sources**
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)

### MOT Integration (MotWanModel)
- Extends DiT blocks to compute joint self-attention across main and MOT latents.
- Maintains separate contexts and time modulations for MOT branch.
- Uses flash attention to fuse Q/K/V from both streams.

```mermaid
classDiagram
class MotWanModel {
+mot_layers
+patch_embedding
+text_embedding
+time_embedding
+blocks : ModuleList[MotWanAttentionBlock]
+forward(wan_block, x, context, t_mod, freqs, x_mot, context_mot, t_mod_mot, freqs_mot, block_id)
}
class MotWanAttentionBlock {
+self_attn : MotSelfAttention
+forward(wan_block, x, context, t_mod, freqs, x_mot, context_mot, t_mod_mot, freqs_mot)
}
class MotSelfAttention {
+forward(x, freqs, is_before_attn)
}
MotWanModel --> MotWanAttentionBlock : "selective layers"
MotWanAttentionBlock --> MotSelfAttention : "joint attn"
```

**Diagram sources**
- [wan_video_mot.py](file://diffsynth/models/wan_video_mot.py)

**Section sources**
- [wan_video_mot.py](file://diffsynth/models/wan_video_mot.py)

### VACE (Video Action Control Encoder)
- Takes control video and mask, optionally reference images, and produces stacked hints.
- Pads and concatenates inactive/reactive latents and mask latents.
- Feeds through VaceWanModel blocks with gradient checkpointing.

```mermaid
flowchart TD
In["vace_video + vace_video_mask + vace_reference_image"] --> Encode["VAE encode inactive/reactive"]
Encode --> Concat["Concat latents + mask latents"]
Concat --> VaceBlocks["VaceWanModel blocks"]
VaceBlocks --> Hints["Hints returned to DiT"]
```

**Diagram sources**
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)

**Section sources**
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)

### VAE (WanVideoVAE)
- Causal 3D convolutions and attention for temporally consistent encoding/decoding.
- Supports tiled inference and feature caching for long sequences.
- Provides framewise decoding option for memory efficiency.

```mermaid
classDiagram
class WanVideoVAE {
+encode(frames, device, tiled, tile_size, tile_stride)
+decode(latents, device, tiled, tile_size, tile_stride)
+encode_framewise(frames, device)
+decode_framewise(latents, device)
}
class Encoder3d {
+downsamples
+middle
+head
+forward(x, feat_cache, feat_idx)
}
class Decoder3d {
+upsamples
+middle
+head
+forward(x, feat_cache, feat_idx)
}
WanVideoVAE --> Encoder3d : "encoder"
WanVideoVAE --> Decoder3d : "decoder"
```

**Diagram sources**
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)

**Section sources**
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)

### Text and Image Encoders
- WanTextEncoder: T5-style encoder with relative positional bias and feed-forward gating.
- WanImageEncoder: Vision Transformer with attention pooling and optional interpolation for variable sequence lengths.

**Section sources**
- [wan_video_text_encoder.py](file://diffsynth/models/wan_video_text_encoder.py)
- [wan_video_image_encoder.py](file://diffsynth/models/wan_video_image_encoder.py)

## Dependency Analysis
Key dependencies and relationships:
- Pipeline depends on all model components and orchestrates data flow.
- DiT depends on attention utilities, RoPE, and optional control adapters.
- MOT and VACE extend DiT blocks selectively.
- Animate adapter injects motion features into DiT at specific block indices.
- Camera controller outputs Plücker embeddings consumed by DiT’s control adapter.

```mermaid
graph TB
PIPE["WanVideoPipeline"] --> DIT["WanModel"]
PIPE --> MOT["MotWanModel"]
PIPE --> VACE["VaceWanModel"]
PIPE --> MOTION["WanMotionControllerModel"]
PIPE --> CAMERA["SimpleAdapter"]
PIPE --> ANIMATE["WanAnimateAdapter"]
PIPE --> VAE["WanVideoVAE"]
PIPE --> TEXT["WanTextEncoder"]
PIPE --> IMG["WanImageEncoder"]
DIT --> ATT["AttentionModule / Flash/SageAttn"]
DIT --> ROPE["RoPE Precompute"]
MOT --> DIT
VACE --> DIT
ANIMATE --> DIT
CAMERA --> DIT
```

**Diagram sources**
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)
- [wan_video_dit.py](file://diffsynth/models/wan_video_dit.py)
- [wan_video_mot.py](file://diffsynth/models/wan_video_mot.py)
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)

**Section sources**
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)

## Performance Considerations
- Attention backends: The DiT uses a prioritized selection among Flash Attention v3/v2, SageAttention, or PyTorch SDPA fallback for optimal throughput.
- Gradient checkpointing: Enabled during training to reduce memory footprint.
- Tiled VAE: Reduces VRAM usage during encoding/decoding with minor quality trade-offs.
- Sequence parallelism: Unified sequence parallel can be enabled to scale across devices.
- Framewise decoding: Useful for very long sequences to avoid OOM.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
Common issues and resolutions:
- VRAM overflow: Enable tiled VAE, reduce tile size/stride, use framewise decoding, or lower resolution/frame count.
- Shape mismatches: Ensure height/width multiples of 16 and num_frames = 4k+1; verify VAE upsampling factors.
- CFG instability: Adjust cfg_scale; consider merging positive/negative predictions if supported.
- Camera control artifacts: Validate direction/speed parameters and origin; ensure input image matches target resolution.
- MOT/VACE integration errors: Check that MOT/VACE latents and contexts match expected shapes and are properly concatenated.

**Section sources**
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)
- [wan_video_vae.py](file://diffsynth/models/wan_video_vae.py)
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)

## Conclusion
WanVideo provides a modular, high-performance video generation framework built around a DiT backbone with rich control interfaces. Motion controllers, camera controllers, animate adapters, MOT, and VACE integrate seamlessly into the pipeline, enabling precise cinematic and character-driven video synthesis. The system supports efficient inference strategies and scalable execution modes suitable for research and production.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Example: Camera Control Operations
- Use camera_control_direction and camera_control_speed to generate Plücker embeddings and inject them into DiT via the control adapter.
- Reference example demonstrates Left/Up directions with an input image.

**Section sources**
- [Wan2.1-Fun-V1.1-14B-Control-Camera.py](file://examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py)
- [wan_video_camera_controller.py](file://diffsynth/models/wan_video_camera_controller.py)
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)

### Example: Motion Specification
- Provide motion_bucket_id to influence motion amplitude via the motion controller.
- Integrates with DiT modulation parameters for consistent temporal dynamics.

**Section sources**
- [wan_video_motion_controller.py](file://diffsynth/models/wan_video_motion_controller.py)
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)

### Example: Multi-modal Input Handling (VACE)
- Combine depth/control video and reference images to steer action and appearance.
- VACE processes inactive/reactive latents and mask latents to produce hints.

**Section sources**
- [Wan2.1-VACE-1.3B.py](file://examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py)
- [wan_video_vace.py](file://diffsynth/models/wan_video_vace.py)
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)

### Example: Character Animation
- Supply pose and face videos to animate a static image with detailed motion.
- Optionally apply local editing via inpaint/mask videos and LoRA relighting.

**Section sources**
- [Wan2.2-Animate-14B.py](file://examples/wanvideo/model_inference/Wan2.2-Animate-14B.py)
- [wan_video_animate_adapter.py](file://diffsynth/models/wan_video_animate_adapter.py)
- [wan_video.py](file://diffsynth/pipelines/wan_video.py)