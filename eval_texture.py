#!/usr/bin/env python3
"""
Texture Reference Evaluation Metrics
Computes Gram-Sim and CLIP-Tex between predicted and reference images.

Usage:
    python eval_texture.py \
        --output_dir path/to/predictions \
        --ref_dir path/to/references \
        --mask_dir path/to/output_masks \
        --ref_mask_dir path/to/ref_masks \
        --metrics gram_sim clip_tex \
        --save_path results.json

Dependencies:
    pip install torch torchvision clip-anytorch pillow numpy
"""

import os
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# ============================================================
# Utility functions
# ============================================================

def get_image_files(directory):
    """Scan a directory for image files, keyed by filename stem."""
    files = {}
    for f in sorted(Path(directory).iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            files[str(f.stem).split('_')[0]] = str(f)
    return files


def load_image(path):
    """Load image as RGB PIL Image."""
    return Image.open(path).convert('RGB')


def load_mask(path, target_size_wh):
    """Load mask as binary numpy array [H, W], resized to target_size_wh=(W, H)."""
    mask = Image.open(path).convert('L')
    mask = mask.resize(target_size_wh, Image.NEAREST)
    mask = np.array(mask)
    mask = (mask > 127).astype(np.float32)
    return mask


def resize_mask_to_feature(mask_np, h, w, device):
    """Resize a numpy mask [H, W] to [h, w] and return as tensor on device."""
    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_t = F.interpolate(mask_t, size=(h, w), mode='nearest')
    return mask_t.squeeze(0).squeeze(0).to(device)  # [h, w]


def crop_to_mask_bbox(img_arr, mask):
    """Crop image to the bounding box of the mask with padding."""
    coords = np.where(mask > 0.5)
    if len(coords[0]) == 0:
        return Image.fromarray(img_arr.astype(np.uint8))

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    h, w = img_arr.shape[:2]
    pad = max(10, int(min(y_max - y_min, x_max - x_min) * 0.05))
    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad + 1)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad + 1)

    cropped = img_arr[y_min:y_max, x_min:x_max]
    return Image.fromarray(cropped.astype(np.uint8))


# ============================================================
# High-Frequency Texture Metric
# ============================================================

class HighFreqTextureMetric:
    """Gradient magnitude based texture strength metric.

    Computes Sobel gradient magnitude as a direct measure of texture intensity.
    Returns the ratio of output texture strength to reference texture strength.
    A score close to 1.0 indicates matched texture intensity.
    """

    def __init__(self):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for HighFreqTextureMetric. "
                "Install via: pip install opencv-python"
            )

    @staticmethod
    def _compute_gradient_magnitude(img_gray, cv2_module):
        grad_x = cv2_module.Sobel(img_gray, cv2_module.CV_32F, 1, 0, ksize=3)
        grad_y = cv2_module.Sobel(img_gray, cv2_module.CV_32F, 0, 1, ksize=3)
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

    def compute(self, output_img, ref_img, output_mask=None, ref_mask=None):
        out_gray = np.array(output_img.convert('L'), dtype=np.float32)
        ref_gray = np.array(ref_img.convert('L'), dtype=np.float32)

        out_grad = self._compute_gradient_magnitude(out_gray, self.cv2)
        ref_grad = self._compute_gradient_magnitude(ref_gray, self.cv2)

        if output_mask is not None:
            valid_out = output_mask > 0.5
            out_strength = float(out_grad[valid_out].mean()) if valid_out.sum() > 0 else 0.0
        else:
            out_strength = float(out_grad.mean())

        if ref_mask is not None:
            valid_ref = ref_mask > 0.5
            ref_strength = float(ref_grad[valid_ref].mean()) if valid_ref.sum() > 0 else 0.0
        else:
            ref_strength = float(ref_grad.mean())

        if ref_strength < 1e-6:
            return 1.0 if out_strength < 1e-6 else 0.0

        ratio = out_strength / ref_strength
        similarity = 1.0 - abs(min(max(ratio, 0.0), 2.0) - 1.0)
        return float(similarity)


# ============================================================
# Gram-Sim Metric
# ============================================================

VGG19_LAYERS = [
    ('relu1_2', 3),
    ('relu2_2', 8),
    ('relu3_4', 17),
    ('relu4_4', 26),
]


class GramSimMetric:
    """Gram Matrix Similarity using VGG19 features.

    Computes cosine similarity of Gram matrices extracted from multiple
    VGG19 layers, optionally restricted to masked regions.
    """

    def __init__(self, device='cuda', model_path=None):
        self.device = device
        if model_path:
            print(f"  Loading VGG19 from local: {model_path}")
            vgg_full = models.vgg19(weights=None)
            state_dict = torch.load(model_path, map_location='cpu')
            vgg_full.load_state_dict(state_dict)
            vgg = vgg_full.features.eval()
        else:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.vgg = vgg.to(device)
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def extract_features(self, img_tensor):
        """Run VGG19 sequentially, capturing features at target layers.

        Args:
            img_tensor: [1, 3, H, W] normalized tensor

        Returns:
            dict: {layer_name: [1, C, H', W']}
        """
        features = {}
        x = img_tensor
        target_map = {idx: name for name, idx in VGG19_LAYERS}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in target_map:
                features[target_map[i]] = x
        return features

    @staticmethod
    def compute_gram(features, mask=None):
        """Compute centered Gram matrix from a feature map.

        Mean subtraction ensures the Gram matrix captures texture covariance
        rather than color/brightness, which is critical for distinguishing
        textured vs smooth regions.
        """
        C, H, W = features.shape
        F_mat = features.reshape(C, -1)  # [C, H*W]

        if mask is not None:
            mask_flat = mask.reshape(-1)
            valid_count = mask_flat.sum()
            if valid_count < 1.0:
                return torch.zeros(C, C, device=features.device)

            valid_indices = mask_flat.bool()
            F_valid = F_mat[:, valid_indices]  # [C, N_valid]

            mean = F_valid.mean(dim=1, keepdim=True)
            F_centered = F_valid - mean
            G = (F_centered @ F_centered.t()) / valid_count
        else:
            mean = F_mat.mean(dim=1, keepdim=True)
            F_centered = F_mat - mean
            G = (F_centered @ F_centered.t()) / float(H * W)

        return G

    def compute(self, output_img, ref_img, output_mask=None, ref_mask=None):
        """Compute Gram-Sim between output and reference images.

        Args:
            output_img: PIL Image (predicted)
            ref_img:    PIL Image (reference)
            output_mask: np.array [H, W] binary, or None
            ref_mask:   np.array [H, W] binary, or None

        Returns:
            float: Gram-Sim score (cosine similarity averaged across layers)
        """
        # Prepare image tensors
        out_t = self.normalize(self.to_tensor(output_img)).unsqueeze(0).to(self.device)
        ref_t = self.normalize(self.to_tensor(ref_img)).unsqueeze(0).to(self.device)

        # Extract VGG features
        out_feats = self.extract_features(out_t)
        ref_feats = self.extract_features(ref_t)

        # Compute per-layer Gram cosine similarity
        sims = []
        for name, _ in VGG19_LAYERS:
            out_feat = out_feats[name][0]  # [C, H, W]
            ref_feat = ref_feats[name][0]  # [C, H', W']

            _, H_out, W_out = out_feat.shape
            _, H_ref, W_ref = ref_feat.shape

            out_m = resize_mask_to_feature(output_mask, H_out, W_out, self.device) if output_mask is not None else None
            ref_m = resize_mask_to_feature(ref_mask, H_ref, W_ref, self.device) if ref_mask is not None else None

            G_out = self.compute_gram(out_feat, out_m)
            G_ref = self.compute_gram(ref_feat, ref_m)

            # Cosine similarity between flattened Gram matrices
            v1 = G_out.flatten().unsqueeze(0)
            v2 = G_ref.flatten().unsqueeze(0)
            sim = F.cosine_similarity(v1, v2, eps=1e-8).item()
            sims.append(sim)

        return float(np.mean(sims))


# ============================================================
# CLIP-Tex Metric
# ============================================================

class CLIPTexMetric:
    """CLIP-based texture similarity on masked regions.

    Applies masks to images, extracts CLIP image embeddings,
    and computes cosine similarity.
    Uses the OpenAI CLIP package (clip-anytorch) which downloads
    from OpenAI CDN, avoiding HuggingFace connectivity issues.
    """

    def __init__(self, device='cuda', model_name='ViT-B/32', model_path=None):
        import clip
        self.device = device
        self.clip_pkg = clip
        if model_path:
            print(f"  Loading CLIP from local: {model_path}")
            # Load JIT model from local path
            try:
                model = torch.jit.load(model_path, map_location=device).eval()
            except Exception:
                # Fallback: try as state dict
                model = clip.build_model(model_name.replace('/', '-').lower())
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model = model.to(device).eval()
            preprocess = clip._transform(model.visual.input_resolution.item())
            self.model = model
            self.preprocess = preprocess
        else:
            print(f"  Loading CLIP model: {model_name}")
            self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def compute(self, output_img, ref_img, output_mask=None, ref_mask=None):
        """Compute CLIP-Tex using mask-bbox cropping for better texture focus."""
        out_arr = np.array(output_img).astype(np.float32)
        ref_arr = np.array(ref_img).astype(np.float32)

        if output_mask is not None:
            out_pil = crop_to_mask_bbox(out_arr, output_mask)
        else:
            out_pil = output_img

        if ref_mask is not None:
            ref_pil = crop_to_mask_bbox(ref_arr, ref_mask)
        else:
            ref_pil = ref_img

        out_t = self.preprocess(out_pil).unsqueeze(0).to(self.device)
        ref_t = self.preprocess(ref_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_out = self.model.encode_image(out_t)
            feat_ref = self.model.encode_image(ref_t)

        feat_out = F.normalize(feat_out, dim=-1)
        feat_ref = F.normalize(feat_ref, dim=-1)

        sim = (feat_out * feat_ref).sum(dim=-1).item()
        return sim


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Texture Reference Evaluation Metrics (Gram-Sim & CLIP-Tex)'
    )
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory of predicted images')
    parser.add_argument('--ref_dir', type=str, required=True,
                        help='Directory of reference images')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory of masks for predicted images')
    parser.add_argument('--ref_mask_dir', type=str, default=None,
                        help='Directory of masks for reference images')
    parser.add_argument('--metrics', nargs='+', default=['gram_sim', 'clip_tex', 'high_freq'],
                        choices=['gram_sim', 'clip_tex', 'high_freq'],
                        help='Metrics to compute (default: all three)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save JSON results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available else cpu)')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32',
                        help='CLIP model name (default: ViT-B/32). Options: ViT-B/32, ViT-B/16, ViT-L/14, RN50')
    parser.add_argument('--vgg_model_path', type=str, default=None,
                        help='Local path to VGG19 weights .pth file (skip auto-download)')
    parser.add_argument('--clip_model_path', type=str, default=None,
                        help='Local path to CLIP model .pt file (skip auto-download)')

    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Collect image files ----
    output_files = get_image_files(args.output_dir)
    ref_files = get_image_files(args.ref_dir)
    mask_files = get_image_files(args.mask_dir) if args.mask_dir else {}
    ref_mask_files = get_image_files(args.ref_mask_dir) if args.ref_mask_dir else {}

    if not output_files:
        print(f"Error: No images found in output_dir: {args.output_dir}")
        return
    if not ref_files:
        print(f"Error: No images found in ref_dir: {args.ref_dir}")
        return

    # Match by filename stem
    common_names = sorted(set(output_files.keys()) & set(ref_files.keys()))

    if not common_names:
        print("Error: No matching images found (match by filename stem).")
        print(f"  Output stems: {list(output_files.keys())[:5]} ...")
        print(f"  Ref stems:    {list(ref_files.keys())[:5]} ...")
        return

    print(f"Found {len(common_names)} matched image pairs.")

    # Warn about missing masks
    if args.mask_dir:
        missing = [n for n in common_names if n not in mask_files]
        if missing:
            print(f"Warning: {len(missing)} pairs missing output masks (will compute without mask).")
    if args.ref_mask_dir:
        missing = [n for n in common_names if n not in ref_mask_files]
        if missing:
            print(f"Warning: {len(missing)} pairs missing ref masks (will compute without mask).")

    # ---- Initialize metrics ----
    metrics = {}
    if 'gram_sim' in args.metrics:
        print("Loading VGG19 for Gram-Sim ...")
        metrics['gram_sim'] = GramSimMetric(device=device, model_path=args.vgg_model_path)
    if 'clip_tex' in args.metrics:
        print("Loading CLIP for CLIP-Tex ...")
        try:
            metrics['clip_tex'] = CLIPTexMetric(
                device=device, model_name=args.clip_model_name,
                model_path=args.clip_model_path
            )
        except Exception as e:
            print(f"  Warning: Failed to load CLIP model: {e}")
            print(f"  Skipping CLIP-Tex. You can set --clip_model_path to a local model dir.")
            args.metrics = [m for m in args.metrics if m != 'clip_tex']
    if 'high_freq' in args.metrics:
        print("Initializing High-Freq Texture Metric ...")
        try:
            metrics['high_freq'] = HighFreqTextureMetric()
        except ImportError as e:
            print(f"  Warning: {e}")
            print(f"  Skipping High-Freq metric.")
            args.metrics = [m for m in args.metrics if m != 'high_freq']

    # ---- Compute metrics per pair ----
    results = {}
    all_scores = {m: [] for m in args.metrics}

    for idx, name in enumerate(common_names):
        print(f"[{idx + 1}/{len(common_names)}] {name}", end=' ... ')

        try:
            # Load images
            output_img = load_image(output_files[name])
            ref_img = load_image(ref_files[name])

            # Load masks (resized to respective image sizes)
            output_mask = None
            ref_mask = None
            if args.mask_dir and name in mask_files:
                output_mask = load_mask(mask_files[name], output_img.size)
            if args.ref_mask_dir and name in ref_mask_files:
                ref_mask = load_mask(ref_mask_files[name], ref_img.size)

            # Compute each metric
            pair_result = {}
            for metric_name, metric_fn in metrics.items():
                score = metric_fn.compute(output_img, ref_img, output_mask, ref_mask)
                pair_result[metric_name] = round(score, 6)
                all_scores[metric_name].append(score)

            results[name] = pair_result
            scores_str = ', '.join(f'{k}={v:.4f}' for k, v in pair_result.items())
            print(scores_str)

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            results[name] = {'error': str(e)}

    # ---- Compute global summary ----
    summary = {}
    for metric_name in args.metrics:
        scores = all_scores[metric_name]
        if scores:
            summary[metric_name] = {
                'mean': round(float(np.mean(scores)), 6),
                'std': round(float(np.std(scores)), 6),
                'min': round(float(np.min(scores)), 6),
                'max': round(float(np.max(scores)), 6),
                'count': len(scores),
            }

    # ---- Save results ----
    output_data = {
        'per_image': results,
        'summary': summary,
    }

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.save_path}")
    print("\n===== Summary =====")
    for metric_name, stats in summary.items():
        print(f"  {metric_name}: mean={stats['mean']:.6f}  std={stats['std']:.6f}  "
              f"min={stats['min']:.6f}  max={stats['max']:.6f}  n={stats['count']}")


if __name__ == '__main__':
    main()
