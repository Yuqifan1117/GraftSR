"""
超分辨率评测指标工具模块

支持的有参考指标（需要 GT）：PSNR、SSIM、LPIPS、DISTS
支持的无参考指标（不需要 GT）：NIQE、MUSIQ、CLIPIQA、MANIQA
支持的数据集级别指标：FID
"""
import json
import os
import time
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL Image 转为 numpy array (H, W, C), uint8, RGB"""
    return np.array(image.convert("RGB"))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image 转为 torch tensor (1, C, H, W), float32, 归一化到 [-1, 1]"""
    array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def pil_to_tensor_01(image: Image.Image) -> torch.Tensor:
    """PIL Image 转为 torch tensor (1, C, H, W), float32, 归一化到 [0, 1]"""
    array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------------------------------------
# 有参考指标（Full-Reference）
# ---------------------------------------------------------------------------

def calculate_psnr(output_image: Image.Image, gt_image: Image.Image, crop_border: int = 0) -> float:
    """计算 PSNR，使用 basicsr 实现，在 Y 通道上计算"""
    from basicsr.metrics.psnr_ssim import calculate_psnr as basicsr_psnr
    return basicsr_psnr(pil_to_numpy(output_image), pil_to_numpy(gt_image),
                        crop_border=crop_border, test_y_channel=True)


def calculate_ssim(output_image: Image.Image, gt_image: Image.Image, crop_border: int = 0) -> float:
    """计算 SSIM，使用 basicsr 实现，在 Y 通道上计算"""
    from basicsr.metrics.psnr_ssim import calculate_ssim as basicsr_ssim
    return basicsr_ssim(pil_to_numpy(output_image), pil_to_numpy(gt_image),
                        crop_border=crop_border, test_y_channel=True)


class LPIPSCalculator:
    """LPIPS 计算器，延迟加载模型"""

    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.net = net
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            import lpips
            from check_lpips_cache import ensure_alexnet_cache
            ensure_alexnet_cache()
            self._model = lpips.LPIPS(net=self.net).to(self.device)
            self._model.eval()

    @torch.no_grad()
    def calculate(self, output_image: Image.Image, gt_image: Image.Image) -> float:
        """计算 LPIPS 感知距离，值越低越好"""
        self._load_model()
        output_tensor = pil_to_tensor(output_image).to(self.device)
        gt_tensor = pil_to_tensor(gt_image).to(self.device)
        return self._model(output_tensor, gt_tensor).item()


class PyIQACalculator:
    """
    基于 pyiqa 库的通用指标计算器，支持有参考和无参考指标。
    延迟加载模型以避免不必要的 GPU 占用。
    """

    def __init__(self, metric_name: str, device: str = "cuda"):
        self.metric_name = metric_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            import pyiqa
            self._model = pyiqa.create_metric(self.metric_name, device=self.device)
            self._model.eval()

    @torch.no_grad()
    def calculate(self, output_image: Image.Image, gt_image: Image.Image = None) -> float:
        """
        计算指标。
        有参考指标需要同时传入 output_image 和 gt_image；
        无参考指标只需传入 output_image。
        """
        self._load_model()
        output_tensor = pil_to_tensor_01(output_image).to(self.device)
        if gt_image is not None:
            gt_tensor = pil_to_tensor_01(gt_image).to(self.device)
            return self._model(output_tensor, gt_tensor).item()
        return self._model(output_tensor).item()


# ---------------------------------------------------------------------------
# 无参考指标（No-Reference）
# ---------------------------------------------------------------------------

def calculate_niqe(output_image: Image.Image, crop_border: int = 0) -> float:
    """计算 NIQE，使用 basicsr 实现，值越低越好"""
    from basicsr.metrics.niqe import calculate_niqe as basicsr_niqe
    return basicsr_niqe(pil_to_numpy(output_image), crop_border=crop_border)


# ---------------------------------------------------------------------------
# 数据集级别指标
# ---------------------------------------------------------------------------

def calculate_fid(output_dir: str, gt_dir: str, device: str = "cuda") -> float:
    """
    计算 FID (Frechet Inception Distance)，值越低越好。
    使用 clean-fid 库，对两个图片目录进行计算。

    Args:
        output_dir: 生成图片所在目录
        gt_dir: GT 图片所在目录
        device: 计算设备
    """
    from cleanfid import fid as cleanfid
    return cleanfid.compute_fid(output_dir, gt_dir, device=torch.device(device))


# ---------------------------------------------------------------------------
# 指标分类
# ---------------------------------------------------------------------------

FULL_REFERENCE_METRICS = {"psnr", "ssim", "lpips", "dists"}
NO_REFERENCE_METRICS = {"niqe", "musiq", "clipiqa", "maniqa", "maniqa-kadid", "maniqa-pipal"}
DATASET_LEVEL_METRICS = {"fid"}
SUPPORTED_METRICS = FULL_REFERENCE_METRICS | NO_REFERENCE_METRICS | DATASET_LEVEL_METRICS


# ---------------------------------------------------------------------------
# MetricsAccumulator
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """
    指标累加器，支持有参考、无参考和数据集级别指标。

    用法:
        accumulator = MetricsAccumulator(metrics=["psnr", "ssim", "niqe"], device="cuda")
        for output_img, gt_img in pairs:
            accumulator.update(output_img, gt_img, image_name="xxx")
        summary = accumulator.summary()
        accumulator.save(output_path)
    """

    def __init__(self, metrics: list, device: str = "cuda",
                 crop_border: int = 0, lpips_net: str = "alex"):
        for metric_name in metrics:
            if metric_name not in SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric: {metric_name}. Supported: {SUPPORTED_METRICS}")

        self.metric_names = [m for m in metrics if m not in DATASET_LEVEL_METRICS]
        self.dataset_metric_names = [m for m in metrics if m in DATASET_LEVEL_METRICS]
        self.all_metric_names = list(metrics)
        self.crop_border = crop_border
        self.device = device
        self.per_image_results = []
        self.dataset_results = {}

        # 初始化需要模型的计算器（延迟加载）
        self._calculators = {}
        if "lpips" in self.metric_names:
            self._calculators["lpips"] = LPIPSCalculator(net=lpips_net, device=device)
        for pyiqa_metric in ("dists", "musiq", "clipiqa", "maniqa", "maniqa-kadid", "maniqa-pipal"):
            if pyiqa_metric in self.metric_names:
                self._calculators[pyiqa_metric] = PyIQACalculator(
                    metric_name=pyiqa_metric, device=device
                )

    def update(self, output_image: Image.Image, gt_image: Image.Image = None,
               image_name: str = "") -> dict:
        """计算一对图像的所有逐图指标并记录"""
        result = {"image_name": image_name}

        for metric_name in self.metric_names:
            # 有参考指标在没有 GT 时跳过
            if metric_name in FULL_REFERENCE_METRICS and gt_image is None:
                continue

            if metric_name == "psnr":
                result["psnr"] = calculate_psnr(output_image, gt_image, crop_border=self.crop_border)
            elif metric_name == "ssim":
                result["ssim"] = calculate_ssim(output_image, gt_image, crop_border=self.crop_border)
            elif metric_name == "lpips":
                result["lpips"] = self._calculators["lpips"].calculate(output_image, gt_image)
            elif metric_name == "dists":
                result["dists"] = self._calculators["dists"].calculate(output_image, gt_image)
            elif metric_name == "niqe":
                result["niqe"] = calculate_niqe(output_image, crop_border=self.crop_border)
            elif metric_name == "musiq":
                result["musiq"] = self._calculators["musiq"].calculate(output_image)
            elif metric_name == "clipiqa":
                result["clipiqa"] = self._calculators["clipiqa"].calculate(output_image)
            elif metric_name == "maniqa":
                result["maniqa"] = self._calculators["maniqa"].calculate(output_image)
            elif metric_name == "maniqa-kadid":
                result["maniqa-kadid"] = self._calculators["maniqa-kadid"].calculate(output_image)
            elif metric_name == "maniqa-pipal":
                result["maniqa-pipal"] = self._calculators["maniqa-pipal"].calculate(output_image)

        self.per_image_results.append(result)
        return result

    def compute_dataset_metrics(self, output_dir: str, gt_dir: str = None):
        """计算数据集级别指标（如 FID），需要在所有图片生成完毕后调用"""
        for metric_name in self.dataset_metric_names:
            if metric_name == "fid":
                if gt_dir is None:
                    print("Warning: FID requires gt_dir, skipping.")
                    continue
                self.dataset_results["fid"] = calculate_fid(
                    output_dir, gt_dir, device=self.device
                )

    def summary(self) -> dict:
        """汇总所有图像的平均指标 + 数据集级别指标"""
        if not self.per_image_results and not self.dataset_results:
            return {}

        averages = {}

        if self.per_image_results:
            for metric_name in self.metric_names:
                values = [r[metric_name] for r in self.per_image_results if metric_name in r]
                if values:
                    averages[f"avg_{metric_name}"] = sum(values) / len(values)
            averages["num_images"] = len(self.per_image_results)

        for key, value in self.dataset_results.items():
            averages[key] = value

        return averages

    def save(self, output_path: str):
        """将逐图指标和汇总结果保存为 JSON 文件"""
        report = {
            "summary": self.summary(),
            "per_image": self.per_image_results,
        }
        with open(output_path, "w") as file_handle:
            json.dump(report, file_handle, indent=2, ensure_ascii=False)

    def merge(self, other_results: list):
        """合并其他 rank 的逐图指标结果"""
        self.per_image_results.extend(other_results)

    def print_summary(self):
        """打印汇总结果"""
        summary = self.summary()
        if not summary:
            print("No metrics collected.")
            return

        print("\n" + "=" * 50)
        print("Evaluation Metrics Summary")
        print("=" * 50)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# 分布式汇总
# ---------------------------------------------------------------------------

def distributed_gather_metrics(accumulator, accelerator, output_dir, gt_dir=None):
    """
    分布式场景下汇总所有 rank 的指标。

    每个 rank 将本地指标保存为临时 JSON，主进程合并后删除临时文件，
    最终在主进程上计算数据集级别指标、保存完整的 metrics.json 并打印汇总。

    Args:
        accumulator: 当前 rank 的 MetricsAccumulator 实例
        accelerator: accelerate.Accelerator 实例
        output_dir: 输出目录路径
        gt_dir: GT 图片目录路径（用于 FID 等数据集级别指标）
    """
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # 每个 rank 保存本地指标到临时文件
    local_metrics_path = os.path.join(output_dir, f"_metrics_rank{rank}.json")
    with open(local_metrics_path, "w") as file_handle:
        json.dump(accumulator.per_image_results, file_handle, ensure_ascii=False)

    # 等待所有 rank 写完
    accelerator.wait_for_everyone()

    # 仅主进程合并所有 rank 的结果
    if accelerator.is_main_process:
        merged_accumulator = MetricsAccumulator(
            metrics=accumulator.all_metric_names,
            device="cuda",
            crop_border=accumulator.crop_border,
        )

        for rank_id in range(world_size):
            rank_path = os.path.join(output_dir, f"_metrics_rank{rank_id}.json")
            # 等待文件可见（应对 NFS 等共享文件系统的同步延迟），超时则跳过该 rank
            max_retries = 15
            for attempt in range(max_retries):
                if os.path.exists(rank_path):
                    break
                accelerator.print(
                    f"[Warning] Waiting for {rank_path} to appear "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(2)

            if not os.path.exists(rank_path):
                accelerator.print(
                    f"[Warning] Metrics file from rank {rank_id} not found after "
                    f"{max_retries * 2}s, skipping. Rank may have crashed."
                )
                continue

            with open(rank_path, "r") as file_handle:
                rank_results = json.load(file_handle)
            merged_accumulator.merge(rank_results)
            os.remove(rank_path)

        # 计算数据集级别指标（如 FID）
        if merged_accumulator.dataset_metric_names:
            merged_accumulator.compute_dataset_metrics(output_dir, gt_dir=gt_dir)

        merged_accumulator.print_summary()
        metrics_output_path = os.path.join(output_dir, "metrics.json")
        merged_accumulator.save(metrics_output_path)
        accelerator.print(f"Metrics saved to {metrics_output_path}")

        return merged_accumulator.summary()

    return None


# ---------------------------------------------------------------------------
# 独立指标计算入口
# ---------------------------------------------------------------------------

def collect_image_paths(directory: str) -> list:
    """收集目录下所有图片文件路径，按文件名排序"""
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    paths = []
    for filename in sorted(os.listdir(directory)):
        if os.path.splitext(filename)[1].lower() in supported_extensions:
            paths.append(os.path.join(directory, filename))
    return paths


def pair_images_by_name(output_paths: list, gt_paths: list) -> list:
    """
    按文件名（不含扩展名）配对 output 和 GT 图片。
    返回 [(output_path, gt_path), ...] 列表，仅包含能配对的图片。
    """
    gt_name_to_path = {}
    for gt_path in gt_paths:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        gt_name_to_path[name] = gt_path

    paired = []
    for output_path in output_paths:
        name = os.path.splitext(os.path.basename(output_path))[0]
        gt_path = gt_name_to_path.get(name)
        if gt_path is not None:
            paired.append((output_path, gt_path))
        else:
            name = os.path.splitext(os.path.basename(output_path))[0].replace('LR4', 'HR').replace('_qualified_0', '_qualified').replace('x1_0', 'x1').replace('lq_0', 'lq').replace('_qualified_sr', '_qualified')
            gt_path = gt_name_to_path.get(name)
            if gt_path is not None:
                paired.append((output_path, gt_path))
            else:
                # with open("/data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/synthetic_250/txt_lists/lq_ref_pairs.txt", 'r') as f:
                with open("/data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/realworld_100/txt_lists/lq_ref_pairs.txt", 'r') as f:
                    origin_output_path = output_path.replace("_test_paired_txt_2.png", ".png")
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            if origin_output_path.split('/')[-1] == parts[1].split('/')[-1]:
                                name = os.path.splitext(os.path.basename(parts[0]))[0]
                                gt_path = gt_name_to_path.get(name)
                                if gt_path is not None:
                                    paired.append((output_path, gt_path))
                                else:
                                    paired.append((output_path, None))
    return paired


def run_standalone_evaluation(output_dir: str, gt_dir: str = None,
                              metrics: list = None, crop_border: int = 0,
                              device: str = "cuda", lpips_net: str = "alex",
                              save_path: str = None):
    """
    独立指标计算入口函数。

    对 output_dir 中的图片逐张计算指标，如果提供了 gt_dir 则按文件名配对。
    结果保存为 JSON 文件并打印汇总。

    Args:
        output_dir: 生成图片所在目录
        gt_dir: GT 图片所在目录（可选，有参考指标和 FID 需要）
        metrics: 要计算的指标列表，如 ["psnr", "ssim", "niqe"]
        crop_border: PSNR/SSIM/NIQE 计算时裁剪的边界像素数
        device: 计算设备
        lpips_net: LPIPS 使用的骨干网络
        save_path: 结果保存路径，默认为 output_dir/metrics.json
    """
    if metrics is None:
        metrics = ["psnr", "ssim", "lpips", "niqe"]

    output_paths = collect_image_paths(output_dir)
    if not output_paths:
        print(f"Error: No images found in {output_dir}")
        return

    print(f"Found {len(output_paths)} images in output dir: {output_dir}")

    # 区分逐图指标和数据集级别指标
    per_image_metric_names = [m for m in metrics if m not in DATASET_LEVEL_METRICS]
    has_fr_metrics = any(m in FULL_REFERENCE_METRICS for m in per_image_metric_names)

    # 配对 GT
    pairs = []
    if gt_dir is not None:
        gt_paths = collect_image_paths(gt_dir)
        print(f"Found {len(gt_paths)} images in GT dir: {gt_dir}")
        pairs = pair_images_by_name(output_paths, gt_paths)
        paired_count = sum(1 for _, gt in pairs if gt is not None)
        print(f"Paired {paired_count}/{len(output_paths)} images by filename")
        if has_fr_metrics and paired_count == 0:
            print("Warning: Full-reference metrics requested but no GT images paired. "
                  "These metrics will be skipped.")
    else:
        pairs = [(p, None) for p in output_paths]
        if has_fr_metrics:
            print("Warning: Full-reference metrics requested but no --gt_dir provided. "
                  "These metrics will be skipped.")

    # 创建累加器
    accumulator = MetricsAccumulator(
        metrics=metrics,
        device=device,
        crop_border=crop_border,
        lpips_net=lpips_net,
    )
    print(f"Metrics to compute: {metrics}, crop_border={crop_border}")
    print("-" * 50)

    # 逐图计算
    for idx, (output_path, gt_path) in enumerate(pairs):
        filename = os.path.splitext(os.path.basename(output_path))[0]
        output_image = Image.open(output_path).convert("RGB")

        gt_image = None
        if gt_path is not None:
            gt_image = Image.open(gt_path).convert("RGB")
            # 如果尺寸不一致，将 GT resize 到与 output 相同大小
            if gt_image.size != output_image.size:
                output_image = output_image.resize(gt_image.size, Image.BICUBIC)
                # gt_image = gt_image.resize(output_image.size, Image.BICUBIC)
            print("output image", output_image.size, "gt image", gt_image.size)
            if gt_image.size[0] * gt_image.size[1] >= 5000*5000:
                continue
        if output_image.size[0] * output_image.size[1] >= 5000*5000:
            continue
        per_image_metrics = accumulator.update(output_image, gt_image, image_name=filename)
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in per_image_metrics.items() if k != "image_name"
        )
        print(f"[{idx + 1}/{len(pairs)}] {filename}: {metrics_str}")

    # 数据集级别指标（如 FID）
    if accumulator.dataset_metric_names:
        accumulator.compute_dataset_metrics(output_dir, gt_dir=gt_dir)

    # 汇总 & 保存
    accumulator.print_summary()

    if save_path is None:
        save_path = os.path.join(output_dir, "metrics.json")
    accumulator.save(save_path)
    print(f"Metrics saved to {save_path}")

    return accumulator.summary()


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Standalone image quality metrics evaluation tool. "
                    "Compute FR/NR/dataset-level metrics on existing image directories."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory containing generated/output images.",
    )
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Directory containing ground-truth images (required for FR metrics and FID).",
    )
    parser.add_argument(
        "--metrics", type=str, default="psnr,ssim,lpips,niqe",
        help="Comma-separated metrics to compute. "
             f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}. "
             "Default: psnr,ssim,lpips,niqe",
    )
    parser.add_argument(
        "--crop_border", type=int, default=0,
        help="Crop border pixels before computing PSNR/SSIM/NIQE (default: 0).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda).",
    )
    parser.add_argument(
        "--lpips_net", type=str, default="alex", choices=["alex", "vgg"],
        help="LPIPS backbone network (default: alex).",
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="Path to save metrics JSON. Default: <output_dir>/metrics.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    run_standalone_evaluation(
        output_dir=args.output_dir,
        gt_dir=args.gt_dir,
        metrics=metric_list,
        crop_border=args.crop_border,
        device=args.device,
        lpips_net=args.lpips_net,
        save_path=args.save_path,
    )
