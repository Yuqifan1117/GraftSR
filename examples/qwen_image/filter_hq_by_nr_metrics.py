"""
根据无参考图像质量指标阈值筛选高质量 HQ 数据（双进度条版）。

筛选条件：NIQE < 3.8, MUSIQ > 75, CLIP-IQA > 0.70, MAN-IQA > 0.5
凑够 num_select 条立即停止。

用法：
    python examples/qwen_image/filter_hq_by_nr_metrics.py \
        --hq_txt /path/to/hq.txt \
        --prompt_txt /path/to/prompt.txt \
        --num_select 2000 \
        --output_dir ./filtered_dataset \
        --device cuda
"""

import argparse
import json
import os
import sys
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from examples.qwen_image.metrics import (
    PyIQACalculator,
    calculate_niqe,
)

# 筛选阈值
NIQE_THRESHOLD = 3.8
MUSIQ_THRESHOLD = 75.0
CLIPIQA_THRESHOLD = 0.70
MANIQA_THRESHOLD = 0.5


def load_path_list(txt_path: str) -> list[str]:
    if txt_path is None or not os.path.exists(txt_path):
        return []
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Filter HQ data by NR IQA thresholds (dual progress)")
    parser.add_argument("--hq_txt", type=str, required=True)
    parser.add_argument("--prompt_txt", type=str, required=True)
    parser.add_argument("--ref_txt", type=str, default=None)
    parser.add_argument("--lq_mask_txt", type=str, default=None)
    parser.add_argument("--ref_mask_txt", type=str, default=None)
    parser.add_argument("--num_select", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="./filtered_dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before filtering")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hq_paths = load_path_list(args.hq_txt)
    prompt_paths = load_path_list(args.prompt_txt)
    ref_paths = load_path_list(args.ref_txt) if args.ref_txt else []
    lq_mask_paths = load_path_list(args.lq_mask_txt) if args.lq_mask_txt else []
    ref_mask_paths = load_path_list(args.ref_mask_txt) if args.ref_mask_txt else []

    num_total = len(hq_paths)
    print(f"Loaded {num_total} HQ images, target: {args.num_select}")
    assert len(prompt_paths) == num_total, \
        f"prompt_txt ({len(prompt_paths)}) != hq_txt ({num_total})"

    indices = list(range(num_total))
    if args.shuffle:
        import random
        random.seed(42)
        random.shuffle(indices)
        print("Shuffled indices for unbiased sampling")

    # 初始化指标计算器
    clipiqa_calc = PyIQACalculator("clipiqa", device=args.device)
    maniqa_calc = PyIQACalculator("maniqa", device=args.device)
    musiq_calc = PyIQACalculator("musiq", device=args.device)

    selected_indices = []
    selected_details = []
    failed_count = 0
    passed_count = 0  # 通过所有阈值的计数（用于观察通过率）

    # 双进度条：外层=扫描进度，内层=选中进度
    scan_bar = tqdm(total=num_total, desc="Scanning ", unit="img", position=0)
    save_bar = tqdm(total=args.num_select, desc="Selected", unit="hit", position=1)

    for idx in indices:
        if len(selected_indices) >= args.num_select:
            break

        hq_path = hq_paths[idx]
        try:
            hq_image = Image.open(hq_path).convert("RGB")

            niqe_val = calculate_niqe(hq_image)
            if niqe_val >= NIQE_THRESHOLD:
                scan_bar.update(1)
                continue

            musiq_val = musiq_calc.calculate(hq_image)
            if musiq_val <= MUSIQ_THRESHOLD:
                scan_bar.update(1)
                continue

            clipiqa_val = clipiqa_calc.calculate(hq_image)
            if clipiqa_val <= CLIPIQA_THRESHOLD:
                scan_bar.update(1)
                continue

            maniqa_val = maniqa_calc.calculate(hq_image)
            if maniqa_val <= MANIQA_THRESHOLD:
                scan_bar.update(1)
                continue

            # 全部通过
            selected_indices.append(idx)
            selected_details.append({
                "index": idx,
                "hq_path": hq_path,
                "niqe": round(niqe_val, 4),
                "clipiqa": round(clipiqa_val, 4),
                "maniqa": round(maniqa_val, 4),
                "musiq": round(musiq_val, 4),
            })
            passed_count += 1
            save_bar.update(1)

        except Exception as e:
            failed_count += 1
            if failed_count <= 10:
                print(f"\n[Warning] Failed {hq_path}: {e}")

        scan_bar.update(1)

        # 每扫描 100 张打印一次实时通过率，方便调整阈值
        scanned_so_far = scan_bar.n
        if scanned_so_far > 0 and scanned_so_far % 100 == 0:
            pass_rate = passed_count / scanned_so_far * 100
            remaining_needed = args.num_select - len(selected_indices)
            estimated_remaining_scan = int(remaining_needed / (pass_rate / 100)) if pass_rate > 0 else float('inf')
            scan_bar.set_postfix({
                "pass%": f"{pass_rate:.1f}%",
                "need": remaining_needed,
                "est_left": estimated_remaining_scan,
            })

    scan_bar.close()
    save_bar.close()

    num_selected = len(selected_indices)
    scanned_total = scan_bar.n
    final_pass_rate = passed_count / scanned_total * 100 if scanned_total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Done: scanned {scanned_total}/{num_total}, "
          f"selected {num_selected}/{args.num_select}, "
          f"failed {failed_count}")
    print(f"Pass rate: {final_pass_rate:.2f}% ({passed_count}/{scanned_total})")

    if num_selected < args.num_select:
        print(f"⚠️  Only got {num_selected} samples. Consider relaxing thresholds:")
        print(f"   Current: NIQE<{NIQE_THRESHOLD}, MUSIQ>{MUSIQ_THRESHOLD}, "
              f"CLIP-IQA>{CLIPIQA_THRESHOLD}, MAN-IQA>{MANIQA_THRESHOLD}")
    elif scanned_total < num_total:
        print(f"✅ Early stopped at {scanned_total}/{num_total} "
              f"(saved {(num_total - scanned_total)} scans)")

    if num_selected == 0:
        print("No samples passed. Exiting without saving.")
        return

    # 统计均值
    avg_niqe = sum(d["niqe"] for d in selected_details) / num_selected
    avg_clipiqa = sum(d["clipiqa"] for d in selected_details) / num_selected
    avg_maniqa = sum(d["maniqa"] for d in selected_details) / num_selected
    avg_musiq = sum(d["musiq"] for d in selected_details) / num_selected
    print(f"\nSelected stats:")
    print(f"  Avg NIQE:     {avg_niqe:.4f} (< {NIQE_THRESHOLD})")
    print(f"  Avg CLIP-IQA: {avg_clipiqa:.4f} (> {CLIPIQA_THRESHOLD})")
    print(f"  Avg MAN-IQA:  {avg_maniqa:.4f} (> {MANIQA_THRESHOLD})")
    print(f"  Avg MUSIQ:    {avg_musiq:.4f} (> {MUSIQ_THRESHOLD})")

    # 写出筛选后的 txt
    sorted_selected = sorted(selected_indices)

    def write_filtered_txt(src_paths: list[str], output_name: str):
        if not src_paths:
            return
        filtered = [src_paths[i] for i in sorted_selected if i < len(src_paths)]
        out_path = os.path.join(args.output_dir, output_name)
        with open(out_path, "w") as f:
            f.write("\n".join(filtered) + "\n")
        print(f"  Wrote {len(filtered)} entries -> {out_path}")

    write_filtered_txt(hq_paths, "hq_filtered.txt")
    write_filtered_txt(prompt_paths, "prompt_filtered.txt")
    write_filtered_txt(ref_paths, "ref_filtered.txt")
    write_filtered_txt(lq_mask_paths, "lq_mask_filtered.txt")
    write_filtered_txt(ref_mask_paths, "ref_mask_filtered.txt")

    # 保存详情
    detail_path = os.path.join(args.output_dir, "filter_metrics_detail.json")
    with open(detail_path, "w") as f:
        json.dump({
            "thresholds": {
                "niqe_max": NIQE_THRESHOLD,
                "musiq_min": MUSIQ_THRESHOLD,
                "clipiqa_min": CLIPIQA_THRESHOLD,
                "maniqa_min": MANIQA_THRESHOLD,
            },
            "num_total": num_total,
            "num_scanned": scanned_total,
            "num_selected": num_selected,
            "num_failed": failed_count,
            "pass_rate": round(final_pass_rate, 2),
            "selected": selected_details,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Details -> {detail_path}")


if __name__ == "__main__":
    main()