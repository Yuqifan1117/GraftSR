#!/bin/bash
# 通过 accelerate 启动训练
# 用法: bash train_edit_mask.sh

set -e
cd "$(dirname "$0")"

# ===== 基本配置 =====
worker_count=8                                    # 进程数（GPU 数）
accl_args="./nebula_configs/accelerate-${worker_count}.yaml"

# ===== 根目录配置（请填写实际路径）=====
CKPT_ROOT="/root/your/path/pretrain"              
DATASET_ROOT="/root/your/path/dataset"           
BENCHMARK_ROOT="/root/your/path/benchmark"        
OUTPUT_DIR="/root/your/path/experiments"      

# 校验路径是否已填写
for p in "$CKPT_ROOT" "$DATASET_ROOT" "$BENCHMARK_ROOT" "$OUTPUT_DIR"; do
    if [[ "$p" == /root/your/path/* ]]; then
        echo "错误: 请先在脚本顶部填写实际的路径 (CKPT_ROOT / DATASET_ROOT / BENCHMARK_ROOT / OUTPUT_DIR)" >&2
        exit 1
    fi
done

# ===== 预训练模型路径 =====
qwen_path="${CKPT_ROOT}/Qwen-Image-Edit-2511"
wan_path="${CKPT_ROOT}/Wan2.1-T2V-1.3B"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_FORCE_DISABLE_TRITON=1
export qwen_path
export wan_path

# ===== 数据集 / 输出路径 =====
DATASET_DIR="${DATASET_ROOT}"

TRAINSET_DIR="${DATASET_DIR}/texrefsr_141k/trainset"
HQ_ORI_IMAGES_TXT="${TRAINSET_DIR}/hq_ori.txt"
REF_CROP_IMAGES_TXT="${TRAINSET_DIR}/ref_crop.txt"
HQ_ORI_PROMPTS_TXT="${TRAINSET_DIR}/prompt_ori.txt"
REF_CROP_MASKS_TXT="${TRAINSET_DIR}/ref_crop_mask_sam3.txt"
LQ_ORI_MASKS_TXT="${TRAINSET_DIR}/hq_ori_mask_sam3.txt"


# ===== 训练配置 =====
mmaigc_dataset_yml="./examples/qwen_image/configs/agb1_g002_r1_5_dynamic_lnnp05_edit_mask_dilate.yaml"
deg_file_path="./examples/qwen_image/configs/deg_pisa.yaml"

train_args=" \
    --mmaigc_dataset_yml ${mmaigc_dataset_yml} \
    --deg_file_path ${deg_file_path} \
    --dataset_txt_paths ${HQ_ORI_IMAGES_TXT} \
    --main_prompt_txt_paths ${HQ_ORI_PROMPTS_TXT} \
    --ref_txt_paths ${REF_CROP_IMAGES_TXT} \
    --hq_ori_mask_txt_paths ${LQ_ORI_MASKS_TXT} \
    --ref_crop_mask_txt_paths ${REF_CROP_MASKS_TXT} \
    --output_dir ${OUTPUT_DIR} \
    --null_text_ratio 0.0001 \
    --ref_dropout_prob 0.1 \
    --task train \
    --clip_grad_norm \
    --use_edit_prompt_prefix \
    --use_resize_before_crop \
    --short_edge_size 2048 \
    --flexible_ref_resolution \
    --flexible_ref_max_pixels 1049088 \
    --use_full_lq_condition \
    --force_valid_crop_prob 0.5 \
    --full_lq_max_pixels 465920 \
"

accelerate launch --config_file=${accl_args} examples/qwen_image/train_gan_edit_mask.py ${train_args}
