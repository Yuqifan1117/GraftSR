#!/bin/bash
# 本地测试脚本：去掉 nebulactl 相关代码，直接运行 test_sr_edit_mask.py
# 用法: bash test_edit_mask.sh

set -e
cd "$(dirname "$0")"

# ===== 基本配置 =====
GPU_IDS=0                                       # 使用的 GPU 编号
TEST_MODE="benchmark_realworld"                 # 测试模式: benchmark_synthetic / benchmark_realworld 
                                                #          benchmark_realsr /  benchmark_drealsr 
ckpt_num=4001


# ===== 根目录配置（请填写实际路径）=====
CKPT_ROOT="/root/your/path/pretrain"            # 预训练模型根目录
DATASET_ROOT="/root/your/path/dataset"          # 数据集根目录
BENCHMARK_ROOT="/root/your/path/benchmark"      # benchmark 根目录
OPEN_BENCH_ROOT="/root/your/path/open_benchmarks"  # 公开 benchmark 根目录
TRAINED_CKPT_ROOT="/root/your/path/trained_ckpt"   # 训练产出 checkpoint 根目录
OUTPUT_ROOT="/root/your/path/experiments"       # 测试结果输出根目录

# 校验路径是否已填写
for p in "$CKPT_ROOT" "$DATASET_ROOT" "$BENCHMARK_ROOT" "$OPEN_BENCH_ROOT" "$TRAINED_CKPT_ROOT" "$OUTPUT_ROOT"; do
    if [[ "$p" == /root/your/path/* ]]; then
        echo "错误: 请先在脚本顶部填写实际的路径" >&2
        exit 1
    fi
done

# ===== 环境变量 =====
qwen_path="${CKPT_ROOT}/Qwen-Image-Edit-2511"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_FORCE_DISABLE_TRITON=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export qwen_path

# ===== 测试集路径 =====
if [ "${TEST_MODE}" == "benchmark_synthetic" ]; then
  DATASET_DIR="${BENCHMARK_ROOT}/synthetic"
  DATASET_NAME="benchmark_synthetic-250"
  DEG_ORI_IMAGES_TXT="${DATASET_DIR}/lq_ori_degraded.txt"
  HQ_ORI_IMAGES_TXT="${DATASET_DIR}/hq_ori.txt"
  HQ_ORI_PROMPTS_TXT="${DATASET_DIR}/prompt_ori.txt"
  REF_CROP_IMAGES_TXT="${DATASET_DIR}/ref_crop.txt"
  HQ_ORI_MASKS_TXT="${DATASET_DIR}/hq_ori_mask_sam3.txt"
  REF_CROP_MASKS_TXT="${DATASET_DIR}/ref_crop_mask_sam3.txt"
elif [ "${TEST_MODE}" == "benchmark_realworld" ]; then
  DATASET_DIR="${BENCHMARK_ROOT}/realworld"
  DATASET_NAME="benchmark_realworld-50"
  REALWORLD_ORI_IMAGES_TXT="${DATASET_DIR}/lq_ori.txt"
  REALWORLD_ORI_PROMPTS_TXT="${DATASET_DIR}/prompt_ori.txt"
  REF_CROP_IMAGES_TXT="${DATASET_DIR}/ref_crop.txt"
  REALWORLD_ORI_MASKS_TXT="${DATASET_DIR}/lq_ori_mask_sam3.txt"
  REF_CROP_MASKS_TXT_0="${DATASET_DIR}/ref_crop_mask_sam3.txt"
elif [ "${TEST_MODE}" == "benchmark_realsr" ]; then
  DATASET_DIR="${OPEN_BENCH_ROOT}/datasets/benchmark_realsr"
  DATASET_NAME="benchmark_realsr"
  HQ_ORI_IMAGES_TXT="${DATASET_DIR}/hq_ori.txt"
  DEG_ORI_IMAGES_TXT="${DATASET_DIR}/lq_ori.txt"
  HQ_ORI_PROMPTS_TXT="${DATASET_DIR}/prompt_ori.txt"
elif [ "${TEST_MODE}" == "benchmark_drealsr" ]; then
  DATASET_DIR="${OPEN_BENCH_ROOT}/datasets/benchmark_drealsr"
  DATASET_NAME="benchmark_drealsr"
  HQ_ORI_IMAGES_TXT="${DATASET_DIR}/hq_ori.txt"
  DEG_ORI_IMAGES_TXT="${DATASET_DIR}/lq_ori.txt"
  HQ_ORI_PROMPTS_TXT="${DATASET_DIR}/prompt_ori.txt"
fi

# ===== 待测试的 checkpoint =====
TRAINED_CKPT="${TRAINED_CKPT_ROOT}/xxxxx/checkpoints/net_gen_iter_${ckpt_num}.pth"

OUTPUT_DIR="${OUTPUT_ROOT}/xxxxx/${DATASET_NAME}_scale_4_fidelity-1.0"

# ===== 测试参数 =====
if [ "${TEST_MODE}" == "benchmark_synthetic" ]; then
  test_args=" \
    --lq_input_txt ${DEG_ORI_IMAGES_TXT} \
    --prompt_input_txt ${HQ_ORI_PROMPTS_TXT} \
    --gt_input_txt ${HQ_ORI_IMAGES_TXT} \
    --output_dir ${OUTPUT_DIR} \
    --trained_ckpt ${TRAINED_CKPT} \
    --gen_start_point 750 \
    --scale 4.0 \
    --cfg 1.0 \
    --fidelity 1.0 \
    --mode one_step \
    --align_method wavelet \
    --crop_border 0 \
    --lora_rank 128 \
    --tiled \
    --tile_prompt_mode global \
    --use_prompt_prefix \
    --ref_input_txt ${REF_CROP_IMAGES_TXT} \
    --ref_max_pixels 1049088 \
    --use_full_lq_condition \
    --full_lq_max_pixels 465920 \
    --hq_ori_mask_txt_paths ${HQ_ORI_MASKS_TXT} \
    --ref_crop_mask_txt_paths ${REF_CROP_MASKS_TXT} \
"
elif [ "${TEST_MODE}" == "benchmark_realworld" ]; then
  test_args=" \
    --lq_input_txt ${REALWORLD_ORI_IMAGES_TXT} \
    --prompt_input_txt ${REALWORLD_ORI_PROMPTS_TXT} \
    --output_dir ${OUTPUT_DIR} \
    --trained_ckpt ${TRAINED_CKPT} \
    --gen_start_point 750 \
    --target_pixels 2073600 \
    --cfg 1.0 \
    --fidelity 1.0 \
    --mode one_step \
    --align_method wavelet \
    --crop_border 0 \
    --lora_rank 128 \
    --tiled \
    --tile_prompt_mode global \
    --use_prompt_prefix \
    --use_full_lq_condition \
    --full_lq_max_pixels 465920 \
    --ref_input_txt ${REF_CROP_IMAGES_TXT} \
    --ref_max_pixels 1049088 \
    --hq_ori_mask_txt_paths ${REALWORLD_ORI_MASKS_TXT} \
    --ref_crop_mask_txt_paths ${REF_CROP_MASKS_TXT_0} \
"
elif [ "${TEST_MODE}" == "benchmark_realsr" ]; then
  test_args=" \
    --lq_input_txt ${DEG_ORI_IMAGES_TXT} \
    --prompt_input_txt ${HQ_ORI_PROMPTS_TXT} \
    --gt_input_txt ${HQ_ORI_IMAGES_TXT} \
    --output_dir ${OUTPUT_DIR} \
    --trained_ckpt ${TRAINED_CKPT} \
    --gen_start_point 750 \
    --scale 4.0 \
    --cfg 1.0 \
    --fidelity 1.0 \
    --mode one_step \
    --align_method wavelet \
    --crop_border 0 \
    --lora_rank 128 \
    --tiled \
    --tile_prompt_mode global \
    --use_prompt_prefix \
    --use_full_lq_condition \
    --full_lq_max_pixels 465920 \
"
elif [ "${TEST_MODE}" == "benchmark_drealsr" ]; then
  test_args=" \
    --lq_input_txt ${DEG_ORI_IMAGES_TXT} \
    --prompt_input_txt ${HQ_ORI_PROMPTS_TXT} \
    --gt_input_txt ${HQ_ORI_IMAGES_TXT} \
    --output_dir ${OUTPUT_DIR} \
    --trained_ckpt ${TRAINED_CKPT} \
    --gen_start_point 750 \
    --scale 4.0 \
    --cfg 1.0 \
    --fidelity 1.0 \
    --mode one_step \
    --align_method wavelet \
    --crop_border 0 \
    --lora_rank 128 \
    --tiled \
    --tile_prompt_mode global \
    --use_prompt_prefix \
    --use_full_lq_condition \
    --full_lq_max_pixels 465920 \
"
fi


python examples/qwen_image/test_sr_edit_mask.py ${test_args}
