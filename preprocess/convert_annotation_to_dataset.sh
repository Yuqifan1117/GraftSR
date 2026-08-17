#!/bin/bash

# ANNOTATION_LIST_JSON='/mnt/workspace/yihu/projects/ODTSR/dataset/cdxq_60w_260320_all-qwen3.5-plus/260320_all-qwen3.5-plus_annotation_list.json'
# DATASET_ROOT='/data/oss_bucket_0/Users/yihu/aigc_img/dataset/cdxq_60w_260320_all-qwen3.5-plus'

DATASET_ROOT='/mnt/workspace/yihu/projects/ODTSR/dataset/cdxq_40-91w_260326_all-qwen3.5-plus'
ANNOTATION_LIST_JSON='/mnt/workspace/yihu/projects/ODTSR/dataset/cdxq_40-91w_260326_all-qwen3.5-plus/annotation_list.json'

python convert_to_test_dataset.py \
    --annotation-list-json "${ANNOTATION_LIST_JSON}" \
    --output-dir "${DATASET_ROOT}/testset_260331" \
    --min-image-side 512 \
    --scale-factor 4 \
    --max-samples 1000

python convert_to_train_dataset.py \
    --annotation-list-json "${ANNOTATION_LIST_JSON}" \
    --output-dir "${DATASET_ROOT}/trainset_260331" \
    --min-image-side 512 \
    --skip-first 1000

echo "Script Executed."