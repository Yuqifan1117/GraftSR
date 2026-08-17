import os
import shutil
import tempfile
import yaml
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--mmaigc_dataset_yml",
        type=str,
        default=None,
        help="the yaml config file for mmagic 's clip data real degradation",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--deg_file_path",
        type=str,
        default=None,
        required=True,
        help="The path of the deg yaml."
    )
    parser.add_argument(
        "--dataset_txt_paths",
        type=str,
        nargs='+',
        default=None,
        required=True,
        help="The path of the images."
    )
    parser.add_argument(
        '--ref_txt_paths',
        type=str,
        nargs='+',
        default=None,
        help='The path of the reference images.'
    )
    parser.add_argument(
        '--similarity_txt_paths',
        type=str,
        nargs='+',
        default=None,
        help='Paths to similarity txt files (one float per line, corresponding to dataset_txt_paths).'
    )
    parser.add_argument(
        '--lq_txt_paths',
        type=str,
        nargs='+',
        default=None,
        help='Paths to txt files containing real LQ image paths (one per line, corresponding to dataset_txt_paths). '
             'When provided, use real LQ images instead of online degradation.'
    )
    parser.add_argument(
        '--highquality_lq_txt_paths',
        type=str,
        nargs='+',
        default=None,
        help='Paths to txt files containing real LQ image paths for highquality dataset (one per line, '
             'corresponding to highquality_dataset_txt_paths).'
    )
    parser.add_argument('--highquality_dataset_txt_paths', 
                        type=str,
                        nargs='+',
                        default=None, 
                        help='Paths to high quality dataset txt files'
    )
    parser.add_argument(
        "--main_prompt_txt_paths",
        type=str,
        nargs='+',
        default=None,
        help="Paths to txt file containing prompt file paths (one per line, corresponding to dataset_txt_paths)."
    )
    parser.add_argument(
        "--highquality_prompt_txt_paths",
        type=str,
        nargs='+',
        default=None,
        help="Paths to txt file containing prompt file paths for high quality dataset (one per line)."
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.5,
        help="Probability of using dataset_txt_paths (with reference) vs highquality_dataset_txt_paths (without reference). Only used when highquality_dataset_txt_paths is provided."
    )
    parser.add_argument(
        "--null_text_ratio",
        type=float,
        default=0,
        help="null_text_ratio",
    )
    parser.add_argument(
        "--use_qwen",
        default=False,
        action="store_true",
        help="Whether to use qwen to get prompt",
    )
    # gradient_accumulation_steps
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--offload_dis_t5",
        default=False,
        action="store_true",
        help="Whether to offload dis's t5 to save gpu memory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--use_resize_before_crop",
        default=False,
        action="store_true",
        help="Whether to resize image to short side size before cropping 512x512. If False, directly crop 512x512."
    )
    parser.add_argument(
        "--short_edge_size",
        type=int,
        default=1024,
        help="Short edge size for resizing image before cropping 512x512."
    )
    parser.add_argument(
        "--random_short_edge",
        default=False,
        action="store_true",
        help="Randomly sample short edge size in [512, min(original_short_edge, short_edge_size)] "
             "for each training sample, instead of using a fixed short_edge_size."
    )
    parser.add_argument(
        "--use_condition_type_embed",
        default=False,
        action="store_true",
        help="Whether to use condition type embed."
    )
    parser.add_argument(
        "--clip_grad_norm",
        default=False,
        action="store_true",
        help="Whether to clip grad norm."
    )
    parser.add_argument(
        '--ref_dropout_prob',
        type=float,
        default=0.0,
        help='Probability of randomly dropping the reference image during training, '
             'forcing the model to generate without ref condition.'
    )
    parser.add_argument(
        '--lq_dropout_similarity_threshold',
        type=float,
        default=0.95,
        help='Similarity threshold for allowing lq dropout. '
             'Only samples with ref-gt similarity > this value are eligible for lq dropout.'
    )
    parser.add_argument(
        '--use_generate_prompt_prefix',
        default=False,
        action="store_true",
        help="Whether to use generate-style prompt prefix (for Qwen-Image model)."
    )
    parser.add_argument(
        '--use_edit_prompt_prefix',
        default=False,
        action="store_true",
        help="Whether to use edit-style prompt prefix (for Qwen-Image-Edit model)."
    )
    parser.add_argument(
        '--aligned_crop_gt_ref',
        default=False,
        action="store_true",
        help="Whether to crop GT and REF at the same spatial position. "
             "Requires GT and REF to have the same resolution."
    )
    parser.add_argument(
        '--hq_mismatched_ref',
        default=False,
        action='store_true',
        help='When using highquality dataset, randomly sample a mismatched ref image '
             'instead of dropout, so the model learns to ignore irrelevant references.'
    )
    parser.add_argument(
        "--enable_tensorboard",
        default=False,
        action="store_true",
        help="Enable TensorBoard logging. Disabled by default to avoid NAS I/O errors."
    )
    parser.add_argument(
        "--flexible_ref_resolution",
        default=False,
        action="store_true",
        help="Whether to allow REF image to have arbitrary resolution (not forced to 512x512). "
             "When enabled, REF is resized by short edge and aligned to 16, then encoded independently."
    )
    parser.add_argument(
        "--flexible_ref_max_pixels",
        type=int,
        default=1366 * 768,
        help="Max total pixel budget for flexible REF resolution. "
             "Image is proportionally scaled so that w*h <= this value, then aligned to 16."
    )
    parser.add_argument(
        "--video_screenshot_degrade_prob",
        type=float,
        default=0.0,
        help="Probability of using video screenshot degradation instead of image degradation. "
             "0.0 = all image degradation (default, backward compatible), "
             "1.0 = all video screenshot degradation. Requires ffmpeg."
    )
    parser.add_argument(
        "--ref_light_degrade_prob",
        type=float,
        default=0.0,
        help="Probability of applying light degradation to REF image during training. "
             "0.0 = no degradation (default), 1.0 = always degrade. "
             "Helps the model be robust to slightly degraded reference images."
    )
    parser.add_argument(
        "--force_valid_crop_prob",
        type=float,
        default=0.0,
        help="以 prob 概率强制裁到有效区域，否则正常随机裁剪"
    )
    parser.add_argument(
        "--real_lq_degrade_prob",
        type=float,
        default=0.0,
        help="Probability of applying light degradation on top of real LQ images. "
             "0.0 = no re-degradation (default), 0.75 = 75%% of real LQ samples get re-degraded. "
             "Used to create training pairs from (pseudo-GT, re-degraded real LQ)."
    )
    parser.add_argument(
        "--dual_noise_cond_degrade",
        default=False,
        action="store_true",
        help="Enable dual degradation mode: run the same degradation pipeline twice on GT, "
             "assign the lighter result to noise flow (lq) and the heavier result to condition flow (lq_for_cond). "
             "This prevents the model from directly copying condition details."
    )
    parser.add_argument(
        "--use_full_lq_condition",
        default=False,
        action="store_true",
        help="Enable triple condition mode: cropped LQ block + full LQ + REF. "
             "When enabled, dataset returns an extra 'full_lq' field containing the "
             "full low-quality image before cropping, providing global layout context."
    )
    parser.add_argument(
        "--full_lq_max_pixels",
        type=int,
        default=1366 * 768,
        help="Max total pixel budget for full LQ in triple condition mode. "
             "Image is proportionally scaled so that w*h <= this value, then aligned to 16."
    )

    # 训练时分布式测试相关参数
    parser.add_argument(
        "--test_lq_txt",
        type=str,
        default=None,
        help="Path to txt file containing test LQ image paths. Enables distributed test after each epoch.",
    )
    parser.add_argument(
        "--test_gt_txt",
        type=str,
        default=None,
        help="Path to txt file containing test GT image paths (for FR metrics and visualization).",
    )
    parser.add_argument(
        "--test_ref_txt",
        type=str,
        default=None,
        help="Path to txt file containing test reference image paths (optional, for dual-condition).",
    )
    parser.add_argument(
        "--test_prompt_txt",
        type=str,
        default=None,
        help="Path to txt file containing test prompt paths (optional).",
    )
    parser.add_argument(
        "--test_scale",
        type=float,
        default=2.0,
        help="SR scale for test.",
    )

    # mask 调制相关参数
    parser.add_argument('--hq_ori_mask_txt_paths', type=str, nargs='+', default=None,
                    help='LQ mask 路径 txt 文件（白色=需要ref参考的区域）')
    parser.add_argument('--ref_crop_mask_txt_paths', type=str, nargs='+', default=None,
                        help='REF mask 路径 txt 文件（白色=有效纹理区域）')

    # 显式指定哪些子数据集有 ref/mask（逗号分隔索引，如 "0,2"）
    # 未指定时自动匹配最后一个子数据集
    parser.add_argument('--ref_dataset_indices', type=str, default=None,
                        help='Comma-separated indices of sub-datasets that have ref/mask. '
                             'E.g., "0,2" means the 1st and 3rd datasets have ref. '
                             'If not specified, auto-detects the last sub-dataset.')

    args = parser.parse_args()
    return args

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])

OSS_MOUNT_PREFIX = "/data/oss_bucket_0"

def _save_image(image, target_path):
    """OSS 挂载路径先写临时文件再复制，否则直接保存。"""
    if target_path.startswith(OSS_MOUNT_PREFIX):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            tmp_path = tmp_f.name
        image.save(tmp_path)
        shutil.copy2(tmp_path, target_path)
        os.remove(tmp_path)
    else:
        image.save(target_path)