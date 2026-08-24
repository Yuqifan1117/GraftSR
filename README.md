# GraftSR

**GraftSR** is a texture-reference-guided real-world image super-resolution framework. Given a low-quality image and a high-quality reference image of the same subject, GraftSR grafts fine-grained textures from the reference onto the degraded input and produces a photorealistic high-resolution result in a single diffusion step.

## 📰 News

<!-- News will be updated soon. -->

## ✅ TODO

- [ ] Upload the inference weights.
- [ ] Upload the dataset and benchmark.

## ⚙ Dependencies and Installation

1. Prepare conda env:
```
conda create -n yourenv python=3.11
```
2. Install ``pytorch`` (we recommend ``torch==2.6.0``): 
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  -f https://mirrors.aliyun.com/pytorch-wheels/cu124/
```
3. Install this repo (based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main)). The required packages will be automatically installed:
```
cd xxxx/ODTSR
pip3 install -e . -v  -i https://mirrors.cloud.tencent.com/pypi/simple
```


4. (For training) Install ``basicsr``:
```
pip install basicsr
```
Note:
You can apply the the following command to fix a bug in ``basicsr``. Make sure to replace ``/opt/conda`` with the path to your own conda environment:
```
sed -i '8s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' /opt/conda/lib/python3.11/site-packages/basicsr/data/degradations.py
```


5. Download base model to your disk: [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/main)

6. (For training) Download base model to your disk: [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main)

## 📦 Data Preparation

We provide the training dataset **TexRefSR-141K** and the evaluation benchmark **TexRefSR-Eval**. Every `*.txt` list file contains the **absolute paths** of the corresponding files (one path per line, line `i` across the txt files corresponds to the same sample `i`). After downloading, please replace the path prefixes inside the txt files with your own paths, e.g.:

```bash
sed -i 's|/original/dataset/root|/your/dataset/root|g' *.txt
```

### TexRefSR-141K (Training Dataset)

TexRefSR-141K contains 141K HQ–reference pairs. Each sample consists of a high-quality target image, a cropped reference image of the same subject, a text prompt, and SAM-3 segmentation masks of the subject region on both the HQ image and the reference crop. The expected directory structure is:

```
<your_dataset_root>/texrefsr_141k/
└── trainset/
    ├── hq_ori/                       # high-quality target (GT) images
    ├── ref_crop/                     # cropped reference images
    ├── prompt_files/                 # per-image text prompt files
    ├── hq_ori_mask_sam3/             # SAM-3 subject masks of hq_ori images
    ├── ref_crop_mask_sam3/           # SAM-3 subject masks of ref_crop images
    ├── hq_ori.txt                    # absolute paths of hq_ori images
    ├── ref_crop.txt                  # absolute paths of ref_crop images
    ├── prompt_ori.txt                # absolute paths of prompt files
    ├── hq_ori_mask_sam3.txt          # absolute paths of hq_ori masks
    └── ref_crop_mask_sam3.txt        # absolute paths of ref_crop masks
```

### TexRefSR-Eval (Evaluation Benchmark)

TexRefSR-Eval consists of two parts:

- **synthetic**: HQ images are degraded with a pre-generated degradation pipeline to obtain `lq_ori_degraded`, so both full-reference and no-reference metrics can be computed.
- **realworld**: collected real-world low-quality images (`lq_ori`). There is **no** `hq_ori` (no ground truth), so only no-reference metrics are reported.

The expected directory structure is:

```
<your_benchmark_root>/
├── synthetic/                        # synthetic degradation benchmark
│   ├── hq_ori/                       # high-quality ground-truth images
│   ├── lq_ori_degraded/              # pre-generated degraded LQ images
│   ├── ref_crop/                     # cropped reference images
│   ├── prompt_files/                 # per-image text prompt files
│   ├── hq_ori_mask_sam3/             # SAM-3 subject masks of hq_ori images
│   ├── ref_crop_mask_sam3/           # SAM-3 subject masks of ref_crop images
│   ├── hq_ori.txt
│   ├── lq_ori_degraded.txt
│   ├── prompt_ori.txt
│   ├── ref_crop.txt
│   ├── hq_ori_mask_sam3.txt
│   └── ref_crop_mask_sam3.txt
└── realworld/                        # real-world benchmark (no ground truth)
    ├── lq_ori/                       # real low-quality input images
    ├── ref_crop/                     # cropped reference images
    ├── prompt_files/                 # per-image text prompt files
    ├── lq_ori_mask_sam3/             # SAM-3 subject masks of lq_ori images
    ├── ref_crop_mask_sam3/           # SAM-3 subject masks of ref_crop images
    ├── lq_ori.txt
    ├── prompt_ori.txt
    ├── ref_crop.txt
    ├── lq_ori_mask_sam3.txt
    └── ref_crop_mask_sam3.txt
```

## 🚂 Training

Training is launched through [train_edit_mask.sh](./train_edit_mask.sh), which starts multi-GPU GAN training of [examples/qwen_image/train_gan_edit_mask.py](./examples/qwen_image/train_gan_edit_mask.py) with `accelerate`.

**Step 1. Fill in the path configuration** at the top of `train_edit_mask.sh`:

```bash
worker_count=8                                    # number of GPUs
CKPT_ROOT="/root/your/path/pretrain"              # contains Qwen-Image-Edit-2511/ and Wan2.1-T2V-1.3B/
DATASET_ROOT="/root/your/path/dataset"            # contains texrefsr_141k/
BENCHMARK_ROOT="/root/your/path/benchmark"        # contains synthetic/ and realworld/
OUTPUT_DIR="/root/your/path/experiments"          # checkpoints and logs
```

The accelerate config is selected by `worker_count` (`nebula_configs/accelerate-8.yaml` / `accelerate-32.yaml`).

**Step 2. (Optional) Adjust the training configuration.**

- Training hyper-parameters (learning rate, loss weights, LoRA rank, GAN settings, etc.) are defined in [examples/qwen_image/configs/agb1_g002_r1_5_dynamic_lnnp05_edit_mask_dilate.yaml](./examples/qwen_image/configs/agb1_g002_r1_5_dynamic_lnnp05_edit_mask_dilate.yaml), e.g. `learning_rate: 5e-5`, `lora_rank: 128`, `gan_loss_weight: 0.02`, `gen_start_point: 750`.
- The online degradation model applied to HQ images during training is configured in [examples/qwen_image/configs/deg_pisa.yaml](./examples/qwen_image/configs/deg_pisa.yaml).
- Key training arguments passed in the shell script, e.g. `--short_edge_size 1024`, `--flexible_ref_resolution`, `--use_full_lq_condition`, `--force_valid_crop_prob 0.5`, can also be tuned in `train_edit_mask.sh`.

**Step 3. Launch training:**

```bash
bash train_edit_mask.sh
```

Generator checkpoints are saved to `${OUTPUT_DIR}/<exp_name>/checkpoints/net_gen_iter_<iter>.pth`.

## 📊 Evaluation and Inference

### Inference

Batch inference on the benchmarks is provided by [test_edit_mask.sh](./test_edit_mask.sh), which runs [examples/qwen_image/test_sr_edit_mask.py](./examples/qwen_image/test_sr_edit_mask.py). Fill in the root paths and the checkpoint iteration at the top of the script, and select one of the four test modes via `TEST_MODE`:

| TEST_MODE | Dataset | Task |
|-|-|-|
| `benchmark_synthetic` | TexRefSR-Eval `synthetic/` | 4× SR on synthetically degraded images (`--scale 4.0`) |
| `benchmark_realworld` | TexRefSR-Eval `realworld/` | SR on real LQ images (`--target_pixels 2073600`) |
| `benchmark_realsr` | public RealSR benchmark | 4× SR |
| `benchmark_drealsr` | public DRealSR benchmark | 4× SR |

```bash
bash test_edit_mask.sh
```

> **Note:** `test_edit_mask.sh` performs batch inference over the txt file lists of the benchmarks. A standalone script that super-resolves a single user-input real image to 4× resolution (or an arbitrary specified resolution) is not provided yet.

### Evaluation

Evaluation is performed on three types of datasets with the scripts under [benchmark/](./benchmark/):

1. **Synthetic benchmark** (full-reference + no-reference metrics, `benchmark/evaluate_synthetic.sh`):
```bash
python evaluate_synthetic.py \
        --output_dir /root/your/path/experiments/your_exp_name/benchmark_synthetic_degraded-250_scale_4_fidelity-1.0 \
        --gt_dir /root/your/path/benchmark/synthetic/hq_ori \
        --metrics psnr,ssim,lpips,dists,niqe,musiq,clipiqa,maniqa-pipal \
        --crop_border 4 \
        --save_path /root/your/path/experiments/summary/your_exp_name_synthetic250_metrics.json
```

2. **Real-world benchmark** (no-reference metrics only, since no ground truth is available, `benchmark/evaluate_real.sh`):
```bash
python evaluate_real.py \
        --output_dir /root/your/path/experiments/your_exp_name/benchmark_realworld-50_scale_4.0_fidelity-1.0 \
        --metrics niqe,musiq,clipiqa,maniqa-pipal \
        --crop_border 0 \
        --save_path /root/your/path/experiments/summary/your_exp_name_realworld_metrics.json
```

3. **General / public benchmarks** such as RealSR and DRealSR (`benchmark/evaluate_synthetic_general.sh`), which reuses `evaluate_synthetic.py` with the corresponding public GT directory.

## 🙏 Acknowledgement

GraftSR is built upon the following great open-source projects: [OSTSR](https://github.com/RedMediaTech/ODTSR). We sincerely thank the authors for their contributions.

## 📖 Citation

BibTeX citation is coming soon.
