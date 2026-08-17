import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=3, collate_fn=lambda x: x[0])
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    from datetime import datetime
    from contextlib import redirect_stdout
    baseworkdir = os.path.join("./experiments", model_logger.output_path)
    if accelerator.is_main_process:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_logger.output_path =os.path.join("./experiments", model_logger.output_path, current_time)
        workdir = model_logger.output_path
        os.makedirs(workdir, exist_ok=False)
        # trt_logger = TensorBoardLogger(workdir, name="tensorboard")

        # backup config file
        # yaml_file_name = os.path.basename(args.mmaigc_dataset_yml)
        # target_path = os.path.join(workdir, yaml_file_name)
        # shutil.copy2(args.mmaigc_dataset_yml, target_path)
        
        # plot model structure
        filename = os.path.join(workdir, 'model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(model.module.pipe)
                
        # 获取生成器可训练参数名字并写入文件
        filename = os.path.join(workdir, 'trainable_parameters.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                for name, param in model.module.pipe.named_parameters():
                    if param.requires_grad:
                        print(name)

    accelerator.wait_for_everyone()
    
    subdirs = [d for d in os.listdir(baseworkdir) if os.path.isdir(os.path.join(baseworkdir, d))]
    latest = max(subdirs)
    baseworkdir = os.path.join(baseworkdir, latest)

    for epoch_id in range(num_epochs):
        for idx, data in enumerate(dataloader):
            accelerator.print("now idx:", idx)
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(loss)
                scheduler.step()

            # 如果iter满足要求，每个进程进行多步推理，参考pipe的call重新写推理
            if idx % 50 == 0:
                gt_img = data['gt']
                lq_img = data['lq']
                prompt = "4k, highly detailed, perfect without deformations, Photographic realism"
                if accelerator.local_process_index == 7:
                    prompt = "beautiful women, " + prompt
                
                accelerator.print(data['text'])

                cfg_scale = accelerator.local_process_index + 1

                res_img = model.module.pipe.TI2I(
                    prompt = prompt,
                    negative_prompt = "jpeg artifacts, oversharpening",
                    cfg_scale = cfg_scale,
                    condition_image = lq_img,
                    num_inference_steps = 40
                )
                
                imgs = [gt_img, lq_img, res_img]
                h = max(im.height for im in imgs)
                total_w = sum(im.width for im in imgs)
                canvas = Image.new('RGB', (total_w, h))

                x = 0
                for im in imgs:
                    canvas.paste(im, (x, 0))
                    x += im.width
                
                canvas.save(os.path.join(baseworkdir, f"iter_{idx}_rank_{accelerator.local_process_index}_cfg_{cfg_scale}.png"))

        model_logger.on_epoch_end(accelerator, model, epoch_id)


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    # for sr task training
    parser.add_argument("--deg_file_path", type=str, default=None, required=True, help="The path of the deg yaml.")
    parser.add_argument("--dataset_txt_paths", type=str, default=None, required=True, help="The path of the images.")
    parser.add_argument('--highquality_dataset_txt_paths', type=str, nargs='?', default=None, help='Paths to high quality dataset txt files')
    parser.add_argument("--null_text_ratio", type=float, default=0, help="null_text_ratio")
    parser.add_argument("--use_qwen", default=False, action="store_true", help="Whether to use qwen to get prompt",)
    # for sr task inference
    parser.add_argument("--input_path", type=str, default=None, required=False, help="The path of the input dir.")
    parser.add_argument("--trained_ckpt", type=str, default=None, required=False, help="Path to trained_ckpt.")
    parser.add_argument("--output_dir", type=str, default="./test_outputs", help="Path to save the results.")
    parser.add_argument("--scale", type=float, default=2.0, help="sr scale")
    parser.add_argument("--cfg", type=float, default=2.5, help="cfg")
    parser.add_argument("--start_end", type=str, default="0,", help="index range")
    return parser
