import argparse
import sys

import torch

sys.path.append(".")

from utils.img_dataloader import ImageDataset
from utils.esd_trainer import (
    ESDConfig,
    run_img_training,
)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="Train SD Image Finetune",
        description="Fine-tune Stable Diffusion using supervised image training.",
    )

    #
    # Model
    #

    parser.add_argument(
        "--basemodel_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="HF diffusers model id",
    )

    #
    # Dataset
    #

    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to metadata json",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing training images",
    )

    #
    # Prompt
    #

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Training prompt shared by all images",
    )

    #
    # Training method
    #

    parser.add_argument(
        "--train_method",
        type=str,
        required=True,
        help=(
            "Training method: "
            "esd-x, esd-u, esd-all, selfattn"
        ),
    )

    #
    # Optimization
    #

    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Number of optimization steps",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size",
    )

    #
    # Resolution
    #

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution",
    )

    #
    # Saving
    #

    parser.add_argument(
        "--save_path",
        type=str,
        default="trained-models/sd-img/",
        help="Checkpoint save directory",
    )

    #
    # Device
    #

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device",
    )

    #
    # Mixed precision
    #

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Training dtype",
    )

    #
    # Gradient checkpointing
    #

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )

    #
    # TF32
    #

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )

    return parser


def parse_dtype(dtype_str: str) -> torch.dtype:

    if dtype_str == "fp16":
        return torch.float16

    if dtype_str == "bf16":
        return torch.bfloat16

    if dtype_str == "fp32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main() -> None:

    args = build_parser().parse_args()

    #
    # Dataset
    #

    dataset = ImageDataset(
        metadata_path=args.metadata_path,
        image_dir=args.image_dir,
        prompt=args.prompt,
        resolution=args.resolution,
    )

    #
    # Config
    #

    config = ESDConfig(
        family="sd_img",
        base_model_id=args.basemodel_id,

        #
        # Reusing erase_concept field
        # as training prompt
        #
        erase_concept=args.prompt,
        erase_from=None,

        train_method=args.train_method,

        iterations=args.iterations,
        lr=args.lr,

        #
        # Unused in supervised diffusion training
        #
        negative_guidance=0,
        num_inference_steps=50,
        guidance_scale=1.0,

        batch_size=args.batchsize,
        resolution=args.resolution,

        save_path=args.save_path,
        device=args.device,

        torch_dtype=parse_dtype(args.dtype),

        gradient_checkpointing=args.gradient_checkpointing,
        allow_tf32=args.allow_tf32,
    )

    #
    # Train
    #

    checkpoint_path = run_img_training(
        config=config,
        dataset=dataset,
    )

    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
    
# python train_sd_img.py \
#   --metadata_path inpaint_ds/metadata.json \
#   --image_dirinpaint_ds \
#   --prompt "nudity" \
#   --train_method esd-x \
#   --iterations 5000 \
#   --resolution 512 \
#   --save_path trained-models/sd-img/