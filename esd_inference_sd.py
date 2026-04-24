import argparse
import random
from copy import deepcopy
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from safetensors.torch import load_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esd_inference_sd.py",
        description="Generate one image with original SD weights and one with an ESD-trained checkpoint.",
    )
    parser.add_argument(
        "--basemodel_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base Stable Diffusion model id.",
    )
    parser.add_argument(
        "--esd_path",
        type=str,
        required=True,
        help="Path to ESD checkpoint (.safetensors) containing UNet weights.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="image of a cowboy drinking a beer",
        help="Prompt used for generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, a random seed is sampled.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images/sd_inference",
        help="Directory where generated images are saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sample",
        help="Filename prefix for saved images.",
    )
    return parser


def make_generator(seed: int, device: str) -> torch.Generator:
    gen_device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return torch.Generator(device=gen_device).manual_seed(seed)


def generate_one(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    device: str,
) -> Image.Image:
    generator = make_generator(seed, device)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    return image


def main() -> None:
    args = build_parser().parse_args()
    torch.set_grad_enabled(False)

    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.basemodel_id,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
    ).to(args.device)

    original_weights = deepcopy(pipe.unet.state_dict())
    esd_weights = load_file(args.esd_path, device=args.device)

    seed = args.seed if args.seed is not None else random.randint(0, 2**15)

    # Original model image
    pipe.unet.load_state_dict(original_weights, strict=False)
    original_image = generate_one(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=seed,
        device=args.device,
    )

    # ESD-trained model image
    pipe.unet.load_state_dict(esd_weights, strict=False)
    trained_image = generate_one(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=seed,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / f"{args.prefix}_original_seed{seed}.png"
    trained_path = output_dir / f"{args.prefix}_trained_seed{seed}.png"
    compare_path = output_dir / f"{args.prefix}_compare_seed{seed}.png"

    original_image.save(original_path)
    trained_image.save(trained_path)

    # Save a side-by-side comparison image for quick inspection.
    compare = Image.new("RGB", (args.width * 2, args.height))
    compare.paste(original_image.resize((args.width, args.height)), (0, 0))
    compare.paste(trained_image.resize((args.width, args.height)), (args.width, 0))
    compare.save(compare_path)

    print(f"Seed: {seed}")
    print(f"Original image saved to: {original_path}")
    print(f"Trained image saved to: {trained_path}")
    print(f"Comparison image saved to: {compare_path}")


if __name__ == "__main__":
    main()
