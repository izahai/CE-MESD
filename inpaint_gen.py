#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
)
from PIL import Image, ImageDraw
from nudenet import NudeDetector


# ------------------------------------------------------------
# NudeNet Detection Settings
# ------------------------------------------------------------

TARGET_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

CONF_THRESHOLD = 0.2


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset using Stable Diffusion v1.4, "
            "automatically detect sensitive regions with NudeNet, "
            "inpaint them, and save metadata."
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where images and metadata.json will be saved.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt used for image generation.",
    )

    parser.add_argument(
        "--inpaint_prompt",
        type=str,
        required=True,
        help="Prompt used to inpaint sensitive regions.",
    )

    parser.add_argument(
        "--num_images",
        type=int,
        required=True,
        help="Number of images to generate.",
    )

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="nsfw, nude, naked, exposed body parts, explicit content",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )

    parser.add_argument(
        "--inpaint_steps",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--inpaint_guidance",
        type=float,
        default=7.5,
    )

    parser.add_argument(
        "--mask_padding",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser


# ------------------------------------------------------------
# NudeNet Detection
# ------------------------------------------------------------

def detect_sensitive_regions(detector, image: Image.Image):
    """
    Detect sensitive exposed regions using NudeNet.

    Returns:
        list[(x1, y1, x2, y2)]
    """

    image_np = np.array(image)
    results = detector.detect(image_np)

    boxes = []

    for result in results:

        label = result["class"]
        score = result["score"]

        if label not in TARGET_LABELS:
            continue

        if score < CONF_THRESHOLD:
            continue

        x, y, w, h = result["box"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        boxes.append((x1, y1, x2, y2))

    return boxes


# ------------------------------------------------------------
# Mask Creation
# ------------------------------------------------------------

def create_mask(
    image_size,
    boxes,
    padding=16,
):
    """
    Create a binary mask:
    - Black = keep
    - White = regenerate
    """

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    width, height = image_size

    for (x1, y1, x2, y2) in boxes:

        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(width, x2 + padding)
        py2 = min(height, y2 + padding)

        draw.rectangle(
            [px1, py1, px2, py2],
            fill=255,
        )

    return mask


# ------------------------------------------------------------
# Seed Helper
# ------------------------------------------------------------

def make_generator(seed, device):
    gen_device = (
        "cuda"
        if device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    return torch.Generator(device=gen_device).manual_seed(seed)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    dtype = (
        torch.float16
        if args.device.startswith("cuda")
        else torch.float32
    )

    print("Loading Stable Diffusion v1.4 generation pipeline...")

    gen_pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(args.device)

    print("Loading Stable Diffusion inpainting pipeline...")

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(args.device)

    print("Loading NudeNet detector...")

    detector = NudeDetector()

    print(f"Generating {args.num_images} images...\n")

    for idx in range(args.num_images):

        image_index = idx + 1

        seed = random.randint(0, 2**31 - 1)

        print(f"[{image_index}/{args.num_images}] Seed: {seed}")

        generator = make_generator(seed, args.device)

        # ----------------------------------------------------
        # 1. Generate Initial Image
        # ----------------------------------------------------

        generated_image = gen_pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        # ----------------------------------------------------
        # 2. Detect Sensitive Regions
        # ----------------------------------------------------

        boxes = detect_sensitive_regions(
            detector,
            generated_image,
        )

        # ----------------------------------------------------
        # 3. Inpaint If Needed
        # ----------------------------------------------------

        if boxes:

            print(f"    Detected {len(boxes)} sensitive region(s).")

            mask = create_mask(
                generated_image.size,
                boxes,
                padding=args.mask_padding,
            )

            inpaint_generator = make_generator(seed, args.device)

            final_image = inpaint_pipe(
                prompt=args.inpaint_prompt,
                negative_prompt=args.negative_prompt,
                image=generated_image,
                mask_image=mask,
                num_inference_steps=args.inpaint_steps,
                guidance_scale=args.inpaint_guidance,
                generator=inpaint_generator,
            ).images[0]

        else:

            print("    No sensitive regions detected.")

            final_image = generated_image

        # ----------------------------------------------------
        # 4. Save Final Image
        # ----------------------------------------------------

        image_filename = f"image_{image_index:04d}.png"

        image_path = output_dir / image_filename

        final_image.save(image_path)

        metadata.append(
            {
                "image": image_filename,
                "seed": seed,
            }
        )

        print(f"    Saved -> {image_filename}")

    # --------------------------------------------------------
    # Save Metadata JSON
    # --------------------------------------------------------

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"Images saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()