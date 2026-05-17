#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoPipelineForInpainting,
)
from PIL import Image, ImageDraw
from nudenet import NudeDetector


# ============================================================
# NudeNet Detection Settings
# ============================================================

TARGET_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

# The baseline threshold inside the detector function
CONF_THRESHOLD = 0.1
# The higher threshold required to accept the image for dataset inclusion
QUALITY_CONF_THRESHOLD = 0.6


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset using Stable Diffusion v1.4, "
            "detect sensitive regions with NudeNet, "
            "inpaint them using Kandinsky 2.2, "
            "and save metadata."
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where images and metadata.json are saved.",
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
        help="Prompt used for inpainting.",
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
        default="nsfw, nude, naked, exposed, bad anatomy",
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
        default=50,
        help="Generation inference steps.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3,
    )

    parser.add_argument(
        "--inpaint_steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--inpaint_guidance",
        type=float,
        default=4.0,
        help="Kandinsky 2.2 works best around 4.0",
    )

    parser.add_argument(
        "--padding_x",
        type=int,
        default=20,
        help="Horizontal mask expansion.",
    )

    parser.add_argument(
        "--padding_y",
        type=int,
        default=10,
        help="Vertical mask expansion.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser


# ============================================================
# Random Seed Helper
# ============================================================

def make_generator(seed: int, device: str):

    gen_device = (
        "cuda"
        if device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    return torch.Generator(device=gen_device).manual_seed(seed)


# ============================================================
# NudeNet Detection
# ============================================================

def detect_nsfw_regions(
    detector,
    image: Image.Image,
):
    """
    Detect NSFW regions.

    Returns:
        raw NudeNet results
    """

    image_np = np.array(image)

    results = detector.detect(image_np)

    filtered = []

    for r in results:

        label = r["class"]
        score = r["score"]

        if label not in TARGET_LABELS:
            continue

        if score < CONF_THRESHOLD:
            continue

        filtered.append(r)

    return filtered


# ============================================================
# Dynamic Mask Creation
# ============================================================

def create_dynamic_mask(
    image_size: tuple[int, int],
    results: list[dict],
    padding_x: int = 20,
    padding_y: int = 10,
):
    """
    Create a dynamic binary mask.

    - Breasts get normal padding.
    - Lower-body regions get wider horizontal padding.
    """

    mask = Image.new("L", image_size, 0)

    draw = ImageDraw.Draw(mask)

    width, height = image_size

    for r in results:

        label = r["class"]

        x, y, w, h = r["box"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        # Wider masks for lower-body regions
        if label == "FEMALE_BREAST_EXPOSED":
            current_padding_x = padding_x
        else:
            current_padding_x = padding_x * 3

        px1 = max(0, x1 - current_padding_x)
        py1 = max(0, y1 - padding_y)

        px2 = min(width, x2 + current_padding_x)
        py2 = min(height, y2 + padding_y)

        draw.rectangle(
            [px1, py1, px2, py2],
            fill=255,
        )

    return mask


# ============================================================
# Main
# ============================================================

def main():

    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = []

    dtype = (
        torch.float16
        if args.device.startswith("cuda")
        else torch.float32
    )

    # --------------------------------------------------------
    # Load SD v1.4 Generation Pipeline
    # --------------------------------------------------------

    print("Loading Stable Diffusion v1.4...")

    gen_pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(args.device)

    # --------------------------------------------------------
    # Load Kandinsky 2.2 Inpainting Pipeline
    # --------------------------------------------------------

    print("Loading Kandinsky 2.2 inpainting pipeline...")

    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        torch_dtype=dtype,
    ).to(args.device)

    # --------------------------------------------------------
    # NudeNet
    # --------------------------------------------------------

    print("Loading NudeNet detector...")

    detector = NudeDetector()

    print(f"\nGenerating {args.num_images} images...\n")

    # ========================================================
    # Main Dataset Loop
    # ========================================================

    for idx in range(args.num_images):

        image_idx = idx + 1
        
        # Inner loop to find a seed that meets the quality threshold
        while True:
            seed = random.randint(0, 2**15)

            print(f"[{image_idx}/{args.num_images}] Trying Seed = {seed}...")

            generator = make_generator(
                seed,
                args.device,
            )

            # ----------------------------------------------------
            # 1. Generate Image
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

            detections = detect_nsfw_regions(
                detector,
                generated_image,
            )

            # Check if any detected region passes our strict quality threshold (0.6)
            has_high_conf_detection = any(
                det["score"] >= QUALITY_CONF_THRESHOLD for det in detections
            )

            if has_high_conf_detection:
                break  # Seed accepted, exit the retry loop
            else:
                print(f"    -> Seed {seed} dropped. No target classes detected above confidence {QUALITY_CONF_THRESHOLD}.")

        # ----------------------------------------------------
        # 3. Inpaint (Since we broke out of the loop, detections exist)
        # ----------------------------------------------------

        print(f"    Accepted! Detected {len(detections)} NSFW region(s).")

        mask_image = create_dynamic_mask(
            generated_image.size,
            detections,
            padding_x=args.padding_x,
            padding_y=args.padding_y,
        )

        inpaint_generator = make_generator(
            seed,
            args.device,
        )

        final_image = inpaint_pipe(
            prompt=args.inpaint_prompt,
            negative_prompt=args.negative_prompt,
            image=generated_image,
            mask_image=mask_image,
            num_inference_steps=args.inpaint_steps,
            guidance_scale=args.inpaint_guidance,
            generator=inpaint_generator,
        ).images[0]

        # ----------------------------------------------------
        # 4. Save Final Image
        # ----------------------------------------------------

        image_filename = f"image_{image_idx:05d}.png"

        image_path = output_dir / image_filename

        final_image.save(image_path)

        metadata.append(
            {
                "image": image_filename,
                "seed": seed,
            }
        )

        print(f"    Saved -> {image_filename}")

    # ========================================================
    # Save Metadata
    # ========================================================

    metadata_path = output_dir / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n================================================")
    print("Dataset generation complete.")
    print(f"Images:   {output_dir}")
    print(f"Metadata: {metadata_path}")
    print("================================================")


if __name__ == "__main__":
    main()