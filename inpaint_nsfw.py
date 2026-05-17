import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForInpainting  # Updated for Kandinsky 2.2
from nudenet import NudeDetector

# Match the labels from your original script
TARGET_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
CONF_THRESHOLD = 0.1


def detect_nsfw_regions(detector, image: Image.Image):
    """
    Detect sensitive exposed regions and return bounding boxes.
    """
    image_np = np.array(image)
    results = detector.detect(image_np)

    boxes = []
    for r in results:
        label = r["class"]
        score = r["score"]

        if label not in TARGET_LABELS or score < CONF_THRESHOLD:
            continue

        x, y, w, h = r["box"]
        boxes.append((int(x), int(y), int(x + w), int(y + h)))

    return boxes

def create_mask_plus(
    image_size: tuple[int, int], 
    results: list[dict], 
    padding_x: int = 40, 
    padding_y: int = 16
) -> Image.Image:
    """
    Create a binary mask for the inpainting pipeline with dynamic padding based on class.
    Expands the width (padding_x) by a factor of 3 for lower-body exposed regions,
    but keeps standard padding for FEMALE_BREAST_EXPOSED.
    White (255) indicates the area to regenerate, Black (0) is kept exactly as is.
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for r in results:
        label = r["class"]
        score = r["score"]

        # Filter out low confidence or unwanted classes
        if label not in TARGET_LABELS or score < CONF_THRESHOLD:
            continue

        x, y, w, h = r["box"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        # Determine horizontal padding based on the specific class
        if label == "FEMALE_BREAST_EXPOSED":
            current_padding_x = padding_x
        else:
            # Multiplies the horizontal padding by 3 for genitalia, buttocks, and anus classes
            current_padding_x = padding_x * 3

        # Apply padding with image boundary safety checks
        px1 = max(0, x1 - current_padding_x)
        py1 = max(0, y1 - padding_y)
        px2 = min(image_size[0], x2 + current_padding_x)
        py2 = min(image_size[1], y2 + padding_y)

        draw.rectangle([px1, py1, px2, py2], fill=255)

    return mask

def create_mask(
    image_size: tuple[int, int], 
    boxes: list[tuple[int, int, int, int]], 
    padding_x: int = 40, 
    padding_y: int = 16
) -> Image.Image:
    """
    Create a binary mask for the inpainting pipeline with asymmetrical padding.
    White (255) indicates the area to regenerate, Black (0) is kept exactly as is.
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for (x1, y1, x2, y2) in boxes:
        # Apply larger padding horizontally (X) and smaller padding vertically (Y)
        px1 = max(0, x1 - padding_x)
        py1 = max(0, y1 - padding_y)
        px2 = min(image_size[0], x2 + padding_x)
        py2 = min(image_size[1], y2 + padding_y)

        draw.rectangle([px1, py1, px2, py2], fill=255)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Inpaint sensitive regions with safe content using Kandinsky 2.2.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the original un-watermarked image")
    parser.add_argument("--output_path", type=str, default="inpainted_output.png")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="person wearing elegant modest clothes, covered up, highly detailed, photorealistic", 
        help="Prompt to guide the inpainted area."
    )
    parser.add_argument("--negative_prompt", type=str, default="nsfw, nude, naked, exposed, bad anatomy")
    parser.add_argument("--padding_x", type=int, default=40, help="Horizontal pixels to expand the mask")
    parser.add_argument("--padding_y", type=int, default=16, help="Vertical pixels to expand the mask")   
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale (Kandinsky 2.2 sweet spot is ~4.0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 1. Load the original image
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Could not find image at {image_path}")
        return

    init_image = Image.open(image_path).convert("RGB")

    # 2. Detect NSFW Regions
    print("Initializing NudeDetector...")
    detector = NudeDetector()
    image_np = np.array(init_image)
    results = detector.detect(image_np)

    # Filter results just to see if we have anything to process
    has_targets = any(r["class"] in TARGET_LABELS and r["score"] >= CONF_THRESHOLD for r in results)

    if not has_targets:
        print("No sensitive regions detected. Saving original image...")
        init_image.save(args.output_path)
        return

    print("Sensitive region(s) detected. Generating dynamic mask...")

    # 3. Create the binary mask (Passing the full results list now)
    mask_image = create_mask_plus(
        init_image.size, 
        results, 
        padding_x=args.padding_x, 
        padding_y=args.padding_y
    )
    
    # 4. Load the Kandinsky 2.2 Inpainting Pipeline
    print(f"Loading Kandinsky 2.2 inpainting pipeline on {args.device}...")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    # Using AutoPipeline handles loading both the prior and decoder variants seamlessly
    pipe = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        torch_dtype=dtype
    ).to(args.device)

    # 5. Run Inpainting
    print(f"Running inpainting with prompt: '{args.prompt}'...")
    generator = torch.Generator(device=args.device).manual_seed(42) # Fixed seed for reproducibility

    # Note: Kandinsky works best natively at 768x768 resolutions. 
    # If your input image sizes vary wildly, you can pass explicit height/width parameters here.
    inpainted_image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator
    ).images[0]

    # 6. Save the Final Image
    inpainted_image.save(args.output_path)
    print(f"Success! Inpainted image saved to {args.output_path}")


if __name__ == "__main__":
    main()