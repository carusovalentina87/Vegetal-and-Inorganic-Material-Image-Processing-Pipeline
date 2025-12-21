---

## pipeline.py 

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vegetal and Inorganic Material Image Processing Pipeline for Archaeological Bitumen-based Samples
========================================================

This script implements a two-stage image processing pipeline for the
enhancement, identification and segmentation of features from archaeological
bitumen-based samples.

The workflow consists of:
1. Load raw RGB images and convert them to 8-bit grayscale.
2. Apply histogram equalization to improve contrast.
3. Apply a global threshold to isolate near-black regions.
4. Perform Retinex-based denoising to enhance fine features.
5. Subtract the denoised background to highlight features.
6. Save automatically filtered images.
7. Load the corresponding manually selected feature image (_msf).
8. Invert the manual mask and subtract it from the filtered image.
9. Remap grayscale values:
   - Values in the range 1–254 → white (255)
   - Values equal to 0 or 255 → black (0)
10. Save the final binary image.

"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps


# ============================================================
# OpenCV-based filtering functions
# ============================================================

def load_image(image_path: str) -> np.ndarray:
    """Load an image and convert it to 8-bit grayscale."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray)  # ensures 8-bit
    return gray


def equalize_histogram(gray_img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to improve contrast (8-bit)."""
    return cv2.equalizeHist(gray_img)


def threshold_image(gray_img: np.ndarray, threshold_value: int) -> np.ndarray:
    """Apply global binary thresholding."""
    _, thresh = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh


def retinex_denoising(gray_img: np.ndarray, sigma: float = 10, iterations: int = 5) -> np.ndarray:
    """Apply Retinex-based denoising to enhance image details."""
    illumination = cv2.GaussianBlur(gray_img, (0, 0), sigma)
    reflection = np.float32(gray_img) / (np.float32(illumination) + 1e-5)
    reflection = np.log1p(reflection)
    reflection = np.uint8(np.clip(reflection * 255, 0, 255))

    for _ in range(iterations):
        reflection = cv2.GaussianBlur(reflection, (0, 0), sigma)
        reflection = np.uint8(np.clip(reflection * 255, 0, 255))

    return reflection


def subtract_background(gray_img: np.ndarray, clean_img: np.ndarray) -> np.ndarray:
    """Subtract background to enhance features in white."""
    highlighted_img = cv2.subtract(gray_img, clean_img)
    highlighted_img = cv2.bitwise_not(highlighted_img)
    return highlighted_img


def preprocess_image(
    image_path: str,
    output_path: str,
    threshold: int,
    sigma: float,
    iterations: int
) -> None:
    """Execute the full OpenCV preprocessing pipeline for a single image."""
    gray_img = load_image(image_path)
    equalized_img = equalize_histogram(gray_img)

    thresh_img = threshold_image(equalized_img, threshold)
    denoised_img = retinex_denoising(thresh_img, sigma=sigma, iterations=iterations)
    highlighted_img = subtract_background(equalized_img, denoised_img)

    cv2.imwrite(output_path, highlighted_img)


# ============================================================
# PIL-based subtraction and final processing
# ============================================================

def get_base_name(filename: str) -> str:
    """Extract the base name before suffixes like '_msf' or '_bkg'."""
    base, _ = os.path.splitext(filename)
    if base.endswith("_msf"):
        return base.replace("_msf", "")
    if base.endswith("_bkg"):
        return base.replace("_bkg", "")
    return base


# ============================================================
# Pipeline execution
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature image processing pipeline")

    parser.add_argument("--input", default="raw_images", help="Folder with raw input images")
    parser.add_argument("--filtered", default="auto_filtered_images", help="Folder for filtered output images")
    parser.add_argument(
        "--manual",
        default="manual_selected_features",
        help="Folder with manual selected features (suffix '_msf')"
    )
    parser.add_argument(
        "--final",
        default="processed_images",
        help="Folder for final processed binary images"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Threshold value for near-black pixel segmentation (default = 5)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10,
        help="Sigma value for Retinex Gaussian blur (default = 10)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of Retinex denoising iterations (default = 5)"
    )

    args = parser.parse_args()

    os.makedirs(args.filtered, exist_ok=True)
    os.makedirs(args.final, exist_ok=True)

    # --------------------------------------------------------
    # Step 1: OpenCV filtering
    # --------------------------------------------------------
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist.")
    else:
        for filename in os.listdir(args.input):
            if filename.lower().endswith(".tif"):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.filtered, filename)

                preprocess_image(
                    input_path,
                    output_path,
                    threshold=args.threshold,
                    sigma=args.sigma,
                    iterations=args.iterations
                )
                print(f"Filtered image saved: {output_path}")

    # --------------------------------------------------------
    # Step 2: PIL subtraction & grayscale remapping
    # --------------------------------------------------------
    image_dict = {}

    for folder in [args.filtered, args.manual]:
        if not os.path.exists(folder):
            print(f"Warning: folder '{folder}' does not exist.")
            continue

        for filename in os.listdir(folder):
            if filename.lower().endswith(".tif"):
                base_name = get_base_name(filename)
                image_dict.setdefault(base_name, []).append((folder, filename))

    for base_name, paths in image_dict.items():
        print(f"\nProcessing prefix: {base_name}")

        try:
            if len(paths) != 2:
                print(f"Prefix {base_name}: invalid number of images ({len(paths)}). Skipping.")
                continue

            # Ensure order: [filtered, manual_selected]
            paths.sort(key=lambda x: 0 if x[0] == args.filtered else 1)

            images = [
                Image.open(os.path.join(folder, filename)).convert("RGB")
                for folder, filename in paths
            ]

            # Resize to match dimensions
            min_width = min(img.width for img in images)
            min_height = min(img.height for img in images)
            images = [img.resize((min_width, min_height), Image.LANCZOS) for img in images]

            # Perform subtraction
            manual_inverted = ImageOps.invert(images[1])
            result = ImageChops.subtract(images[0], manual_inverted)

            # Pixel mapping logic
            result_gray = result.convert("L")
            final_binarized = result_gray.point(lambda p: 255 if 0 < p < 255 else 0)

            output_path = os.path.join(args.final, f"{base_name}_msf_final.tif")
            final_binarized.save(output_path)
            print(f"Final result saved: {output_path}")

        except Exception as e:
            print(f"Error processing {base_name}: {e}")


if __name__ == "__main__":
    main()
