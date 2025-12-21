# Vegetal and Inorganic Material Image Processing Pipeline for Archaeological Bitumen-based Samples

This repository provides a two-stage image processing pipeline for the
enhancement, identification and segmentation of features from archaeological
bitumen-based samples.

## Workflow

1. Load raw RGB images and convert them to 8-bit grayscale.
2. Apply histogram equalization to improve contrast.
3. Apply a global threshold to isolate near-black regions.
4. Perform Retinex-based denoising to enhance fine features.
5. Subtract the denoised background to highlight features.
6. Save automatically filtered images.
7. Load the corresponding manually selected feature image (`_msf`).
8. Invert the manual mask and subtract it from the filtered image.
9. Remap grayscale values:
   - Values in the range 1–254 → white (255)
   - Values equal to 0 or 255 → black (0)
10. Save the final binary image.

## Input requirements

- `--input` (default: `raw_images`)  
  Raw sample images. Non-sample areas must already be removed and set to black (0).

- `--filtered` (default: `auto_filtered_images`)  
  Automatically generated OpenCV outputs.

- `--manual` (default: `manual_selected_features`)  
  Manually selected feature images. Files must share the same base name as the input image and use the suffix `_msf`.

- `--final` (default: `processed_images`)  
  Final binarized subtraction results.

### Supported image format
- Input: `.tif`  
- Output: `.tif`

## Optional parameters

- `--threshold` (default: `5`)  
  Global threshold value for near-black segmentation.

- `--sigma` (default: `10`)  
  Gaussian blur sigma used by the Retinex denoising step.

- `--iterations` (default: `5`)  
  Number of Retinex denoising iterations.

## Usage

```bash
python pipeline.py \
  --input raw_images \
  --filtered auto_filtered_images \
  --manual manual_selected_features \
  --final processed_images \
  --threshold 5 \
  --sigma 10 \
  --iterations 5
