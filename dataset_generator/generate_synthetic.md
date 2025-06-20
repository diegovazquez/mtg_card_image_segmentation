# MTG Card Synthetic Image Generator

A comprehensive BlenderProc2-based system for generating synthetic Magic The Gathering card images with segmentation masks for training computer vision models.

## Overview

This tool generates realistic synthetic images of MTG cards with the following features:
- **Photorealistic rendering** using BlenderProc2 and HDRI environments
- **Precise card geometry** with rounded corners
- **Random transformations** for data augmentation
- **Segmentation masks** for semantic segmentation training
- **Configurable output** formats and directories

## Requirements

- Python 3.10+
- BlenderProc2 >= 2.6.0
- HDRI environment files (included in `hdri/` directory)
- Reference card images (745x1040 pixels)

## Usage

### Command Line Usage

```bash
# Single card generation

python -m blenderproc run dataset_generator/generate_synthetic.py --input dataset_generator/references/test/full_art_0a35bb96-89de-4a0a-a53e-aa97f800e92f.png --count 10 --hdri dataset_generator/hdri --output synthetic_output_test
```

### Advanced Configuration

```python
generator = MTGCardSynthetic(
    reference_image_path="card.png",
    output_base_dir="synthetic_output",
    hdri_dir="hdri",
    images_per_reference=8,
    output_resolution=(480, 640)  # Higher resolution
)
```

## Parameters

### MTGCardSynthetic Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_image_path` | str | Required | Path to reference card image (745x1040 px) |
| `output_base_dir` | str | "synthetic_output" | Base directory for output files |
| `hdri_dir` | str | "hdri" | Directory containing HDRI files |
| `images_per_reference` | int | 4 | Number of images to generate per reference |
| `output_resolution` | tuple | (720, 1280) | Output image resolution (width, height) |

### Command Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | str | Required | Input reference image or directory |
| `--output` | `-o` | str | "synthetic_output" | Output directory |
| `--hdri` | | str | "hdri" | HDRI directory |
| `--count` | `-c` | int | 4 | Images per reference |

# Developer Notes

The file sample.blend is the blender reference script for generating the 3d model of the card.

The file prompt.md is my personal prompts notes for CLINE, in spanish.


