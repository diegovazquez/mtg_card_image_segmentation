
import blenderproc as bproc
"""
Synthetic MTG Card Dataset Generator

This script generates synthetic MTG card images using the MTGCardSynthetic class.
It processes reference images from references/train and references/test directories
and generates synthetic datasets in dataset/train and dataset/test respectively.

Usage:
    python -m blenderproc run dataset_generator/03_generate_synthetic_dataset.py
"""
import os
import sys
from pathlib import Path

parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)
from generate_synthetic import MTGCardSynthetic

def main():
    """
    Generates a synthetic dataset of MTG cards using reference images.
    It processes both 'train' and 'test' sets from predefined directories.
    """
    # Get the absolute path of the directory where this script is located
    script_dir = Path(__file__).resolve().parent

    # Define configurations for dataset generation
    configs = [
        {
            "input_dir": script_dir / "references" / "train",
            "output_dir": script_dir.parent / "dataset" / "train"
        },
        {
            "input_dir": script_dir / "references" / "test",
            "output_dir": script_dir.parent / "dataset" / "test"
        }
    ]

    # Directory containing HDRI files for lighting and background
    hdri_dir = script_dir / "hdri"
    
    # Number of synthetic images to generate for each reference image
    images_per_reference = 4

    # Process each configuration
    for config in configs:
        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        
        print(f"Processing directory: {input_dir}")
        print(f"Output will be saved to: {output_dir}")

        # A dummy reference image path is required for initialization,
        # but it will be ignored when using batch_process_directory.
        # We find the first .png file in the directory to use as a placeholder.
        try:
            dummy_ref_image = next(input_dir.glob("*.png"))
        except StopIteration:
            print(f"Warning: No reference images found in {input_dir}. Skipping.")
            continue

        # Initialize the synthetic image generator
        try:
            generator = MTGCardSynthetic(
                reference_image_path=str(dummy_ref_image),
                output_base_dir=str(output_dir),
                hdri_dir=str(hdri_dir),
                images_per_reference=images_per_reference
            )
            
            # Process all reference images in the specified directory
            generator.batch_process_directory(str(input_dir))
            
        except Exception as e:
            print(f"An error occurred while processing {input_dir}: {e}")
            continue

    print("Synthetic dataset generation complete.")

if __name__ == "__main__":
    main()
