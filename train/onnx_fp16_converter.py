#!/usr/bin/env python3
"""
ONNX Model FP32 to FP16 Converter

This script converts an ONNX model from FP32 (float32) to FP16 (float16) precision
to reduce model size and potentially improve inference speed on compatible hardware.

Requirements:
    pip install onnx onnxconverter-common
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import onnx
    from onnxconverter_common import float16
except ImportError as e:
    print(f"Error: Required packages not installed. Please run:")
    print("pip install onnx onnxconverter-common")
    sys.exit(1)


def convert_onnx_to_fp16(input_path, output_path=None, keep_io_types=True):
    """
    Convert ONNX model from FP32 to FP16.
    
    Args:
        input_path (str): Path to input ONNX model
        output_path (str): Path for output FP16 model (optional)
        keep_io_types (bool): Whether to keep input/output in FP32 for compatibility
    
    Returns:
        str: Path to the converted model
    """
    
    # Validate input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.lower().endswith('.onnx'):
        raise ValueError("Input file must be an ONNX model (.onnx)")
    
    # Generate output path if not provided
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_fp16.onnx")
    
    print(f"Loading ONNX model from: {input_path}")
    
    try:
        # Load the original model
        model = onnx.load(input_path)
        
        # Check if model is valid
        onnx.checker.check_model(model)
        print("✓ Model validation passed")
        
        # Get original model info
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        print(f"Original model size: {original_size:.2f} MB")
        
        # Convert to FP16
        print("Converting model to FP16...")
        
        if keep_io_types:
            # Keep inputs and outputs in FP32 for better compatibility
            model_fp16 = float16.convert_float_to_float16(
                model, 
                keep_io_types=True,
                disable_shape_infer=False
            )
            print("✓ Conversion completed (I/O kept as FP32)")
        else:
            # Convert everything including I/O to FP16
            model_fp16 = float16.convert_float_to_float16(model)
            print("✓ Conversion completed (full FP16)")
        
        # Save the converted model
        print(f"Saving FP16 model to: {output_path}")
        onnx.save(model_fp16, output_path)
        
        # Get converted model info
        converted_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = (1 - converted_size / original_size) * 100
        
        print(f"✓ Model saved successfully!")
        print(f"Converted model size: {converted_size:.2f} MB")
        print(f"Size reduction: {compression_ratio:.1f}%")
        
        # Validate converted model
        try:
            converted_model = onnx.load(output_path)
            onnx.checker.check_model(converted_model)
            print("✓ Converted model validation passed")
        except Exception as e:
            print(f"⚠ Warning: Converted model validation failed: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model from FP32 to FP16 precision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python onnx_fp16_converter.py model.onnx
  python onnx_fp16_converter.py model.onnx -o model_half.onnx
  python onnx_fp16_converter.py model.onnx --full-fp16
        """
    )
    
    parser.add_argument(
        "input_model",
        help="Path to input ONNX model file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path for output FP16 model (default: input_name_fp16.onnx)"
    )
    
    parser.add_argument(
        "--full-fp16",
        action="store_true",
        help="Convert inputs/outputs to FP16 too (default: keep I/O as FP32)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists"
    )
    
    args = parser.parse_args()
    
    # Check if output file exists
    output_path = args.output
    if output_path is None:
        input_path_obj = Path(args.input_model)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_fp16.onnx")
    
    if os.path.exists(output_path) and not args.force:
        response = input(f"Output file '{output_path}' already exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Conversion cancelled.")
            return
    
    try:
        # Perform conversion
        result_path = convert_onnx_to_fp16(
            args.input_model,
            output_path,
            keep_io_types=not args.full_fp16
        )
        
        print(f"\n🎉 Conversion successful!")
        print(f"FP16 model saved to: {result_path}")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()