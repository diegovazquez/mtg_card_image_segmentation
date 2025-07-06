#!/usr/bin/env python3
"""
Export trained corner detection model to ONNX format
"""

import os
import sys
import argparse
import torch
import torch.onnx
import numpy as np
from typing import Tuple, Optional
from onnxconverter_common import auto_mixed_precision
import onnx

# Import local modules
from model import create_model, load_model


def export_to_onnx(
    model_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (640, 480),
    batch_size: int = 1,
    opset_version: int = 19,
    dynamic_axes: bool = True,
    optimize: bool = True,
    do_constant_folding: bool = True,
    auto_convert_mixed_precision: bool = True
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model_path: Path to the trained model checkpoint
        output_path: Path where to save the ONNX model
        input_size: Input image size (width, height)
        batch_size: Batch size for export (use 1 for single image inference)
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes for variable batch size
        optimize: Whether to optimize the ONNX model
        do_constant_folding: Whether to apply constant folding optimization
        auto_convert_mixed_precision: Whether to automatically convert mixed precision models
    """
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model  
    # Try to get heatmap_size from model checkpoint config, fallback to input parameter
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    checkpoint_heatmap_size = input_size  # Default fallback
    if 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
        if 'config' in checkpoint['metrics'] and 'heatmap_size' in checkpoint['metrics']['config']:
            checkpoint_heatmap_size = tuple(checkpoint['metrics']['config']['heatmap_size'])
    
    model = load_model(model_path, num_keypoints=4, heatmap_size=checkpoint_heatmap_size)
    model.eval()

    # Create dummy input tensor
    width, height = input_size
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    
    # Define dynamic axes if requested
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )
    
    print(f"Model exported successfully to: {output_path}")

    if auto_convert_mixed_precision:
        output_path_fp16 = str(output_path).replace(".onnx", "_fp16.onnx")
        model_onnx = onnx.load(output_path)
        feed_dict = {'input': dummy_input.numpy()}
        
        model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model_onnx, feed_dict, rtol=0.01, atol=0.001, keep_io_types=True)
        onnx.save(model_fp16, output_path_fp16)

        print(f"Model FP16 exported successfully to: {output_path_fp16}")


def get_model_info(model_path: str):
    """
    Get information about the model checkpoint
    
    Args:
        model_path: Path to the model checkpoint
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("Model Checkpoint Information:")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'Unknown')}")
    
    if 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
        metrics = checkpoint['metrics']
        print(f"  Config: {metrics.get('config', {})}")
        if 'train_loss' in metrics:
            print(f"  Train Loss: {metrics['train_loss']}")
        if 'val_loss' in metrics:
            print(f"  Val Loss: {metrics['val_loss']}")


def main():
    """
    Main function for ONNX export
    """
    parser = argparse.ArgumentParser(description='Export trained model to ONNX format')
    parser.add_argument('model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, help='Output ONNX file path')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 480], 
                        help='Input image size (width height)')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Batch size for export')
    parser.add_argument('--opset-version', type=int, default=19, 
                        help='ONNX opset version')
    parser.add_argument('--no-dynamic-axes', action='store_true',
                        help='Disable dynamic axes (fixed batch size)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Disable ONNX optimization')
    parser.add_argument('--no-constant-folding', action='store_true',
                        help='Disable constant folding optimization')
    parser.add_argument('--info', action='store_true',
                        help='Show model checkpoint information')
    parser.add_argument('--no-auto-convert-mixed-precision', action='store_true',
                        help='Disable auto_convert_mixed_precision in ONNX conversion')
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = os.path.realpath(args.model_path)
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    # Show model info if requested
    if args.info:
        get_model_info(model_path)
        return
    
    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.dirname(model_path)
        args.output = os.path.join(output_dir, f"{base_name}.onnx")
    
    output_path = os.path.realpath(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("ONNX Export Configuration:")
    print(f"  Input model: {model_path}")
    print(f"  Output ONNX: {output_path}")
    print(f"  Input size: {args.input_size[0]}x{args.input_size[1]}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Opset version: {args.opset_version}")
    print(f"  Dynamic axes: {not args.no_dynamic_axes}")
    print(f"  Optimize: {not args.no_optimize}")
    print(f"  Constant folding: {not args.no_constant_folding}")
    print(f"  Auto convert mixed precision: {not args.no_auto_convert_mixed_precision}")
    print()
    
    try:
        # Export to ONNX
        export_to_onnx(
            model_path=model_path,
            output_path=output_path,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            opset_version=args.opset_version,
            dynamic_axes=not args.no_dynamic_axes,
            optimize=not args.no_optimize,
            do_constant_folding=not args.no_constant_folding,
            auto_convert_mixed_precision=not args.no_auto_convert_mixed_precision
        )
        
        print(f"\n✓ Export completed successfully!")
        print(f"  ONNX model saved to: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()