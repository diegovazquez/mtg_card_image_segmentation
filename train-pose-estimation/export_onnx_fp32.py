#!/usr/bin/env python3
"""
Alternative ONNX export script that avoids the double free error.
Uses torch.onnx.export directly with memory management.
"""

import os
import sys
import argparse
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

def export_to_onnx_alternative(model_path: str, output_path: str, input_size: int = 640):
    """
    Export YOLO model to ONNX using torch.onnx.export directly.
    
    Args:
        model_path: Path to the YOLO model
        output_path: Output ONNX file path
        input_size: Input image size
    """
    print(f"Loading YOLO model: {model_path}")
    
    # Load the model
    yolo_model = YOLO(model_path)
    model = yolo_model.model
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    print(f"Exporting to ONNX with input shape: {dummy_input.shape}")
    
    # Export arguments
    input_names = ['images']
    output_names = ['output']
    
    # Use torch.onnx.export directly
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX export completed: {output_path}")
    
    # Validate the exported model
    validate_onnx_model(output_path)
    
    return output_path

def validate_onnx_model(onnx_path: str):
    """Validate the exported ONNX model."""
    try:
        # Load and check the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed")
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_shape = ort_session.get_inputs()[0].shape
        print(f"✅ ONNX Runtime loaded successfully")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {ort_session.get_outputs()[0].shape}")
        
        # Test inference
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = ort_session.run(None, {'images': dummy_input})
        print(f"✅ Test inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Alternative ONNX export for YOLO models")
    parser.add_argument('--model', required=True, help='Path to YOLO model')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    print("Alternative ONNX Export Tool")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Input size: {args.input_size}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        export_to_onnx_alternative(args.model, args.output, args.input_size)
        print(f"\n🎉 Export completed successfully!")
        print(f"ONNX model saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()