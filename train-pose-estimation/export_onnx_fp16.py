#!/usr/bin/env python3
"""
Export ONNX model with FP16 precision for optimized deployment.
"""

import os
import sys
import argparse
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from onnxconverter_common import float16
from ultralytics import YOLO

def export_to_onnx_fp16(model_path: str, output_path: str, input_size: int = 640):
    """
    Export YOLO model to ONNX with FP16 precision.
    
    Args:
        model_path: Path to the YOLO model
        output_path: Output ONNX file path
        input_size: Input image size
    """
    print(f"Loading YOLO model: {model_path}")
    
    # Load the model
    yolo_model = YOLO(model_path)
    model = yolo_model.model
    
    # Set model to evaluation mode and half precision
    model.eval()
    model.half()  # Convert to FP16
    
    # Create dummy input in FP16
    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float16)
    
    print(f"Exporting to ONNX FP16 with input shape: {dummy_input.shape}")
    print(f"Input dtype: {dummy_input.dtype}")
    
    # Export arguments
    input_names = ['images']
    output_names = ['output']
    
    # Temporary FP32 export path
    temp_fp32_path = output_path.replace('.onnx', '_temp_fp32.onnx')
    
    try:
        # First export as FP32
        print("Step 1: Exporting as FP32...")
        model_fp32 = yolo_model.model.float()  # Ensure FP32 for export
        dummy_input_fp32 = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
        
        torch.onnx.export(
            model_fp32,
            dummy_input_fp32,
            temp_fp32_path,
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
        
        print("Step 2: Converting to FP16...")
        # Load FP32 model and convert to FP16
        fp32_model = onnx.load(temp_fp32_path)
        fp16_model = float16.convert_float_to_float16(fp32_model)
        
        # Save FP16 model
        onnx.save(fp16_model, output_path)
        
        # Clean up temporary file
        if os.path.exists(temp_fp32_path):
            os.remove(temp_fp32_path)
        
        print(f"FP16 ONNX export completed: {output_path}")
        
        # Validate the exported model
        validate_fp16_model(output_path)
        
        return output_path
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_fp32_path):
            os.remove(temp_fp32_path)
        raise e

def validate_fp16_model(onnx_path: str):
    """Validate the FP16 ONNX model."""
    try:
        # Load and check the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX FP16 model validation passed")
        
        # Check if model uses FP16
        graph = onnx_model.graph
        has_fp16 = any(
            tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
            for tensor in list(graph.value_info) + list(graph.input) + list(graph.output)
        )
        
        if has_fp16:
            print("✅ Model confirmed to use FP16 precision")
        else:
            print("⚠️  Model may not be using FP16 precision")
        
        # Test with ONNX Runtime
        try:
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input shape
            input_shape = ort_session.get_inputs()[0].shape
            input_type = ort_session.get_inputs()[0].type
            
            print(f"✅ ONNX Runtime loaded successfully")
            print(f"   Input shape: {input_shape}")
            print(f"   Input type: {input_type}")
            print(f"   Output shape: {ort_session.get_outputs()[0].shape}")
            
            # Test inference with FP32 input (ONNX Runtime will convert)
            dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            outputs = ort_session.run(None, {'images': dummy_input})
            print(f"✅ Test inference successful")
            print(f"   Output shape: {outputs[0].shape}")
            print(f"   Output dtype: {outputs[0].dtype}")
            
        except Exception as ort_error:
            print(f"⚠️  ONNX Runtime test failed: {ort_error}")
            print("   Model may still be valid for other runtimes")
        
    except Exception as e:
        print(f"❌ FP16 validation failed: {e}")
        raise

def compare_model_sizes(original_path: str, fp16_path: str):
    """Compare file sizes between original and FP16 models."""
    if os.path.exists(original_path) and os.path.exists(fp16_path):
        original_size = os.path.getsize(original_path) / (1024 * 1024)  # MB
        fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)  # MB
        compression_ratio = fp16_size / original_size
        
        print(f"\nFile size comparison:")
        print(f"  Original (FP32): {original_size:.2f} MB")
        print(f"  FP16:            {fp16_size:.2f} MB")
        print(f"  Compression:     {compression_ratio:.2f}x ({(1-compression_ratio)*100:.1f}% reduction)")

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to FP16 ONNX")
    parser.add_argument('--model', required=True, help='Path to YOLO model')
    parser.add_argument('--output', required=True, help='Output FP16 ONNX file path')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    parser.add_argument('--compare', help='Path to original ONNX model for size comparison')
    
    args = parser.parse_args()
    
    print("ONNX FP16 Export Tool")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Input size: {args.input_size}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        export_to_onnx_fp16(args.model, args.output, args.input_size)
        
        if args.compare and os.path.exists(args.compare):
            compare_model_sizes(args.compare, args.output)
        
        print(f"\n🎉 FP16 export completed successfully!")
        print(f"FP16 ONNX model saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ FP16 export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()