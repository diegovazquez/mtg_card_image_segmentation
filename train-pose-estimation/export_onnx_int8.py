#!/usr/bin/env python3
"""
Quantize ONNX model to reduce size and improve inference speed.
"""

import os
import argparse
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np

def quantize_onnx_model(input_path: str, output_path: str):
    """
    Quantize ONNX model using dynamic quantization.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
    """
    print(f"Loading ONNX model: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Applying dynamic quantization...")
    
    # Apply dynamic quantization
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,  # Quantize weights to 8-bit unsigned integers
    )
    
    print(f"Quantized model saved: {output_path}")
    
    # Compare file sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    compression_ratio = quantized_size / original_size
    
    print(f"\nFile size comparison:")
    print(f"  Original:  {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Compression: {compression_ratio:.2f}x ({(1-compression_ratio)*100:.1f}% reduction)")
    
    return output_path

def validate_quantized_model(quantized_path: str):
    """Validate the quantized model."""
    try:
        import onnxruntime as ort
        
        print("\nValidating quantized model...")
        
        # Load quantized model
        ort_session = ort.InferenceSession(quantized_path)
        
        # Get input/output info
        input_shape = ort_session.get_inputs()[0].shape
        output_shape = ort_session.get_outputs()[0].shape
        
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shape: {output_shape}")
        
        # Test inference
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = ort_session.run(None, {'images': dummy_input})
        
        print(f"✅ Test inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model")
    parser.add_argument('--input', required=True, help='Input ONNX model path')
    parser.add_argument('--output', required=True, help='Output quantized model path')
    
    args = parser.parse_args()
    
    print("ONNX Model Quantization Tool")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    try:
        quantized_path = quantize_onnx_model(args.input, args.output)
        validate_quantized_model(quantized_path)
        
        print(f"\n🎉 Quantization completed successfully!")
        print(f"Quantized model: {quantized_path}")
        
    except Exception as e:
        print(f"\n❌ Quantization failed: {e}")
        raise

if __name__ == "__main__":
    main()