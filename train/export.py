"""
Model export script for semantic segmentation.
Exports trained models to various formats including ONNX for deployment.
"""
import os
import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple, Optional

from config import Config
from model import create_model, get_model_size

class ModelExporter:
    """
    Class for exporting trained models to different formats.
    """
    
    def __init__(self, model, device):
        """
        Initialize model exporter.
        
        Args:
            model (nn.Module): Trained model to export
            device: Device the model is on
        """
        self.model = model
        self.device = device
    
    def export_to_onnx(self, output_path: str, input_shape: Tuple[int, int, int, int] = None,
                      opset_version: int = 11, dynamic_axes: bool = False,
                      optimize: bool = True) -> None:
        """
        Export model to ONNX format.
        
        Args:
            output_path (str): Path to save ONNX model
            input_shape (tuple): Input shape (B, C, H, W)
            opset_version (int): ONNX opset version
            dynamic_axes (bool): Use dynamic batch size
            optimize (bool): Apply ONNX optimizations
        """
        if input_shape is None:
            input_shape = (1, 3, Config.INPUT_HEIGHT, Config.INPUT_WIDTH)
        
        print(f"Exporting model to ONNX format...")
        print(f"Input shape: {input_shape}")
        print(f"Output path: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Define dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        
        # Verify ONNX model
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed!")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")
            return
        
        # Optimize ONNX model if requested
        if optimize:
            self._optimize_onnx_model(output_path)
        
        # Get model size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"ONNX model saved successfully!")
        print(f"File size: {file_size:.2f} MB")
        
        # Test inference
        self._test_onnx_inference(output_path, dummy_input)
    
    def _optimize_onnx_model(self, model_path: str) -> None:
        """
        Optimize ONNX model for better performance.
        
        Args:
            model_path (str): Path to ONNX model
        """
        try:
            import onnxoptimizer
            
            print("Optimizing ONNX model...")
            onnx_model = onnx.load(model_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(onnx_model)
            
            # Save optimized model
            optimized_path = model_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            # Replace original with optimized
            os.replace(optimized_path, model_path)
            print("ONNX model optimization completed!")
            
        except ImportError:
            print("onnxoptimizer not available, skipping optimization")
        except Exception as e:
            print(f"ONNX optimization failed: {e}")
    
    def _test_onnx_inference(self, model_path: str, test_input: torch.Tensor) -> None:
        """
        Test ONNX model inference and compare with PyTorch.
        
        Args:
            model_path (str): Path to ONNX model
            test_input (torch.Tensor): Test input tensor
        """
        try:
            print("Testing ONNX inference...")
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = self.model(test_input).cpu().numpy()
            
            # ONNX inference
            ort_session = ort.InferenceSession(model_path)
            onnx_input = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]
            
            # Compare outputs
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            
            print(f"Output comparison:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print("✓ ONNX inference test passed!")
            else:
                print("⚠ ONNX inference test shows significant differences")
                
        except Exception as e:
            print(f"ONNX inference test failed: {e}")
    
    def export_to_torchscript(self, output_path: str, method: str = 'trace',
                             input_shape: Tuple[int, int, int, int] = None) -> None:
        """
        Export model to TorchScript format.
        
        Args:
            output_path (str): Path to save TorchScript model
            method (str): Export method ('trace' or 'script')
            input_shape (tuple): Input shape for tracing
        """
        if input_shape is None:
            input_shape = (1, 3, Config.INPUT_HEIGHT, Config.INPUT_WIDTH)
        
        print(f"Exporting model to TorchScript using {method} method...")
        
        self.model.eval()
        
        if method == 'trace':
            # Trace the model
            dummy_input = torch.randn(input_shape).to(self.device)
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(output_path)
            
        elif method == 'script':
            # Script the model
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(output_path)
            
        else:
            raise ValueError(f"Unknown export method: {method}")
        
        # Get model size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"TorchScript model saved successfully!")
        print(f"File size: {file_size:.2f} MB")
        
        # Test inference
        self._test_torchscript_inference(output_path, input_shape)
    
    def _test_torchscript_inference(self, model_path: str, input_shape: Tuple[int, int, int, int]) -> None:
        """
        Test TorchScript model inference.
        
        Args:
            model_path (str): Path to TorchScript model
            input_shape (tuple): Input shape for testing
        """
        try:
            print("Testing TorchScript inference...")
            
            # Load TorchScript model
            scripted_model = torch.jit.load(model_path, map_location=self.device)
            scripted_model.eval()
            
            # Test inference
            test_input = torch.randn(input_shape).to(self.device)
            
            with torch.no_grad():
                # Original model
                original_output = self.model(test_input)
                # TorchScript model
                scripted_output = scripted_model(test_input)
            
            # Compare outputs
            max_diff = torch.max(torch.abs(original_output - scripted_output)).item()
            mean_diff = torch.mean(torch.abs(original_output - scripted_output)).item()
            
            print(f"Output comparison:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            if max_diff < 1e-5:
                print("✓ TorchScript inference test passed!")
            else:
                print("⚠ TorchScript inference test shows differences")
                
        except Exception as e:
            print(f"TorchScript inference test failed: {e}")
    
    def export_state_dict(self, output_path: str, include_optimizer: bool = False,
                         optimizer=None, epoch: int = 0, metrics: dict = None) -> None:
        """
        Export PyTorch state dict with metadata.
        
        Args:
            output_path (str): Path to save state dict
            include_optimizer (bool): Include optimizer state
            optimizer: Optimizer object
            epoch (int): Current epoch
            metrics (dict): Training metrics
        """
        print("Exporting PyTorch state dict...")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_classes': Config.NUM_CLASSES,
                'input_height': Config.INPUT_HEIGHT,
                'input_width': Config.INPUT_WIDTH,
                'model_name': Config.MODEL_NAME
            },
            'epoch': epoch,
            'metrics': metrics or {}
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, output_path)
        
        # Get model size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"State dict saved successfully!")
        print(f"File size: {file_size:.2f} MB")
    
    def create_deployment_package(self, output_dir: str, model_name: str = "card_segmentation") -> None:
        """
        Create a complete deployment package with all necessary files.
        
        Args:
            output_dir (str): Directory to save deployment package
            model_name (str): Name for the model files
        """
        print("Creating deployment package...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to different formats
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        torchscript_path = os.path.join(output_dir, f"{model_name}.pt")
        statedict_path = os.path.join(output_dir, f"{model_name}_state_dict.pth")
        
        # Export models
        self.export_to_onnx(onnx_path, optimize=True)
        self.export_to_torchscript(torchscript_path, method='trace')
        self.export_state_dict(statedict_path)
        
        # Create model info file
        self._create_model_info_file(output_dir, model_name)
        
        # Create example inference script
        self._create_inference_example(output_dir, model_name)
        
        print(f"Deployment package created in: {output_dir}")
    
    def _create_model_info_file(self, output_dir: str, model_name: str) -> None:
        """Create model information file."""
        info_content = f"""# {model_name.title()} Model Information

## Model Details
- Architecture: {Config.MODEL_NAME}
- Input Size: {Config.INPUT_HEIGHT}x{Config.INPUT_WIDTH} (Height x Width)
- Number of Classes: {Config.NUM_CLASSES}
- Classes: Background (0), Card (1)

## Model Files
- `{model_name}.onnx`: ONNX format for cross-platform deployment
- `{model_name}.pt`: TorchScript format for PyTorch deployment
- `{model_name}_state_dict.pth`: PyTorch state dict for training/fine-tuning

## Input/Output
- Input: RGB image tensor of shape (1, 3, {Config.INPUT_HEIGHT}, {Config.INPUT_WIDTH})
- Input normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Output: Segmentation logits of shape (1, 2, {Config.INPUT_HEIGHT}, {Config.INPUT_WIDTH})

## Usage
See `inference_example.py` for example usage.

## Requirements
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- onnxruntime (for ONNX inference)
- opencv-python
- numpy
- Pillow
"""
        
        info_path = os.path.join(output_dir, "README.md")
        with open(info_path, 'w') as f:
            f.write(info_content)
    
    def _create_inference_example(self, output_dir: str, model_name: str) -> None:
        """Create example inference script."""
        example_content = f'''"""
Example inference script for {model_name} model.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

def preprocess_image(image_path, target_size=({Config.INPUT_HEIGHT}, {Config.INPUT_WIDTH})):
    """
    Preprocess image for model inference.
    
    Args:
        image_path (str): Path to input image
        target_size (tuple): Target image size (H, W)
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def postprocess_output(output):
    """
    Postprocess model output to get segmentation mask.
    
    Args:
        output: Model output tensor
        
    Returns:
        np.ndarray: Binary segmentation mask
    """
    # Apply softmax and get predictions
    probs = F.softmax(output, dim=1)
    pred_mask = torch.argmax(probs, dim=1)
    
    return pred_mask.cpu().numpy()[0]

def inference_pytorch(model_path, image_path):
    """
    Run inference using PyTorch model.
    """
    # Load model
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    mask = postprocess_output(output)
    
    return mask

def inference_onnx(model_path, image_path):
    """
    Run inference using ONNX model.
    """
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    input_array = input_tensor.numpy()
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {{input_name: input_array}})[0]
    
    # Postprocess
    output_tensor = torch.from_numpy(output)
    mask = postprocess_output(output_tensor)
    
    return mask

def save_mask(mask, output_path):
    """Save segmentation mask as image."""
    # Convert to 0-255 range
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on card segmentation model')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output_mask.png', help='Output mask path')
    parser.add_argument('--format', type=str, choices=['pytorch', 'onnx'], default='onnx',
                       help='Model format')
    
    args = parser.parse_args()
    
    # Run inference
    if args.format == 'pytorch':
        mask = inference_pytorch(args.model, args.image)
    else:
        mask = inference_onnx(args.model, args.image)
    
    # Save result
    save_mask(mask, args.output)
    print(f"Segmentation mask saved to: {{args.output}}")
'''
        
        example_path = os.path.join(output_dir, "inference_example.py")
        with open(example_path, 'w') as f:
            f.write(example_content)

def main(args):
    """Main export function."""
    
    print("=" * 50)
    print("MODEL EXPORT")
    print("=" * 50)
    
    # Set device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = create_model(
        num_classes=Config.NUM_CLASSES,
        pretrained=False  # We're loading trained weights
    ).to(device)
    
    # Load model weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {args.model_path}")
    
    # Print model info
    original_size = get_model_size(model)
    print(f"Model size: {original_size:.2f} MB")
    
    # Create exporter
    exporter = ModelExporter(model, device)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if args.create_package:
        # Create complete deployment package
        exporter.create_deployment_package(output_dir, args.model_name)
    else:
        # Export individual formats
        if args.export_onnx:
            onnx_path = os.path.join(output_dir, f"{args.model_name}.onnx")
            exporter.export_to_onnx(
                onnx_path,
                input_shape=(1, 3, Config.INPUT_HEIGHT, Config.INPUT_WIDTH),
                dynamic_axes=args.dynamic_batch,
                optimize=args.optimize
            )
        
        if args.export_torchscript:
            ts_path = os.path.join(output_dir, f"{args.model_name}.pt")
            exporter.export_to_torchscript(ts_path, method=args.torchscript_method)
        
        if args.export_state_dict:
            sd_path = os.path.join(output_dir, f"{args.model_name}_state_dict.pth")
            exporter.export_state_dict(
                sd_path,
                include_optimizer=False,
                epoch=checkpoint.get('epoch', 0),
                metrics=checkpoint.get('metrics', {})
            )
    
    print("\nExport completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export semantic segmentation model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='train/exported_models',
                        help='Output directory for exported models')
    parser.add_argument('--model-name', type=str, default='card_segmentation',
                        help='Name for exported model files')
    
    # Export format options
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export to ONNX format')
    parser.add_argument('--export-torchscript', action='store_true',
                        help='Export to TorchScript format')
    parser.add_argument('--export-state-dict', action='store_true',
                        help='Export PyTorch state dict')
    parser.add_argument('--create-package', action='store_true',
                        help='Create complete deployment package (all formats)')
    
    # ONNX options
    parser.add_argument('--dynamic-batch', action='store_true',
                        help='Use dynamic batch size for ONNX export')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='Optimize ONNX model')
    
    # TorchScript options
    parser.add_argument('--torchscript-method', type=str, choices=['trace', 'script'],
                        default='trace', help='TorchScript export method')
    
    args = parser.parse_args()
    
    # Set default exports if none specified
    if not any([args.export_onnx, args.export_torchscript, args.export_state_dict, args.create_package]):
        args.create_package = True
    
    try:
        main(args)
    except Exception as e:
        print(f"Export failed with error: {e}")
        raise
