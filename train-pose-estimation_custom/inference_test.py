#!/usr/bin/env python3
"""
Inference test script for corner detection model
Supports both PyTorch (.pth) and ONNX (.onnx) models
"""

import os
import sys
import argparse
import time
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import local modules
from model import create_model, load_model


class CornerDetectionInference:
    """
    Corner detection inference class supporting both PyTorch and ONNX models
    """
    
    def __init__(self, 
                 model_path: str, 
                 input_size: Tuple[int, int] = (640, 480),
                 heatmap_size: Tuple[int, int] = (160, 120),
                 device: str = 'auto'):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to model file (.pth or .onnx)
            input_size: Input image size (width, height)
            heatmap_size: Heatmap output size (width, height)
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = os.path.realpath(model_path)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.model_type = self._detect_model_type()
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = None
        self.onnx_session = None
        self._load_model()
        
        print(f"Loaded {self.model_type} model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Input size: {self.input_size}")
        print(f"Heatmap size: {self.heatmap_size}")
    
    def _detect_model_type(self) -> str:
        """Detect model type based on file extension"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if self.model_path.endswith('.pth'):
            return 'pytorch'
        elif self.model_path.endswith('.onnx'):
            return 'onnx'
        else:
            raise ValueError(f"Unsupported model format. Use .pth or .onnx")
    
    def _load_model(self):
        """Load the model based on type"""
        if self.model_type == 'pytorch':
            self._load_pytorch_model()
        elif self.model_type == 'onnx':
            self._load_onnx_model()
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        print("Loading PyTorch model...")
        self.model = load_model(self.model_path, num_keypoints=4, heatmap_size=self.heatmap_size)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            print("Loading ONNX model...")
            
            # Set providers based on device
            providers = ['CPUExecutionProvider']
            if self.device.type == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Try loading with different session options for FP16 models
            session_options = ort.SessionOptions()
            
            # First try with optimizations disabled (for mixed precision models)
            try:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                self.onnx_session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=providers)
            except Exception as e:
                if "float16" in str(e) and "float" in str(e):
                    print("Warning: FP16 model detected with type conflicts. Trying fallback options...")
                    # Try with basic optimizations
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    try:
                        self.onnx_session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=providers)
                    except Exception:
                        # Last resort: try without session options
                        try:
                            self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
                        except Exception:
                            # Suggest alternative model
                            fp32_model = self.model_path.replace('_fp16.onnx', '.onnx')
                            if os.path.exists(fp32_model):
                                raise RuntimeError(f"FP16 model incompatible with this ONNX Runtime version. "
                                                 f"Use the FP32 model instead: {fp32_model}")
                            else:
                                raise RuntimeError(f"FP16 model incompatible with this ONNX Runtime version. "
                                                 f"Original error: {str(e)}")
                else:
                    raise e
            
            # Get input/output info
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            
            print(f"ONNX input: {input_info.name} {input_info.shape} {input_info.type}")
            print(f"ONNX output: {output_info.name} {output_info.shape} {output_info.type}")
            
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (preprocessed_tensor, original_image, original_size)
        """
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image with OpenCV
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
        
        # Resize image
        resized_image = cv2.resize(original_image, self.input_size)
        
        # Normalize to [0, 1] and convert to tensor
        image_tensor = torch.from_numpy(resized_image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        return image_tensor, original_image, original_size
    
    def inference(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Run inference on preprocessed image
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Heatmap output as numpy array
        """
        if self.model_type == 'pytorch':
            return self._inference_pytorch(image_tensor)
        elif self.model_type == 'onnx':
            return self._inference_onnx(image_tensor)
    
    def _inference_pytorch(self, image_tensor: torch.Tensor) -> np.ndarray:
        """PyTorch inference"""
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            heatmaps = self.model(image_tensor)
            inference_time = time.time() - start_time
            
            print(f"PyTorch inference time: {inference_time:.4f}s")
            
            return heatmaps.cpu().numpy()
    
    def _inference_onnx(self, image_tensor: torch.Tensor) -> np.ndarray:
        """ONNX inference"""
        input_name = self.onnx_session.get_inputs()[0].name
        input_numpy = image_tensor.numpy()
        
        # Check if model expects FP16 input
        input_type = self.onnx_session.get_inputs()[0].type
        if 'float16' in input_type and input_numpy.dtype != np.float16:
            input_numpy = input_numpy.astype(np.float16)
        elif 'float16' not in input_type and input_numpy.dtype == np.float16:
            input_numpy = input_numpy.astype(np.float32)
        
        start_time = time.time()
        outputs = self.onnx_session.run(None, {input_name: input_numpy})
        inference_time = time.time() - start_time
        
        print(f"ONNX inference time: {inference_time:.4f}s")
        
        return outputs[0]
    
    def extract_keypoints(self, heatmaps: np.ndarray, threshold: float = 0.1) -> List[Tuple[float, float]]:
        """
        Extract keypoints from heatmaps
        
        Args:
            heatmaps: Heatmap array (B, N, H, W)
            threshold: Minimum confidence threshold
            
        Returns:
            List of (x, y) coordinates in heatmap space
        """
        batch_size, num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
        keypoints = []
        
        # Process first batch
        for i in range(num_keypoints):
            heatmap = heatmaps[0, i]  # (H, W)
            
            # Find peak
            max_val = np.max(heatmap)
            if max_val < threshold:
                keypoints.append((0, 0))  # No detection
                continue
            
            # Find peak location
            peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            peak_y, peak_x = peak_idx
            
            # Convert to coordinates (0-1 range)
            x = peak_x / heatmap_w
            y = peak_y / heatmap_h
            
            keypoints.append((x, y))
        
        return keypoints
    
    def scale_keypoints(self, keypoints: List[Tuple[float, float]], 
                       original_size: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        Scale keypoints from heatmap space to original image space
        
        Args:
            keypoints: List of (x, y) coordinates in heatmap space (0-1 range)
            original_size: Original image size (width, height)
            
        Returns:
            List of (x, y) coordinates in original image space
        """
        original_w, original_h = original_size
        scaled_keypoints = []
        
        for x, y in keypoints:
            scaled_x = x * original_w
            scaled_y = y * original_h
            scaled_keypoints.append((scaled_x, scaled_y))
        
        return scaled_keypoints
    
    def visualize_results(self, image: np.ndarray, keypoints: List[Tuple[float, float]], 
                         output_path: str, show_heatmap: bool = True, heatmaps: Optional[np.ndarray] = None):
        """
        Visualize detection results
        
        Args:
            image: Original image
            keypoints: List of (x, y) coordinates
            output_path: Path to save output image
            show_heatmap: Whether to show heatmap overlay
            heatmaps: Heatmap array for visualization
        """
        fig, axes = plt.subplots(1, 2 if show_heatmap and heatmaps is not None else 1, 
                                figsize=(15, 8))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot original image with keypoints
        axes[0].imshow(image)
        axes[0].set_title('Corner Detection Results')
        axes[0].axis('off')
        
        # Define colors for corners
        colors = ['red', 'green', 'blue', 'yellow']
        labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        
        # Plot keypoints
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # Valid detection
                axes[0].plot(x, y, 'o', color=colors[i % len(colors)], 
                           markersize=10, markeredgewidth=2, markeredgecolor='white')
                axes[0].annotate(labels[i % len(labels)], (x, y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='white', weight='bold')
        
        # Draw bounding box if we have 4 valid corners
        valid_points = [(x, y) for x, y in keypoints if x > 0 and y > 0]
        if len(valid_points) == 4:
            # Sort points to create proper rectangle
            # This is a simple sorting, might need adjustment based on your corner order
            points = np.array(valid_points)
            rect = patches.Polygon(points, linewidth=2, edgecolor='cyan', 
                                 facecolor='none', linestyle='--')
            axes[0].add_patch(rect)
        
        # Plot heatmap if requested
        if show_heatmap and heatmaps is not None and len(axes) > 1:
            # Combine all heatmaps
            combined_heatmap = np.max(heatmaps[0], axis=0)
            
            # Resize heatmap to match image size
            h, w = image.shape[:2]
            heatmap_resized = cv2.resize(combined_heatmap, (w, h))
            
            # Create overlay
            overlay = axes[1].imshow(image, alpha=0.7)
            heatmap_overlay = axes[1].imshow(heatmap_resized, alpha=0.5, cmap='jet')
            axes[1].set_title('Heatmap Overlay')
            axes[1].axis('off')
            
            # Add colorbar
            plt.colorbar(heatmap_overlay, ax=axes[1])
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=colors[i], markersize=10, 
                                    label=labels[i]) for i in range(len(labels))]
        axes[0].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results saved to: {output_path}")
    
    def run_inference(self, image_path: str, output_path: str, 
                     threshold: float = 0.1, show_heatmap: bool = True) -> List[Tuple[float, float]]:
        """
        Run complete inference pipeline
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            threshold: Detection threshold
            show_heatmap: Whether to show heatmap overlay
            
        Returns:
            List of detected keypoints in original image coordinates
        """
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        
        # Run inference
        heatmaps = self.inference(image_tensor)
        
        # Extract keypoints
        keypoints_normalized = self.extract_keypoints(heatmaps, threshold)
        keypoints_scaled = self.scale_keypoints(keypoints_normalized, original_size)
        
        # Print results
        print(f"Detected {len([kp for kp in keypoints_scaled if kp[0] > 0 and kp[1] > 0])} corners:")
        for i, (x, y) in enumerate(keypoints_scaled):
            if x > 0 and y > 0:
                print(f"  Corner {i+1}: ({x:.1f}, {y:.1f})")
        
        # Visualize results
        self.visualize_results(original_image, keypoints_scaled, output_path, 
                             show_heatmap, heatmaps if show_heatmap else None)
        
        return keypoints_scaled


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test corner detection model inference')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('model_path', type=str, help='Path to model file (.pth or .onnx)')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 480],
                        help='Input image size (width height)')
    parser.add_argument('--heatmap-size', type=int, nargs=2, default=[160, 120],
                        help='Heatmap output size (width height)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Detection threshold')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--no-heatmap', action='store_true',
                        help='Disable heatmap visualization')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_dir = os.path.dirname(args.image_path)
        args.output = os.path.join(output_dir, f"{base_name}_detection_result.png")
    
    try:
        # Create inference engine
        inference_engine = CornerDetectionInference(
            model_path=args.model_path,
            input_size=tuple(args.input_size),
            heatmap_size=tuple(args.heatmap_size),
            device=args.device
        )
        
        # Run inference
        keypoints = inference_engine.run_inference(
            image_path=args.image_path,
            output_path=args.output,
            threshold=args.threshold,
            show_heatmap=not args.no_heatmap
        )
        
        print(f"\n✓ Inference completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()