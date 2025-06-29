#!/usr/bin/env python3
"""
Comprehensive ONNX inference testing script for MTG corner detection.
Tests different ONNX formats (FP32, FP16, Quantized) with real images and benchmarks.
"""

import os
import sys
import argparse
import time
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ONNXCornerDetector:
    """ONNX-based corner detection inference."""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize ONNX corner detector.
        
        Args:
            model_path: Path to ONNX model
            providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        
        # Set up providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
        
        print(f"Loading ONNX model: {self.model_name}")
        print(f"Providers: {providers}")
        
        # Load model
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model info
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        
        print(f"Input: {self.input_info.name}, shape: {self.input_info.shape}, type: {self.input_info.type}")
        print(f"Output: {self.output_info.name}, shape: {self.output_info.shape}, type: {self.output_info.type}")
        
        # Determine input data type
        self.input_dtype = np.float16 if 'float16' in self.input_info.type else np.float32
        print(f"Input data type: {self.input_dtype}")
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for ONNX inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (640, 640))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        
        # Convert to appropriate data type
        image = image.astype(self.input_dtype)
        
        return image
    
    def postprocess_output(self, output: np.ndarray) -> Dict:
        """
        Postprocess ONNX model output to extract corners.
        
        Args:
            output: Raw model output
            
        Returns:
            Dictionary with detection results
        """
        # Remove batch dimension
        output = output[0]  # Shape: (17, 8400)
        
        # Get class scores (first channel)
        class_scores = output[0]
        
        # Get bounding boxes (channels 1-4)
        bboxes = output[1:5]
        
        # Get keypoints (channels 5-16, 4 points * 3 values each)
        keypoints = output[5:17].reshape(4, 3, -1)  # (4 corners, 3 values, 8400 anchors)
        
        # Find best detection (highest class score)
        best_idx = np.argmax(class_scores)
        best_score = class_scores[best_idx]
        
        # Extract best bounding box (center_x, center_y, width, height)
        best_bbox = bboxes[:, best_idx]
        
        # Extract best keypoints (x, y, visibility for each corner)
        best_keypoints = keypoints[:, :, best_idx]  # (4, 3)
        
        # Convert to corner format
        corners = []
        for i in range(4):
            x, y, visibility = best_keypoints[i]
            corners.append({
                'x': float(x),
                'y': float(y),
                'visibility': float(visibility),
                'corner_name': ['top_left', 'top_right', 'bottom_right', 'bottom_left'][i]
            })
        
        return {
            'confidence': float(best_score),
            'bbox': {
                'center_x': float(best_bbox[0]),
                'center_y': float(best_bbox[1]),
                'width': float(best_bbox[2]),
                'height': float(best_bbox[3])
            },
            'corners': corners,
            'detection_index': int(best_idx)
        }
    
    def predict(self, image_path: str) -> Dict:
        """
        Run inference on image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Detection results with timing info
        """
        # Preprocess
        start_time = time.time()
        input_tensor = self.preprocess_image(image_path)
        preprocess_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        outputs = self.session.run(None, {self.input_info.name: input_tensor})
        inference_time = time.time() - start_time
        
        # Postprocess
        start_time = time.time()
        results = self.postprocess_output(outputs[0])
        postprocess_time = time.time() - start_time
        
        # Add timing info
        results['timing'] = {
            'preprocess': preprocess_time * 1000,  # ms
            'inference': inference_time * 1000,    # ms
            'postprocess': postprocess_time * 1000, # ms
            'total': (preprocess_time + inference_time + postprocess_time) * 1000
        }
        
        return results

class ONNXBenchmark:
    """Benchmark multiple ONNX models."""
    
    def __init__(self, models_dir: str):
        """
        Initialize benchmark with models directory.
        
        Args:
            models_dir: Directory containing ONNX models
        """
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
        
    def discover_models(self) -> Dict[str, str]:
        """Discover ONNX models in directory."""
        models = {}
        
        for file_path in Path(self.models_dir).glob("*.onnx"):
            model_name = file_path.stem
            if 'fp16' in model_name.lower():
                model_type = 'FP16'
            elif 'quantized' in model_name.lower() or 'quant' in model_name.lower():
                model_type = 'Quantized'
            elif 'alternative' in model_name.lower():
                model_type = 'FP32'
            else:
                model_type = 'Unknown'
            
            models[f"{model_type} ({model_name})"] = str(file_path)
        
        return models
    
    def benchmark_model(self, model_path: str, test_images: List[str], num_runs: int = 10) -> Dict:
        """
        Benchmark single model.
        
        Args:
            model_path: Path to ONNX model
            test_images: List of test image paths
            num_runs: Number of runs for timing benchmark
            
        Returns:
            Benchmark results
        """
        try:
            detector = ONNXCornerDetector(model_path)
            
            # Test on images
            image_results = []
            for img_path in test_images[:3]:  # Test on first 3 images
                if os.path.exists(img_path):
                    result = detector.predict(img_path)
                    image_results.append(result)
            
            # Timing benchmark with synthetic data
            timing_results = []
            dummy_image_path = test_images[0] if test_images and os.path.exists(test_images[0]) else None
            
            if dummy_image_path:
                for _ in range(num_runs):
                    result = detector.predict(dummy_image_path)
                    timing_results.append(result['timing'])
            
            # Calculate statistics
            if timing_results:
                timings = {
                    'preprocess': [r['preprocess'] for r in timing_results],
                    'inference': [r['inference'] for r in timing_results],
                    'postprocess': [r['postprocess'] for r in timing_results],
                    'total': [r['total'] for r in timing_results]
                }
                
                timing_stats = {}
                for key, values in timings.items():
                    timing_stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
            else:
                timing_stats = {}
            
            return {
                'model_path': model_path,
                'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'input_dtype': str(detector.input_dtype),
                'image_results': image_results,
                'timing_stats': timing_stats,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'model_path': model_path,
                'model_size_mb': os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0,
                'error': str(e),
                'status': 'failed'
            }
    
    def run_benchmark(self, test_images: List[str], output_dir: str = "benchmark_results"):
        """
        Run comprehensive benchmark.
        
        Args:
            test_images: List of test image paths
            output_dir: Output directory for results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Discover models
        models = self.discover_models()
        print(f"Found {len(models)} models:")
        for name, path in models.items():
            print(f"  {name}: {path}")
        
        # Benchmark each model
        results = {}
        for name, path in models.items():
            print(f"\nBenchmarking {name}...")
            results[name] = self.benchmark_model(path, test_images)
            
            if results[name]['status'] == 'success':
                print(f"✅ Success - Size: {results[name]['model_size_mb']:.1f} MB")
                if results[name]['timing_stats']:
                    mean_inference = results[name]['timing_stats']['inference']['mean']
                    print(f"   Average inference: {mean_inference:.2f} ms")
            else:
                print(f"❌ Failed: {results[name]['error']}")
        
        # Save results
        results_file = os.path.join(output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Generate comparison
        self.generate_comparison_report(results, output_dir)
        
        return results
    
    def generate_comparison_report(self, results: Dict, output_dir: str):
        """Generate comparison report and visualizations."""
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
        
        if not successful_results:
            print("No successful results to compare")
            return
        
        # Create comparison table
        comparison_data = []
        for name, result in successful_results.items():
            if result['timing_stats']:
                comparison_data.append({
                    'Model': name,
                    'Size (MB)': result['model_size_mb'],
                    'Input Type': result['input_dtype'],
                    'Avg Inference (ms)': result['timing_stats']['inference']['mean'],
                    'Std Inference (ms)': result['timing_stats']['inference']['std'],
                    'Total Time (ms)': result['timing_stats']['total']['mean']
                })
        
        # Save comparison table
        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Avg Inference (ms)')
            
            # Save as CSV
            df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
            
            # Print comparison
            print("\n" + "="*80)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*80)
            print(df.to_string(index=False, float_format='%.2f'))
            print("="*80)
            
            # Generate plots
            self.plot_comparison(df, output_dir)
    
    def plot_comparison(self, df, output_dir: str):
        """Generate comparison plots."""
        plt.style.use('seaborn-v0_8')
        
        # Size vs Speed plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['Size (MB)'], df['Avg Inference (ms)'], s=100, alpha=0.7)
        for i, row in df.iterrows():
            plt.annotate(row['Model'].split('(')[0], 
                        (row['Size (MB)'], row['Avg Inference (ms)']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Average Inference Time (ms)')
        plt.title('Model Size vs Inference Speed')
        plt.grid(True, alpha=0.3)
        
        # Performance comparison bar chart
        plt.subplot(1, 2, 2)
        models = df['Model'].str.split('(').str[0]
        plt.bar(range(len(models)), df['Avg Inference (ms)'], alpha=0.7)
        plt.xlabel('Model Type')
        plt.ylabel('Average Inference Time (ms)')
        plt.title('Inference Speed Comparison')
        plt.xticks(range(len(models)), models, rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {os.path.join(output_dir, 'model_comparison.png')}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test ONNX corner detection inference")
    parser.add_argument('--models-dir', required=True, help='Directory containing ONNX models')
    parser.add_argument('--test-images', nargs='+', help='Test image paths')
    parser.add_argument('--test-dir', help='Directory containing test images')
    parser.add_argument('--single-model', help='Test single model only')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--output-dir', default='onnx_test_results', help='Output directory')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of timing runs')
    
    args = parser.parse_args()
    
    # Gather test images
    test_images = []
    if args.test_images:
        test_images.extend(args.test_images)
    if args.test_dir and os.path.exists(args.test_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(list(Path(args.test_dir).glob(ext)))
        test_images = [str(p) for p in test_images[:10]]  # Limit to 10 images
    
    if not test_images:
        print("No test images provided. Creating synthetic test data...")
        # Create a dummy image for testing
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dummy_path = os.path.join(args.output_dir, 'dummy_test.jpg')
        os.makedirs(args.output_dir, exist_ok=True)
        cv2.imwrite(dummy_path, dummy_img)
        test_images = [dummy_path]
    
    print(f"ONNX Corner Detection Inference Test")
    print(f"Models directory: {args.models_dir}")
    print(f"Test images: {len(test_images)} images")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    if args.single_model:
        # Test single model
        print(f"Testing single model: {args.single_model}")
        detector = ONNXCornerDetector(args.single_model)
        
        for img_path in test_images[:3]:
            print(f"\nTesting image: {img_path}")
            result = detector.predict(img_path)
            
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Timing: {result['timing']['total']:.2f} ms")
            print("Corners:")
            for corner in result['corners']:
                print(f"  {corner['corner_name']}: ({corner['x']:.1f}, {corner['y']:.1f}) "
                      f"visibility: {corner['visibility']:.3f}")
    else:
        # Benchmark all models
        benchmark = ONNXBenchmark(args.models_dir)
        results = benchmark.run_benchmark(test_images, args.output_dir)
        
        print(f"\n🎉 Benchmark completed!")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()