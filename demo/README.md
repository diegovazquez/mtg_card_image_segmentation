---
title: Card Segmentation
emoji: 🎴
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
license: mit
---

# Card Segmentation - Real-time Detection

A real-time card segmentation application that uses semantic segmentation to detect and highlight cards in camera feeds. Built with ONNX Runtime Web for in-browser AI inference.

## Features

- **Real-time Detection**: Continuous card segmentation on live camera feed
- **Browser-based AI**: All processing happens in your browser using ONNX Runtime Web
- **Multi-camera Support**: Automatic camera detection and selection
- **Adaptive Resolution**: Supports both 480x640 and 640x480 resolutions with automatic rotation
- **Visual Feedback**: Transparent cyan overlay showing detected card regions
- **Performance Monitoring**: Real-time FPS and inference statistics

## How It Works

1. **Camera Selection**: Choose from available cameras on your device
2. **Video Processing**: Live video feed with automatic resolution handling
3. **AI Inference**: Real-time semantic segmentation using a lightweight MobileNetV3 model
4. **Visual Overlay**: Detected cards are highlighted with a transparent cyan mask

## Technical Details

- **Model**: LRASPP MobileNetV3 Large architecture
- **Input Size**: 480x640 pixels with ImageNet normalization
- **Output**: 2-class segmentation (Background, Card)
- **Runtime**: ONNX Runtime Web with GPU acceleration support
- **Performance**: Optimized for real-time inference with automatic provider selection

### GPU Acceleration

The demo automatically detects and uses the best available acceleration:

- **WebGPU**: Next-generation GPU acceleration (Chrome/Edge experimental)
- **WebGL**: Widely supported GPU acceleration (2-5x performance boost when compatible)
- **WebAssembly**: CPU fallback ensuring universal compatibility

**Note on GPU Compatibility**: Some models may use operators not supported by GPU providers (like 'HardSigmoid' in WebGL). In such cases, the demo automatically falls back to CPU processing while clearly indicating the reason in the acceleration status.

The active acceleration method is displayed in the interface, showing which provider is being used, GPU information when available, and the reason for any fallbacks.

## Usage

1. Grant camera permissions when prompted
2. Select your preferred camera from the dropdown
3. Click "Start Camera" to begin video feed
4. Click "Start Detection" to enable real-time card segmentation
5. Point your camera at cards to see the segmentation overlay

## Browser Requirements

- Modern browser with WebRTC support (Chrome, Firefox, Safari, Edge)
- Camera access permissions
- WebAssembly support (available in all modern browsers)

## Privacy

- All processing happens locally in your browser
- No data is sent to external servers
- Camera access is only used for real-time processing

## Performance Tips

- Use good lighting for better detection accuracy
- Keep cards clearly visible and unobstructed
- Horizontal cameras will be automatically rotated for optimal viewing
- Close other browser tabs for better performance

## Technical Architecture

The application consists of several modular components:

- **Camera Manager**: Handles camera enumeration, selection, and streaming
- **Model Inference**: ONNX Runtime Web integration for AI inference
- **Image Utils**: Image preprocessing, rotation, and overlay rendering
- **Main App**: Coordinates all components and manages application state

## Model Information

The segmentation model was trained specifically for card detection with the following specifications:

- **Architecture**: LRASPP (Lite Reduced Atrous Spatial Pyramid Pooling) with MobileNetV3 Large backbone
- **Training Data**: Synthetic dataset of various card types and backgrounds
- **Input Preprocessing**: RGB normalization with ImageNet statistics
- **Output**: Pixel-wise binary classification (card vs background)

## Development

The application is built with vanilla JavaScript and modern web APIs:

- **No Build Process**: Direct deployment of HTML, CSS, and JavaScript
- **Modular Design**: Separate modules for different functionality
- **Error Handling**: Comprehensive error handling and user feedback
- **Responsive Design**: Works on both desktop and mobile devices

## Troubleshooting

**Camera not detected:**
- Ensure camera permissions are granted
- Try refreshing the page
- Check if camera is being used by another application

**Low performance:**
- Close other browser tabs and applications
- Use a device with better hardware acceleration
- Ensure good lighting conditions

**Model not loading:**
- Check internet connection for initial model download
- Clear browser cache and reload
- Try a different browser

---

Built with ❤️ using ONNX Runtime Web and modern web technologies.
