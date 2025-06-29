# Developer Documentation

## Project Overview

Card Segmentation is a real-time card detection application that uses semantic segmentation with ONNX Runtime Web for in-browser AI inference. The application is built with vanilla JavaScript using modern web APIs and includes a comprehensive build system with Rollup.

## Architecture

### Core Modules

The application follows a modular architecture with the following main components:

- **`CardSegmentationApp`** (`src/app.js`) - Main application controller that coordinates all components
- **`CameraManager`** (`src/camera-manager.js`) - Handles camera enumeration, selection, and streaming
- **`ModelInference`** (`src/model-inference.js`) - ONNX Runtime Web integration for AI inference
- **`ImageUtils`** (`src/image-utils.js`) - Image preprocessing, rotation, and overlay rendering

### Application Flow

1. **Initialization**: DOM setup, event listeners, camera enumeration
2. **Camera Management**: Device detection, stream initialization, resolution handling
3. **Model Loading**: ONNX model loading with provider selection (WebGPU → WebGL → WASM)
4. **Real-time Inference**: Continuous video processing and overlay rendering
5. **Performance Monitoring**: FPS tracking and inference statistics

## Build System

The project uses Rollup for bundling with the following features:

### Build Configuration (`rollup.config.js`)

- **Input**: `src/app.js` (main entry point)
- **Output**: 
  - `dist/bundle.js` - Development bundle with source maps
  - `dist/bundle.min.js` - Minified production bundle
- **Plugins**:
  - `@rollup/plugin-node-resolve` - Module resolution
  - `@rollup/plugin-commonjs` - CommonJS support
  - `@rollup/plugin-terser` - Minification
  - `rollup-plugin-copy` - Asset copying
  - `rollup-plugin-gzip` - Compression

### Build Scripts

```bash
# Standard build
npm run build

# Watch mode for development
npm run build:watch

# Production build with NODE_ENV=production
npm run build:prod

# Clean dist folder
npm run clean
```

### Build Output

- **bundle.js** (~18KB) - Development version with comments
- **bundle.min.js** (~8.4KB) - Minified version (53% size reduction)
- **Gzipped versions** (87% compression ratio)
- **Source maps** for debugging
- **Assets** copied to dist/ (HTML, CSS, SVG, ONNX models)

## Development Setup

### Prerequisites

- Node.js (v14+)
- Modern browser with WebRTC support
- Camera device for testing

### Installation

```bash
# Clone repository
git clone <repository-url>
cd mtg_scanner/demo

# Install dependencies
npm install

# Build the project
npm run build

# Serve the application
# Use any static file server, e.g.:
npx http-server dist
# or
python -m http.server 8000 -d dist
```

### Development Workflow

1. **Development Mode**:
   ```bash
   npm run build:watch
   ```
   This will rebuild automatically when source files change.

2. **Testing Changes**:
   - Serve the `dist/` folder with any static file server
   - Open in browser and test camera functionality
   - Check browser console for errors

3. **Production Build**:
   ```bash
   npm run build:prod
   ```

## Code Structure

### Main Application (`src/app.js`)

```javascript
class CardSegmentationApp {
    constructor() {
        // Core modules initialization
        this.cameraManager = new CameraManager();
        this.modelInference = new ModelInference();
        
        // Application state management
        this.state = {
            isInitialized: false,
            isModelLoaded: false,
            isCameraActive: false,
            isInferenceRunning: false,
            currentError: null
        };
    }
    
    // Key methods:
    // - init(): Application initialization
    // - setupDOMElements(): DOM element binding
    // - setupEventListeners(): Event handler setup
    // - runInferenceLoop(): Main inference loop
}
```

### Camera Management (`src/camera-manager.js`)

Handles all camera-related operations:
- Device enumeration and filtering
- Stream initialization with constraints
- Resolution handling (480x640, 640x480)
- Automatic rotation detection
- Error handling and recovery

### Model Inference (`src/model-inference.js`)

ONNX Runtime Web integration:
- Model loading with provider fallback
- GPU acceleration (WebGPU → WebGL → WASM)
- Input preprocessing (normalization, tensor conversion)
- Output processing (segmentation mask)
- Performance optimization

### Image Processing (`src/image-utils.js`)

Utilities for image manipulation:
- Canvas operations
- Image rotation and resizing  
- Overlay rendering
- Color space conversions
- Performance optimizations

## AI Model Details

### Model Architecture
- **Type**: LRASPP (Lite Reduced Atrous Spatial Pyramid Pooling)
- **Backbone**: MobileNetV3 Large
- **Input**: 480x640 RGB image
- **Output**: 2-class segmentation (Background, Card)
- **Normalization**: ImageNet statistics

### Model Files
- **`card_segmentation.onnx`** - Full precision model
- **`card_segmentation_fp16.onnx`** - Half precision model (smaller, faster)

### Provider Selection Strategy
1. **WebGPU** - Best performance, future-proof
2. **WebGL** - Wide compatibility, 2-5x speedup
3. **WebAssembly** - Universal fallback, CPU-based

## Performance Optimization

### Build Optimizations
- **Tree shaking** via Rollup
- **Minification** with Terser
- **Gzip compression** for network transfer
- **Source maps** for debugging

### Runtime Optimizations
- **GPU acceleration** with automatic fallback
- **Efficient canvas operations**
- **Memory management** for tensors
- **Optimized inference loop** with requestAnimationFrame

### Performance Monitoring
- Real-time FPS counter
- Inference time tracking
- Memory usage monitoring
- GPU provider status

## Browser Compatibility

### Minimum Requirements
- **WebRTC** support for camera access
- **WebAssembly** support (universal in modern browsers)
- **Canvas 2D** API support
- **ES6+** JavaScript features

### Tested Browsers
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

### GPU Acceleration Support
- **WebGPU**: Chrome 113+, Edge 113+
- **WebGL**: All modern browsers
- **WebAssembly**: Universal fallback

## Debugging and Troubleshooting

### Development Tools
- Browser DevTools console for logs
- Network tab for model loading issues
- Performance tab for bottlenecks
- Source maps for debugging minified code

### Common Issues

**Build Failures**:
- Check Node.js version compatibility
- Verify all dependencies installed
- Clear node_modules and reinstall if needed

**Camera Issues**:
- Ensure HTTPS or localhost for camera permissions
- Check camera permissions in browser settings
- Verify camera not in use by other applications

**Model Loading Issues**:
- Check network connectivity
- Verify ONNX files in models/ directory
- Clear browser cache
- Check CORS headers if serving from different domain

**Performance Issues**:
- Monitor GPU provider fallbacks
- Check browser compatibility
- Close other tabs/applications
- Ensure adequate system resources

### Debug Mode

Enable verbose logging by setting localStorage:
```javascript
localStorage.setItem('debug', 'true');
```

## Deployment

### Static Hosting
The built application in `dist/` folder can be deployed to any static hosting service:
- GitHub Pages
- Netlify
- Vercel
- AWS S3
- Any web server

### Requirements
- **HTTPS** required for camera access (except localhost)
- **CORS headers** if models served from different domain
- **Gzip encoding** supported for optimal performance

### Production Checklist
- [ ] Run `npm run build:prod`
- [ ] Test all functionality in target browsers
- [ ] Verify camera permissions work
- [ ] Check model loading on target domain
- [ ] Test GPU acceleration fallbacks
- [ ] Verify gzip compression enabled
- [ ] Monitor console for errors

## Contributing

### Code Style
- Use ES6+ modern JavaScript
- Follow existing naming conventions
- Add JSDoc comments for public methods
- Maintain modular architecture
- Handle errors gracefully

### Testing
- Test on multiple browsers
- Verify camera functionality
- Check model loading and inference
- Test error handling scenarios
- Validate performance metrics

### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Run build and verify output
4. Test in multiple browsers
5. Submit PR with description

## License

MIT License - see project root for details.