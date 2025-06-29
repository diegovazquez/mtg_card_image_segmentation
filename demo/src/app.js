/**
 * Main application module
 * Coordinates camera, model inference, and UI interactions
 */

import CameraManager from './camera-manager.js';
import ModelInference from './model-inference.js';
import ImageUtils from './image-utils.js';

class CardSegmentationApp {
    constructor() {
        // Core modules
        this.cameraManager = new CameraManager();
        this.modelInference = new ModelInference();
        
        // DOM elements
        this.elements = {};
        
        // Application state
        this.state = {
            isInitialized: false,
            isModelLoaded: false,
            isCameraActive: false,
            isInferenceRunning: false,
            currentError: null
        };
        
        // Animation frame ID for inference loop
        this.inferenceLoopId = null;
        
        // Performance tracking
        this.performanceStats = {
            frameCount: 0,
            lastFpsUpdate: 0,
            fps: 0
        };
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            console.log('Initializing Card Segmentation App...');
            
            // Get DOM elements
            this.setupDOMElements();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Show loading indicator
            this.showLoading('Initializing cameras...');
            
            // Initialize camera manager
            await this.cameraManager.initialize();
            this.populateCameraDropdown();
            
            // Initialize model
            this.showLoading('Loading AI model...');
            await this.modelInference.initialize();
            
            // Update acceleration info display
            this.updateAccelerationInfo();
            
            // Hide loading, show camera selection
            this.hideLoading();
            this.showCameraSelection();
            
            this.state.isInitialized = true;
            this.state.isModelLoaded = true;
            
            console.log('Application initialized successfully');
            
        } catch (error) {
            console.error('Application initialization failed:', error);
            this.showError('Failed to initialize application', error.message);
        }
    }

    /**
     * Set up DOM element references
     */
    setupDOMElements() {
        this.elements = {
            // Main containers
            cameraSelection: document.getElementById('cameraSelection'),
            videoContainer: document.getElementById('videoContainer'),
            loadingIndicator: document.getElementById('loadingIndicator'),
            errorDisplay: document.getElementById('errorDisplay'),
            
            // Camera selection
            cameraSelect: document.getElementById('cameraSelect'),
            startCamera: document.getElementById('startCamera'),
            
            // Video and controls
            videoElement: document.getElementById('videoElement'),
            overlayCanvas: document.getElementById('overlayCanvas'),
            toggleInference: document.getElementById('toggleInference'),
            switchCamera: document.getElementById('switchCamera'),
            
            // Info displays
            resolutionInfo: document.getElementById('resolutionInfo'),
            fpsInfo: document.getElementById('fpsInfo'),
            inferenceStatus: document.getElementById('inferenceStatus'),
            accelerationInfo: document.getElementById('accelerationInfo'),
            
            // Loading and error
            loadingText: document.getElementById('loadingText'),
            errorMessage: document.getElementById('errorMessage'),
            retryButton: document.getElementById('retryButton')
        };
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Camera selection
        this.elements.cameraSelect.addEventListener('change', (e) => {
            this.elements.startCamera.disabled = !e.target.value;
        });

        this.elements.startCamera.addEventListener('click', () => {
            this.startCamera();
        });

        // Video controls
        this.elements.toggleInference.addEventListener('click', () => {
            this.toggleInference();
        });

        this.elements.switchCamera.addEventListener('click', () => {
            this.showCameraSelection();
        });

        // Error handling
        this.elements.retryButton.addEventListener('click', () => {
            this.hideError();
            this.init();
        });

        // Handle video loaded
        this.elements.videoElement.addEventListener('loadedmetadata', () => {
            this.setupOverlayCanvas();
        });

        // Handle camera device changes
        this.cameraManager.onDeviceChange(() => {
            this.populateCameraDropdown();
        });

        // Handle page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    /**
     * Populate camera selection dropdown
     */
    populateCameraDropdown() {
        this.cameraManager.populateCameraSelect(this.elements.cameraSelect);
        this.elements.startCamera.disabled = this.elements.cameraSelect.value === '';
    }

    /**
     * Start camera with selected device
     */
    async startCamera() {
        try {
            const selectedCameraId = this.elements.cameraSelect.value;
            if (!selectedCameraId) {
                throw new Error('No camera selected');
            }

            this.showLoading('Starting camera...');

            // Start camera stream
            const streamInfo = await this.cameraManager.startCamera(
                this.elements.videoElement, 
                selectedCameraId
            );

            // Update UI
            this.updateResolutionInfo(streamInfo);
            this.setupOverlayCanvas();
            
            // Show video container
            this.hideLoading();
            this.hideCameraSelection();
            this.showVideoContainer();
            
            this.state.isCameraActive = true;
            
            console.log('Camera started successfully:', streamInfo);
            
        } catch (error) {
            console.error('Failed to start camera:', error);
            this.showError('Failed to start camera', error.message);
        }
    }

    /**
     * Set up overlay canvas to match video dimensions
     */
    setupOverlayCanvas() {
        const video = this.elements.videoElement;
        const canvas = this.elements.overlayCanvas;
        
        if (video.videoWidth && video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Apply same rotation class as video if needed
            if (video.classList.contains('rotate-90ccw')) {
                canvas.classList.add('rotate-90ccw');
            } else {
                canvas.classList.remove('rotate-90ccw');
            }
        }
    }

    /**
     * Toggle inference on/off
     */
    toggleInference() {
        if (this.state.isInferenceRunning) {
            this.stopInference();
        } else {
            this.startInference();
        }
    }

    /**
     * Start continuous inference
     */
    startInference() {
        if (!this.state.isModelLoaded || !this.state.isCameraActive) {
            this.showError('Cannot start inference', 'Camera or model not ready');
            return;
        }

        this.state.isInferenceRunning = true;
        this.elements.toggleInference.textContent = 'Stop Detection';
        this.elements.toggleInference.className = 'btn btn-danger';
        this.updateInferenceStatus('Detection: Running');
        
        // Reset performance stats
        this.performanceStats.frameCount = 0;
        this.performanceStats.lastFpsUpdate = performance.now();
        
        // Start inference loop
        this.runInferenceLoop();
        
        console.log('Inference started');
    }

    /**
     * Stop continuous inference
     */
    stopInference() {
        this.state.isInferenceRunning = false;
        this.elements.toggleInference.textContent = 'Start Detection';
        this.elements.toggleInference.className = 'btn btn-success';
        this.updateInferenceStatus('Detection: Off');
        
        // Cancel animation frame
        if (this.inferenceLoopId) {
            cancelAnimationFrame(this.inferenceLoopId);
            this.inferenceLoopId = null;
        }
        
        // Clear overlay
        const ctx = this.elements.overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.elements.overlayCanvas.width, this.elements.overlayCanvas.height);
        
        console.log('Inference stopped');
    }

    /**
     * Main inference loop
     */
    async runInferenceLoop() {
        if (!this.state.isInferenceRunning) {
            return;
        }

        try {
            // Process video frame
            const result = await this.modelInference.processVideoFrame(this.elements.videoElement);
            
            if (result && result.mask) {
                // Draw segmentation overlay
                ImageUtils.drawSegmentationOverlay(
                    result.mask,
                    this.modelInference.inputWidth,
                    this.modelInference.inputHeight,
                    this.elements.overlayCanvas,
                    result.shouldRotateBack
                );
                
                // Update performance stats
                this.updatePerformanceStats(result.stats);
            }
            
        } catch (error) {
            console.error('Inference loop error:', error);
            // Don't stop inference for individual frame errors
        }

        // Schedule next frame
        if (this.state.isInferenceRunning) {
            this.inferenceLoopId = requestAnimationFrame(() => this.runInferenceLoop());
        }
    }

    /**
     * Update performance statistics display
     */
    updatePerformanceStats(inferenceStats) {
        this.performanceStats.frameCount++;
        const now = performance.now();
        
        // Update FPS every second
        if (now - this.performanceStats.lastFpsUpdate >= 1000) {
            this.performanceStats.fps = this.performanceStats.frameCount;
            this.performanceStats.frameCount = 0;
            this.performanceStats.lastFpsUpdate = now;
            
            // Update UI
            this.elements.fpsInfo.textContent = `FPS: ${this.performanceStats.fps}`;
        }
    }

    /**
     * Update resolution information display
     */
    updateResolutionInfo(streamInfo) {
        const resText = `Resolution: ${streamInfo.width}x${streamInfo.height}${streamInfo.rotated ? ' (rotated)' : ''}`;
        this.elements.resolutionInfo.textContent = resText;
    }

    /**
     * Update inference status display
     */
    updateInferenceStatus(status) {
        this.elements.inferenceStatus.textContent = status;
        
        // Update status styling
        this.elements.inferenceStatus.className = 'status-inactive';
        if (status.includes('Running')) {
            this.elements.inferenceStatus.className = 'status-active';
        } else if (status.includes('Processing')) {
            this.elements.inferenceStatus.className = 'status-processing';
        }
    }

    /**
     * Update acceleration information display
     */
    updateAccelerationInfo() {
        if (!this.elements.accelerationInfo) return;
        
        const accelerationInfo = this.modelInference.getAccelerationInfo();
        
        // Create acceleration status text
        let accelerationText = `Acceleration: ${accelerationInfo.activeProvider.toUpperCase()}`;
        
        // Add GPU info if available
        if (accelerationInfo.gpuInfo && accelerationInfo.activeProvider !== 'wasm') {
            const gpu = accelerationInfo.gpuInfo;
            const vendor = gpu.vendor.includes('Google') ? gpu.renderer.split(' ')[0] : gpu.vendor;
            accelerationText += ` (${vendor})`;
        }
        
        // Add note if using CPU fallback due to GPU limitations
        if (accelerationInfo.activeProvider === 'wasm') {
            if (accelerationInfo.webglError && accelerationInfo.webglError.includes('cannot resolve operator')) {
                accelerationText += ' (GPU unsupported operators)';
            } else if (accelerationInfo.supportsWebGL || accelerationInfo.supportsWebGPU) {
                accelerationText += ' (GPU provider failed)';
            } else {
                accelerationText += ' (No GPU support)';
            }
        }
        
        this.elements.accelerationInfo.textContent = accelerationText;
        
        // Update styling based on acceleration type
        this.elements.accelerationInfo.className = 'acceleration-info';
        if (accelerationInfo.activeProvider === 'webgpu') {
            this.elements.accelerationInfo.classList.add('acceleration-webgpu');
        } else if (accelerationInfo.activeProvider === 'webgl') {
            this.elements.accelerationInfo.classList.add('acceleration-webgl');
        } else {
            this.elements.accelerationInfo.classList.add('acceleration-wasm');
        }
    }

    /**
     * Show loading indicator
     */
    showLoading(message = 'Loading...') {
        this.elements.loadingText.textContent = message;
        this.elements.loadingIndicator.style.display = 'block';
        this.elements.cameraSelection.style.display = 'none';
        this.elements.videoContainer.style.display = 'none';
        this.elements.errorDisplay.style.display = 'none';
    }

    /**
     * Hide loading indicator
     */
    hideLoading() {
        this.elements.loadingIndicator.style.display = 'none';
    }

    /**
     * Show camera selection
     */
    showCameraSelection() {
        this.stopInference();
        this.cameraManager.stopCamera();
        this.state.isCameraActive = false;
        
        this.elements.cameraSelection.style.display = 'block';
        this.elements.videoContainer.style.display = 'none';
        this.elements.errorDisplay.style.display = 'none';
    }

    /**
     * Hide camera selection
     */
    hideCameraSelection() {
        this.elements.cameraSelection.style.display = 'none';
    }

    /**
     * Show video container
     */
    showVideoContainer() {
        this.elements.videoContainer.style.display = 'block';
        this.elements.errorDisplay.style.display = 'none';
    }

    /**
     * Show error message
     */
    showError(title, message) {
        this.elements.errorMessage.innerHTML = `<strong>${title}</strong><br>${message}`;
        this.elements.errorDisplay.style.display = 'block';
        this.elements.loadingIndicator.style.display = 'none';
        this.elements.cameraSelection.style.display = 'none';
        this.elements.videoContainer.style.display = 'none';
        
        this.state.currentError = { title, message };
    }

    /**
     * Hide error display
     */
    hideError() {
        this.elements.errorDisplay.style.display = 'none';
        this.state.currentError = null;
    }

    /**
     * Clean up resources
     */
    cleanup() {
        console.log('Cleaning up application...');
        
        this.stopInference();
        this.cameraManager.dispose();
        this.modelInference.dispose();
        
        this.state = {
            isInitialized: false,
            isModelLoaded: false,
            isCameraActive: false,
            isInferenceRunning: false,
            currentError: null
        };
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new CardSegmentationApp();
    app.init();
    
    // Make app globally available for debugging
    window.cardSegmentationApp = app;
});
