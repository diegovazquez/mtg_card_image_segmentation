/**
 * Model inference module using ONNX Runtime Web
 */

class ModelInference {
    constructor() {
        this.session = null;
        this.isModelLoaded = false;
        this.isInferring = false;
        this.modelPath = 'models/card_segmentation.onnx';
        
        // Model specifications
        this.inputHeight = 480;
        this.inputWidth = 640;
        this.numClasses = 2;
        
        // GPU acceleration info
        this.accelerationInfo = {
            activeProvider: 'unknown',
            availableProviders: [],
            gpuInfo: null,
            supportsWebGL: false,
            supportsWebGPU: false
        };
        
        // Performance tracking
        this.inferenceStats = {
            totalInferences: 0,
            totalTime: 0,
            averageTime: 0,
            lastInferenceTime: 0
        };
    }

    /**
     * Initialize ONNX Runtime and load the model with GPU acceleration
     * @returns {Promise<boolean>}
     */
    async initialize() {
        try {
            console.log('Initializing ONNX Runtime with GPU acceleration...');
            
            // Configure ONNX Runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
            ort.env.wasm.numThreads = 1; // Single thread for stability
            
            // Detect GPU capabilities
            await this.detectGPUCapabilities();
            
            // Determine optimal execution providers
            const executionProviders = this.getOptimalExecutionProviders();
            
            console.log('Attempting to load model with providers:', executionProviders);
            console.log(`Loading model from: ${this.modelPath}`);
            
            // Try to load model with optimal providers
            this.session = await this.loadModelWithFallback(executionProviders);
            
            console.log(`Model loaded successfully with provider: ${this.accelerationInfo.activeProvider}`);
            this.isModelLoaded = true;
            
            // Log model and acceleration information
            this.logModelInfo();
            this.logAccelerationInfo();
            
            return true;
        } catch (error) {
            console.error('Failed to initialize model:', error);
            this.isModelLoaded = false;
            throw new Error(`Model initialization failed: ${error.message}`);
        }
    }

    /**
     * Detect GPU capabilities and available acceleration
     */
    async detectGPUCapabilities() {
        console.log('Detecting GPU capabilities...');
        
        // Check WebGL support
        this.accelerationInfo.supportsWebGL = this.checkWebGLSupport();
        
        // Check WebGPU support
        this.accelerationInfo.supportsWebGPU = await this.checkWebGPUSupport();
        
        // Get GPU info if available
        if (this.accelerationInfo.supportsWebGL) {
            this.accelerationInfo.gpuInfo = this.getWebGLInfo();
        }
        
        console.log('GPU Capabilities:', {
            WebGL: this.accelerationInfo.supportsWebGL,
            WebGPU: this.accelerationInfo.supportsWebGPU,
            GPU: this.accelerationInfo.gpuInfo
        });
    }

    /**
     * Check WebGL support
     * @returns {boolean}
     */
    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!gl;
        } catch (error) {
            console.warn('WebGL check failed:', error);
            return false;
        }
    }

    /**
     * Check WebGPU support
     * @returns {Promise<boolean>}
     */
    async checkWebGPUSupport() {
        try {
            if (!navigator.gpu) {
                return false;
            }
            
            const adapter = await navigator.gpu.requestAdapter();
            return !!adapter;
        } catch (error) {
            console.warn('WebGPU check failed:', error);
            return false;
        }
    }

    /**
     * Get WebGL renderer information
     * @returns {Object|null}
     */
    getWebGLInfo() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (!gl) return null;
            
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            
            return {
                vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
                renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
                version: gl.getParameter(gl.VERSION),
                shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION)
            };
        } catch (error) {
            console.warn('Failed to get WebGL info:', error);
            return null;
        }
    }

    /**
     * Determine optimal execution providers based on capabilities
     * @returns {Array<string>}
     */
    getOptimalExecutionProviders() {
        const providers = [];
        
        // Prioritize WebGL over WebGPU for better compatibility
        if (this.accelerationInfo.supportsWebGL) {
            providers.push('webgl');
            this.accelerationInfo.availableProviders.push('webgl');
        }
        
        // Only add WebGPU if explicitly enabled and WebGL failed
        // (WebGPU is still experimental and often fails)
        if (this.accelerationInfo.supportsWebGPU && !this.accelerationInfo.supportsWebGL) {
            providers.push('webgpu');
            this.accelerationInfo.availableProviders.push('webgpu');
        }
        
        // Always add WASM as fallback
        providers.push('wasm');
        this.accelerationInfo.availableProviders.push('wasm');
        
        return providers;
    }

    /**
     * Load model with fallback mechanism
     * @param {Array<string>} executionProviders 
     * @returns {Promise<InferenceSession>}
     */
    async loadModelWithFallback(executionProviders) {
        const baseOptions = {
            enableMemPattern: false,
            enableCpuMemArena: false,
            graphOptimizationLevel: 'all'
        };

        // Try each provider in order
        for (const provider of executionProviders) {
            try {
                console.log(`Attempting to load model with ${provider} provider...`);
                
                const options = {
                    ...baseOptions,
                    executionProviders: [provider]
                };

                const session = await ort.InferenceSession.create(this.modelPath, options);
                this.accelerationInfo.activeProvider = provider;
                
                console.log(`✓ Successfully loaded with ${provider} provider`);
                return session;
            } catch (error) {
                const errorMsg = error.message;
                
                // Log specific information for operator compatibility issues
                if (errorMsg.includes('cannot resolve operator')) {
                    const operatorName = errorMsg.match(/operator '(\w+)'/)?.[1] || 'unknown';
                    console.warn(`✗ ${provider.toUpperCase()} provider doesn't support operator '${operatorName}' - falling back to next provider`);
                } else {
                    console.warn(`✗ Failed to load with ${provider} provider:`, errorMsg);
                }
                
                // Store failed provider info for reporting
                this.accelerationInfo[`${provider}Error`] = errorMsg;
                continue;
            }
        }
        
        throw new Error('Failed to load model with any execution provider');
    }

    /**
     * Log acceleration information
     */
    logAccelerationInfo() {
        console.log('=== Acceleration Information ===');
        console.log(`Active Provider: ${this.accelerationInfo.activeProvider}`);
        console.log(`Available Providers: ${this.accelerationInfo.availableProviders.join(', ')}`);
        console.log(`WebGL Support: ${this.accelerationInfo.supportsWebGL}`);
        console.log(`WebGPU Support: ${this.accelerationInfo.supportsWebGPU}`);
        
        if (this.accelerationInfo.gpuInfo) {
            console.log('GPU Information:');
            console.log(`  Vendor: ${this.accelerationInfo.gpuInfo.vendor}`);
            console.log(`  Renderer: ${this.accelerationInfo.gpuInfo.renderer}`);
            console.log(`  Version: ${this.accelerationInfo.gpuInfo.version}`);
        }
        console.log('===============================');
    }

    /**
     * Get acceleration information
     * @returns {Object}
     */
    getAccelerationInfo() {
        return { ...this.accelerationInfo };
    }

    /**
     * Log model information for debugging
     */
    logModelInfo() {
        if (!this.session) return;
        
        console.log('=== Model Information ===');
        console.log('Input tensors:');
        this.session.inputNames.forEach((name, index) => {
            // Try to access input metadata safely
            try {
                if (this.session.inputs && this.session.inputs[index]) {
                    const input = this.session.inputs[index];
                    console.log(`  ${name}: ${input.dims} (${input.type})`);
                } else {
                    console.log(`  ${name}: metadata not available`);
                }
            } catch (error) {
                console.log(`  ${name}: metadata not available`);
            }
        });
        
        console.log('Output tensors:');
        this.session.outputNames.forEach((name, index) => {
            // Try to access output metadata safely
            try {
                if (this.session.outputs && this.session.outputs[index]) {
                    const output = this.session.outputs[index];
                    console.log(`  ${name}: ${output.dims} (${output.type})`);
                } else {
                    console.log(`  ${name}: metadata not available`);
                }
            } catch (error) {
                console.log(`  ${name}: metadata not available`);
            }
        });
        console.log('========================');
    }

    /**
     * Run inference on preprocessed image data
     * @param {Float32Array} inputData 
     * @returns {Promise<Float32Array>}
     */
    async runInference(inputData) {
        if (!this.isModelLoaded || !this.session) {
            throw new Error('Model not loaded. Call initialize() first.');
        }

        if (this.isInferring) {
            console.warn('Inference already in progress, skipping...');
            return null;
        }

        try {
            this.isInferring = true;
            const startTime = performance.now();

            // Create input tensor
            const inputTensor = new ort.Tensor('float32', inputData, [1, 3, this.inputHeight, this.inputWidth]);
            
            // Get input name from model
            const inputName = this.session.inputNames[0];
            
            // Run inference
            const results = await this.session.run({ [inputName]: inputTensor });
            
            // Get output tensor
            const outputName = this.session.outputNames[0];
            const outputTensor = results[outputName];
            
            // Calculate inference time
            const endTime = performance.now();
            const inferenceTime = endTime - startTime;
            
            // Update statistics
            this.updateInferenceStats(inferenceTime);
            
            console.log(`Inference completed in ${inferenceTime.toFixed(2)}ms`);
            
            return outputTensor.data;
        } catch (error) {
            console.error('Inference failed:', error);
            throw new Error(`Inference failed: ${error.message}`);
        } finally {
            this.isInferring = false;
        }
    }

    /**
     * Process video frame and return segmentation mask
     * @param {HTMLVideoElement} video 
     * @returns {Promise<Object>} - {mask: Uint8Array, stats: Object}
     */
    async processVideoFrame(video) {
        try {
            // Check if model is ready
            if (!this.isModelLoaded) {
                throw new Error('Model not initialized');
            }

            // Get video dimensions
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;
            
            if (videoWidth === 0 || videoHeight === 0) {
                throw new Error('Video not ready');
            }

            // Determine if rotation is needed for model input
            const shouldRotateForModel = ImageUtils.shouldRotateForModel(videoWidth, videoHeight);
            
            // Preprocess video frame
            const preprocessedData = ImageUtils.preprocessVideoFrame(
                video, 
                this.inputWidth, 
                this.inputHeight, 
                shouldRotateForModel
            );

            // Run inference
            const modelOutput = await this.runInference(preprocessedData);
            
            if (!modelOutput) {
                return null; // Inference skipped
            }

            // Process model output to create mask
            const mask = ImageUtils.processModelOutput(
                modelOutput, 
                this.inputHeight, 
                this.inputWidth
            );

            return {
                mask: mask,
                shouldRotateBack: shouldRotateForModel,
                stats: this.getInferenceStats()
            };
        } catch (error) {
            console.error('Error processing video frame:', error);
            throw error;
        }
    }

    /**
     * Update inference performance statistics
     * @param {number} inferenceTime 
     */
    updateInferenceStats(inferenceTime) {
        this.inferenceStats.totalInferences++;
        this.inferenceStats.totalTime += inferenceTime;
        this.inferenceStats.averageTime = this.inferenceStats.totalTime / this.inferenceStats.totalInferences;
        this.inferenceStats.lastInferenceTime = inferenceTime;
    }

    /**
     * Get current inference statistics
     * @returns {Object}
     */
    getInferenceStats() {
        return {
            ...this.inferenceStats,
            fps: this.inferenceStats.lastInferenceTime > 0 ? 
                 1000 / this.inferenceStats.lastInferenceTime : 0
        };
    }

    /**
     * Reset inference statistics
     */
    resetStats() {
        this.inferenceStats = {
            totalInferences: 0,
            totalTime: 0,
            averageTime: 0,
            lastInferenceTime: 0
        };
    }

    /**
     * Check if model is ready for inference
     * @returns {boolean}
     */
    isReady() {
        return this.isModelLoaded && !this.isInferring;
    }

    /**
     * Get model specifications
     * @returns {Object}
     */
    getModelSpecs() {
        return {
            inputHeight: this.inputHeight,
            inputWidth: this.inputWidth,
            numClasses: this.numClasses,
            isLoaded: this.isModelLoaded
        };
    }

    /**
     * Dispose of the model and free resources
     */
    dispose() {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
        this.isModelLoaded = false;
        this.isInferring = false;
        console.log('Model resources disposed');
    }
}

// Export for use in other modules
window.ModelInference = ModelInference;
