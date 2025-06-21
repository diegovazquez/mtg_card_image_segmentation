/**
 * Image utilities for preprocessing and transformations
 */

class ImageUtils {
    /**
     * Create a canvas element for image processing
     * @param {number} width 
     * @param {number} height 
     * @returns {HTMLCanvasElement}
     */
    static createCanvas(width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        return canvas;
    }

    /**
     * Rotate image 90 degrees counterclockwise
     * @param {ImageData} imageData 
     * @returns {ImageData}
     */
    static rotateImage90CCW(imageData) {
        const { width, height, data } = imageData;
        const rotatedCanvas = this.createCanvas(height, width);
        const ctx = rotatedCanvas.getContext('2d');
        
        // Create temporary canvas with original image
        const tempCanvas = this.createCanvas(width, height);
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);
        
        // Apply rotation transformation
        ctx.translate(height, 0);
        ctx.rotate(Math.PI / 2);
        ctx.drawImage(tempCanvas, 0, 0);
        
        return ctx.getImageData(0, 0, height, width);
    }

    /**
     * Resize image to target dimensions
     * @param {ImageData} imageData 
     * @param {number} targetWidth 
     * @param {number} targetHeight 
     * @returns {ImageData}
     */
    static resizeImage(imageData, targetWidth, targetHeight) {
        const { width, height } = imageData;
        
        // Create canvas with original image
        const sourceCanvas = this.createCanvas(width, height);
        const sourceCtx = sourceCanvas.getContext('2d');
        sourceCtx.putImageData(imageData, 0, 0);
        
        // Create target canvas
        const targetCanvas = this.createCanvas(targetWidth, targetHeight);
        const targetCtx = targetCanvas.getContext('2d');
        
        // Resize using canvas scaling
        targetCtx.drawImage(sourceCanvas, 0, 0, width, height, 0, 0, targetWidth, targetHeight);
        
        return targetCtx.getImageData(0, 0, targetWidth, targetHeight);
    }

    /**
     * Extract image data from video element
     * @param {HTMLVideoElement} video 
     * @returns {ImageData}
     */
    static getVideoFrame(video) {
        const canvas = this.createCanvas(video.videoWidth, video.videoHeight);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        return ctx.getImageData(0, 0, video.videoWidth, video.videoHeight);
    }

    /**
     * Normalize image data for model inference
     * Uses ImageNet normalization values
     * @param {ImageData} imageData 
     * @returns {Float32Array}
     */
    static normalizeImageData(imageData) {
        const { width, height, data } = imageData;
        const normalizedData = new Float32Array(3 * width * height);
        
        // ImageNet normalization constants
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        
        let pixelIndex = 0;
        for (let i = 0; i < data.length; i += 4) {
            // Extract RGB values (skip alpha channel)
            const r = data[i] / 255.0;
            const g = data[i + 1] / 255.0;
            const b = data[i + 2] / 255.0;
            
            // Apply normalization: (pixel - mean) / std
            normalizedData[pixelIndex] = (r - mean[0]) / std[0]; // R channel
            normalizedData[pixelIndex + width * height] = (g - mean[1]) / std[1]; // G channel
            normalizedData[pixelIndex + 2 * width * height] = (b - mean[2]) / std[2]; // B channel
            
            pixelIndex++;
        }
        
        return normalizedData;
    }

    /**
     * Preprocess image for model inference
     * @param {HTMLVideoElement} video 
     * @param {number} targetWidth 
     * @param {number} targetHeight 
     * @param {boolean} shouldRotate - Whether to rotate 90 degrees CCW
     * @returns {Float32Array}
     */
    static preprocessVideoFrame(video, targetWidth, targetHeight, shouldRotate = false) {
        try {
            // Extract frame from video
            let imageData = this.getVideoFrame(video);
            
            // Rotate if needed (for 640x480 -> 480x640)
            if (shouldRotate) {
                imageData = this.rotateImage90CCW(imageData);
            }
            
            // Resize to target dimensions
            if (imageData.width !== targetWidth || imageData.height !== targetHeight) {
                imageData = this.resizeImage(imageData, targetWidth, targetHeight);
            }
            
            // Normalize for model input
            return this.normalizeImageData(imageData);
        } catch (error) {
            console.error('Error preprocessing video frame:', error);
            throw error;
        }
    }

    /**
     * Create tensor from preprocessed data
     * @param {Float32Array} normalizedData 
     * @param {number} height 
     * @param {number} width 
     * @returns {Object} - Tensor-like object for ONNX
     */
    static createInputTensor(normalizedData, height, width) {
        return {
            data: normalizedData,
            dims: [1, 3, height, width],
            type: 'float32'
        };
    }

    /**
     * Process model output to create segmentation mask
     * @param {Float32Array} modelOutput 
     * @param {number} height 
     * @param {number} width 
     * @returns {Uint8Array} - Binary mask
     */
    static processModelOutput(modelOutput, height, width) {
        const mask = new Uint8Array(height * width);
        
        // Apply softmax and get argmax for each pixel
        for (let i = 0; i < height * width; i++) {
            const backgroundLogit = modelOutput[i];
            const cardLogit = modelOutput[i + height * width];
            
            // Simple argmax (card class = 1, background = 0)
            mask[i] = cardLogit > backgroundLogit ? 255 : 0;
        }
        
        return mask;
    }

    /**
     * Create overlay canvas with segmentation mask
     * @param {Uint8Array} mask 
     * @param {number} width 
     * @param {number} height 
     * @param {HTMLCanvasElement} targetCanvas 
     * @param {boolean} shouldRotate - Whether to rotate mask back
     */
    static drawSegmentationOverlay(mask, width, height, targetCanvas, shouldRotate = false) {
        const ctx = targetCanvas.getContext('2d');
        
        // Create mask canvas
        const maskCanvas = this.createCanvas(width, height);
        const maskCtx = maskCanvas.getContext('2d');
        const maskImageData = maskCtx.createImageData(width, height);
        
        // Fill mask with cyan color where mask is active
        for (let i = 0; i < mask.length; i++) {
            const pixelIndex = i * 4;
            if (mask[i] > 0) {
                maskImageData.data[pixelIndex] = 0;     // R
                maskImageData.data[pixelIndex + 1] = 255; // G (cyan)
                maskImageData.data[pixelIndex + 2] = 255; // B (cyan)
                maskImageData.data[pixelIndex + 3] = 128; // A (50% transparent)
            } else {
                maskImageData.data[pixelIndex + 3] = 0; // Fully transparent
            }
        }
        
        maskCtx.putImageData(maskImageData, 0, 0);
        
        // Clear target canvas
        ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        
        // Draw mask with rotation if needed
        if (shouldRotate) {
            ctx.save();
            ctx.translate(targetCanvas.width, 0);
            ctx.rotate(Math.PI / 2);
            ctx.drawImage(maskCanvas, 0, 0, height, width, 0, 0, targetCanvas.height, targetCanvas.width);
            ctx.restore();
        } else {
            ctx.drawImage(maskCanvas, 0, 0, width, height, 0, 0, targetCanvas.width, targetCanvas.height);
        }
    }

    /**
     * Check if camera orientation suggests it should be rotated
     * @param {number} width 
     * @param {number} height 
     * @returns {boolean}
     */
    static shouldRotateCamera(width, height) {
        return width > height; // Horizontal camera should be rotated to vertical
    }

    /**
     * Check if input image needs rotation for model (640x480 -> 480x640)
     * @param {number} width 
     * @param {number} height 
     * @returns {boolean}
     */
    static shouldRotateForModel(width, height) {
        return width === 640 && height === 480;
    }
}

// Export for use in other modules
window.ImageUtils = ImageUtils;
