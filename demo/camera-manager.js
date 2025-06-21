/**
 * Camera management module for video stream handling
 */

class CameraManager {
    constructor() {
        this.currentStream = null;
        this.availableCameras = [];
        this.selectedCameraId = null;
        this.videoElement = null;
        this.preferredConstraints = {
            width: { ideal: 480, min: 320 },
            height: { ideal: 640, min: 240 }
        };
        this.fallbackConstraints = {
            width: { ideal: 640, min: 320 },
            height: { ideal: 480, min: 240 }
        };
    }

    /**
     * Initialize camera manager and enumerate devices
     * @returns {Promise<void>}
     */
    async initialize() {
        try {
            // Request permission first
            await this.requestCameraPermission();
            
            // Enumerate cameras
            await this.enumerateCameras();
            
            console.log(`Found ${this.availableCameras.length} camera(s)`);
        } catch (error) {
            console.error('Failed to initialize camera manager:', error);
            throw new Error(`Camera initialization failed: ${error.message}`);
        }
    }

    /**
     * Request camera permission
     * @returns {Promise<void>}
     */
    async requestCameraPermission() {
        try {
            // Request basic camera access to get permissions
            const tempStream = await navigator.mediaDevices.getUserMedia({ 
                video: true 
            });
            
            // Stop the temporary stream
            tempStream.getTracks().forEach(track => track.stop());
            
            console.log('Camera permission granted');
        } catch (error) {
            throw new Error('Camera permission denied or not available');
        }
    }

    /**
     * Enumerate available cameras
     * @returns {Promise<void>}
     */
    async enumerateCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            
            this.availableCameras = devices
                .filter(device => device.kind === 'videoinput')
                .map((device, index) => ({
                    id: device.deviceId,
                    label: device.label || `Camera ${index + 1}`,
                    groupId: device.groupId
                }));

            if (this.availableCameras.length === 0) {
                throw new Error('No cameras found');
            }

            // Select first camera by default
            this.selectedCameraId = this.availableCameras[0].id;
            
        } catch (error) {
            throw new Error(`Failed to enumerate cameras: ${error.message}`);
        }
    }

    /**
     * Get list of available cameras
     * @returns {Array}
     */
    getAvailableCameras() {
        return [...this.availableCameras];
    }

    /**
     * Start video stream with selected camera
     * @param {HTMLVideoElement} videoElement 
     * @param {string} cameraId 
     * @returns {Promise<Object>} - Stream info
     */
    async startCamera(videoElement, cameraId = null) {
        try {
            // Stop current stream if exists
            if (this.currentStream) {
                this.stopCamera();
            }

            // Use provided camera or current selection
            const targetCameraId = cameraId || this.selectedCameraId;
            this.selectedCameraId = targetCameraId;
            
            // Store video element reference
            this.videoElement = videoElement;

            // Try preferred resolution first (480x640)
            let stream = null;
            let usedConstraints = null;
            
            try {
                const constraints = {
                    video: {
                        deviceId: { exact: targetCameraId },
                        ...this.preferredConstraints
                    }
                };
                
                console.log('Trying preferred resolution (480x640)...');
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                usedConstraints = this.preferredConstraints;
                
            } catch (error) {
                console.log('Preferred resolution failed, trying fallback (640x480)...');
                
                const constraints = {
                    video: {
                        deviceId: { exact: targetCameraId },
                        ...this.fallbackConstraints
                    }
                };
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                usedConstraints = this.fallbackConstraints;
            }

            // Set up video element
            videoElement.srcObject = stream;
            this.currentStream = stream;

            // Wait for video metadata to load
            await new Promise((resolve, reject) => {
                videoElement.addEventListener('loadedmetadata', resolve, { once: true });
                videoElement.addEventListener('error', reject, { once: true });
                
                // Timeout after 10 seconds
                setTimeout(() => reject(new Error('Video load timeout')), 10000);
            });

            // Get actual video dimensions
            const videoWidth = videoElement.videoWidth;
            const videoHeight = videoElement.videoHeight;
            
            console.log(`Video stream started: ${videoWidth}x${videoHeight}`);

            // Determine if camera should be rotated for display
            const shouldRotateDisplay = ImageUtils.shouldRotateCamera(videoWidth, videoHeight);
            
            // Apply rotation CSS if needed
            if (shouldRotateDisplay) {
                videoElement.classList.add('rotate-90ccw');
                console.log('Applied 90° CCW rotation to video display');
            } else {
                videoElement.classList.remove('rotate-90ccw');
            }

            return {
                width: videoWidth,
                height: videoHeight,
                rotated: shouldRotateDisplay,
                constraints: usedConstraints,
                cameraId: targetCameraId,
                cameraLabel: this.getCameraLabel(targetCameraId)
            };
            
        } catch (error) {
            console.error('Failed to start camera:', error);
            throw new Error(`Camera start failed: ${error.message}`);
        }
    }

    /**
     * Stop current video stream
     */
    stopCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => {
                track.stop();
                console.log(`Stopped ${track.kind} track`);
            });
            this.currentStream = null;
        }

        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement.classList.remove('rotate-90ccw');
        }

        console.log('Camera stopped');
    }

    /**
     * Switch to a different camera
     * @param {string} cameraId 
     * @returns {Promise<Object>}
     */
    async switchCamera(cameraId) {
        if (!this.videoElement) {
            throw new Error('No video element available');
        }

        return await this.startCamera(this.videoElement, cameraId);
    }

    /**
     * Get label for camera ID
     * @param {string} cameraId 
     * @returns {string}
     */
    getCameraLabel(cameraId) {
        const camera = this.availableCameras.find(cam => cam.id === cameraId);
        return camera ? camera.label : 'Unknown Camera';
    }

    /**
     * Get current camera information
     * @returns {Object|null}
     */
    getCurrentCameraInfo() {
        if (!this.selectedCameraId || !this.videoElement) {
            return null;
        }

        return {
            id: this.selectedCameraId,
            label: this.getCameraLabel(this.selectedCameraId),
            width: this.videoElement.videoWidth,
            height: this.videoElement.videoHeight,
            isRotated: this.videoElement.classList.contains('rotate-90ccw')
        };
    }

    /**
     * Check if camera is currently active
     * @returns {boolean}
     */
    isActive() {
        return this.currentStream !== null && 
               this.currentStream.getTracks().some(track => track.readyState === 'live');
    }

    /**
     * Get video element reference
     * @returns {HTMLVideoElement|null}
     */
    getVideoElement() {
        return this.videoElement;
    }

    /**
     * Set up camera selection dropdown
     * @param {HTMLSelectElement} selectElement 
     */
    populateCameraSelect(selectElement) {
        // Clear existing options
        selectElement.innerHTML = '';

        if (this.availableCameras.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No cameras available';
            option.disabled = true;
            selectElement.appendChild(option);
            return;
        }

        // Add camera options
        this.availableCameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.id;
            option.textContent = camera.label;
            selectElement.appendChild(option);
        });

        // Select current camera
        if (this.selectedCameraId) {
            selectElement.value = this.selectedCameraId;
        }
    }

    /**
     * Handle device change events (camera plugged/unplugged)
     * @param {Function} callback 
     */
    onDeviceChange(callback) {
        navigator.mediaDevices.addEventListener('devicechange', async () => {
            console.log('Camera devices changed');
            try {
                await this.enumerateCameras();
                callback(this.availableCameras);
            } catch (error) {
                console.error('Error handling device change:', error);
                callback([]);
            }
        });
    }

    /**
     * Dispose of resources
     */
    dispose() {
        this.stopCamera();
        this.availableCameras = [];
        this.selectedCameraId = null;
        this.videoElement = null;
        console.log('Camera manager disposed');
    }
}

// Export for use in other modules
window.CameraManager = CameraManager;
