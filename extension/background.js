// Background script for AI Deepfake Detector extension

class DeepfakeDetectorBackground {
    constructor() {
        this.apiUrl = 'http://localhost:8001';
        this.init();
    }

    init() {
        this.bindEvents();
        this.createContextMenu();
    }

    bindEvents() {
        // Listen for messages from content scripts
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'detectImage') {
                this.detectImage(request.imageUrl).then(sendResponse);
                return true; // Keep message channel open for async response
            }
        });

        // Handle context menu clicks
        chrome.contextMenus.onClicked.addListener((info, tab) => {
            if (info.menuItemId === 'detectDeepfake') {
                this.handleContextMenuClick(info, tab);
            }
        });

        // Handle extension installation
        chrome.runtime.onInstalled.addListener((details) => {
            if (details.reason === 'install') {
                this.onInstall();
            }
        });
    }

    createContextMenu() {
        // Remove existing menu items
        chrome.contextMenus.removeAll(() => {
            // Create new context menu
            chrome.contextMenus.create({
                id: 'detectDeepfake',
                title: '🔍 Detect Deepfake',
                contexts: ['image']
            });
        });
    }

    async handleContextMenuClick(info, tab) {
        if (info.srcUrl) {
            try {
                // Show notification that detection is starting
                this.showNotification('Analyzing image...', 'info');
                
                // Detect the image
                const result = await this.detectImage(info.srcUrl);
                
                if (result.success) {
                    const status = result.result.is_deepfake ? 'AI Generated' : 'Real';
                    const confidence = Math.round(result.result.confidence * 100);
                    const detectionType = result.result.detection_type || 'unknown';
                    
                    let message = `${status} (${confidence}% confidence)`;
                    
                    // Add more specific messaging based on detection type
                    if (detectionType === 'deepfake') {
                        message = `🤖 Deepfake Detected (${confidence}% confidence)`;
                    } else if (detectionType === 'possible_ai') {
                        message = `⚠️ Possible AI Content (${confidence}% confidence)`;
                    } else if (detectionType === 'real') {
                        message = `✅ Real Image (${confidence}% confidence)`;
                    } else if (detectionType === 'uncertain') {
                        message = `❓ Uncertain (${confidence}% confidence)`;
                    }
                    
                    this.showNotification(message, result.result.is_deepfake ? 'warning' : 'success');
                } else {
                    this.showNotification('Detection failed', 'error');
                }
            } catch (error) {
                console.error('Context menu detection error:', error);
                this.showNotification('Error occurred during detection', 'error');
            }
        }
    }

    showNotification(message, type = 'info') {
        // Create a simple notification using chrome.notifications if available
        // For now, we'll use a more basic approach
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // You could also inject a notification into the page
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'showNotification',
                    message: message,
                    type: type
                });
            }
        });
    }

    async detectImage(imageUrl) {
        try {
            // Fetch the image
            const response = await fetch(imageUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch image: ${response.status}`);
            }

            const blob = await response.blob();
            
            // Create form data for API
            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');

            // Send to our FastAPI backend
            const apiResponse = await fetch(`${this.apiUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!apiResponse.ok) {
                throw new Error(`API error: ${apiResponse.status}`);
            }

            const result = await apiResponse.json();
            
            return {
                success: true,
                result: {
                    is_deepfake: result.is_deepfake,
                    confidence: result.confidence,
                    processing_time: result.processing_time,
                    model_used: result.model_used
                }
            };

        } catch (error) {
            console.error('Background detection error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    onInstall() {
        // Set default settings
        chrome.storage.sync.set({
            autoScan: false,  // Changed to false since we're using context menu
            showOverlay: false  // Changed to false since we're using context menu
        });

        // Open welcome page
        chrome.tabs.create({
            url: chrome.runtime.getURL('welcome.html')
        });
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
}

// Initialize background script
new DeepfakeDetectorBackground();
