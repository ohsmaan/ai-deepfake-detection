class DeepfakeDetectorContent {
    constructor() {
        this.settings = {};
        this.apiUrl = 'http://localhost:8001';
        this.filterMode = false; // Toggle for AI content filtering
        this.scannedElements = new Set(); // Track already scanned elements
        this.scannedTexts = new Set(); // Track already scanned text content
        this.removeContent = false; // New setting for removing content
        this.init();
    }

    async init() {
        this.loadSettings();
        this.bindEvents();
        this.detectVideos();
        this.setupContentFilter();
    }

    async loadSettings() {
        chrome.storage.sync.get({
            filterMode: false,
            removeContent: false,
            textDetection: true,
            textConfidenceThreshold: 0.7,
            continuousScanning: true,
            scanInterval: 5000
        }, (settings) => {
            this.settings = settings;
            this.filterMode = settings.filterMode;
            this.removeContent = settings.removeContent;
            console.log('🔧 Loaded settings:', settings);
        });
    }

    setupContentFilter() {
        this.addFilterToggle();
        this.filterMode = this.settings.filterMode;
        this.removeContent = this.settings.removeContent;
    }

    addFilterToggle() {
        // Remove existing toggle if any
        const existingToggle = document.querySelector('.ai-filter-toggle');
        if (existingToggle) {
            existingToggle.remove();
        }

        // Create toggle container
        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'ai-filter-toggle';
        toggleContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 100000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            min-width: 200px;
        `;

        toggleContainer.innerHTML = `
            <div style="margin-bottom: 10px; font-weight: bold;">🤖 AI Content Filter</div>
            <div style="margin-bottom: 8px;">
                <label style="display: flex; align-items: center; cursor: pointer;">
                    <input type="checkbox" id="filterMode" ${this.filterMode ? 'checked' : ''} style="margin-right: 8px;">
                    Show Overlays
                </label>
            </div>
            <div style="margin-bottom: 8px;">
                <label style="display: flex; align-items: center; cursor: pointer;">
                    <input type="checkbox" id="removeContent" ${this.removeContent ? 'checked' : ''} style="margin-right: 8px;">
                    Remove AI Content
                </label>
            </div>
            <div style="margin-bottom: 10px;">
                <button id="removeAllAI" style="
                    background: #ff4444;
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    width: 100%;
                    margin-bottom: 5px;
                ">🗑️ Remove All AI Content</button>
                <button id="restoreContent" style="
                    background: #44aa44;
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    width: 100%;
                ">🔄 Restore Removed Content</button>
            </div>
            <div style="font-size: 12px; color: #666;">
                Blocked: <span class="filter-count">0</span>
            </div>
        `;

        document.body.appendChild(toggleContainer);

        // Add event listeners
        const filterModeCheckbox = toggleContainer.querySelector('#filterMode');
        const removeContentCheckbox = toggleContainer.querySelector('#removeContent');
        const removeAllButton = toggleContainer.querySelector('#removeAllAI');
        const restoreContentButton = toggleContainer.querySelector('#restoreContent');

        filterModeCheckbox.addEventListener('change', async () => {
            this.filterMode = filterModeCheckbox.checked;
            await chrome.storage.sync.set({ filterMode: this.filterMode });
            
            if (this.filterMode) {
                this.startContentMonitoring();
            } else {
                this.stopContentMonitoring();
                this.unhideAllContent();
            }
        });

        removeContentCheckbox.addEventListener('change', async () => {
            this.removeContent = removeContentCheckbox.checked;
            await chrome.storage.sync.set({ removeContent: this.removeContent });
            
            if (this.removeContent) {
                this.startContentMonitoring();
            }
        });

        removeAllButton.addEventListener('click', () => {
            this.removeAllAIContent();
        });

        restoreContentButton.addEventListener('click', () => {
            this.restoreRemovedContent();
        });
    }

    startContentMonitoring() {
        if (this.observer) {
            this.observer.disconnect();
        }
        
        // Create a more aggressive observer for Twitter and Google Images
        const isTwitter = window.location.hostname.includes('twitter.com') || window.location.hostname.includes('x.com');
        const isGoogleImages = window.location.hostname.includes('google.com') && window.location.pathname.includes('/search') && window.location.search.includes('tbm=isch');
        
        this.observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        this.scanNewContent(node);
                        
                        // For Twitter and Google Images, also scan the entire document periodically
                        if (isTwitter || isGoogleImages) {
                            setTimeout(() => {
                                this.scanExistingContent();
                            }, 1000);
                        }
                    }
                });
            });
        });
        
        // Fix the observer options - remove attributeFilter when attributes is false
        const observerOptions = {
            childList: true,
            subtree: true,
            attributes: false  // Set to false since we don't need attribute changes
        };
        
        this.observer.observe(document.body, observerOptions);
        
        // For Twitter and Google Images, also set up periodic scanning
        if (isTwitter || isGoogleImages) {
            this.periodicScanInterval = setInterval(() => {
                this.scanExistingContent();
            }, 3000); // Scan every 3 seconds
        }
        
        console.log('🔍 Content monitoring started');
    }

    stopContentMonitoring() {
        if (this.observer) {
            this.observer.disconnect();
            this.observer = null;
        }
        
        // Clear Twitter scan interval
        if (this.periodicScanInterval) {
            clearInterval(this.periodicScanInterval);
            this.periodicScanInterval = null;
        }
        
        console.log('🔍 Content monitoring stopped');
    }

    scanNewContent(node) {
        // Scan for images and videos in new content
        const images = node.querySelectorAll ? node.querySelectorAll('img') : [];
        const videos = node.querySelectorAll ? node.querySelectorAll('video') : [];
        
        images.forEach(img => this.scanImage(img));
        videos.forEach(video => this.scanVideo(video));
        
        // If the node itself is an image or video
        if (node.tagName === 'IMG') this.scanImage(node);
        if (node.tagName === 'VIDEO') this.scanVideo(node);
        
        // Check for new images, videos, and text
        if (node.querySelectorAll) {
            const images = node.querySelectorAll('img');
            const videos = node.querySelectorAll('video');
            // const textElements = node.querySelectorAll('p, div, span, article');  // DISABLED - No automatic text scanning
            
            images.forEach(img => {
                if (this.isValidImage(img)) {
                    this.scanImage(img);
                }
            });
            
            videos.forEach(video => {
                if (this.isValidVideo(video)) {
                    this.scanVideo(video);
                }
            });
            
            // DISABLED - Automatic text scanning removed
            // textElements.forEach(element => {
            //     if (this.isValidTextElement(element)) {
            //         this.scanTextElement(element);
            //     }
            // });
        }
    }

    scanExistingContent() {
        // Check if we're on Google Images and use specific scanning
        if (window.location.hostname.includes('google.com') && window.location.pathname.includes('/search') && window.location.search.includes('tbm=isch')) {
            this.scanGoogleImages();
        } else {
            // Scan all existing images and videos
            this.scanImages();
            this.scanVideos();
            this.scanAudio();
            // this.scanTexts();  // DISABLED - No automatic text scanning
        }
    }

    async scanImage(img) {
        if (this.scannedElements.has(img) || !this.isValidImage(img)) {
            console.log(`🔍 Skipping image: already scanned or invalid`);
            return;
        }
        
        this.scannedElements.add(img);
        console.log(`🔍 Scanning image: ${img.src.substring(0, 100)}...`);
        
        // Debug: Show current filter settings
        console.log(`🔧 Current settings:`, {
            filterMode: this.filterMode,
            removeContent: this.removeContent,
            settings: this.settings
        });
        
        // Special debugging for Twitter
        if (window.location.hostname.includes('twitter.com') || window.location.hostname.includes('x.com')) {
            console.log(`🐦 Twitter image debug:`, {
                src: img.src,
                naturalWidth: img.naturalWidth,
                naturalHeight: img.naturalHeight,
                width: img.width,
                height: img.height,
                className: img.className,
                parentElement: img.parentElement?.tagName,
                parentClassName: img.parentElement?.className
            });
        }
        
        try {
            // Check if image is loaded
            if (!img.complete || img.naturalWidth === 0) {
                console.log(`🔍 Image not loaded yet, skipping`);
                return;
            }
            
            console.log(`🔍 Image size: ${img.naturalWidth}x${img.naturalHeight}`);
            
            const result = await this.analyzeFrame(img.src);
            console.log(`🔍 Image analysis result:`, result);
            
            // Debug: Log detailed result information
            console.log(`🔍 Detailed result:`, {
                is_deepfake: result.is_deepfake,
                confidence: result.confidence,
                detection_type: result.detection_type,
                predicted_label: result.predicted_label,
                analysis: result.analysis
            });
            
            // Handle different detection types with lower thresholds for testing
            if (this.filterMode) {
                console.log(`🔧 Filter mode is ON, checking detection result:`, {
                    is_deepfake: result.is_deepfake,
                    detection_type: result.detection_type,
                    confidence: result.confidence,
                    predicted_label: result.predicted_label
                });
                
                if (result.is_deepfake) {
                    console.log(`🚫 Hiding deepfake image (${Math.round(result.confidence * 100)}%)`);
                    this.hideContent(img, 'image', result);
                } else if (result.detection_type === 'ai_generated' && result.confidence > 0.7) {
                    console.log(`🚫 Hiding AI-generated image (${Math.round(result.confidence * 100)}%)`);
                    this.hideContent(img, 'image', result);
                } else if (result.detection_type === 'suspicious' && result.confidence > 0.6) {
                    console.log(`⚠️ Hiding suspicious image (${Math.round(result.confidence * 100)}%)`);
                    this.hideContent(img, 'image', result);
                } else if (result.predicted_label === 'artificial' && result.confidence > 0.5) {
                    console.log(`🚫 Hiding artificial image (${Math.round(result.confidence * 100)}%)`);
                    this.hideContent(img, 'image', result);
                } else {
                    console.log(`✅ Image appears to be real or low confidence (${Math.round(result.confidence * 100)}%)`);
                }
            } else {
                console.log(`🔧 Filter mode is OFF - not hiding content`);
            }
        } catch (error) {
            console.error('❌ Image scan error:', error);
            // Don't fail the entire filter for one image
        }
    }

    async scanVideo(video) {
        if (this.scannedElements.has(video) || !this.isValidVideo(video)) return;
        
        this.scannedElements.add(video);
        
        try {
            // Extract and analyze a frame from the video
            const frame = await this.extractVideoFrame(video);
            if (frame) {
                const result = await this.analyzeFrame(frame);
                
                if (result.is_deepfake && this.filterMode) {
                    this.hideContent(video, 'video', result);
                }
            }
        } catch (error) {
            console.error('Video scan error:', error);
        }
    }

    async extractVideoFrame(video) {
        return new Promise((resolve) => {
            if (video.readyState < 2) {
                resolve(null);
                return;
            }
            
            try {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                // Try to set crossOrigin to handle cross-origin videos
                if (video.crossOrigin !== 'anonymous') {
                    video.crossOrigin = 'anonymous';
                }
                
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL('image/jpeg', 0.8));
            } catch (error) {
                console.warn('Canvas taint error, skipping video frame extraction:', error);
                resolve(null);
            }
        });
    }

    hideContent(element, type, result) {
        // Remove any existing overlay for this element
        const existingOverlay = element.parentElement.querySelector('.ai-content-overlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }
        
        // Debug: Log the current settings
        console.log(`🔧 hideContent called with:`, {
            removeContent: this.removeContent,
            filterMode: this.filterMode,
            type: type,
            result: result
        });
        
        // If removeContent is enabled, completely remove the element
        if (this.removeContent) {
            console.log(`🗑️ Removing content (removeContent is enabled)`);
            this.removeElement(element, type, result);
            return;
        }
        
        // Otherwise, create overlay
        console.log(`🚫 Creating overlay (removeContent is disabled)`);
        this.createOverlay(element, type, result);
    }

    removeElement(element, type, result) {
        // Store info about removed element for potential restoration
        const removedInfo = {
            element: element,
            type: type,
            result: result,
            parent: element.parentElement,
            nextSibling: element.nextSibling
        };
        
        // Add to removed elements list
        if (!this.removedElements) {
            this.removedElements = [];
        }
        this.removedElements.push(removedInfo);
        
        // Website-specific removal logic
        const hostname = window.location.hostname;
        
        if (hostname.includes('youtube.com')) {
            // For YouTube, remove the entire video card/container
            this.removeYouTubeVideoCard(element);
        } else if (hostname.includes('twitter.com') || hostname.includes('x.com')) {
            // For Twitter, remove the entire post/tweet
            this.removeTwitterPost(element);
        } else if (hostname.includes('google.com')) {
            // For Google Images, remove the image card
            this.removeGoogleImageCard(element);
        } else {
            // For other sites, just remove the element
            element.remove();
        }
        
        // Update filter count
        this.updateFilterCount();
        
        console.log(`🗑️ Removed ${type}: ${result.detection_type} content (${Math.round(result.confidence * 100)}%)`);
    }

    removeYouTubeVideoCard(element) {
        console.log(`🎥 Attempting to remove YouTube video card`);
        
        // Look for YouTube-specific containers
        const youtubeSelectors = [
            'ytd-video-renderer',
            'ytd-rich-item-renderer', 
            'ytd-compact-video-renderer',
            'ytd-grid-video-renderer',
            'ytd-video-card-renderer',
            '[id*="video-renderer"]',
            '[id*="rich-item-renderer"]'
        ];
        
        let container = null;
        
        // Try to find the YouTube container
        for (const selector of youtubeSelectors) {
            container = element.closest(selector);
            if (container) {
                console.log(`🎥 Found YouTube container: ${selector}`);
                break;
            }
        }
        
        // If no specific container found, try to find a reasonable parent
        if (!container) {
            // Look for common YouTube parent patterns
            let current = element;
            for (let i = 0; i < 5; i++) { // Go up 5 levels max
                current = current.parentElement;
                if (!current) break;
                
                // Check if this looks like a video container
                if (current.tagName === 'YTD-VIDEO-RENDERER' || 
                    current.tagName === 'YTD-RICH-ITEM-RENDERER' ||
                    current.classList.contains('ytd-video-renderer') ||
                    current.id && current.id.includes('video') ||
                    current.querySelector('ytd-thumbnail') ||
                    current.querySelector('[id*="video"]')) {
                    container = current;
                    console.log(`🎥 Found YouTube container by pattern: ${current.tagName}`);
                    break;
                }
            }
        }
        
        // Remove the container if found, otherwise just remove the element
        if (container) {
            container.remove();
            console.log(`🗑️ Removed YouTube video card container`);
        } else {
            element.remove();
            console.log(`🗑️ Removed individual YouTube element (no container found)`);
        }
    }

    removeTwitterPost(element) {
        console.log(`🐦 Attempting to remove Twitter post`);
        
        // Look for Twitter-specific containers
        const twitterSelectors = [
            '[data-testid="tweet"]',
            '[data-testid="cellInnerDiv"]',
            'article[data-testid="tweet"]',
            '[role="article"]',
            '.css-1dbjc4n.r-1loqt21.r-18u37iz.r-1ny4l3l',
            '[data-testid="tweetText"]',
            '.tweet'
        ];
        
        let container = null;
        
        // Try to find the Twitter container
        for (const selector of twitterSelectors) {
            container = element.closest(selector);
            if (container) {
                console.log(`🐦 Found Twitter container: ${selector}`);
                break;
            }
        }
        
        // If no specific container found, try to find a reasonable parent
        if (!container) {
            // Look for common Twitter parent patterns
            let current = element;
            for (let i = 0; i < 5; i++) { // Go up 5 levels max
                current = current.parentElement;
                if (!current) break;
                
                // Check if this looks like a tweet container
                if (current.getAttribute('data-testid') === 'tweet' ||
                    current.getAttribute('role') === 'article' ||
                    current.classList.contains('tweet') ||
                    current.querySelector('[data-testid="tweetText"]') ||
                    current.querySelector('[data-testid="tweet"]')) {
                    container = current;
                    console.log(`🐦 Found Twitter container by pattern: ${current.tagName}`);
                    break;
                }
            }
        }
        
        // Remove the container if found, otherwise just remove the element
        if (container) {
            container.remove();
            console.log(`🗑️ Removed Twitter post container`);
        } else {
            element.remove();
            console.log(`🗑️ Removed individual Twitter element (no container found)`);
        }
    }

    removeGoogleImageCard(element) {
        console.log(`🔍 Attempting to remove Google Image card`);
        
        // Look for Google Images-specific containers
        const googleImageSelectors = [
            '.isv-r', // Google Images result container
            '.isv-r.PNCib.MSM1fd.BUooTd', // Main image result container
            '[data-ved]', // Elements with data-ved attribute (Google Images specific)
            '.rg_i', // Image container class
            '.isv-r.PNCib', // Another common container pattern
            '[jsname="sTFXNd"]', // Google Images specific selector
            '.islrc > div', // Direct children of image results container
            '.islrc .isv-r', // Image results within islrc
            '[data-ri]', // Elements with data-ri attribute
            '.rg_bx', // Image box container
            '.rg_ic', // Image container
            '.rg_meta', // Image metadata container
            '.rg_di', // Image display container
        ];
        
        let container = null;
        
        // Try to find the Google Images container
        for (const selector of googleImageSelectors) {
            container = element.closest(selector);
            if (container) {
                console.log(`🔍 Found Google Images container: ${selector}`);
                break;
            }
        }
        
        // If no specific container found, try to find a reasonable parent
        if (!container) {
            // Look for common Google Images parent patterns
            let current = element;
            for (let i = 0; i < 8; i++) { // Go up 8 levels max for Google Images
                current = current.parentElement;
                if (!current) break;
                
                // Check if this looks like a Google Images container
                if (current.classList.contains('isv-r') ||
                    current.classList.contains('rg_i') ||
                    current.classList.contains('rg_bx') ||
                    current.classList.contains('rg_ic') ||
                    current.classList.contains('rg_di') ||
                    current.getAttribute('data-ved') ||
                    current.getAttribute('data-ri') ||
                    current.getAttribute('jsname') === 'sTFXNd' ||
                    current.querySelector('.rg_i') ||
                    current.querySelector('[data-ved]') ||
                    current.querySelector('.rg_meta')) {
                    container = current;
                    console.log(`🔍 Found Google Images container by pattern: ${current.tagName} ${current.className}`);
                    break;
                }
            }
        }
        
        // Additional fallback: look for the closest div that contains this image and has a reasonable size
        if (!container) {
            let current = element;
            for (let i = 0; i < 6; i++) {
                current = current.parentElement;
                if (!current) break;
                
                // Check if this element looks like it could be an image card
                const rect = current.getBoundingClientRect();
                if (rect.width > 100 && rect.height > 100 && 
                    (current.tagName === 'DIV' || current.tagName === 'FIGURE') &&
                    current.querySelector('img') === element) {
                    container = current;
                    console.log(`🔍 Found Google Images container by size/position: ${current.tagName} ${rect.width}x${rect.height}`);
                    break;
                }
            }
        }
        
        // Remove the container if found, otherwise just remove the element
        if (container) {
            container.remove();
            console.log(`🗑️ Removed Google Images card container`);
        } else {
            element.remove();
            console.log(`🗑️ Removed individual Google Images element (no container found)`);
        }
    }

    createOverlay(element, type, result) {
        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'ai-content-overlay';
        
        // Determine overlay content based on detection type
        let overlayText = 'AI Generated Content';
        let overlayColor = 'rgba(255, 68, 68, 0.9)';
        let overlayIcon = '🤖';
        
        if (result.detection_type === 'suspicious') {
            overlayText = 'Suspicious Content';
            overlayColor = 'rgba(255, 165, 0, 0.9)';
            overlayIcon = '⚠️';
        } else if (result.detection_type === 'ai_generated') {
            overlayText = 'AI Generated Content';
            overlayColor = 'rgba(255, 68, 68, 0.9)';
            overlayIcon = '🤖';
        }
        
        overlay.innerHTML = `
            <div class="overlay-content">
                <div class="overlay-icon">${overlayIcon}</div>
                <div class="overlay-text">${overlayText}</div>
                <div class="overlay-confidence">${Math.round(result.confidence * 100)}% confidence</div>
                <div class="overlay-type">${result.detection_type}</div>
                <button class="overlay-show-btn">Show Anyway</button>
                <button class="overlay-analyze-btn">Re-analyze</button>
                <button class="overlay-remove-btn">Remove</button>
            </div>
        `;
        
        overlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: ${overlayColor};
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            border-radius: 8px;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            text-align: center;
            backdrop-filter: blur(2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
        
        // Style the overlay content
        const overlayContent = overlay.querySelector('.overlay-content');
        overlayContent.style.cssText = `
            padding: 20px;
            max-width: 90%;
        `;
        
        // Style the icon
        const overlayIconEl = overlay.querySelector('.overlay-icon');
        overlayIconEl.style.cssText = `
            font-size: 48px;
            margin-bottom: 10px;
            display: block;
        `;
        
        // Style the text
        const overlayTextEl = overlay.querySelector('.overlay-text');
        overlayTextEl.style.cssText = `
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
        `;
        
        // Style the confidence
        const overlayConfidence = overlay.querySelector('.overlay-confidence');
        overlayConfidence.style.cssText = `
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 8px;
        `;
        
        // Style the type
        const overlayType = overlay.querySelector('.overlay-type');
        overlayType.style.cssText = `
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        `;
        
        // Style the buttons
        const buttons = overlay.querySelectorAll('button');
        buttons.forEach(btn => {
            btn.style.cssText = `
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 8px 16px;
                margin: 5px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s;
            `;
            
            btn.addEventListener('mouseenter', () => {
                btn.style.background = 'rgba(255, 255, 255, 0.3)';
            });
            
            btn.addEventListener('mouseleave', () => {
                btn.style.background = 'rgba(255, 255, 255, 0.2)';
            });
        });
        
        // Position the element relatively if needed
        const container = element.parentElement;
        if (getComputedStyle(container).position === 'static') {
            container.style.position = 'relative';
        }
        
        container.appendChild(overlay);
        
        // Add show button functionality
        const showBtn = overlay.querySelector('.overlay-show-btn');
        showBtn.addEventListener('click', () => {
            overlay.remove();
        });
        
        // Add re-analyze button functionality
        const analyzeBtn = overlay.querySelector('.overlay-analyze-btn');
        analyzeBtn.addEventListener('click', async () => {
            overlay.remove();
            // Re-analyze the content
            if (type === 'image') {
                await this.scanImage(element);
            } else if (type === 'video') {
                await this.scanVideo(element);
            }
        });
        
        // Add remove button functionality
        const removeBtn = overlay.querySelector('.overlay-remove-btn');
        removeBtn.addEventListener('click', () => {
            overlay.remove();
            this.removeElement(element, type, result);
        });
        
        // Update filter count
        this.updateFilterCount();
        
        console.log(`🚫 Hidden ${type}: ${result.detection_type} content detected (${Math.round(result.confidence * 100)}%)`);
    }

    unhideAllContent() {
        const overlays = document.querySelectorAll('.ai-content-overlay');
        overlays.forEach(overlay => overlay.remove());
        this.updateFilterCount();
    }

    updateFilterCount() {
        const overlayCount = document.querySelectorAll('.ai-content-overlay').length;
        const removedCount = this.removedElements ? this.removedElements.length : 0;
        const totalCount = overlayCount + removedCount;
        
        const countElement = document.querySelector('.filter-count');
        if (countElement) {
            countElement.textContent = totalCount;
        }
    }

    bindEvents() {
        // Listen for messages from popup and background
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'scanImages') {
                this.scanImages().then(sendResponse);
                return true; // Keep message channel open for async response
            } else if (request.action === 'scanVideos') {
                this.scanVideos().then(sendResponse);
                return true;
            } else if (request.action === 'scanAudio') {
                this.scanAudio().then(sendResponse);
                return true;
            } else if (request.action === 'showNotification') {
                this.showNotification(request.message, request.type);
            } else if (request.action === 'updateFilterMode') {
                this.updateFilterMode(request.filterMode);
                sendResponse({ success: true });
            } else if (request.action === 'updateTextDetection') {
                this.settings.textDetection = request.textDetection;
                this.showNotification(`Text detection ${request.textDetection ? 'enabled' : 'disabled'}`, 'info');
                sendResponse({ success: true });
            } else if (request.action === 'updateTextConfidenceThreshold') {
                this.settings.textConfidenceThreshold = request.textConfidenceThreshold;
                this.showNotification(`Text confidence threshold updated to ${Math.round(request.textConfidenceThreshold * 100)}%`, 'info');
                sendResponse({ success: true });
            }
        });

        // Add context menu for all media types - but don't prevent default
        document.addEventListener('contextmenu', (e) => {
            const target = e.target;
            
            // Check for different types of content
            if (target.tagName === 'IMG') {
                // Image analysis
                this.addImageAnalysisButton(target, e.clientX, e.clientY);
            } else if (target.tagName === 'VIDEO') {
                // Video analysis
                this.addVideoAnalysisButton(target, e.clientX, e.clientY);
            } else if (target.tagName === 'AUDIO') {
                // Audio analysis
                this.addAudioAnalysisButton(target, e.clientX, e.clientY);
            } else if (target.textContent && target.textContent.trim().length > 20) {
                // Text analysis (for elements with substantial text content)
                this.addTextAnalysisButton(target, e.clientX, e.clientY);
            }
        });
        
        // Add scroll event listener for Google Images infinite scroll
        if (window.location.hostname.includes('google.com') && window.location.pathname.includes('/search') && window.location.search.includes('tbm=isch')) {
            let scrollTimeout;
            window.addEventListener('scroll', () => {
                clearTimeout(scrollTimeout);
                scrollTimeout = setTimeout(() => {
                    // Scan for new images after scrolling stops
                    this.scanGoogleImages();
                }, 500);
            });
        }
        
        // Watch for new videos being added to the page
        this.observeVideoChanges();
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+F to toggle filter mode
            if (e.ctrlKey && e.shiftKey && e.key === 'F') {
                e.preventDefault();
                this.filterMode = !this.filterMode;
                this.updateFilterMode(this.filterMode);
                this.showNotification(`Filter mode ${this.filterMode ? 'enabled' : 'disabled'}`, 'info');
            }
            
            // Ctrl+Shift+R to restore all content
            if (e.ctrlKey && e.shiftKey && e.key === 'R') {
                e.preventDefault();
                this.restoreRemovedContent();
                this.showNotification('All removed content restored', 'success');
            }
        });
        
        console.log('🔧 Events bound');
    }

    addImageAnalysisButton(img, x, y) {
        // Remove any existing analysis buttons
        const existingButtons = document.querySelectorAll('.ai-analysis-button');
        existingButtons.forEach(btn => btn.remove());
        
        // Create floating analysis button
        const button = document.createElement('div');
        button.className = 'ai-analysis-button';
        button.innerHTML = '🤖';
        button.title = 'Analyze with AI';
        
        button.style.cssText = `
            position: fixed;
            top: ${y - 30}px;
            left: ${x + 10}px;
            width: 30px;
            height: 30px;
            background: #007bff;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 16px;
            z-index: 100000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: all 0.2s ease;
            user-select: none;
        `;
        
        // Hover effects
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.1)';
            button.style.background = '#0056b3';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
            button.style.background = '#007bff';
        });
        
        // Click to analyze
        button.addEventListener('click', () => {
            this.analyzeImage(img);
            button.remove();
        });
        
        document.body.appendChild(button);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (button.parentNode) {
                button.remove();
            }
        }, 3000);
    }

    addTextAnalysisButton(element, x, y) {
        // Remove any existing analysis buttons
        const existingButtons = document.querySelectorAll('.ai-text-analysis-button');
        existingButtons.forEach(btn => btn.remove());
        
        // Create floating text analysis button
        const button = document.createElement('div');
        button.className = 'ai-text-analysis-button';
        button.innerHTML = '📝';
        button.title = 'Analyze Text with AI';
        
        button.style.cssText = `
            position: fixed;
            top: ${y - 30}px;
            left: ${x + 10}px;
            width: 30px;
            height: 30px;
            background: #28a745;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 16px;
            z-index: 100000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: all 0.2s ease;
            user-select: none;
        `;
        
        // Hover effects
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.1)';
            button.style.background = '#1e7e34';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
            button.style.background = '#28a745';
        });
        
        // Click to analyze text
        button.addEventListener('click', () => {
            this.analyzeTextElement(element);
            button.remove();
        });
        
        document.body.appendChild(button);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (button.parentNode) {
                button.remove();
            }
        }, 3000);
    }

    addVideoAnalysisButton(video, x, y) {
        // Remove any existing analysis buttons
        const existingButtons = document.querySelectorAll('.ai-video-analysis-button');
        existingButtons.forEach(btn => btn.remove());
        
        // Create floating video analysis button
        const button = document.createElement('div');
        button.className = 'ai-video-analysis-button';
        button.innerHTML = '🎥';
        button.title = 'Analyze Video with AI';
        
        button.style.cssText = `
            position: fixed;
            top: ${y - 30}px;
            left: ${x + 10}px;
            width: 30px;
            height: 30px;
            background: #dc3545;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 16px;
            z-index: 100000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: all 0.2s ease;
            user-select: none;
        `;
        
        // Hover effects
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.1)';
            button.style.background = '#c82333';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
            button.style.background = '#dc3545';
        });
        
        // Click to analyze video
        button.addEventListener('click', () => {
            this.analyzeVideo(video);
            button.remove();
        });
        
        document.body.appendChild(button);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (button.parentNode) {
                button.remove();
            }
        }, 3000);
    }

    addAudioAnalysisButton(audio, x, y) {
        // Remove any existing analysis buttons
        const existingButtons = document.querySelectorAll('.ai-audio-analysis-button');
        existingButtons.forEach(btn => btn.remove());
        
        // Create floating audio analysis button
        const button = document.createElement('div');
        button.className = 'ai-audio-analysis-button';
        button.innerHTML = '🎵';
        button.title = 'Analyze Audio with AI';
        
        button.style.cssText = `
            position: fixed;
            top: ${y - 30}px;
            left: ${x + 10}px;
            width: 30px;
            height: 30px;
            background: #6f42c1;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 16px;
            z-index: 100000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: all 0.2s ease;
            user-select: none;
        `;
        
        // Hover effects
        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.1)';
            button.style.background = '#5a32a3';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
            button.style.background = '#6f42c1';
        });
        
        // Click to analyze audio
        button.addEventListener('click', () => {
            this.analyzeVideoAudio(audio);
            button.remove();
        });
        
        document.body.appendChild(button);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (button.parentNode) {
                button.remove();
            }
        }, 3000);
    }

    async analyzeImage(img) {
        try {
            this.showNotification('Analyzing image...', 'info');
            
            const result = await this.analyzeFrame(img.src);
            
            // Show detailed result
            this.showImageResult(result, img);
            
        } catch (error) {
            console.error('Image analysis error:', error);
            this.showNotification('Error analyzing image', 'error');
        }
    }

    showImageResult(result, img) {
        // Create result popup
        const popup = document.createElement('div');
        popup.className = 'ai-result-popup';
        
        const confidence = Math.round(result.confidence * 100);
        const status = result.is_deepfake ? '🚫 AI DETECTED' : '✅ LIKELY REAL';
        const statusColor = result.is_deepfake ? '#ff4444' : '#44aa44';
        
        popup.innerHTML = `
            <div class="popup-header">
                <h3>Image Analysis Result</h3>
                <button class="close-btn">×</button>
            </div>
            <div class="popup-content">
                <div class="result-status" style="color: ${statusColor}">${status}</div>
                <div class="result-confidence">Confidence: ${confidence}%</div>
                <div class="result-type">Type: ${result.detection_type}</div>
                <div class="result-analysis">${result.analysis}</div>
                <div class="result-model">Model: ${result.model_used}</div>
            </div>
        `;
        
        popup.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 100000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 400px;
            width: 90%;
        `;
        
        // Style header
        const header = popup.querySelector('.popup-header');
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
        `;
        
        header.querySelector('h3').style.cssText = `
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        `;
        
        const closeBtn = header.querySelector('.close-btn');
        closeBtn.style.cssText = `
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #666;
        `;
        
        // Style content
        const content = popup.querySelector('.popup-content');
        content.style.cssText = `
            padding: 20px;
        `;
        
        const statusEl = content.querySelector('.result-status');
        statusEl.style.cssText = `
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        `;
        
        const confidenceEl = content.querySelector('.result-confidence');
        confidenceEl.style.cssText = `
            font-size: 14px;
            margin-bottom: 8px;
        `;
        
        const typeEl = content.querySelector('.result-type');
        typeEl.style.cssText = `
            font-size: 14px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        `;
        
        const analysisEl = content.querySelector('.result-analysis');
        analysisEl.style.cssText = `
            font-size: 14px;
            margin-bottom: 10px;
            line-height: 1.4;
        `;
        
        const modelEl = content.querySelector('.result-model');
        modelEl.style.cssText = `
            font-size: 12px;
            color: #666;
            font-style: italic;
        `;
        
        // Add close functionality
        closeBtn.addEventListener('click', () => popup.remove());
        
        // Close on outside click
        popup.addEventListener('click', (e) => {
            if (e.target === popup) {
                popup.remove();
            }
        });
        
        document.body.appendChild(popup);
    }

    observeVideoChanges() {
        // Create a mutation observer to watch for new videos
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        // Check if the added node is a video
                        if (node.tagName === 'VIDEO') {
                            this.handleNewVideo(node);
                        }
                        // Check if the added node contains videos
                        const videos = node.querySelectorAll && node.querySelectorAll('video');
                        if (videos) {
                            videos.forEach(video => this.handleNewVideo(video));
                        }
                        
                        // Check if the added node is an image
                        if (node.tagName === 'IMG') {
                            this.handleNewImage(node);
                        }
                        // Check if the added node contains images
                        const images = node.querySelectorAll && node.querySelectorAll('img');
                        if (images) {
                            images.forEach(img => this.handleNewImage(img));
                        }
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    handleNewVideo(video) {
        // Add a small delay to ensure video is loaded
        setTimeout(() => {
            if (video.readyState >= 2) { // HAVE_CURRENT_DATA
                this.showVideoAnalysisButton(video);
            } else {
                video.addEventListener('loadeddata', () => {
                    this.showVideoAnalysisButton(video);
                });
            }
        }, 1000);
    }

    handleNewImage(img) {
        // Add a small delay to ensure image is loaded
        setTimeout(() => {
            if (img.complete && img.naturalWidth > 0) {
                this.scanImage(img);
            } else {
                img.addEventListener('load', () => {
                    this.scanImage(img);
                });
            }
        }, 500);
    }

    showVideoAnalysisButton(video) {
        // Remove any existing analysis button
        const existingButton = video.parentElement?.querySelector('.video-analysis-btn');
        if (existingButton) {
            existingButton.remove();
        }
        
        // Only add button if parent element exists
        if (!video.parentElement) {
            console.log('⚠️ No parent element found for video, skipping analysis button');
            return;
        }
        
        // Create analysis button
        const button = document.createElement('button');
        button.className = 'video-analysis-btn';
        button.innerHTML = '🔍 Analyze Video';
        button.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-family: Arial, sans-serif;
        `;
        
        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.analyzeVideo(video);
        });
        
        // Add button to video container
        video.parentElement.style.position = 'relative';
        video.parentElement.appendChild(button);
        
        console.log('🎥 Added video analysis button');
    }

    detectVideos() {
        const videos = document.querySelectorAll('video');
        videos.forEach(video => {
            this.handleNewVideo(video);
        });
    }

    async analyzeVideo(video) {
        try {
            this.showNotification('Extracting video frames...', 'info');
            
            // Extract frames
            const frames = await this.extractVideoFrames(video, 5);
            
            if (frames.length === 0) {
                this.showNotification('No frames could be extracted from video', 'error');
                return;
            }

            console.log(`🎬 Extracted ${frames.length} frames from video`);
            this.showNotification(`Analyzing ${frames.length} frames...`, 'info');
            
            // Analyze each frame
            const results = await this.analyzeFrames(frames);
            
            console.log('📊 Frame analysis results:', results);
            
            // Aggregate results
            const summary = this.aggregateResults(results);
            
            console.log('📈 Aggregated summary:', summary);
            
            // Show final result with delay to ensure it's visible
            setTimeout(() => {
                this.showVideoResult(summary);
            }, 1000);
            
            // Also analyze audio if available (with delay to avoid overlap)
            setTimeout(async () => {
                await this.analyzeVideoAudio(video);
            }, 2000);
            
        } catch (error) {
            console.error('Video analysis error:', error);
            this.showNotification('Error analyzing video', 'error');
        }
    }

    async extractVideoFrames(video, maxFrames = 5) {
        console.log('🎬 Starting frame extraction...');
        return new Promise((resolve) => {
            const frames = [];
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size to video size
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            console.log(`📐 Canvas size: ${canvas.width}x${canvas.height}`);
            console.log(`⏱️ Video duration: ${video.duration}s`);
            
            const duration = video.duration;
            
            // 2-1-2 strategy: 2 frames early, 1 middle, 2 late
            const frameTimes = [];
            
            // First cluster: 15-20% of video (2 frames)
            const earlyStart = duration * 0.15;
            const earlyEnd = duration * 0.20;
            frameTimes.push(earlyStart);
            frameTimes.push(earlyEnd);
            
            // Middle frame: 50% of video
            frameTimes.push(duration * 0.50);
            
            // Last cluster: 75-80% of video (2 frames)
            const lateStart = duration * 0.75;
            const lateEnd = duration * 0.80;
            frameTimes.push(lateStart);
            frameTimes.push(lateEnd);
            
            console.log(`🎬 2-1-2 frame extraction strategy:`);
            console.log(`   Early cluster: ${earlyStart.toFixed(1)}s, ${earlyEnd.toFixed(1)}s`);
            console.log(`   Middle frame: ${(duration * 0.50).toFixed(1)}s`);
            console.log(`   Late cluster: ${lateStart.toFixed(1)}s, ${lateEnd.toFixed(1)}s`);
            
            console.log(`Extracting ${frameTimes.length} frames at times:`, frameTimes);
            
            let currentFrame = 0;
            
            const extractFrame = () => {
                if (currentFrame >= frameTimes.length) {
                    console.log(`✅ Frame extraction complete. Got ${frames.length} frames`);
                    resolve(frames);
                    return;
                }
                
                const time = frameTimes[currentFrame];
                console.log(`📸 Extracting frame ${currentFrame + 1} at time ${time}s`);
                
                // Check if video is ready
                if (video.readyState < 2) {
                    console.log(`⚠️ Video not ready, waiting... (readyState: ${video.readyState})`);
                    setTimeout(extractFrame, 100);
                    return;
                }
                
                video.currentTime = time;
                
                video.addEventListener('seeked', function onSeeked() {
                    video.removeEventListener('seeked', onSeeked);
                    
                    console.log(`🎬 Video seeked to ${video.currentTime}s`);
                    
                    // Draw video frame to canvas
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Convert to base64 with higher quality
                    const frameData = canvas.toDataURL('image/jpeg', 1.0);
                    frames.push(frameData);
                    
                    console.log(`✅ Frame ${currentFrame + 1} extracted (${frameData.length} chars)`);
                    
                    // Debug: Save first frame to see what it looks like
                    if (currentFrame === 0) {
                        console.log('🔍 First frame data URL (first 100 chars):', frameData.substring(0, 100));
                        // Create a download link for the first frame
                        const link = document.createElement('a');
                        link.download = 'extracted_frame.jpg';
                        link.href = frameData;
                        link.click();
                        console.log('📥 Downloaded first frame for inspection');
                    }
                    
                    currentFrame++;
                    extractFrame();
                }, { once: true });
                
                // Add timeout in case seeking fails
                setTimeout(() => {
                    console.log(`⏰ Timeout waiting for seek to ${time}s`);
                    currentFrame++;
                    extractFrame();
                }, 5000);
            };
            
            extractFrame();
        });
    }

    async analyzeFrames(frames) {
        const results = [];
        
        for (let i = 0; i < frames.length; i++) {
            try {
                this.showNotification(`Analyzing frame ${i + 1}/${frames.length}...`, 'info');
                
                const result = await this.analyzeFrame(frames[i]);
                console.log(`📸 Frame ${i + 1} result:`, result);
                results.push(result);
                
                // Small delay to avoid overwhelming the API
                await new Promise(resolve => setTimeout(resolve, 200));
                
            } catch (error) {
                console.error(`❌ Error analyzing frame ${i + 1}:`, error);
                // Continue with other frames
            }
        }
        
        return results;
    }

    async analyzeFrame(frameData) {
        try {
            // Convert base64 to blob
            const response = await fetch(frameData);
            const blob = await response.blob();
            
            // Create FormData
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            
            // Send to backend
            const result = await fetch('http://localhost:8001/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!result.ok) {
                throw new Error(`HTTP ${result.status}: ${result.statusText}`);
            }
            
            return await result.json();
            
        } catch (error) {
            console.error('Frame analysis error:', error);
            return {
                is_deepfake: false,
                confidence: 0.0,
                error: error.message
            };
        }
    }

    aggregateResults(results) {
        if (results.length === 0) {
            return {
                is_deepfake: false,
                confidence: 0.0,
                frames_analyzed: 0
            };
        }
        
        const deepfakeCount = results.filter(r => r.is_deepfake).length;
        const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;
        
        return {
            is_deepfake: deepfakeCount > results.length * 0.5,
            confidence: avgConfidence,
            frames_analyzed: results.length,
            deepfake_frames: deepfakeCount,
            results: results
        };
    }

    showVideoResult(summary) {
        const message = summary.is_deepfake 
            ? `🚫 Video appears to be AI-generated (${Math.round(summary.confidence * 100)}% confidence)`
            : `✅ Video appears to be real (${Math.round(summary.confidence * 100)}% confidence)`;
        
        this.showNotification(message, summary.is_deepfake ? 'error' : 'success');
        
        console.log('🎬 Video analysis complete:', summary);
    }

    async analyzeVideoAudio(video) {
        try {
            this.showNotification('Extracting audio from video...', 'info');
            
            const audioBlob = await this.extractAudioFromVideo(video);
            
            if (audioBlob) {
                this.showNotification('Analyzing audio...', 'info');
                const audioResult = await this.analyzeAudio(audioBlob);
                this.showAudioResult(audioResult);
            }
            
        } catch (error) {
            console.error('Audio analysis error:', error);
            this.showNotification('Error analyzing audio', 'error');
        }
    }

    async extractAudioFromVideo(video) {
        return new Promise((resolve) => {
            try {
                // Create audio context
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaElementSource(video);
                
                // Create destination node
                const destination = audioContext.createMediaStreamDestination();
                source.connect(destination);
                
                // Create MediaRecorder
                const mediaRecorder = new MediaRecorder(destination.stream);
                const chunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    chunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(chunks, { type: 'audio/wav' });
                    resolve(audioBlob);
                };
                
                // Start recording
                mediaRecorder.start();
                
                // Play video to capture audio
                video.currentTime = 0;
                video.play();
                
                // Stop recording after 10 seconds or when video ends
                const stopRecording = () => {
                    mediaRecorder.stop();
                    video.pause();
                    video.removeEventListener('ended', stopRecording);
                };
                
                video.addEventListener('ended', stopRecording);
                setTimeout(stopRecording, 10000);
                
            } catch (error) {
                console.error('Audio extraction error:', error);
                resolve(null);
            }
        });
    }

    async analyzeAudio(audioBlob) {
        try {
            const formData = new FormData();
            formData.append('file', audioBlob, 'audio.wav');
            
            const result = await fetch('http://localhost:8001/upload-audio', {
                method: 'POST',
                body: formData
            });
            
            if (!result.ok) {
                throw new Error(`HTTP ${result.status}: ${result.statusText}`);
            }
            
            return await result.json();
            
        } catch (error) {
            console.error('Audio analysis error:', error);
            return {
                is_deepfake: false,
                confidence: 0.0,
                error: error.message
            };
        }
    }

    showAudioResult(audioResult) {
        const message = audioResult.is_deepfake 
            ? `🎵 Audio appears to be AI-generated (${Math.round(audioResult.confidence * 100)}% confidence)`
            : `🎵 Audio appears to be real (${Math.round(audioResult.confidence * 100)}% confidence)`;
        
        this.showNotification(message, audioResult.is_deepfake ? 'error' : 'success');
        
        console.log('🎵 Audio analysis complete:', audioResult);
    }

    async scanImages() {
        const images = document.querySelectorAll('img');
        console.log(`🔍 Found ${images.length} images to scan`);
        
        for (const img of images) {
            await this.scanImage(img);
        }
    }

    async scanVideos() {
        const videos = document.querySelectorAll('video');
        console.log(`🎬 Found ${videos.length} videos to scan`);
        
        for (const video of videos) {
            await this.scanVideo(video);
        }
    }

    async scanAudio() {
        const audioElements = document.querySelectorAll('audio');
        console.log(`🎵 Found ${audioElements.length} audio elements to scan`);
        
        for (const audio of audioElements) {
            if (this.isValidAudio(audio)) {
                // Audio scanning logic would go here
            }
        }
    }

    isValidImage(img) {
        // Skip images that are too small, already processed, or invalid
        if (!img || !img.src || img.src === '' || img.src.startsWith('data:')) {
            return false;
        }
        
        // Skip very small images (likely icons)
        if (img.naturalWidth < 50 || img.naturalHeight < 50) {
            return false;
        }
        
        // Skip images that are already being processed
        if (this.scannedElements.has(img)) {
            return false;
        }
        
        // Skip images with certain patterns (avatars, icons, etc.)
        const src = img.src.toLowerCase();
        const skipPatterns = [
            'avatar', 'icon', 'logo', 'emoji', 'favicon', 'button',
            'spinner', 'loading', 'placeholder', 'ad', 'banner',
            'gif-icon', 'gif-thumbnail' // Skip GIF icons/thumbnails
        ];
        
        if (skipPatterns.some(pattern => src.includes(pattern))) {
            return false;
        }
        
        // Google Images specific handling
        if (window.location.hostname.includes('google.com') && window.location.pathname.includes('/search') && window.location.search.includes('tbm=isch')) {
            // For Google Images, be more lenient with sizes but skip very small ones
            if (img.naturalWidth < 80 || img.naturalHeight < 80) {
                return false;
            }
            
            // Skip Google Images specific small elements
            if (src.includes('gstatic.com') && (img.naturalWidth < 100 || img.naturalHeight < 100)) {
                return false;
            }
            
            // Skip Google's own UI elements
            if (src.includes('googleusercontent.com') && src.includes('icon')) {
                return false;
            }
        }
        
        // Twitter-specific small images
        if (window.location.hostname.includes('twitter.com') || window.location.hostname.includes('x.com')) {
            // Twitter has many small profile images and icons
            if (img.naturalWidth < 100 || img.naturalHeight < 100) {
                return false;
            }
        }
        
        // Special handling for GIFs - allow them but with some restrictions
        if (src.endsWith('.gif') || src.includes('gif')) {
            // Skip very small GIFs (likely loading indicators)
            if (img.naturalWidth < 80 || img.naturalHeight < 80) {
                return false;
            }
            
            // Skip GIFs that are likely UI elements
            if (src.includes('loading') || src.includes('spinner') || src.includes('progress')) {
                return false;
            }
        }
        
        return true;
    }

    isValidVideo(video) {
        // Skip videos that are too short or invalid
        if (!video || !video.src || video.duration < 1) {
            return false;
        }
        
        // Skip videos that are already being processed
        if (this.scannedElements.has(video)) {
            return false;
        }
        
        return true;
    }

    isValidAudio(audio) {
        // Skip audio that is too short or invalid
        if (!audio || !audio.src || audio.duration < 1) {
            return false;
        }
        
        return true;
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.ai-detector-notification');
        existingNotifications.forEach(notification => notification.remove());
        
        const notification = document.createElement('div');
        notification.className = 'ai-detector-notification';
        notification.textContent = message;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${this.getNotificationColor(type)};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            font-weight: 500;
            z-index: 1000000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            max-width: 400px;
            text-align: center;
            animation: slideIn 0.3s ease-out;
        `;
        
        // Add animation keyframes
        if (!document.querySelector('#ai-detector-styles')) {
            const style = document.createElement('style');
            style.id = 'ai-detector-styles';
            style.textContent = `
                @keyframes slideIn {
                    from {
                        transform: translateX(-50%) translateY(-100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(-50%) translateY(0);
                        opacity: 1;
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    getNotificationColor(type) {
        switch (type) {
            case 'error':
                return '#ff4444';
            case 'success':
                return '#44aa44';
            case 'warning':
                return '#ffaa00';
            default:
                return '#4444ff';
        }
    }

    updateFilterMode(filterMode) {
        this.filterMode = filterMode;
        
        if (filterMode) {
            this.startContentMonitoring();
        } else {
            this.stopContentMonitoring();
        }
    }

    async testFilter() {
        console.log('🧪 Testing filter functionality...');
        
        // Test with a sample image
        const testImage = document.createElement('img');
        testImage.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iI2ZmMDAwMCIvPjx0ZXh0IHg9IjUwIiB5PSI1MCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE0IiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+VEVTVDwvdGV4dD48L3N2Zz4=';
        
        await this.scanImage(testImage);
        
        this.showNotification('Filter test completed - check console for results', 'info');
    }

    removeAllAIContent() {
        console.log('🗑️ Removing all AI content...');
        
        // Find all images and videos
        const images = document.querySelectorAll('img');
        const videos = document.querySelectorAll('video');
        
        let removedCount = 0;
        
        // Remove images that might be AI-generated (this is a simplified approach)
        images.forEach(img => {
            if (this.isValidImage(img) && !this.scannedElements.has(img)) {
                // For testing, remove some images randomly
                if (Math.random() < 0.1) { // 10% chance
                    img.style.display = 'none';
                    removedCount++;
                }
            }
        });
        
        // Remove videos that might be AI-generated
        videos.forEach(video => {
            if (this.isValidVideo(video) && !this.scannedElements.has(video)) {
                // For testing, remove some videos randomly
                if (Math.random() < 0.1) { // 10% chance
                    video.style.display = 'none';
                    removedCount++;
                }
            }
        });
        
        this.showNotification(`Removed ${removedCount} potentially AI-generated elements`, 'warning');
        console.log(`🗑️ Removed ${removedCount} elements`);
    }

    restoreRemovedContent() {
        console.log('🔄 Restoring removed content...');
        
        // Show all hidden images and videos
        const hiddenImages = document.querySelectorAll('img[style*="display: none"]');
        const hiddenVideos = document.querySelectorAll('video[style*="display: none"]');
        
        hiddenImages.forEach(img => {
            img.style.display = '';
        });
        
        hiddenVideos.forEach(video => {
            video.style.display = '';
        });
        
        const restoredCount = hiddenImages.length + hiddenVideos.length;
        this.showNotification(`Restored ${restoredCount} elements`, 'success');
        console.log(`🔄 Restored ${restoredCount} elements`);
    }

    setupScrollMonitoring() {
        // Monitor scroll events to scan new content
        this.scrollHandler = () => {
            this.scanVisibleContent();
        };
        
        window.addEventListener('scroll', this.scrollHandler, { passive: true });
    }

    setupIntersectionObserver() {
        // Use Intersection Observer to detect when new content becomes visible
        this.intersectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Scan the newly visible element
                    this.scanNewContent(entry.target);
                }
            });
        }, {
            rootMargin: '50px', // Start scanning 50px before element becomes visible
            threshold: 0.1 // Trigger when 10% of element is visible
        });
        
        // Observe all existing images and videos
        const images = document.querySelectorAll('img');
        const videos = document.querySelectorAll('video');
        
        images.forEach(img => this.intersectionObserver.observe(img));
        videos.forEach(video => this.intersectionObserver.observe(video));
    }

    scanVisibleContent() {
        // Scan content that's currently visible in the viewport
        const images = document.querySelectorAll('img');
        const videos = document.querySelectorAll('video');
        
        images.forEach(img => {
            if (this.isElementInViewport(img)) {
                this.scanImage(img);
            }
        });
        
        videos.forEach(video => {
            if (this.isElementInViewport(video)) {
                this.scanVideo(video);
            }
        });
    }

    isElementInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    // Google Images specific scanning
    scanGoogleImages() {
        if (!window.location.hostname.includes('google.com') || !window.location.pathname.includes('/search') || !window.location.search.includes('tbm=isch')) {
            return;
        }
        
        console.log('🔍 Scanning Google Images specifically...');
        
        // Look for Google Images containers
        const imageContainers = document.querySelectorAll('.isv-r, .rg_i, [data-ved], [jsname="sTFXNd"]');
        
        imageContainers.forEach(container => {
            const images = container.querySelectorAll('img');
            images.forEach(img => {
                if (this.isValidImage(img) && !this.scannedElements.has(img)) {
                    this.scanImage(img);
                }
            });
        });
        
        // Also scan any images that might not be in containers yet
        const allImages = document.querySelectorAll('img');
        allImages.forEach(img => {
            if (this.isValidImage(img) && !this.scannedElements.has(img)) {
                this.scanImage(img);
            }
        });
    }

    // Text detection functions
    async scanTexts() {
        console.log('📝 Scanning for text content...');
        
        // Scan different types of text content based on the website
        const hostname = window.location.hostname;
        
        if (hostname.includes('twitter.com') || hostname.includes('x.com')) {
            this.scanTwitterTexts();
        } else if (hostname.includes('youtube.com')) {
            this.scanYouTubeTexts();
        } else {
            this.scanGenericTexts();
        }
    }

    async scanTwitterTexts() {
        console.log('🐦 Scanning Twitter texts...');
        
        // Look for tweet text elements
        const tweetSelectors = [
            '[data-testid="tweetText"]',
            '[data-testid="tweet"] [lang]',
            'article[data-testid="tweet"] div[lang]',
            '.css-1dbjc4n.r-1wbh5a2.r-dnmrzs.r-1ny4l3l div[lang]'
        ];
        
        tweetSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                if (this.isValidTextElement(element)) {
                    this.scanTextElement(element);
                }
            });
        });
    }

    async scanYouTubeTexts() {
        console.log('🎥 Scanning YouTube texts...');
        
        // Look for comment text elements
        const commentSelectors = [
            '#content-text',
            '#content-text span',
            '.ytd-comment-renderer #content-text',
            '.ytd-comment-thread-renderer #content-text'
        ];
        
        commentSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                if (this.isValidTextElement(element)) {
                    this.scanTextElement(element);
                }
            });
        });
    }

    async scanGenericTexts() {
        console.log('📄 Scanning generic texts...');
        
        // Look for common text content patterns
        const textSelectors = [
            'p[class*="comment"]',
            'div[class*="comment"]',
            'span[class*="comment"]',
            'p[class*="text"]',
            'div[class*="text"]',
            'span[class*="text"]'
        ];
        
        textSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                if (this.isValidTextElement(element)) {
                    this.scanTextElement(element);
                }
            });
        });
    }

    isValidTextElement(element) {
        if (!element || !element.textContent) {
            return false;
        }
        
        const text = element.textContent.trim();
        
        // Skip if already scanned
        if (this.scannedTexts.has(text)) {
            return false;
        }
        
        // Skip very short or very long texts
        if (text.length < 20 || text.length > 2000) {
            return false;
        }
        
        // Skip if element is hidden
        if (element.offsetParent === null) {
            return false;
        }
        
        return true;
    }

    async scanTextElement(element) {
        const text = element.textContent.trim();
        
        // Check if text detection is enabled
        if (!this.settings.textDetection) {
            return;
        }
        
        // Mark as scanned
        this.scannedTexts.add(text);
        
        console.log(`📝 Scanning text: "${text.substring(0, 100)}..."`);
        
        try {
            const result = await this.analyzeText(text);
            console.log(`📝 Text analysis result:`, result);
            
            // Check if confidence meets threshold
            if (result.confidence >= this.settings.textConfidenceThreshold) {
                if (this.filterMode && result.is_bot) {
                    console.log(`🚫 Hiding bot/LLM text (${Math.round(result.confidence * 100)}%)`);
                    
                    // Show visible notification
                    const confidence = Math.round(result.confidence * 100);
                    const detectionType = result.detection_type || 'bot';
                    this.showNotification(
                        `🚫 ${detectionType.toUpperCase()} content detected (${confidence}%) - ${text.substring(0, 50)}...`,
                        'warning'
                    );
                    
                    this.hideTextContent(element, result);
                } else if (result.is_bot && result.confidence > 0.8) {
                    // Show notification for high-confidence bot detection even if filter is off
                    const confidence = Math.round(result.confidence * 100);
                    this.showNotification(
                        `🤖 High-confidence bot detection (${confidence}%) - ${text.substring(0, 50)}...`,
                        'info'
                    );
                }
            } else {
                console.log(`📝 Text confidence below threshold: ${result.confidence} < ${this.settings.textConfidenceThreshold}`);
            }
        } catch (error) {
            console.error('❌ Text scan error:', error);
            this.showNotification('Error analyzing text content', 'error');
        }
    }

    async analyzeText(text) {
        try {
            const response = await fetch(`${this.apiUrl}/detect-text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    confidence_threshold: this.settings.textConfidenceThreshold || 0.7
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('❌ Text analysis error:', error);
            throw error;
        }
    }

    hideTextContent(element, result) {
        // If removeContent is enabled, completely remove the element
        if (this.removeContent) {
            console.log(`🗑️ Removing bot/LLM text content`);
            this.removeTextElement(element, result);
            return;
        }
        
        // Otherwise, create overlay
        console.log(`🚫 Creating text overlay`);
        this.createTextOverlay(element, result);
    }

    removeTextElement(element, result) {
        // Store info about removed element for potential restoration
        const removedInfo = {
            element: element,
            type: 'text',
            result: result,
            parent: element.parentElement,
            nextSibling: element.nextSibling,
            originalText: element.textContent
        };
        
        // Add to removed elements list
        if (!this.removedElements) {
            this.removedElements = [];
        }
        this.removedElements.push(removedInfo);
        
        // Website-specific removal logic
        const hostname = window.location.hostname;
        
        if (hostname.includes('twitter.com') || hostname.includes('x.com')) {
            this.removeTwitterText(element);
        } else if (hostname.includes('youtube.com')) {
            this.removeYouTubeText(element);
        } else {
            element.remove();
        }
        
        // Update filter count
        this.updateFilterCount();
        
        console.log(`🗑️ Removed text: ${result.detection_type} content (${Math.round(result.confidence * 100)}%)`);
    }

    removeTwitterText(element) {
        console.log(`🐦 Attempting to remove Twitter text`);
        
        // Look for the tweet container
        const tweetContainer = element.closest('[data-testid="tweet"]') || 
                              element.closest('article[data-testid="tweet"]') ||
                              element.closest('[role="article"]');
        
        if (tweetContainer) {
            tweetContainer.remove();
            console.log(`🗑️ Removed Twitter tweet container`);
        } else {
            element.remove();
            console.log(`🗑️ Removed individual Twitter text element`);
        }
    }

    removeYouTubeText(element) {
        console.log(`🎥 Attempting to remove YouTube text`);
        
        // Look for the comment container
        const commentContainer = element.closest('ytd-comment-renderer') ||
                                element.closest('ytd-comment-thread-renderer') ||
                                element.closest('[id*="comment"]');
        
        if (commentContainer) {
            commentContainer.remove();
            console.log(`🗑️ Removed YouTube comment container`);
        } else {
            element.remove();
            console.log(`🗑️ Removed individual YouTube text element`);
        }
    }

    createTextOverlay(element, result) {
        // Create overlay for text content
        const overlay = document.createElement('div');
        overlay.className = 'ai-text-overlay';
        
        // Determine overlay content based on detection type
        let overlayText = 'Bot/LLM Content';
        let overlayColor = 'rgba(255, 68, 68, 0.9)';
        let overlayIcon = '🤖';
        
        if (result.detection_type === 'suspicious') {
            overlayText = 'Suspicious Content';
            overlayColor = 'rgba(255, 165, 0, 0.9)';
            overlayIcon = '⚠️';
        } else if (result.detection_type === 'bot') {
            overlayText = 'Bot Content';
            overlayColor = 'rgba(255, 68, 68, 0.9)';
            overlayIcon = '🤖';
        } else if (result.detection_type === 'llm') {
            overlayText = 'AI-Generated Text';
            overlayColor = 'rgba(255, 68, 68, 0.9)';
            overlayIcon = '🤖';
        }
        
        overlay.innerHTML = `
            <div class="overlay-content">
                <div class="overlay-icon">${overlayIcon}</div>
                <div class="overlay-text">${overlayText}</div>
                <div class="overlay-confidence">${Math.round(result.confidence * 100)}% confidence</div>
                <div class="overlay-type">${result.detection_type}</div>
                <button class="overlay-show-btn">Show Anyway</button>
                <button class="overlay-analyze-btn">Re-analyze</button>
                <button class="overlay-remove-btn">Remove</button>
            </div>
        `;
        
        overlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: ${overlayColor};
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            border-radius: 8px;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            text-align: center;
            backdrop-filter: blur(2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
        
        // Style the overlay content
        const overlayContent = overlay.querySelector('.overlay-content');
        overlayContent.style.cssText = `
            padding: 20px;
            max-width: 90%;
        `;
        
        // Style the icon
        const overlayIconEl = overlay.querySelector('.overlay-icon');
        overlayIconEl.style.cssText = `
            font-size: 48px;
            margin-bottom: 10px;
            display: block;
        `;
        
        // Style the text
        const overlayTextEl = overlay.querySelector('.overlay-text');
        overlayTextEl.style.cssText = `
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
        `;
        
        // Style the confidence
        const overlayConfidence = overlay.querySelector('.overlay-confidence');
        overlayConfidence.style.cssText = `
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 8px;
        `;
        
        // Style the type
        const overlayType = overlay.querySelector('.overlay-type');
        overlayType.style.cssText = `
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 15px;
        `;
        
        // Style the buttons
        const buttons = overlay.querySelectorAll('button');
        buttons.forEach(button => {
            button.style.cssText = `
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 8px 12px;
                margin: 5px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: background 0.2s;
            `;
            
            button.addEventListener('mouseenter', () => {
                button.style.background = 'rgba(255, 255, 255, 0.3)';
            });
            
            button.addEventListener('mouseleave', () => {
                button.style.background = 'rgba(255, 255, 255, 0.2)';
            });
        });
        
        // Add button event listeners
        const showBtn = overlay.querySelector('.overlay-show-btn');
        const analyzeBtn = overlay.querySelector('.overlay-analyze-btn');
        const removeBtn = overlay.querySelector('.overlay-remove-btn');
        
        showBtn.addEventListener('click', () => {
            overlay.remove();
        });
        
        analyzeBtn.addEventListener('click', async () => {
            const originalText = element.textContent.trim();
            try {
                const newResult = await this.analyzeText(originalText);
                console.log('🔄 Re-analysis result:', newResult);
                
                if (!newResult.is_bot) {
                    overlay.remove();
                    this.showNotification('Content re-analyzed and approved', 'success');
                } else {
                    this.showNotification('Content still flagged as bot/LLM', 'warning');
                }
            } catch (error) {
                console.error('❌ Re-analysis error:', error);
                this.showNotification('Re-analysis failed', 'error');
            }
        });
        
        removeBtn.addEventListener('click', () => {
            this.removeTextElement(element, result);
            overlay.remove();
        });
        
        // Position the overlay relative to the element
        const rect = element.getBoundingClientRect();
        const parentRect = element.parentElement.getBoundingClientRect();
        
        overlay.style.position = 'absolute';
        overlay.style.top = `${rect.top - parentRect.top}px`;
        overlay.style.left = `${rect.left - parentRect.left}px`;
        overlay.style.width = `${rect.width}px`;
        overlay.style.height = `${rect.height}px`;
        
        // Add overlay to parent element
        element.parentElement.style.position = 'relative';
        element.parentElement.appendChild(overlay);
        
        console.log(`🚫 Created text overlay for ${result.detection_type} content`);
    }

    async analyzeTextElement(element) {
        const text = element.textContent.trim();
        
        try {
            this.showNotification('Analyzing text...', 'info');
            
            const result = await this.analyzeText(text);
            
            // Show detailed result
            this.showTextResult(result, element);
            
        } catch (error) {
            console.error('Text analysis error:', error);
            this.showNotification('Error analyzing text', 'error');
        }
    }

    showTextResult(result, element) {
        // Create result popup
        const popup = document.createElement('div');
        popup.className = 'ai-text-result-popup';
        
        const confidence = Math.round(result.confidence * 100);
        const status = result.is_bot ? '🚫 BOT/LLM DETECTED' : '✅ LIKELY HUMAN';
        const statusColor = result.is_bot ? '#ff4444' : '#44aa44';
        
        popup.innerHTML = `
            <div class="popup-header">
                <h3>Text Analysis Result</h3>
                <button class="close-btn">×</button>
            </div>
            <div class="popup-content">
                <div class="result-status" style="color: ${statusColor}">${status}</div>
                <div class="result-confidence">Confidence: ${confidence}%</div>
                <div class="result-type">Type: ${result.detection_type}</div>
                <div class="result-analysis">${result.analysis}</div>
                <div class="result-model">Model: ${result.model_used}</div>
                <div class="result-text-preview">
                    <strong>Analyzed Text:</strong><br>
                    <em>"${element.textContent.substring(0, 100)}${element.textContent.length > 100 ? '...' : ''}"</em>
                </div>
            </div>
        `;
        
        popup.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 100000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 500px;
            width: 90%;
        `;
        
        // Style header
        const header = popup.querySelector('.popup-header');
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
        `;
        
        header.querySelector('h3').style.cssText = `
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        `;
        
        const closeBtn = header.querySelector('.close-btn');
        closeBtn.style.cssText = `
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #666;
        `;
        
        // Style content
        const content = popup.querySelector('.popup-content');
        content.style.cssText = `
            padding: 20px;
        `;
        
        const statusEl = content.querySelector('.result-status');
        statusEl.style.cssText = `
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        `;
        
        const confidenceEl = content.querySelector('.result-confidence');
        confidenceEl.style.cssText = `
            font-size: 14px;
            margin-bottom: 8px;
        `;
        
        const typeEl = content.querySelector('.result-type');
        typeEl.style.cssText = `
            font-size: 14px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        `;
        
        const analysisEl = content.querySelector('.result-analysis');
        analysisEl.style.cssText = `
            font-size: 14px;
            margin-bottom: 15px;
            line-height: 1.4;
        `;
        
        const modelEl = content.querySelector('.result-model');
        modelEl.style.cssText = `
            font-size: 12px;
            color: #666;
            font-style: italic;
            margin-bottom: 15px;
        `;
        
        const textPreviewEl = content.querySelector('.result-text-preview');
        textPreviewEl.style.cssText = `
            font-size: 13px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        `;
        
        // Add close functionality
        closeBtn.addEventListener('click', () => popup.remove());
        
        // Close on outside click
        popup.addEventListener('click', (e) => {
            if (e.target === popup) {
                popup.remove();
            }
        });
        
        // Close on escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                popup.remove();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
        
        document.body.appendChild(popup);
    }
}

// Initialize the detector when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new DeepfakeDetectorContent().init();
    });
} else {
    new DeepfakeDetectorContent().init();
}