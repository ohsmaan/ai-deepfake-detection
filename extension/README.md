# AI Deepfake Detector Chrome Extension

A Chrome extension that detects AI-generated deepfakes in images on web pages using a FastAPI backend with Hugging Face models.

## Features

- 🔍 **Automatic Image Scanning**: Scans web pages for images automatically
- ⚡ **Real-time Detection**: Click on image overlays for instant analysis
- 📊 **Confidence Scores**: Get detailed confidence scores for each detection
- 🎯 **Visual Overlays**: Clear visual indicators on images
- 🔧 **Customizable Settings**: Toggle auto-scan and overlay features

## Installation

### 1. Start the Backend

First, make sure your FastAPI backend is running:

```bash
cd backend
poetry run python app.py
```

The backend should be running on `http://localhost:8000`

### 2. Load the Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The extension should now appear in your extensions list

### 3. Test the Extension

1. Navigate to any webpage with images (e.g., Twitter, Facebook, news sites)
2. Click the extension icon in your toolbar
3. Click "Scan Page Images" to analyze all images
4. Or click on individual image overlays for instant detection

## Usage

### Manual Scanning
- Click the extension icon
- Click "Scan Page Images" to analyze all images on the current page
- View results in the popup

### Click-to-Detect
- Images on web pages will show overlays (if enabled in settings)
- Click any overlay to instantly analyze that specific image
- Results appear directly on the image for 3 seconds

### Settings
- **Auto-scan**: Automatically scan images when pages load
- **Show Overlay**: Display clickable overlays on images

## API Integration

The extension communicates with your FastAPI backend at `http://localhost:8000`:

- `POST /upload` - Upload images for detection
- `GET /health` - Check API status

## File Structure

```
extension/
├── manifest.json          # Extension configuration
├── popup.html            # Extension popup UI
├── popup.css             # Popup styling
├── popup.js              # Popup functionality
├── content.js            # Content script for web pages
├── content.css           # Overlay styling
├── background.js         # Background script
├── welcome.html          # Welcome page
├── assets/               # Icons and assets
└── README.md            # This file
```

## Troubleshooting

### Extension Not Working
1. Check that the backend is running on `localhost:8000`
2. Verify the extension is loaded in Chrome
3. Check the browser console for errors
4. Ensure the API health endpoint responds

### Images Not Detected
1. Make sure images are larger than 100x100 pixels
2. Check that images are from valid URLs (not data: or blob:)
3. Verify the page has loaded completely

### API Connection Issues
1. Ensure the FastAPI backend is running
2. Check that CORS is properly configured
3. Verify the API URL in the extension code

## Development

To modify the extension:

1. Edit the relevant files in the `extension/` folder
2. Go to `chrome://extensions/`
3. Click the refresh icon on the extension
4. Test your changes

## Security Notes

- The extension only communicates with `localhost:8000` for security
- Image data is sent directly to your backend for processing
- No data is stored or transmitted to third parties

## Demo Sites

Great places to test the extension:
- Twitter/X (profile pictures, media)
- Facebook (profile photos, posts)
- Instagram (posts, stories)
- News websites (article images)
- Reddit (image posts) 