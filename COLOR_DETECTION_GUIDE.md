# Color Detection Guide - Simple & Effective Method

## How to Use Color Detection

### Method 1: Use the Color Picker

1. **Enable Tracking**
   - Check the "Enable Tracking" checkbox at the top

2. **Select a Color**
   - Click the color picker box to choose any color
   - Or type the hex code directly (e.g., `FF0000` for red)
   - The preview box shows your selected color

3. **Select a Video**
   - Click "Choose a video" or drag-and-drop a video file
   - Adjust the frame interval (default 0.1 seconds = 10 FPS)

4. **Upload & Extract**
   - Click "Upload & Extract" button
   - Wait for processing
   - View results with detected color markers (green circles)

---

### Method 2: Use the Eyedropper Tool (RECOMMENDED)

This is the easiest way to get exact colors from your video!

**Step 1: Click the Eyedropper Button**
- Look for the **🎨 Eyedropper** button next to the color picker
- Click it to activate color sampling

**Step 2: Select an Image**
- A file dialog opens
- Choose any image file (JPG, PNG, etc.)
- The image loads on screen

**Step 3: Click on a Pixel**
- Move your cursor over the image
- Click exactly on the color you want to detect
- The color is automatically extracted and set

**Step 4: Process Your Video**
- The color is now selected
- Choose your video
- Click "Upload & Extract"
- All frames with that color get marked with green circles

---

## Understanding the Results

### Output Frames
Each extracted frame shows:
- **Green Circle**: Center point of detected color
- **Coordinates**: (X, Y) position in pixels, origin (0, 0) at bottom-left, X extends right, Y extends up
- **Files**: Named `frame_00001.jpg`, `frame_00002.jpg`, etc.

### Detection Accuracy
The "Detection Accuracy (Min Pixels)" slider controls sensitivity:
- **Lower values (10-20)**: Finds small colored objects
- **Higher values (50-100)**: Finds larger colored regions

### Download Results
- Click **Download frames.zip** to get all frames
- Click **Show thumbnails** to preview in browser
- Each frame has the detected color marked

---

## Color Tolerance Guide

The system automatically uses **tolerance=20** (balanced accuracy):
- Detects exact color matches
- Works even with slight lighting variations
- Rejects nearby but different colors

For different scenarios:
- **Exact matches only**: Use pure colors (red, green, blue)
- **With lighting variation**: Use eyedropper on actual video frame
- **Multiple shades**: Adjust frame interval to find all instances

---

## Tips for Best Results

### ✓ Do This:
1. Use the eyedropper on actual video frames
2. Select colors with good lighting in the video
3. Use higher min_pixels if getting false positives
4. Test with small interval (0.1s) first

### ✗ Avoid This:
1. Picking colors from a different source (different lighting)
2. Using very dark or very light colors (hard to detect)
3. Selecting colors that appear throughout the frame (will mark everything)
4. Using colors that match the background

---

## Example Workflows

### Scenario 1: Track Red Ball in Video
1. Check "Enable Tracking"
2. Click Eyedropper button
3. Select a frame from the video
4. Click on the red ball to extract that exact color
5. Upload the video
6. All red balls now marked with green circles

### Scenario 2: Find All Green Markers
1. Check "Enable Tracking"
2. Use color picker and choose `#00FF00` (pure green)
3. Or use eyedropper on a green marker image
4. Upload video
5. All green regions detected and marked

### Scenario 3: Detect Multiple Colors
1. Process video once for red (`#FF0000`)
2. Download frames
3. Process again for blue (`#0000FF`)
4. Compare results to track both colors

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **No colors detected** | Try eyedropper on actual video frame |
| **Too many false positives** | Increase "Min Pixels" slider |
| **Missing some instances** | Lower "Min Pixels" or increase interval |
| **Colors not matching** | Use eyedropper instead of color picker |
| **Upload fails** | Check video format (MP4, MOV, MKV supported) |

---

## Technical Details

**Detection Method**: Simple but Effective
- Uses RGB color space for speed and accuracy
- Euclidean distance matching for color similarity
- Morphological cleanup to remove noise
- Finds largest connected color region per frame

**Processing**: 
- Extracts frames at your specified interval
- Analyzes each frame for target color
- Marks detected regions with green circles
- Returns ZIP with all processed frames

**Performance**:
- ~50-150ms per frame depending on resolution
- Works with HD (1920x1080) and higher resolutions
- No GPU required

---

## Features

✅ **Eyedropper Color Sampling** - Pick exact colors from images  
✅ **Color Picker** - Choose from any hex color  
✅ **Real-time Preview** - See color as you select  
✅ **Adjustable Sensitivity** - Control detection threshold  
✅ **Frame Extraction** - At any interval you choose  
✅ **Marked Results** - Clear green circle indicators  
✅ **Batch Processing** - Process entire videos at once  
✅ **Download All** - Get ZIP with all frames  

---

## Getting Started Now

1. **Open the app**: http://127.0.0.1:5000
2. **Enable Tracking**: Check the checkbox
3. **Click Eyedropper**: 🎨 Eyedropper button
4. **Pick a Color**: Select your image and click on color
5. **Upload Video**: Choose your video file
6. **Extract**: Click "Upload & Extract"
7. **Download**: Get your frames!

That's it! Your color detection is ready to use.

