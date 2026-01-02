# Y-Axis Coordinate System Change

## Summary
The coordinate system has been updated so that **Y-axis extends UPWARD** instead of downward. This matches mathematical and physics conventions where the origin is at the bottom-left.

## Before (Image Coordinates)
- **Origin (0, 0)**: Top-left corner
- **X-axis**: Extends rightward → increases to the right
- **Y-axis**: Extends downward → increases downward
- **Example**: A point near the top of image = small Y value

## After (Cartesian Coordinates - NEW)
- **Origin (0, 0)**: Bottom-left corner  
- **X-axis**: Extends rightward → increases to the right
- **Y-axis**: Extends upward → increases upward
- **Example**: A point near the top of image = large Y value

## Visual Example (320×240 frame)

### Before:
```
(0,0)─────────────────(320,0)
  │                       │
  │  Red dot at top       │
  │  center: (256, 50)    │
  │                       │
  │                       │
  │  Red dot at bottom    │
  │  center: (256, 190)   │
  │                       │
(0,240)───────────────(320,240)
```

### After (NEW):
```
(0,240)───────────────(320,240)
  │                       │
  │  Red dot at top       │
  │  center: (256, 190)   │ ← High Y (near top)
  │                       │
  │                       │
  │  Red dot at bottom    │
  │  center: (256, 50)    │ ← Low Y (near bottom)
  │                       │
(0,0)──────────────────(320,0)
```

## Implementation Details

### Detection Functions Updated
All color detection functions now flip Y-coordinates before returning:

```python
# In all detect_color_*() functions:
cy = h - cy  # Flip Y to Cartesian coordinates
return float(cx), float(cy), int(area), float(confidence)
```

### Drawing Code Adjusted
When drawing on image output, coordinates are flipped back to image space:

```python
# Before rendering on image:
h, w = img.shape[:2]
cy_draw = h - cy  # Flip back to image coordinates for drawing
center_int = (int(round(cx)), int(round(cy_draw)))
cv2.circle(img, center_int, ...)
```

### API Responses
All API responses now return Y-coordinates in the new Cartesian system:
- JSON detection data includes flipped Y values
- Coordinate labels in images show the new Cartesian values

## Testing

### Frame: 320×240 pixels
- Red object detected at original image Y=96.5
- After flip: Y = 240 - 96.5 = **143.5**
- Server log: `centroid=(256.4,143.5)` ✓
- This is now 143.5 pixels UP from the bottom

## Files Modified
1. **backend/app.py**: 
   - Updated all detection functions: `detect_color_simple_effective()`, `detect_color_ultimate_accuracy()`, `detect_color_opencv_hsv()`, `detect_color_pillow_coords()`
   - Modified drawing code to flip Y back when rendering
   
2. **COLOR_DETECTION_GUIDE.md**: 
   - Updated coordinate system documentation

## User Impact
- **Frontend**: Coordinate labels on images now show Cartesian-style values
- **API Responses**: JSON includes flipped Y-coordinates
- **Visual Markers**: Green circles still appear at correct visual positions (drawing code handles flip)

## Example API Response
```json
{
  "frames": {
    "frame_00001.jpg": {
      "cx": 256.4,
      "cy": 143.5,
      "area": 4753.5,
      "confidence": 0.619
    }
  }
}
```

For a 320×240 frame:
- X=256.4: 256.4 pixels from LEFT edge
- Y=143.5: 143.5 pixels UP from BOTTOM edge (or 96.5 pixels DOWN from TOP)
