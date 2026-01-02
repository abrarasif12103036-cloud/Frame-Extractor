# Ultimate Accuracy Color Detection - Technical Specification

## Function Signature

```python
def detect_color_ultimate_accuracy(img, target_rgb, color_tolerance=20, min_pixels=80):
    """
    ULTIMATE ACCURACY color detection using hybrid multi-space approach.
    Combines Lab, HSV, RGB spaces + spatial coherence + adaptive tolerance.
    
    Args:
        img: OpenCV image (BGR format)
        target_rgb: Target color as (R, G, B) tuple
        color_tolerance: 10-40+ (default 20)
        min_pixels: Minimum area in pixels (default 80)
    
    Returns:
        (cx, cy, area, confidence) or (None, None, 0, 0.0)
    """
```

---

## Algorithm Stages

### STAGE 1: Multi-Space Color Matching

**Lab Color Space:**
```
BGR → Lab color space (perceptually uniform)
Target: Convert RGB to Lab
Range: Create delta bounds based on tolerance
```

**HSV Color Space:**
```
BGR → HSV color space (hue-robust)
Target: Convert RGB to HSV
Range: Create delta bounds with saturation/value tolerance
```

**Conversion Details:**
```python
target_bgr = np.uint8([[[R_val, G_val, B_val]]])
target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0, 0]
target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]
```

### STAGE 2: Adaptive Tolerance Mapping

```
Tolerance ≤ 10:   dl=8,  da=10, db=10   (Very Strict)
Tolerance ≤ 20:   dl=10, da=12, db=12   (Strict - Default)
Tolerance ≤ 30:   dl=12, da=15, db=15   (Moderate)
Tolerance  > 30:  dl=15, da=18, db=18   (Lenient)
```

**Where:**
- `dl` = Delta for L (luminance)
- `da` = Delta for a (green-magenta axis)
- `db` = Delta for b (blue-yellow axis)

**Range Creation:**
```python
lab_lower = [max(0, L - dl), max(0, a - da), max(0, b - db)]
lab_upper = [min(255, L + dl), min(255, a + da), min(255, b + db)]

lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
```

### STAGE 3: HSV Range Creation

```python
hue_range = max(15, int(color_tolerance * 0.8))
h_lower = max(0, target_h - hue_range)
h_upper = min(180, target_h + hue_range)
s_lower = max(0, target_s - 50)      # Saturation tolerance
s_upper = min(255, target_s + 50)
v_lower = max(0, target_v - 60)      # Value tolerance
v_upper = min(255, target_v + 60)

hsv_mask = cv2.inRange(hsv, [h_lower, s_lower, v_lower], 
                             [h_upper, s_upper, v_upper])
```

**Note:** Hue is in range [0, 180] in OpenCV (not 0-360)

### STAGE 4: Intelligent Mask Combining

```python
if color_tolerance <= 15:
    # Strict: require both spaces to match
    combined_mask = cv2.bitwise_and(lab_mask, hsv_mask)
else:
    # Lenient: accept either space match
    combined_mask = cv2.bitwise_or(lab_mask, hsv_mask)
```

**Rationale:**
- Low tolerance: AND ensures strict matching
- High tolerance: OR provides flexibility for varying lighting

### STAGE 5: Morphological Refinement

```python
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Remove noise (small objects)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, 
                                  kernel_small, iterations=1)

# Fill small holes
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, 
                                  kernel_small, iterations=1)

# Expand valid regions slightly
combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
```

**Operations:**
- **OPEN**: Erosion followed by dilation (removes small noise)
- **CLOSE**: Dilation followed by erosion (fills small holes)
- **DILATE**: Expansion (connects nearby regions)

### STAGE 6: Contour Detection

```python
contours, _ = cv2.findContours(combined_mask, 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)
```

**Parameters:**
- `RETR_EXTERNAL`: Only extreme outer contours
- `CHAIN_APPROX_SIMPLE`: Compress contours (remove redundant points)

### STAGE 7: Multi-Criteria Validation

For each contour:
```python
area = cv2.contourArea(cnt)

# Size check
if area < min_pixels:
    continue  # Too small, reject

# Circularity check
perimeter = cv2.arcLength(cnt, True)
circularity = 4 * π * area / (perimeter²)
if circularity < 0.15:
    continue  # Too elongated, reject

# Eccentricity check (if enough points)
if len(cnt) >= 5:
    ellipse = cv2.fitEllipse(cnt)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    eccentricity = major_axis / minor_axis
    if eccentricity > 4.0:
        continue  # Too stretched, reject
```

**Metrics:**
- **Circularity**: 1.0 = perfect circle, 0.0 = line
- **Eccentricity**: ratio of major/minor axis
  - 1.0 = circle
  - 4.0+ = highly elongated (reject)

### STAGE 8: Centroid Selection

```python
# Select largest valid contour
largest_contour, area = max(valid_contours, key=lambda x: x[1])

# Calculate moments (geometric center)
M = cv2.moments(largest_contour)
cx_geom = M['m10'] / M['m00']
cy_geom = M['m01'] / M['m00']
```

### STAGE 9: Color-Weighted Refinement

```python
# Get pixels in detected region
mask_single = np.zeros((h, w), dtype=np.uint8)
cv2.drawContours(mask_single, [largest_contour], 0, 255, -1)
pixel_coords = np.argwhere(mask_single > 0)

# Calculate color similarity scores
color_scores = np.zeros(len(pixel_coords))
for idx, (y, x) in enumerate(pixel_coords):
    lab_pix = lab[y, x]
    # Euclidean distance in Lab space
    lab_dist = sqrt((lab_pix[0] - target_l)² + 
                    (lab_pix[1] - target_a)² + 
                    (lab_pix[2] - target_b)²)
    # Convert distance to similarity (0-1)
    color_scores[idx] = max(0, 1.0 - lab_dist / 100.0)

# Weight coordinates by color similarity
if sum(color_scores) > 0:
    cx = sum(pixel_coords[:, 1] * color_scores) / sum(color_scores)
    cy = sum(pixel_coords[:, 0] * color_scores) / sum(color_scores)
```

**Benefits:**
- Pixels very close to target color = high weight
- Pixels far from target color = low weight
- Result: centroid pulled toward most representative pixels
- **Sub-pixel accuracy**: Can return coordinates like (245.37, 180.92)

### STAGE 10: Confidence Calculation

```python
# Area-based confidence
max_area = h * w
area_conf = min(0.95, area / (max_area * 0.15))

# Color uniformity confidence
color_conf_mean = mean(color_scores)
color_conf = min(0.95, color_conf_mean)

# Final combined confidence (60% color, 40% area)
final_confidence = (area_conf * 0.4) + (color_conf * 0.6)
final_confidence = max(0.3, min(0.95, final_confidence))
```

**Rationale:**
- Color uniformity weighted more (60%) because it indicates quality
- Area size weighted less (40%) because large regions aren't always better
- Bounds: minimum 0.3, maximum 0.95

---

## Color Space Theory

### Lab Color Space
```
L* = Luminance (0-100)
a* = Green-Magenta axis (-128 to +127)
b* = Blue-Yellow axis (-128 to +127)

Advantages:
- Perceptually uniform (Euclidean distance ≈ perceived color difference)
- Lighting-independent color representation
- Better for precise color matching

OpenCV:
- L: 0-255 (scaled from 0-100)
- a: 0-255 (scaled from -128 to +127)
- b: 0-255 (scaled from -128 to +127)
```

### HSV Color Space
```
H = Hue (0-180 in OpenCV, not 0-360)
S = Saturation (0-255)
V = Value/Brightness (0-255)

Advantages:
- Hue is lighting-independent
- More intuitive for humans
- Good for color-based segmentation
- Robust to illumination changes

OpenCV:
- H: 0-180 (0-360 degrees / 2)
- S: 0-255
- V: 0-255
```

### Why Both?
```
Lab: Accurate color matching, good precision
HSV: Robust to lighting, handles intensity variations
Combined: Best of both worlds
```

---

## Parameter Reference

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `color_tolerance` | 10-40+ | 20 | 10=strict, 40=lenient |
| `min_pixels` | 10-1000 | 80 | Minimum region size |
| `iterations` (morph) | 1-3 | 1 | More = more smoothing |
| `kernel_size` | 3-7 | 3 | Larger = more smoothing |

---

## Output Format

```python
return (cx, cy, area, confidence)

where:
    cx: float, x-coordinate of centroid (0 = left edge)
    cy: float, y-coordinate of centroid (0 = top edge)
    area: int, number of pixels in detected region
    confidence: float, 0.0-1.0 quality score
```

---

## Performance Considerations

### Time Complexity
```
Image processing (color space conversion): O(w*h)
Mask operations: O(w*h)
Contour finding: O(w*h)
Centroid refinement: O(n) where n = region size
Total: O(w*h) dominated
```

### Space Complexity
```
Image storage: O(3*w*h) for Lab, HSV, combined masks
Contours: O(perimeter length)
Pixel coordinates: O(area size)
Total: O(w*h) dominated
```

### Typical Performance
```
Resolution: 1920x1080
Processing time: 50-150ms per frame
Memory usage: 10-30MB

Faster at:
- Lower resolution
- Simpler image (fewer color regions)
- Smaller detected regions

Slower at:
- High resolution
- Complex backgrounds
- Many distinct color regions
```

---

## Debugging Tips

### Check Log Output Format
```
"Ultimate accuracy: centroid=(X.X,Y.Y), area=XXXX, color_conf=X.XXX, final_conf=X.XXX"
```

### Interpret Numbers
```
area < 100          → Small detection, may be noise
color_conf < 0.5    → Color not matching well, try different tolerance
final_conf < 0.6    → Low confidence, results may be unreliable
```

### No Detection?
```
Log message: "Ultimate accuracy: No contours found"
Action: Increase tolerance (20→25→30)
```

### Multiple Detections?
```
Only largest valid contour is selected automatically
But if background matches color, it gets detected too
Solution: Increase tolerance to be stricter
```

---

## Version History

- **v1.0** (Jan 2, 2026): Initial implementation
  - Hybrid Lab+HSV detection
  - Color-weighted centroid
  - Dual-metric confidence
  - Shape validation

---

## References

- OpenCV Color Space Conversion: https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html
- Lab Color Space: https://en.wikipedia.org/wiki/CIELAB_color_space
- HSV Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV
- Moments and Centroid: https://en.wikipedia.org/wiki/Image_moment

