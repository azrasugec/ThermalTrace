# Hidden Thermal Target Detection

---

## How It Works

Most people's first instinct with noisy multi-frame data is to average the frames and apply contrast enhancement. That works fine when the camera is stationary — but here the UAV is moving. Averaging misaligned frames just blurs everything together and the target stays hidden.

The breakthrough comes from recognizing that the two "problem" objects (the super-hot and super-cold anchors) are actually **natural GPS markers**. They appear in every single frame. By tracking where they land pixel-by-pixel across all 100 frames, we can measure exactly how much the camera shifted between shots — and correct for it.

Once all frames are aligned to the same coordinate system, the math becomes simple:
- **Noise** is random → it cancels out when you take the median across 100 frames
- **The target** is a real physical object on the ground → it shows up consistently in the same place and survives the median

After stacking, the image only uses 29 out of 256 possible brightness levels. Histogram Projection spreads those 29 levels across the full 0–255 range without distorting their relative order — think of it as removing the empty space between occupied values. Finally, CLAHE boosts local contrast in small 8×8 tiles, which is what makes the target pop visually without amplifying the background noise.

The end result: a Y-shaped thermal object that was completely invisible in any single frame becomes clearly identifiable after processing.

---

## Problem

100 thermal UAV images are given. In every frame:
- A **super-hot** stationary object saturates the sensor (pixel value → 255)
- A **super-cold** stationary object desaturates the sensor (pixel value → 0)

These two anchor objects dominate the histogram, making the actual target object **completely invisible** to the naked eye.

**Goal:** Reveal the hidden target.

---

## Solution Pipeline

```
100 raw frames
      │
      ▼
┌─────────────────────────┐
│  1. Anchor detection    │  → find hot/cold pixel per frame
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  2. Frame registration  │  → align via hot anchor (UAV drift correction)
│     Δy = r₀ − rᵢ       │    Δx = c₀ − cᵢ
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  3. Extreme masking     │  → replace hot/cold pixels with NaN
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  4. Median stacking     │  → pixel-wise nanmedian (SNR ≈ √100 = 10×)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  5. Histogram Projection│  → 29 occupied bins → stretched to 0–255
│     (Zhang et al. 2014) │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  6. CLAHE               │  → local contrast enhancement
│     (Pizer et al. 1987) │    clipLimit=3.0, tileGridSize=(8,8)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  7. Anomaly detection   │  → Gaussian background subtraction (σ=20)
└─────────┬───────────────┘
          │
          ▼
   Y-shaped target revealed
   at pixel (254, 344)  ✓
```

---

## Key Insight

The hot and cold anchor objects — initially appearing as the **problem** — are actually the **solution**. Their consistent presence in every frame makes them ideal natural **fiducial markers** for image registration. Without registration, frame stacking blurs the scene and the target remains invisible.

---


## Visual Results

### Before & After
| Raw Frame | Final Result |
|-----------|-------------|
| ![raw](outputs/raw.png) | ![result](outputs/final_result.png) |

> The target object is completely invisible in the raw frame. After processing, a Y-shaped thermal structure is clearly revealed.

---

### Pipeline Overview

![pipeline](outputs/9_pipeline_result.png)

**(a)** Raw frame with hot (red ★) and cold (blue ★) anchor markers
**(b)** Anchor drift across 100 frames — each dot is one frame, showing UAV movement
**(c)** After registration + median stack — Y-shape emerges from noise
**(d)** After Histogram Projection — full contrast range utilized
**(e)** After CLAHE — target clearly visible (green circle)
**(f)** Background-subtracted anomaly map — red = warm, blue = cold

---

### Naive vs Proposed Method

![comparison](outputs/10_naive_vs_advanced.png)

Without registration, averaging 100 frames just blurs the scene. The proposed method corrects UAV drift first, then stacks — revealing the target clearly.

---

### Histogram Evolution

![histogram](outputs/6_histogram_projection.png)

**Left:** After median stacking — only 29 of 256 gray level bins are occupied.
**Right:** After Histogram Projection — all 29 bins are stretched across the full 0–255 range.

---

## Methods Explained

### 1. Anchor-Based Registration
The hot and cold objects are physically stationary on the ground. Even though they appear at different pixel locations in each frame (due to UAV movement), they can be used as **natural fiducial markers**.

For each frame *i*, we detect the hot anchor position *(rᵢ, cᵢ)* and compute the required shift to align it with the reference frame:

```
Δy = r₀ - rᵢ
Δx = c₀ - cᵢ
```

This corrects the UAV drift exactly, so all 100 frames look at the same ground location from the same reference point.

---

### 2. Temporal Median Stacking
After alignment, we compute the pixel-wise **median** across all 100 frames.

- Random thermal noise → different value each frame → **cancels out**
- The target → same location every frame → **survives**
- SNR improvement: **√100 = 10×**

The median is preferred over the mean because it is robust to outlier pixels (residual anchor artifacts).

---

### 3. Histogram Projection
*(Zhang et al., 2014)*

The stacked image uses only 29 of 256 possible gray levels — 227 bins are completely empty. Standard histogram equalization would redistribute pixel frequencies and distort the image. Histogram Projection instead:

1. Builds a binary occupancy map: `H(i) = 1` if any pixel has value `i`, else `0`
2. Spreads the occupied levels **uniformly** across 0–255
3. Empty bins are eliminated — relative ordering is preserved

```
dₖ = floor( 255 × Σ H(i) / S )
```
where S = total number of occupied bins.

---

### 4. CLAHE
*(Pizer et al., 1987)*

Contrast Limited Adaptive Histogram Equalization enhances **local** contrast by:
- Dividing the image into small 8×8 pixel tiles
- Applying histogram equalization **independently** within each tile
- Using a **clip limit** (3.0) to prevent noise amplification in flat regions
- Blending tile boundaries with bilinear interpolation

This makes the Y-shaped target — which has only subtle thermal contrast — clearly distinguishable from the background.

---

## Results

| Metric | Raw Frame | After Processing |
|--------|-----------|-----------------|
| Occupied gray levels | 63 / 256 | 256 / 256 |
| True scene dynamic range | — | 27.5 GL |
| Noise std per frame | 14.4 GL | ~1.4 GL |
| Target visibility | ❌ Not visible | ✅ Clearly visible |
| Warm anomaly strength | — | +16.77 GL |
| SNR improvement | 1× | ~10× |
| Target location | — | pixel (254, 344) |

---

## Output Images

| File | Description |
|------|-------------|
| `1_raw_frame.png` | Single raw thermal frame |
| `2_histogram_raw.png` | Raw frame histogram analysis |
| `3_anchor_drift.png` | Anchor object positions across 100 frames |
| `5_median_stack.png` | Registered median stack result |
| `6_histogram_projection.png` | Before/after histogram projection |
| `7_clahe_result.png` | Final CLAHE result |
| `8_anomaly_map.png` | Background-subtracted anomaly map |
| `9_pipeline_result.png` | Full 6-panel pipeline overview |
| `10_naive_vs_advanced.png` | Naive vs proposed method comparison |
| `final_result.png` | Final output image |

---

## Requirements

```bash
pip install numpy pillow matplotlib opencv-python scipy
```

---

## Usage

1. Set your image folder path in the script:

```python
IMAGE_DIR  = r'path/to/thermal_images/'
OUTPUT_DIR = r'path/to/outputs/'
```

2. Run:

```bash
python AzraSugec_2220674062_solution.py
```

All output images are saved to `OUTPUT_DIR`.

---


## Project Structure

```
📁 project/
  ├── thermaltrace.ipynb   ← main script
  ├── README.md
  ├── 📁 img/                     ← input images
  │     ├── thermal_image_1.png
  │     ├── thermal_image_2.png
  │     └── ... (100 files)
  └── 📁 outputs/                        ← generated automatically
        ├── final_result.png
        └── ...
```

---

## References

- Zhang, F., Xie, W., Ma, G., & Qin, Q. (2016). *High dynamic range compression and detail enhancement of infrared images in the gradient domain.* Infrared Physics & Technology. Elsevier.
- Fattal, R., Lischinski, D., & Werman, M. (2002). *Gradient domain high dynamic range compression.* ACM SIGGRAPH.
- Pizer, S. M., et al. (1987). *Adaptive histogram equalization and its variations.* Computer Vision, Graphics, and Image Processing, 39(3), 355–368.
- Chen, S.-D. & Ramli, A. R. (2004). *Preserving brightness in histogram equalization based contrast enhancement techniques.* Digital Signal Processing, 14(5), 413–428.

---

## Author

**Azra Sugeç**
