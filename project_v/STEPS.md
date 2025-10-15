## 🧩 Folder Structure Inputs

| Type                  | Example Path                                                                                      | Purpose                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| TTI inference code    | `/cluster/projects/madanigroup/lorenz/tti/optimized_eval.py`                                      | Runs the trained TTI model and outputs segmentation masks         |
| Extracted RGB frames  | `/cluster/projects/madanigroup/lorenz/tti/frame_extraction/LapChol Case 0001/frame_sec_020.jpg`   | True video frames                                                 |
| GNG prediction videos | `/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0001 03.MP4` | Each pixel colored as go (green), no-go (red), background (black) |

---

## ⚙️ Step-by-Step Workflow

### **Step 1. Generate GNG masks per frame**

1. For each GNG prediction video, extract frames using the same timing index as your extracted RGB frames.

   ```bash
   python3 -c "import cv2; cap=cv2.VideoCapture('LapChol Case 0001 03.MP4'); cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FPS)*20)); ret,frame=cap.read(); cv2.imwrite('gng_frame_sec_020.jpg', frame); cap.release()"
   ```
2. Now you have e.g.:

   * `frame_sec_020.jpg`  (RGB)
   * `gng_frame_sec_020.jpg` (Go/No-Go mask image)

---

### **Step 2. Run TTI inference on each extracted RGB frame**

Use your inference code:

```bash
python optimized_eval.py --input frame_sec_020.jpg --output tti_pred_020.png
```

Make sure it outputs a **binary or RGB mask** where TTI pixels = 1 (or bright color).

---

### **Step 3. Align frame resolutions and color spaces**

Before overlap calculation, confirm dimensions match:

```python
import cv2
gng = cv2.imread("gng_frame_sec_020.jpg")
tti = cv2.imread("tti_pred_020.png")
if gng.shape[:2] != tti.shape[:2]:
    tti = cv2.resize(tti, (gng.shape[1], gng.shape[0]))
```

---

### **Step 4. Compute pixel-level overlaps**

Identify zones in the GNG mask (e.g., using HSV or BGR thresholds):

```python
import numpy as np

# Example color masks (tune as needed)
go_mask   = cv2.inRange(gng,  (0,100,0),  (0,255,0))   # green
nogo_mask = cv2.inRange(gng,  (0,0,100),  (100,0,255)) # red

tti_mask  = cv2.inRange(tti, (1,1,1), (255,255,255))   # non-black = TTI

go_overlap   = np.logical_and(tti_mask>0, go_mask>0).sum()
nogo_overlap = np.logical_and(tti_mask>0, nogo_mask>0).sum()
total_tti    = (tti_mask>0).sum()

print("Go %",   go_overlap/total_tti*100)
print("No-Go %", nogo_overlap/total_tti*100)
```

---

### **Step 5. Assign classification rule**

Define a threshold (e.g. > 50 % overlap → classified as that zone):

```python
if go_overlap/total_tti >= 0.5:
    zone = "Go"
elif nogo_overlap/total_tti >= 0.5:
    zone = "No-Go"
else:
    zone = "Background"
```

Store this classification per frame or per interaction (if multiple TTIs).

---

### **Step 6. Automate over all videos**

Loop through all videos and extracted frames:

```bash
for vid in /cluster/projects/madanigroup/lorenz/tti/frame_extraction/LapChol*; do
    python projectV_overlap.py --video "$vid"
done
```

Where `projectV_overlap.py` wraps Steps 2–5 and outputs a CSV:

```
video,frame,go_percent,nogo_percent,classification
LapChol Case 0001 03,20,72.3,27.1,Go
...
```

---

### **Step 7. Aggregate & Visualize**

Combine across all frames/videos to produce:

* Mean % TTI in Go vs No-Go for Safe vs BDI videos
* Bar/box plots (`matplotlib`/`seaborn`)

Example goal pattern:

| Video Type | Mean % Go Zone TTI | Mean % No-Go Zone TTI |
| ---------- | ------------------ | --------------------- |
| Safe       | 80 %               | 15 %                  |
| BDI        | 40 %               | 50 %                  |

---

### **Step 8. Deliverables (per the PDF)**

✅ CSV / summary table
✅ Overlap threshold justification (report IoU/Dice if needed)
✅ Visual plots comparing Safe vs BDI TTI distribution
✅ Short interpretation (e.g., “AI detects higher no-go TTI rates in BDI videos”)
