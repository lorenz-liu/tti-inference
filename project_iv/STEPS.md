# Step-by-step plan

## 0) What goes in / what comes out

**Inputs**

* Ground truth (GT) JSONs (Labelbox-style) in `ground_truths/`
  • For each frame, there may be 0..n objects labeled “Start of TTI/End of TTI” with a **bounding box**.
* Model prediction videos in `inferences/`
  • `pred_*.mp4` produced by your TTI model (mask/overlay video; non-black pixels ≈ predicted TTI).
  • If you also have `pred_*.json`, it can help identify frames with TTI but is optional for pixel metrics.

**Outputs**

1. `pixel_metrics_per_frame.csv` — TP/FP/FN counts + IoU/Dice per frame (for frames that have GT or prediction).
2. `pixel_metrics_per_video.csv` — mean IoU/Dice per video (+ std, #frames used).
3. `sample_confusion_table.png` — a small 2×2 pixel confusion table from one representative frame (pretty print).
4. `pr_curve.png` — **Precision-Recall curve** created by **sweeping IoU thresholds** over frames.
5. (Optional) `qualitative_overlays/…` — side-by-side overlay images for quick visual sanity checks.

---

## 1) Enumerate videos and match GT ↔ predictions

* Scan `ground_truths/*.json` and `inferences/pred_*.mp4`.
* Normalize filenames (strip `pred_` prefix, ignore extension, collapse spaces) to pair each GT JSON with its prediction video.
* Skip pairs where either side is missing.

---

## 2) Parse GT JSON → per-frame GT masks

For each GT JSON:

* Handle **list-root or dict-root** formats.
* Descend into `data_units[...] → labels` mapping.
* For each `frame_id`:

  * Collect all objects whose **name/value indicates TTI** (e.g., `start_of_tti`).
  * Extract each object’s **boundingBox** (`x, y, w, h` are **normalized** to [0,1]).
  * Convert to **pixel coordinates** once we know the video frame size (Step 3 gives width/height).
  * Build a **binary GT mask** (same H×W as video): fill 1s inside every TTI box; 0 elsewhere.
  * If multiple boxes exist, union them.

> Edge case: If a frame has no TTI object, its GT mask is all zeros (i.e., negative).

---

## 3) Read prediction video → per-frame Pred masks

For each `pred_*.mp4`:

* Open with OpenCV; get `width`, `height`, `fps`, and `frame_count`.
* For each frame `i`:

  * Read the BGR image.
  * Convert to **grayscale**, then set **non-zero pixels** to 1 → this is the **Pred binary mask** (H×W).
  * If frames contain probabilities/heatmaps (not just binary), retain the grayscale intensity as a **confidence map** (0–255) and create a binary mask by `> 0`. (We’ll still use the binary mask for IoU/Dice; the intensity can later be used for alternative PR definitions if needed.)

> Note: If GT JSON provides only normalized coords, you can’t build the exact H×W GT mask until you know `W,H`. So either (a) pre-read frame 0 from the prediction video to get `W,H`, or (b) pre-open the original RGB video. Using the prediction video’s geometry is fine if they match.

---

## 4) Align frames (indexing and size)

* Ensure **GT mask** and **Pred mask** share the same size `(H, W)`.
* If needed, resize the GT mask from its initial canvas to `(W, H)` using **nearest-neighbor** (to preserve binary integrity).
* Make sure frame indices align: GT frames 0..N and prediction frames 0..M are compared on the **intersection** of indices.

---

## 5) Compute pixel confusion counts per frame

For each aligned frame `i`:

* Let `G` = GT binary mask (0/1), `P` = Pred binary mask (0/1).
* Compute:

  * `TP = count((G==1) & (P==1))`
  * `FP = count((G==0) & (P==1))`
  * `FN = count((G==1) & (P==0))`
  * (Optionally `TN = count((G==0) & (P==0))` — usually not needed for IoU/Dice)
* From TP/FP/FN:

  * **IoU** = `TP / (TP + FP + FN)`  (define as 0 if denominator=0)
  * **Dice** = `2*TP / (2*TP + FP + FN)`  (define as 0 if denominator=0)
  * (Optional) **Precision** = `TP / (TP + FP)`, **Recall** = `TP / (TP + FN)`

> Tip: If a frame has **no GT positive and no Pred positive**, it’s an **uninformative** background frame for IoU/Dice (denominator 0). You can either discard such frames from aggregation or record IoU=1 for “empty=empty” (but most segmentation papers discard).

* Write one row per frame to `pixel_metrics_per_frame.csv`:

  ```
  video, frame_idx, tp, fp, fn, iou, dice, gt_area, pred_area
  ```

---

## 6) Aggregate per video (mean ± std)

For each video:

* Compute **mean IoU**, **mean Dice** across **informative** frames (denominator>0).
* Also store `n_frames_used`.
* Append one row per video to `pixel_metrics_per_video.csv`:

  ```
  video, frames_used, iou_mean, iou_std, dice_mean, dice_std
  ```

---

## 7) Produce a PR curve (frame-level, by IoU threshold)

We want a **Precision-Recall curve** even if predictions are binary.
A standard way in segmentation is to treat **IoU per frame** as a **quality score**, then:

* Define **frame-level ground truth**: a frame is **positive** if `GT_area > 0` (there is TTI in GT).
* For each frame with `GT_area > 0`, we have an **IoU score** between its GT and prediction.
* **Sweep a threshold τ over IoU** (e.g., τ ∈ {0.0, 0.05, 0.10, …, 1.0}):

  * Predict **frame-positive** iff `IoU ≥ τ`.
  * Then compute:

    * `TP_frames` = #frames with `GT_positive` and `IoU ≥ τ`
    * `FP_frames` = #frames with `GT_negative` but `pred_area > 0` *and* we choose to treat any predicted mask as candidate (or simpler: frames with `GT_negative` but `IoU` is defined and ≥ τ — practically `IoU=0` if GT is empty; so FP_frames will be those with any prediction when GT empty if you base on `pred_area>0`).
    * `FN_frames` = #frames with `GT_positive` and `IoU < τ`
  * **Precision(τ)** = `TP_frames / (TP_frames + FP_frames)`
  * **Recall(τ)** = `TP_frames / (TP_frames + FN_frames)`
* Plot **Recall(τ)** vs **Precision(τ)**, or **τ** on X-axis with three curves (Precision/Recall/F1) — whichever you prefer (your earlier Project III used τ vs scores; both are acceptable).

> Rationale: IoU acts like a **confidence score** for localization quality on frames that have GT positives. Sweeping τ shows how strict you must be about overlap to count a frame as correctly localized.

---

## 8) Save a representative pixel confusion table (optional but requested in deliverables)

* Choose a **representative frame** (e.g., median IoU frame or a user-selected index).
* Render a small 2×2 table image with numbers:

  ```
  GT\Pred   TTI      No-TTI
  TTI       TP       FN
  No-TTI    FP       TN   (optional)
  ```
* Save as `sample_confusion_table.png`.

---

## 9) (Optional) Qualitative overlays for sanity check

* For a handful of frames per video, save side-by-sides:

  * Original (if accessible) / Pred mask / GT mask / overlay.
* Helps visually verify that thresholds and masks are behaving as expected.

---

## 10) Basic robustness & edge cases to handle

* **Top-level list JSONs** (use the first element).
* **Trailing spaces** in names (“End of TTI ”) — normalize strings.
* **Multiple GT boxes** per frame — union them.
* **Mismatched frame counts** — only compare overlapping range.
* **Zero-area masks** — guard divisions; drop uninformative frames from mean IoU/Dice.
* **Different resolutions** — resize GT mask to match prediction frame size with nearest-neighbor.
* **Performance** — iterate frames once; avoid copying large arrays when possible.

---

## 11) Final artifacts checklist

* ✅ `pixel_metrics_per_frame.csv`
* ✅ `pixel_metrics_per_video.csv`
* ✅ `pr_curve.png` (Precision–Recall or Threshold–Precision/Recall/F1)
* ✅ `sample_confusion_table.png`
* ✅ (Optional) `qualitative_overlays/*.png`
