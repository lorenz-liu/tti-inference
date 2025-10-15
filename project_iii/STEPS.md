## 📂 Your Data Mapping

| Type                    | Path                                                                           | Description                                          |
| ----------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **Ground-truth labels** | `/cluster/projects/madanigroup/lorenz/tti/ground_truths/*.json`                | Each JSON lists frames where interactions begin/end. |
| **Model predictions**   | `/cluster/projects/madanigroup/lorenz/tti/inferences/pred_*.json` (and `.mp4`) | Frame-wise predicted TTIs.                           |
| **Goal**                | Match the two for each video → determine interaction-level accuracy.           |                                                      |

---

## 🧮 Step-by-Step Plan

### **Step 1. Parse ground-truth intervals**

Each JSON file encodes events like:

* `"Start of TTI"` → frame `s`
* `"End of TTI"` → frame `e`

From these, build a list:

```python
gt_intervals = [(start_frame, end_frame), ...]
```

(Optionally, also store `Start of No Interaction`/`End of No Interaction` to verify complementarity.)

---

### **Step 2. Parse model detections**

From each `pred_*.json` (or threshold your `pred_*.mp4` mask frames):

* Extract frames where the model predicted any TTI.
* Build a binary array `pred[frame] = 1 if detected`.

---

### **Step 3. Compute detection ratio per interaction**

For every ground-truth interval `[s, e]`:

```python
frames_in_gt = range(s, e+1)
detected_frames = sum(pred[f] for f in frames_in_gt)
ratio = detected_frames / len(frames_in_gt)
```

Store each `ratio`.

---

### **Step 4. Sweep detection thresholds**

Define a range (0 → 1 in 0.05 increments).
For each threshold `t`, mark interaction *i* as **detected** if `ratio ≥ t`.

Then compute:

```python
TP = interactions correctly detected
FP = predicted interactions not matched to any GT
FN = GT interactions missed
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (P * R) / (P + R)
```

---

### **Step 5. Plot results**

* **PR curve:** threshold vs Precision/Recall/F1.
* **Histogram:** distribution of all `ratio` values across interactions.

---

### **Step 6. Define the rule**

Pick threshold where F1 is maximal.
Example outcome:

> “We define an interaction as detected if ≥ 50 % of its frames are predicted, achieving F1 = 0.84.”

---

## 📊 Deliverables

| Deliverable                 | Output                                                                    |
| --------------------------- | ------------------------------------------------------------------------- |
| **1. Threshold rule**       | “Detected if ≥ x % of frames predicted.”                                  |
| **2. Histogram / PR curve** | Plots saved (e.g. `interaction_pr_curve.png`, `frame_ratio_hist.png`).    |
| **3. CSV summary**          | One row per GT interaction: start_frame, end_frame, ratio, detected(0/1). |

---

## 🧠 Interpretation Example

| Threshold t | Precision | Recall | F1              |
| ----------- | --------- | ------ | --------------- |
| 0.3         | 0.78      | 0.92   | 0.84            |
| 0.5         | 0.85      | 0.88   | **0.86 (peak)** |
| 0.7         | 0.89      | 0.70   | 0.78            |

✅ **Conclusion:** A detection threshold of 50 % is optimal — if half the frames of an interaction are detected, count it as a hit.
