### 🧩 Inputs

* `TTI_model_outputs/` → segmentation masks (TTI pixels = 1)
* `GNGNet_outputs/` → segmentation masks (green = Go, red = No-Go)
* `video_labels.csv` → marks each video as `safe` or `BDI`

---

### 🧮 Algorithm Outline (Pseudocode)

```python
# ------------------------------------------
# Load video lists
# ------------------------------------------
videos = load_video_list("video_labels.csv")

results = []

for video in videos:
    gng_frames = load_gng_masks(video)   # one per frame
    tti_frames = load_tti_masks(video)

    go_ratios = []
    nogo_ratios = []

    for i in range(len(tti_frames)):
        tti = tti_frames[i]          # binary mask: 1 where tool-tissue contact
        gng = gng_frames[i]          # RGB mask: green=Go, red=No-Go

        # Extract zone masks (by color thresholds)
        go_zone   = hsv_inrange(gng, low=(40,40,40),  high=(85,255,255))
        nogo_zone = hsv_inrange(gng, low=(0,70,50),   high=(10,255,255))
        nogo_zone |= hsv_inrange(gng, low=(170,70,50), high=(180,255,255))

        # Compute pixel overlap
        total_tti  = np.count_nonzero(tti)
        go_overlap = np.count_nonzero(tti & go_zone)
        ng_overlap = np.count_nonzero(tti & nogo_zone)

        if total_tti > 0:
            go_ratio   = go_overlap / total_tti
            nogo_ratio = ng_overlap / total_tti
            go_ratios.append(go_ratio)
            nogo_ratios.append(nogo_ratio)

    # Per-video averages
    mean_go   = np.mean(go_ratios)
    mean_nogo = np.mean(nogo_ratios)

    results.append({
        "video": video.name,
        "group": video.group,       # Safe or BDI
        "mean_go": mean_go,
        "mean_nogo": mean_nogo
    })

# ------------------------------------------
# Aggregate group statistics
# ------------------------------------------
df = pd.DataFrame(results)
safe = df[df.group == "Safe"]
bdi  = df[df.group == "BDI"]

print("Average % TTI in Go zone – Safe:", safe.mean_go.mean())
print("Average % TTI in Go zone – BDI:",  bdi.mean_go.mean())
print("Average % TTI in NoGo zone – Safe:", safe.mean_nogo.mean())
print("Average % TTI in NoGo zone – BDI:",  bdi.mean_nogo.mean())

# ------------------------------------------
# Statistical test (optional)
# ------------------------------------------
from scipy.stats import ttest_ind
p_go   = ttest_ind(safe.mean_go,   bdi.mean_go)
p_nogo = ttest_ind(safe.mean_nogo, bdi.mean_nogo)

# ------------------------------------------
# Visualization
# ------------------------------------------
plt.figure()
sns.boxplot(data=df, x="group", y="mean_nogo")
plt.title("Proportion of TTIs in No-Go Zone")
plt.savefig("tti_gng_overlap_boxplot.png")
```

---

### 📈 Output Deliverables

| Output File                   | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| `tti_gng_overlap.csv`         | per-video: mean % TTI in Go / No-Go              |
| `tti_gng_overlap_boxplot.png` | visual comparison of Safe vs BDI                 |
| `report.txt`                  | statistical difference summary (p-values, means) |

---

### ✅ Interpretation Goal

| Group    | Mean TTI in Go Zone | Mean TTI in No-Go Zone |
| -------- | ------------------- | ---------------------- |
| **Safe** | High (≈ 0.8)        | Low (≈ 0.1 – 0.2)      |
| **BDI**  | Low (≈ 0.4)         | High (≈ 0.5 – 0.6)     |

**Conclusion:**

> BDI videos show more tool–tissue contact in No-Go zones → the AI can distinguish safe vs unsafe behavior.
