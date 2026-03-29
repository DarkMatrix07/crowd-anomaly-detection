# Demo Walkthrough

A scripted guide for presenting the system during the capstone review. Follow this order for maximum impact.

---

## Before You Start

Make sure both servers are running:

```bash
# Terminal 1
uvicorn src.api.app:app --reload --port 8000

# Terminal 2
cd web && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser. Confirm the **"Live"** badge is green in the top-right corner.

Recommended browser: Chrome or Edge, fullscreen (F11).

---

## Part 1 — Dashboard Overview (2–3 minutes)

**What to show:** The Dashboard tab with live alert data.

**Steps:**
1. Open the dashboard — point out the three metric cards (LOW / MEDIUM / HIGH alert counts)
2. Point to the **Risk Score Timeline** chart — explain the Y-axis is the anomaly score (0–1), X-axis is time
3. Scroll down to the **Alert History** table — show the alert rows with camera ID, risk level badge, score bar, and timestamp
4. Point to the **Alert Thresholds** panel on the right
   - Drag the "High" threshold slider down to show it updates live
   - Click **Strict** preset — show thresholds change to tighter values
   - Click **Default** to reset
5. Show the **Acknowledge Alert** form — enter an Alert ID, operator name, and a note, then submit. The alert status updates in the table.

**What to say:**
> "This is the operator dashboard. In a real deployment, alerts would be pushed here automatically as the surveillance cameras feed into the inference pipeline. The operator can filter by camera, adjust sensitivity thresholds without restarting anything, and log acknowledgements for every alert with their name and notes — giving a full audit trail."

---

## Part 2 — Live Detection (5–7 minutes)

This is the most impressive part. Walk through it carefully.

### Step 1 — Select the best demo clip

Click the **Live Detection** tab.

In the **Video Clip** dropdown, select:
> `01_0130 — Scene 01 — Sudden crowd rush (peaks 0.997)`

This clip has the highest peak score (0.997) and is the most visually striking.

### Step 2 — Select the model

Click **ResNet18 + MLP** (it should already be selected — the 0.97 AUC model).

Set speed to **15–20 fps** using the slider (fast enough to be dynamic, slow enough to follow).

### Step 3 — Click Analyze Clip

Click the **Analyze Clip** button.

**While waiting (~45 seconds), explain to the panel:**
> "The system is now running ResNet18 feature extraction on every frame of this 337-frame clip. For each 30-frame window, it aggregates spatial features into a 2048-dimensional vector and passes it through our trained MLP classifier. This produces a per-frame anomaly score."

### Step 4 — Watch the playback

Once analysis completes, playback starts automatically. Point out:

- The **score badge** (LOW / MEDIUM / HIGH) in the top-left of the video — it changes colour as the scene evolves
- The **numerical score** (e.g. "0.847") updating in real time
- The **GT label** (Ground Truth: normal / ANOMALY) in the top-right — this is the hand-annotated label from the dataset
- The **score bar** at the bottom of the video
- The **live stats sidebar** on the right — Current Score (large), Peak, Anomalous frame count, ROC-AUC, Accuracy
- The **Score history strip** — mini bar chart of the score so far

**When the score goes HIGH (red):**
> "Notice the score jump here — this is where the crowd rush begins. The model detects the sudden change in motion patterns and raises a HIGH alert. The ground truth label confirms this is indeed anomalous."

### Step 5 — Show the Analysis Summary

After playback ends, scroll down to the **Analysis Complete** section. Point out:

- **Peak Score: 1.000** — the model was maximally confident at the peak anomaly
- **Anomalous Frames** — how many frames were flagged
- **ROC-AUC: ~0.947** — this specific clip's AUC
- **Accuracy: ~91%** — vs the ground truth mask
- The **Frame-by-Frame Score chart** with the GT overlay (red shaded regions = ground truth anomaly, blue line = model score). Show how well they align.
- Bottom bar: **"ResNet18+MLP 0.9466 vs RF baseline 0.8313"**

---

## Part 3 — Model Comparison (2 minutes)

**Show the difference between the two models:**

1. Without leaving the results, click **Random Forest** model button
2. Click **Re-analyze**
3. Wait for it to complete (~5 seconds — RF is much faster)
4. Compare the scores — the RF will detect the anomaly but with lower peak confidence and slightly different timing

**What to say:**
> "The Random Forest uses hand-crafted optical flow features and takes 5 seconds. The ResNet model uses deep features from a pretrained ImageNet network and takes 45 seconds on CPU. The trade-off is clear: ResNet gives 0.97 AUC vs 0.83 for the Random Forest. On a GPU, the ResNet would also run in under 5 seconds."

---

## Part 4 — Try a Normal Clip (1 minute)

Select:
> `05_0018 — Scene 05 — Normal pedestrian flow (baseline)`

Click **Analyze Clip** with ResNet18+MLP.

**While waiting:**
> "This is a clip of normal campus activity — students walking, no anomalies. Let's see if the model correctly stays LOW throughout."

When playback runs — the score should stay mostly in the LOW range (green). This demonstrates the model does not just flag everything.

---

## Part 5 — Try a Different Scene (optional, if time allows)

Select:
> `06_0144 — Scene 06 — High-density anomaly (58% anomaly)`

This clip is from a completely different camera angle and scene from the previous ones. Running the model on it and getting good results demonstrates the model generalises across scenes.

---

## Closing Statement

> "To summarise — we have built an end-to-end abnormal crowd behaviour detection system: from raw camera footage through deep learning inference to a live operator dashboard. Our ResNet18+MLP model achieves 0.97 ROC-AUC and 91% accuracy on the ShanghaiTech benchmark. The system is fully deployed with a production-quality web interface, configurable alert thresholds, and an operator acknowledgement workflow. Thank you."

---

## Backup Plan (if something breaks)

| Problem | Fix |
|---------|-----|
| API shows "Offline" | Restart: `uvicorn src.api.app:app --reload --port 8000` |
| Analysis hangs >2 minutes | Refresh the page, try a shorter clip (01_0063 — 193 frames) |
| Dashboard shows no alerts | The DB resets on restart — run a clip analysis first to generate alerts |
| Browser looks broken | Hard refresh: Ctrl+Shift+R |

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| ResNet18+MLP ROC-AUC | **0.9715** |
| ResNet18+MLP Accuracy | **91.96%** |
| Random Forest ROC-AUC | **0.8313** |
| Temporal window size | **30 frames** |
| Feature vector size | **2048-d** |
| Demo clips included | **10 clips, 6 scenes** |
| Demo data size | **57 MB** |
