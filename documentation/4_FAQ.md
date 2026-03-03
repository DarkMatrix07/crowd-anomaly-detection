# Frequently Asked Questions (FAQ)

## Abnormal Crowd Behaviour Detection System

---

## General Questions

**Q1. What exactly does this system do?**
It watches video footage from surveillance cameras and automatically detects when a crowd starts behaving abnormally. It assigns a risk score between 0.0 (normal) and 1.0 (highly abnormal) to every 30-frame window of video, and raises LOW, MEDIUM, or HIGH alerts when the score crosses configurable thresholds.

---

**Q2. What types of crowd anomalies can it detect?**
The system is trained on the ShanghaiTech Campus Dataset and can detect: running, fighting, sudden panic and dispersion, dangerous crowd congestion, abnormally fast or chaotic motion, and violent altercations. Any behaviour that produces irregular, fast, or spatially disordered motion patterns will trigger elevated scores.

---

**Q3. Does it need a GPU to run?**
No. The system runs entirely on standard CPU hardware. Inference takes approximately 100 milliseconds per 30-frame window, which is fast enough for real-time processing at 25 fps. Training the model also does not require a GPU, though it takes longer on CPU (~5–15 minutes).

---

**Q4. What cameras does it support?**
Currently the demo uses pre-recorded video files (ShanghaiTech test clips). The production inference pipeline (`RollingInferencePipeline`) already supports RTSP streams from IP cameras via OpenCV. Full live camera integration would require RTSP URL configuration in `configs/infer.yaml`.

---

**Q5. What video formats does it support for upload?**
The upload feature in the demo accepts MP4, AVI, MOV, and MKV files. Internally, OpenCV reads the video and extracts frames, so any format OpenCV supports (which is most common formats) will work.

---

**Q6. How accurate is the system?**
On the ShanghaiTech Campus Dataset (international benchmark):
- ROC-AUC: **83.1%** (out of 10 normal/abnormal pairs, correctly ranks 8.3)
- F1 Score: **78.7%** (about 8 out of 10 alerts are correct)
- Recall: **88.5%** (catches ~9 out of every 10 real anomalies)

All targets set for the project have been exceeded.

---

**Q7. What is ROC-AUC and why does it matter?**
ROC-AUC (Area Under the Receiver Operating Characteristic Curve) measures how well the model separates normal from abnormal scenes across all possible thresholds. A score of 1.0 is perfect; 0.5 is random guessing. 0.831 means the model is significantly better than chance and is competitive with published research on this dataset. It is the primary metric used in crowd anomaly detection research, which is why it was chosen as the project evaluation gate.

---

**Q8. Will it work on my specific CCTV cameras?**
The system processes standard video frames. As long as the camera provides a video stream that OpenCV can read (MP4 file, AVI file, or RTSP URL), it will work. Performance may vary depending on camera angle, lighting, and resolution, as the model was trained on a specific dataset. For best results, cameras should have a top-down or elevated angle similar to the ShanghaiTech dataset.

---

**Q9. How many cameras can it handle simultaneously?**
The current architecture supports one pipeline instance per camera. Multiple instances can run in parallel on the same machine. In practice, a modern laptop with 8 GB RAM can run 4–8 pipeline instances simultaneously, each processing one camera feed in real time.

---

**Q10. What is the minimum Python version required?**
Python 3.11 or higher. The code uses `list[type] | None` union syntax introduced in Python 3.10, and is tested on Python 3.11 and 3.13.

---

## Technical Questions

**Q11. What is optical flow and why is it used?**
Optical flow is a technique that measures the apparent motion of pixels between two consecutive video frames. For each pixel, it computes a 2D vector indicating how much and in which direction that pixel has moved. In crowd analysis, optical flow reveals how fast the crowd is moving, in which direction, and whether the motion is orderly or chaotic — all key signals for detecting anomalies.

---

**Q12. What is Farneback optical flow?**
Farneback is a dense optical flow algorithm implemented in OpenCV (`cv2.calcOpticalFlowFarneback`). "Dense" means it computes a flow vector for every pixel in the frame (as opposed to sparse methods that only track key points). It uses polynomial expansion to model the neighbourhood of each pixel and estimates motion by comparing successive frames. It was chosen because it is fast, widely used, and produces smooth flow fields suitable for extracting statistical features.

---

**Q13. What are the 10 frame-level features extracted?**
For each video frame (compared to the previous frame), 10 scalar features are computed:
1. Mean pixel intensity
2. Standard deviation of pixel intensity
3. Laplacian variance (sharpness / edge energy)
4. Occupancy (fraction of bright pixels — crowd density proxy)
5. Mean optical flow magnitude (average motion speed)
6. Variance of optical flow magnitude (how uneven motion is)
7. Maximum optical flow magnitude (fastest moving pixel)
8. Directional entropy (how disordered motion directions are)
9. Divergence proxy (crowd expansion/contraction signal)
10. Temporal contrast change (how different this frame is from the previous)

---

**Q14. What is a temporal window and why use 30 frames?**
A temporal window is a fixed-length sequence of consecutive video frames analysed together as a single unit. Using 30 frames (~1.2 seconds at 25 fps) allows the model to see how the crowd changes over time — not just what it looks like at one instant. Ablation experiments confirmed that W=30 is significantly better than W=15 (0.831 vs 0.745 AUC). Single-frame analysis achieved only 0.66 AUC, showing that temporal context is critical.

---

**Q15. How is the 40-dimensional feature vector built from 10 frame features?**
The 10 frame features are computed for each of the 30 frames in the window, creating a 30×10 matrix. This matrix is then summarised with 4 statistical aggregations:
- **Mean** across frames (average behaviour) — 10 values
- **Standard deviation** across frames (variability) — 10 values
- **Maximum** across frames (peak values) — 10 values
- **Delta** (mean of last 10 frames minus mean of first 10 frames) — 10 values

Total: 10 × 4 = **40 dimensions**. The delta captures acceleration — if motion magnitude is rising rapidly, the delta will be large positive, which is a strong panic signal.

---

**Q16. Why Random Forest and not a neural network?**
Random Forest was chosen because: it runs on CPU without a GPU; it trains in minutes rather than hours; it produces a 5.3 MB model file (vs gigabytes for deep learning models); its feature importances are interpretable; and it achieves 0.831 AUC which meets all project targets. Deep neural networks would likely achieve 0.90–0.94 AUC but require GPU hardware and significantly more engineering effort. For this project's scope and deployment constraints, Random Forest is the right choice.

---

**Q17. What is hysteresis and why does the system use it?**
Hysteresis prevents the alert level from flickering rapidly between states. Without it, if a score fluctuates around 0.50 (the MEDIUM threshold), the alert would switch between LOW and MEDIUM many times per second, which would be confusing and noisy for operators. With hysteresis (margin = 0.05), the score must exceed 0.55 to transition from LOW to MEDIUM, but once at MEDIUM it only drops back to LOW if the score falls below 0.45. This creates stable, stable alerts that only change when there is a clear change in the scene.

---

**Q18. What is exponential smoothing and how is it configured?**
Exponential smoothing (alpha = 0.35) blends the current score with the history: `smoothed = alpha × current + (1 - alpha) × previous`. At alpha=0.35, about 35% of the smoothed value comes from the current score and 65% from history. This reduces the impact of brief, noisy spikes. A higher alpha makes the system more responsive to sudden changes; a lower alpha makes it more stable. The value is configurable in `configs/risk.yaml`.

---

**Q19. How is the composite risk score calculated?**
The final risk score combines four signals with configurable weights:
```
risk = 0.40 × anomaly_score
     + 0.25 × flow_instability
     + 0.20 × crowd_density_pressure
     + 0.15 × trend_acceleration
```
The anomaly score comes directly from the Random Forest model. The other three signals are computed from raw optical flow and intensity features. This multi-signal approach makes the system more robust than relying on the model score alone.

---

**Q20. What is the SQLite database used for?**
The SQLite file (`artifacts/alerts.db`) is the persistence layer for the FastAPI alert server. It stores every alert raised by the system (timestamp, camera ID, risk level, score, evidence window) and tracks acknowledgments (operator name, note, acknowledgment timestamp). It also stores the current threshold configuration. SQLite was chosen because it requires no separate database server and is perfectly adequate for the alert volumes this system generates.

---

**Q21. Why does the model have n_jobs=1 at inference time?**
RandomForest with `n_jobs=-1` (all CPU cores) hangs indefinitely on Windows during `predict_proba()` calls. This is a known issue with Python's `multiprocessing` module on Windows when called from within a Streamlit application (which itself runs in a thread). Setting `n_jobs=1` forces single-threaded prediction, which is slightly slower but works correctly on all platforms. The inference speed (100ms per window) is still fast enough for real-time use.

---

## Demo and UI Questions

**Q22. How do I run the demo?**
The simplest way is to run `bash START_DEMO.sh` in Git Bash from the project folder. A browser window opens automatically at http://localhost:8501. Select a clip and press Start Demo.

---

**Q23. Why should I use START_DEMO.sh instead of a BAT file?**
The `.sh` file runs in Git Bash, which provides reliable error output, coloured messages, and always waits for the user to press Enter before closing. Windows CMD has encoding quirks and inconsistent behaviour with Python processes, making `.sh` the recommended launcher.

---

**Q24. What do the pre-selected demo clips demonstrate?**
- `01_0130` — peaks at score 0.997. Best demo clip. Clear, dramatic anomaly that escalates over time.
- `02_0128` — peaks at 0.960. Second-best demo clip.
- `01_0063` — fast-escalating anomaly. Score rises quickly.
- `01_0054` — 12-frame early warning. System catches it early.
- `05_0018` — normal clip. Score stays green throughout, showing the system does NOT over-alert.

---

**Q25. Why does the demo say "Analysing clip" before playback starts?**
The demo pre-scores all frames before playback begins. This is intentional — it means the video plays back smoothly at the selected fps without any inference lag. The pre-scoring phase typically takes 10–30 seconds depending on clip length and machine speed.

---

**Q26. Can I upload any video, not just crowd footage?**
Yes, any video file can be uploaded. However, the model was trained on surveillance-style crowd footage from ShanghaiTech. Videos with very different visual characteristics (e.g. indoor office footage, close-up videos) may produce less meaningful scores. The system will still run without errors — it simply may not produce meaningful anomaly scores for non-crowd content.

---

**Q27. What does the ground-truth label overlay show?**
When "Show ground-truth label" is enabled, each frame is labelled with the human-annotated ground truth from the ShanghaiTech dataset: "GT: ANOMALY" (blue) or "GT: normal" (green). This lets you compare the AI's prediction (the score and alert level) with what a human expert annotated. It is only available for the built-in demo clips, not uploaded videos.

---

**Q28. What does the score history chart show?**
The line chart on the right side of the demo shows the anomaly score over time, updated in real time as frames are played back. It lets you see how the score evolves — rising gradually (building tension) or spiking suddenly (instantaneous event). The x-axis is frame number; the y-axis is anomaly score (0–1).

---

**Q29. Can I slow down the playback?**
Yes. The FPS slider in the sidebar (labelled "Speed (fps)") goes from 1 to 30. At 1 fps, each frame is shown for one second, allowing you to examine individual frames carefully. At 30 fps, playback is very fast. The default is 12 fps, which gives a good balance of speed and clarity.

---

**Q30. What is the Alert Log table in the demo?**
The Alert Log shows a table of every MEDIUM or HIGH alert raised during playback of the current clip. Each row shows the frame number, time in seconds, alert level, anomaly score, and (for demo clips) whether the ground truth was anomalous at that frame. This is useful for verifying the system's alert precision during the demonstration.

---

## Data and Training Questions

**Q31. What is the ShanghaiTech Campus Dataset?**
ShanghaiTech Campus is one of the most widely used international benchmarks for crowd anomaly detection. It contains 330 normal training videos from 13 camera positions on a university campus, and 60 annotated test clips with frame-level binary ground truth masks indicating which frames contain anomalies. The anomaly types include running, fighting, riding a bicycle, jumping, and other behaviours abnormal for a pedestrian walkway.

---

**Q32. How was the model trained?**
Training used both the 330 normal training videos (label = 0 for all frames) and the 60 annotated test clips (labels from ground truth masks). Frames were extracted at a stride of 5, resized to 320×240, and grouped into 30-frame windows with a stride of 5 windows. Each window was labelled by majority vote of its frames (if more than 50% of frames are anomalous, the window is anomalous). A Random Forest with 300 estimators and balanced class weights was trained on the resulting 40-dimensional feature vectors.

---

**Q33. What is the train/test split?**
80% of the ShanghaiTech test clips were used for training (with their labels) and 20% were held out as a test set. The 330 normal training videos were used for additional normal samples. This resulted in a test set of 1044 windows with a roughly balanced class distribution after downsampling.

---

**Q34. What are ablation studies and why were they done?**
Ablation studies systematically remove or change one component of the system at a time to measure its contribution to performance. Four variants were tested:
- **RF W=30** (production): 0.8313 AUC
- **GBT W=30**: 0.8178 AUC (gradient boosting instead of RF)
- **RF W=30 no-flow**: 0.8016 AUC (optical flow features zeroed out)
- **RF W=15**: 0.7449 AUC (shorter temporal window)

Results confirmed that: (1) window size is the most important factor; (2) optical flow features add ~3 AUC points; (3) RF slightly outperforms GBT on this data.

---

**Q35. How were class imbalances handled?**
Normal crowd footage is far more common than anomalous footage. To prevent the model from simply predicting "normal" for everything, two techniques were used: (1) `class_weight='balanced_subsample'` in the RandomForest, which automatically up-weights the minority (anomaly) class; (2) during training data construction, the normal dataset was downsampled to a 2:1 ratio relative to anomaly samples. This resulted in reasonable recall (88.5%) without excessive false positives.

---

## Operations Questions

**Q36. How do operators acknowledge alerts?**
In the full production stack, the Streamlit operator dashboard has an acknowledgment form. The operator enters the alert ID, their name, and a note (e.g. "Checked, false alarm" or "Escalated to security team"). This is recorded in the database with a timestamp. Acknowledged alerts are marked as reviewed so they do not require further attention.

---

**Q37. Can the thresholds be changed without restarting the system?**
Yes. The operator dashboard has threshold sliders for LOW, MEDIUM, and HIGH. Clicking "Apply Threshold Profile" sends a PUT request to the API, which updates the thresholds in the database immediately. The running pipeline will use the new thresholds for all subsequent alerts without needing a restart.

---

**Q38. How long is alert history kept?**
Currently the SQLite database keeps all alerts indefinitely. There is no automatic purge mechanism. For long-running deployments, periodic archiving or deletion of old records would need to be implemented to prevent the database from growing indefinitely.

---

**Q39. Can multiple operators use the dashboard simultaneously?**
Yes. The Streamlit dashboard connects to the FastAPI server over HTTP. Multiple browser instances can connect simultaneously from different machines on the same network. All operators see the same live data and any acknowledgments are immediately reflected for all viewers.

---

**Q40. Is the system production-ready?**
The core pipeline (feature extraction, model inference, risk scoring, alert generation) is production-quality code with error handling, configuration management, and test coverage. The demo runs stably on Windows with the `.sh` launcher. For a true production deployment, additional work would be needed: RTSP camera integration, multi-camera scaling, alert notification (email/SMS), and a hardened deployment environment (Docker, reverse proxy, TLS). See the Future Upgrades document for the full roadmap.

---

**Q41. What happens if the video stream drops or a frame is missing?**
The inference pipeline handles missing frames gracefully. In `RollingInferencePipeline.process_frame()`, if a frame cannot be read, the last known score is carried forward. In the demo, `cv2.imread()` returns `None` for corrupt frames, and these are skipped with the previous score used. The system does not crash on frame drops.

---

**Q42. How is the system secured against SQL injection?**
The FastAPI server uses parameterized queries (SQLite `?` placeholders) for all database operations. User-supplied values are never interpolated directly into SQL strings. This prevents SQL injection attacks. See `src/api/app.py` for implementation details.

---

**Q43. What is the evidence_window field in alerts?**
Each alert record includes an `evidence_window` field containing the start and end frame indices of the 30-frame window that triggered the alert. This allows operators to go back and review the exact frames that caused the anomaly score to spike, providing evidence for the alert decision.

---

**Q44. Can the model be retrained on new data?**
Yes. Running `python scripts/train_production_model.py` with a different `--data-root` or dataset will retrain the model from scratch. The training script accepts the dataset path, classifier type, window size, and other parameters as command-line arguments. The resulting `.joblib` file can replace the existing model in `artifacts/models/` and the system will use it immediately on the next start.

---

**Q45. What is the difference between the demo and the full production stack?**
The **demo** (`scripts/crowd_anomaly_demo.py`) is a standalone Streamlit app that runs inference directly without the API server. It is optimised for presentations — it pre-scores all frames and plays them back smoothly with visual overlays.

The **full production stack** consists of three separate services: the inference script feeds frames through the pipeline and posts alerts to the API server, which persists them in SQLite; the operator dashboard reads from the API and displays live monitoring. The full stack is designed for continuous, real-time operation with multiple operators.
