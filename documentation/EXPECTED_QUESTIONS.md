# Expected Review Questions and How to Answer Them

30+ questions across all likely topics: model choices, dataset, evaluation, system design, limitations, and future work.

---

## Section 1 — Project Understanding

**Q1. What is the problem you are solving?**

We are building an automated system to detect abnormal crowd behaviour in surveillance video. Traditional CCTV requires humans to watch screens continuously, which is unreliable and expensive. Our system analyses every frame automatically and raises alerts only when something unusual is detected, so human operators only need to respond when something actually happens.

---

**Q2. What counts as "abnormal" in your dataset?**

The ShanghaiTech Campus dataset defines anomalies as behaviours that deviate from normal pedestrian movement. This includes: running, fighting, chasing, loitering, jumping from heights, throwing objects, cycling in pedestrian zones, and sudden crowd gatherings. All ground truth labels were manually annotated by the dataset authors at frame level.

---

**Q3. Why did you choose the ShanghaiTech dataset?**

It is one of the most widely used benchmarks for video anomaly detection in academic research. It has frame-level ground truth (not just clip-level), covers 13 different real campus scenes, and has a standard train/test split used across the literature, making our results directly comparable to published work.

---

**Q4. What is the difference between your two models?**

The Random Forest uses manually crafted features — optical flow and HOG descriptors aggregated over 30 frames. The ResNet18+MLP uses a pretrained deep neural network to extract rich 512-d spatial features per frame, which are then aggregated and classified by a small MLP. The deep model learns features automatically from data rather than relying on hand-designed descriptors, resulting in significantly better performance (0.97 vs 0.83 AUC).

---

**Q5. Why not just use a single model?**

We built two to demonstrate the value of deep learning over traditional ML for this problem. The Random Forest serves as a strong interpretable baseline. The comparison also forms part of our ablation study and shows how much is gained from feature learning versus feature engineering.

---

## Section 2 — Deep Learning Model

**Q6. What is ResNet18 and why did you use it?**

ResNet18 is a convolutional neural network with 18 layers, pretrained on ImageNet (1.2 million images, 1000 classes). It uses residual connections (skip connections) to avoid the vanishing gradient problem in deep networks. We chose it because: (a) it is computationally efficient, (b) its pretrained weights already encode rich visual representations, and (c) transfer learning on it consistently yields strong results on limited data.

---

**Q7. What is transfer learning and why does it help here?**

Transfer learning reuses a model trained on a large dataset (ImageNet) as a starting point for a different task (anomaly detection). The early layers of ResNet18 learn universal visual features — edges, textures, shapes — that are useful for almost any vision task. Instead of training from scratch and needing millions of crowd behaviour examples, we leverage these pre-learned features and only train a small classifier on top.

---

**Q8. Why did you freeze the ResNet18 weights?**

We froze the backbone because: (a) our training set is relatively small — unfreezing the full network could lead to overfitting, (b) ImageNet features already capture the visual information we need, and (c) it significantly reduces training time. Fine-tuning the full network is a potential improvement for future work if more data becomes available.

---

**Q9. What is the MLP architecture and why those dimensions?**

The MLP is: 2048 → 256 → 64 → 1. The input is 2048-d from our temporal window aggregation. We progressively reduce dimensionality to force the network to learn compact representations. BatchNorm is used after each layer for stable training. Dropout (0.3) prevents overfitting. The final layer uses sigmoid to output a probability in [0, 1].

---

**Q10. Why use a 30-frame window?**

A single frame cannot capture motion or crowd dynamics. Our ablation study directly tested this: reducing the window from 30 to 15 frames dropped ROC-AUC from 0.831 to 0.745 — a loss of 8.6 percentage points. 30 frames at roughly 30fps corresponds to about 1 second of video, which is enough to observe acceleration, direction changes, and crowd dispersion patterns.

---

**Q11. What is the temporal aggregation (mean/std/max/delta) doing?**

For each of the 512 ResNet feature dimensions across 30 frames:
- **Mean** captures the average visual state of the window
- **Std** captures how much it varies (high std = dynamic scene)
- **Max** captures peak activations (extreme events)
- **Delta** (last frame minus first) captures the direction of change over the window

Together these four statistics summarise both the content and the dynamics of the 30-frame sequence in 2048 numbers.

---

**Q12. How did you handle class imbalance?**

Normal frames outnumber anomaly frames significantly in the dataset. If we train naively, the model just learns to always predict "normal." We addressed this using `pos_weight` in the Binary Cross Entropy loss, which up-weights anomaly samples proportionally. This forces the model to take anomaly examples seriously during training.

---

## Section 3 — Random Forest Baseline

**Q13. What features does the Random Forest use?**

Two types of hand-crafted features:
1. **Optical flow (Lucas-Kanade):** Computes pixel-level motion vectors between consecutive frames. We extract mean flow magnitude, max magnitude, flow direction histogram (8 bins), and flow variance per frame.
2. **HOG (Histogram of Oriented Gradients):** Captures edge and shape patterns in each frame.

These are aggregated using mean/std/max/delta over 30 frames, giving a 40-dimensional feature vector per window.

---

**Q14. Why is the Random Forest weaker than ResNet18+MLP?**

Because hand-crafted features lose information that deep features retain. Optical flow captures motion direction and magnitude but misses context — it cannot tell the difference between a person running normally on a track versus running in a panicked crowd. ResNet18's deep features capture spatial relationships, object arrangements, and scene-level context that are invisible to hand-crafted descriptors.

---

**Q15. What is optical flow?**

Optical flow is a technique that computes how pixels move between two consecutive video frames. For each pixel, it estimates a velocity vector (dx, dy) indicating how far and in which direction it moved. In crowd analysis, high optical flow magnitude means people are moving fast; chaotic flow directions mean the crowd is disorganised — both indicators of potential anomaly.

---

## Section 4 — Evaluation

**Q16. What is ROC-AUC and why is it your primary metric?**

ROC (Receiver Operating Characteristic) curve plots True Positive Rate against False Positive Rate at every possible decision threshold. AUC (Area Under the Curve) summarises this in a single number from 0 to 1. We use it as the primary metric because: (a) it is threshold-independent — it measures the model's discriminating ability regardless of what threshold you choose, and (b) it handles class imbalance better than accuracy.

---

**Q17. What does 0.9715 AUC actually mean in practice?**

If you randomly pick one anomalous frame and one normal frame from the test set, there is a 97.15% chance our model assigns a higher score to the anomalous frame. It is a measure of how well the model separates the two classes.

---

**Q18. Why not just report accuracy?**

Accuracy is misleading on imbalanced datasets. If 90% of frames are normal, a model that always predicts "normal" gets 90% accuracy but detects zero anomalies — useless for our purpose. ROC-AUC and F1 score are more informative because they account for both false positives and false negatives.

---

**Q19. What is the train/test split?**

We use a clip-level 80/20 split — clips are assigned to train or test, not individual frames. This is important to avoid data leakage: frames from the same clip are very similar, so a frame-level split would give artificially inflated results (the model would effectively have seen test data during training).

---

**Q20. How did you validate that your results are reliable?**

We tested on the standard ShanghaiTech test set (107 annotated clips) with frame-level ground truth. Results are consistent across multiple runs. We also ran an ablation study testing 4 model variants, and the ranking of results is consistent with theoretical expectations (more context = better, deep features > hand-crafted).

---

## Section 5 — System Design

**Q21. Why use FastAPI instead of Flask or Django?**

FastAPI is: (a) significantly faster than Flask due to async support and Pydantic validation, (b) auto-generates OpenAPI documentation, (c) has built-in type validation, and (d) is the current industry standard for Python ML APIs. Django is a full-stack framework with too much overhead for a REST API serving inference results.

---

**Q22. Why SQLite and not a full database like PostgreSQL?**

SQLite is appropriate for this use case — a single-machine deployment with moderate alert volume. It requires zero configuration, runs in-process, and the database is a single file that is easy to back up. For a production multi-server deployment, migrating to PostgreSQL would be straightforward as the SQLAlchemy-compatible schema is already structured for it.

---

**Q23. Why Next.js for the frontend?**

Next.js provides server-side rendering for fast initial load, TypeScript support for type safety, and the App Router for clean page organisation. It also handles the API proxy (rewrites `/api/*` to FastAPI) cleanly, keeping the frontend and backend decoupled. The React ecosystem has mature animation (Framer Motion) and charting (Recharts) libraries well-suited to a real-time monitoring dashboard.

---

**Q24. How does the frontend communicate with the backend?**

The Next.js config has a rewrite rule: any request to `/api/*` is proxied to `http://127.0.0.1:8000/*`. This means the browser only ever talks to the Next.js server, which forwards requests to FastAPI. This avoids CORS issues and keeps the backend address configurable via environment variable.

---

**Q25. What happens if the API server is down?**

The frontend has an offline state: if the health check fails, it shows an "Offline" badge in the header and a banner with instructions to restart the server. All dashboard data fetching is wrapped in try/catch — failures set `connected = false` rather than crashing the UI.

---

## Section 6 — Limitations and Future Work

**Q26. Can this work on a live camera stream?**

Currently, no — it works on pre-recorded clips. For live deployment, you would need a frame capture loop that reads from an RTSP camera stream (e.g., using OpenCV's `VideoCapture` with an RTSP URL), buffers 30 frames at a time, and feeds them to the inference pipeline. The inference pipeline (`src/inference/pipeline.py`) is already designed for streaming input — it just needs the frame source replaced.

---

**Q27. What would happen if you deployed this on a new campus with different cameras?**

Performance would likely degrade because the model was trained on ShanghaiTech's specific 13 scenes. The feature distributions of a completely different campus — different lighting, camera angles, crowd densities — would not match the training distribution. The solution is domain adaptation: fine-tune the MLP on a small labeled dataset from the new campus, or use unsupervised domain adaptation techniques. The frozen ResNet18 backbone would still transfer well.

---

**Q28. Why does the model sometimes flag sparse scenes (3 people running) as anomalous?**

The model learned that running is associated with anomalous events in the training data. It does not have explicit knowledge of crowd density — it only sees motion patterns and frame features. A person running in an otherwise empty scene triggers similar motion features as running in a crowd. To fix this, you could add a crowd density estimator as an additional input feature.

---

**Q29. What are the ethical concerns with automated surveillance?**

This is an important question. Automated surveillance systems raise concerns around: (a) false positives leading to unfair targeting, (b) disproportionate impact on certain groups, (c) normalisation of mass surveillance, and (d) data privacy. Our system keeps humans in the loop — no automated action is taken without operator review. Alerts require acknowledgement, and the system is designed for institutional security use with appropriate oversight, not mass public monitoring.

---

**Q30. What would you improve if you had more time?**

Several directions:
1. **Live camera integration** — RTSP stream input for real-time monitoring
2. **Fine-tuning ResNet18** — unfreeze the backbone with careful learning rate scheduling
3. **Crowd density as input** — add a density estimator to reduce false positives in sparse scenes
4. **Active learning loop** — use operator acknowledgements to continuously retrain the model on new examples
5. **Multi-camera tracking** — correlate events across multiple cameras to detect distributed anomalies
6. **Transformer-based temporal modelling** — replace window aggregation with a temporal self-attention mechanism for richer temporal context

---

**Q31. How would you scale this to 100 cameras?**

The current single-process FastAPI server would not scale. For 100 cameras you would need: (a) a message queue (Redis/Kafka) where each camera pushes frames, (b) a pool of inference workers consuming from the queue, (c) a load-balanced API tier, and (d) a proper database (PostgreSQL). The ML inference itself can be parallelised with GPU batching — processing multiple clips simultaneously on a single GPU.

---

**Q32. How long does inference take per clip?**

On CPU, the ResNet18+MLP model processes a 457-frame clip in approximately 45–60 seconds (about 8 frames/second). On a GPU this would be 5–10x faster. For real-time deployment at 30fps, GPU inference is required. The Random Forest is significantly faster on CPU (~2 seconds for the same clip) but less accurate.

---

**Q33. What makes your project different from existing solutions?**

Most existing anomaly detection research focuses on either (a) unsupervised reconstruction methods (autoencoders) or (b) large end-to-end 3D CNN architectures that require significant compute. Our approach is a practical middle ground: we combine a pretrained feature extractor (computationally efficient, no training required) with a lightweight classifier (fast to train, easy to update) and wrap it in a production-ready monitoring system with a professional web dashboard. The full stack from raw video to operator dashboard is demonstrated end-to-end.

---

*Good luck with your capstone review. Know your numbers: 0.97 AUC, 91% accuracy, 30-frame window, 2048-d features. These are the figures reviewers will ask about most.*
