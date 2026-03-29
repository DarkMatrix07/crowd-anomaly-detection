# The Project Explained in Simple Terms

---

## The Big Idea

Imagine you are a security guard watching 10 CCTV screens at once. After a few hours, your eyes get tired. You might miss something important. Now imagine if a computer could watch all those screens 24/7, never get tired, and tap you on the shoulder only when something actually looks wrong. That is exactly what this system does.

---

## What is "Abnormal Behaviour"?

Normal behaviour on a campus looks like this:
- People walking calmly from one place to another
- Small groups of students chatting or sitting
- Regular foot traffic at predictable times

Abnormal behaviour looks like this:
- Someone running at high speed through a crowd
- A group of people suddenly scattering in all directions
- Two people chasing each other
- A crowd rapidly gathering around one point
- Someone jumping from a height

The AI learns the difference between these two types of scenes and raises an alert when it sees something that looks abnormal.

---

## How Does the AI Learn?

Think of it like teaching a child to spot "odd" behaviour.

**Step 1 — Show examples of normal:**
We give the model hundreds of videos of normal campus activity. It learns what a "normal day" looks like — how people move, how fast they walk, typical crowd sizes and patterns.

**Step 2 — Show examples of anomalies:**
We also show it clips where something abnormal happened, with labels telling it "this frame is anomalous, this one is not."

**Step 3 — The model generalises:**
After training, when it sees a new video it has never seen before, it can judge each frame — "this looks normal" or "this looks unusual" — and give a confidence score between 0 and 1.

---

## What is a "Score"?

Every frame of a video gets a score between 0.0 and 1.0:

- **0.0 to 0.30** → LOW — everything looks normal
- **0.30 to 0.60** → MEDIUM — something slightly unusual
- **0.60 to 0.85** → HIGH — strong indication of anomaly
- **0.85 to 1.00** → CRITICAL — almost certainly anomalous

These thresholds can be adjusted by the operator depending on how sensitive the system needs to be.

---

## Why Use ResNet18?

ResNet18 is a neural network that was trained on 1.2 million images from the internet (called ImageNet) to recognise thousands of different objects. Even though it was not trained on crowd behaviour specifically, it already knows how to extract rich visual information from images — edges, shapes, textures, poses, spatial relationships.

We take this pre-trained knowledge and repurpose it: instead of asking "is this a cat or a dog?", we ask "does this frame contain signs of abnormal motion?"

This approach (called **transfer learning**) lets us get excellent results without needing millions of crowd behaviour examples.

---

## Why Look at 30 Frames at Once?

A single frame tells you very little. Someone running in frame 1 looks the same as someone jogging in frame 1. But if you look at 30 frames (about 1 second of video), you can see:
- How fast people are accelerating
- Whether the crowd is scattering or gathering
- Whether the direction of movement is chaotic or orderly

Looking at 30 frames together is called a **temporal window**. Our ablation study confirmed that reducing the window from 30 to 15 frames dropped the accuracy significantly — proving that context over time is critical.

---

## What Happens When an Alert is Raised?

1. The model scores a window of frames as HIGH risk
2. A new alert is created in the database with: camera ID, timestamp, risk level, and score
3. The alert appears on the dashboard in real time
4. An operator reviews it and either:
   - Acknowledges it (confirms it was real) and escalates
   - Dismisses it (false alarm)
5. The acknowledgement is logged with the operator's name and notes

This human-in-the-loop approach ensures no automated action is taken without human confirmation.

---

## The Dashboard — What Can You See?

**Dashboard Tab:**

- Three counters at the top: how many LOW, MEDIUM, and HIGH alerts have been recorded
- A timeline chart showing how risk scores changed over time
- A table of all recent alerts with camera, time, score, and an Acknowledge button
- A panel on the right to adjust thresholds and acknowledge specific alerts

**Live Detection Tab:**

- Pick any of 10 pre-loaded video clips from the ShanghaiTech dataset
- Choose a model: the DL model (ResNet18+MLP) or the ML baseline (Random Forest)
- Click Analyze — the system runs inference on every frame
- Watch the video play back with a live score overlay showing LOW/MEDIUM/HIGH
- See a full summary: peak score, anomalous frame count, ROC-AUC, accuracy vs ground truth

---

## What is ROC-AUC and Why Does it Matter?

Imagine a dial that goes from "never raise an alarm" to "always raise an alarm". At one extreme you miss everything; at the other you cry wolf constantly.

ROC-AUC measures how well the model separates normal from anomalous frames across ALL possible positions of that dial. A score of:
- **1.0** = perfect separation — model always knows the difference
- **0.5** = random guessing — no better than a coin flip
- **0.97** = our model — extremely strong discrimination

We use ROC-AUC because it does not depend on which threshold you choose. It measures the underlying quality of the model's predictions.

---

## Limitations (What the System Cannot Do Yet)

- **No live camera stream** — currently works on pre-recorded clips. A real deployment would need a frame capture loop from an IP camera.
- **Fixed camera angles** — trained on ShanghaiTech scenes. A new camera in a different location would need fine-tuning.
- **No automatic action** — the system alerts but does not trigger alarms or contact authorities automatically. A human operator always decides.
- **Sparse crowd false positives** — if just 2-3 people are running in an otherwise empty scene, the model may flag it. Context about crowd density is not explicitly modelled.

---

*These limitations are normal for a research-grade prototype and are standard topics for academic discussion at a capstone review.*
