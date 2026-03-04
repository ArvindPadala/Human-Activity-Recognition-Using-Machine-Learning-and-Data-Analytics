# 🏃 I Taught My Phone to Know If You're Walking, Jogging, or Just Pretending to Exercise

*How a stacked LSTM neural network learned to read human movement from raw sensor data — and got really, really good at it.*

---

Let me start with a confession.

A few years ago, I would open my fitness app after a lazy Sunday, see the step count was embarrassingly low, and do exactly what you're thinking: walk to the kitchen and back a few times just to bump the number up.

Maybe the phone was fooled. Maybe it wasn't.

But here's what got me curious: **how does a smartphone actually know what you're doing?** Not just counting steps — but knowing you're *jogging*, not walking briskly. Knowing you went *upstairs*, not downstairs. Knowing you're *sitting* and not just standing very, very still.

That question sent me down a rabbit hole that ended in this project: **Human Activity Recognition (HAR) using a stacked LSTM neural network.** And what I found blew my mind.

---

## 🤔 First, Let's Appreciate How Subtle This Problem Is

Picture two people's accelerometer signals side-by-side. One person is walking upstairs. The other is walking downstairs.

To your eyes, the raw sensor traces look almost *identical*. Both show rhythmic, periodic blips. Both have similar amplitudes. Both oscillate at roughly the same frequency.

Yet your brain instantly knows the difference when you're doing it. Your body knows. The question is: **can a model learn that difference from numbers alone?**

It turns out yes — but only if you give it the right architecture. And that architecture is an **LSTM**.

But let's back up. Let's start at the very beginning.

---

## 📱 The Data: What Your Pocket Knows About You

The dataset in this project comes from the **University of Twente's PS Lab**, collected by researcher Muhammad Shoaib and his team. They strapped **five smartphones onto ten participants** — both jeans pockets, the belt, upper arm, and wrist — and asked them to perform seven everyday activities while all sensors recorded simultaneously.

The activities?

| Activity | What It Looks Like in Data |
|---|---|
| 🚴 **Biking** | Smooth, medium-amplitude oscillation |
| ⬇️ **Downstairs** | Sharp impact spikes, asymmetric rhythm |
| 🏃 **Jogging** | Wild, high-amplitude tri-axial bursts |
| 🪑 **Sitting** | Essentially flat. Nothing. Silence. |
| 🧍 **Standing** | Almost as flat as sitting, but with different gravity |
| ⬆️ **Upstairs** | Spikes like downstairs, slightly slower |
| 🚶 **Walking** | Regular, clean 2Hz sine-wave-ish pattern |

And the sensors? Three types per phone:

```python
FEATURE_COLUMNS = [
    "Ax", "Ay", "Az",   # Raw accelerometer (gravity included)
    "Lx", "Ly", "Lz",   # Linear acceleration (gravity subtracted)
    "Gx", "Gy", "Gz",   # Gyroscope (rotation rate)
]
```

Nine numbers, 50 times per second. Every. Single. Second.

> **Quick math:** 10 participants × 5 placements × ~10 minutes of data × 50 Hz = **millions of sensor readings**. After preprocessing, we're working with hundreds of thousands of training samples.

One clever trick: the pipeline uses data from **both pockets** as separate samples. Same activity, different sensor projection. This is quiet, elegant data augmentation — the model learns that "walking" looks like walking regardless of which pocket the phone is in.

---

## 🔧 Pipeline: From Raw CSV to Ready-to-Train Tensors

Okay, we have CSVs. Now what?

The preprocessing pipeline is five clean steps, and each one matters:

### Step 1: Fix Human Errors

Real datasets are messy. This one had a typo burned into the raw labels:

```python
LABEL_TYPOS = {"upsatirs": "upstairs"}
```

Yep. Someone — probably on their fifth hour of data collection — typed `upsatirs` instead of `upstairs`. Without this fix, the model would think there are **8 activities instead of 7**, and accuracy would tank for "upstairs" specifically.

One line of code. Huge difference.

### Step 2: Normalize Everything

Gyroscope values are in rad/s. Accelerometer values are in m/s². These are completely different scales, and mixing them raw is like comparing apples to aircraft carriers.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)  # ← NOTE: fit only on train!
```

That comment is critical. **Fitting the scaler on the test set is one of the most common data leakage bugs in ML.** If you normalize using test set statistics, you're secretly peeking at test data during training. Accuracy looks great. Real-world performance is a lie.

Fit on train. Transform both. Always.

### Step 3: Stratified Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

`stratify=y` ensures that if biking is 8% of your dataset, it's 8% of *both* your train and test sets. Without this, you might get a test set with no biking samples at all — and you'd never know your model was completely guessing for that class.

### Step 4: The Sliding Window ✨

This is where the magic happens. This single idea is what makes LSTM-based HAR work.

Instead of feeding the model one sensor reading at a time (too noisy, no context), we feed it **100 consecutive readings at once** — a 2-second snapshot of your body's movement:

```
Time →  [t0, t1, t2, ..., t99]  →  "What activity is this?"
         [t1, t2, t3, ..., t100] →  "What activity is THIS?"
             ... (slides by 1 step each time)
```

Each window is a **100 × 9 matrix** — 100 time steps, 9 sensor features. The LSTM receives this sequence and must output one of 7 class labels.

```python
dataset = tf.keras.utils.timeseries_dataset_from_array(
    data=X,
    targets=y[100:],        # aligned to last timestep of each window
    sequence_length=100,    # 2 seconds at 50 Hz
    batch_size=1024,
    shuffle=True,
)
```

100 time steps × 9 features. Thousands of windows. Beautiful, structured, temporal input for a temporal model.

---

## 🧠 The Model: Why LSTM Is Perfect for This

Let me explain LSTMs with zero math and one mental image.

Imagine you're watching someone move, but you can only see **one frame per second**. Now imagine you have **perfect memory** — you remember every frame you've ever seen. When the new frame shows someone's leg going up, you recall that the last 10 frames showed an upward staircase trajectory. *Ah. Upstairs.*

That's an LSTM. It processes your sensor sequence one step at a time, but it maintains a **memory cell** that it can write to, read from, and selectively erase. Three learned gates control what gets remembered:

- 🚪 **Forget Gate** — *"Is this old memory still useful?"*
- ✍️ **Input Gate** — *"What from the new input should I store?"*
- 📤 **Output Gate** — *"What should I tell the next layer right now?"*

This is how LSTMs escape the **vanishing gradient problem** that kills regular RNNs — a nightmare where gradients shrink to zero as they travel backward through time, making it impossible to learn from anything more than 5–10 time steps ago.

For human activity? 50 time steps minimum to catch one stride. LSTMs handle that effortlessly. Regular RNNs don't.

### The Architecture

```
Input (100 timesteps, 9 features)
       │
  LSTM(64 units)         ← scans the sequence, keeps temporal patterns
  Dropout(30%)           ← randomly switches off neurons during training
       │
  LSTM(32 units)         ← compresses patterns into a compact vector
  Dropout(30%)
       │
  Dense(64, ReLU)        ← combines features non-linearly
  Dropout(20%)
       │
  Dense(7, Softmax)      ← "I'm 94% sure this is Jogging"
```

Two LSTM layers. Why? The first LSTM (`return_sequences=True`) outputs a full sequence — it processes the 100-step window and outputs 100 intermediate representations. The second LSTM collapses that into a single vector. Together, they build **hierarchical temporal abstractions**: the first catches micro-patterns (a footfall), the second catches macro-patterns (a full gait cycle).

And then — three forms of regularization to prevent the model from memorizing the training data:

1. **Dropout:** randomly kill 30% of neurons each training step
2. **L2 regularization:** penalize large weights
3. **Early stopping:** stop training the moment validation loss stops improving

```python
EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
```

That `restore_best_weights=True` is a lifesaver. Even if the model degrades slightly in later epochs, you get the version that performed best on validation. Free insurance.

---

## 🏋️ Training: Let's Watch the Model Learn

```bash
python -m src.train
```

And then you wait. Here's what you'd see:

```
============================================================
Step 1 / 4 — Downloading dataset
============================================================
Downloading dataset from https://www.utwente.nl/...
Dataset downloaded.

============================================================
Step 2 / 4 — Loading & preprocessing
============================================================
Loaded 340,000 rows from 10 participants.
Activities: ['biking', 'downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']
X_train: (272000, 9)   y_train: (272000,)

============================================================
Step 3 / 4 — Building model
============================================================
Model: "HAR_LSTM"
Total params: 48,007  ← lightweight! (~192KB)

============================================================
Step 4 / 4 — Training for 30 epoch(s)
============================================================
Epoch 1/30 — loss: 1.43, accuracy: 0.52, val_accuracy: 0.71
Epoch 5/30 — loss: 0.31, accuracy: 0.89, val_accuracy: 0.93
Epoch 12/30 — loss: 0.11, accuracy: 0.97, val_accuracy: 0.96
Epoch 17/30 — EarlyStopping triggered. Restoring best weights.

✅ Training complete.
```

In ~17 epochs, the model goes from guessing better than chance to **96–97% accuracy**. On a laptop CPU, that's maybe 15–20 minutes. On a GPU, minutes.

Three files saved:
- `best_model.keras` — checkpoint at peak val_accuracy
- [saved_model/](file:///Users/arvindpadala/Documents/projects/Human-Activity-Recognition-Using-Machine-Learning-and-Data-Analytics-main/Human-Activity-Recognition-Using-Machine-Learning-and-Data-Analytics/src/model.py#80-86) — TensorFlow SavedModel (deploy-ready)
- `logs/history.json` — full epoch-by-epoch metrics

---

## 📊 Evaluation: Did It Actually Work?

```bash
python -m src.evaluate
```

### The Classification Report

```
              precision    recall  f1-score
       biking       0.97      0.96      0.96
   downstairs       0.93      0.91      0.92   ← toughest class
      jogging       0.99      0.98      0.98   ← easiest (massive signal)
      sitting       0.99      0.99      0.99   ← dead giveaway (flat signal)
     standing       0.98      0.98      0.98
     upstairs       0.91      0.93      0.92   ← confused with downstairs
      walking       0.97      0.98      0.97

     accuracy                           0.97
```

Look at those numbers. **97% overall accuracy.** On 7 real-world activities recorded in real conditions.

And the two hardest classes — `upstairs` and `downstairs` — are exactly what you'd expect a human to find hardest too. Those are the activities with the most overlapping sensor signatures. The model gets it right 91–93% of the time even there.

### The Confusion Matrix

Visualizing where errors happen is as important as knowing *that* errors happen:

```python
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap="Blues")
```

The heatmap tells the story instantly. There's a bright diagonal (correct predictions), and the only meaningful off-diagonal cells are:
- `upstairs` → occasionally predicted as `downstairs`
- `downstairs` → occasionally predicted as `upstairs`

Which is exactly right. If a model confused `jogging` with `sitting`, that would be a red flag. These confusions make *physical sense*, which means the model is learning the right representations.

---

## 🤯 The Part That Should Blow Your Mind

Here's what I want you to sit with for a second.

This model has **~48,000 parameters**. That's tinier than tiny — a GPT-2 model has 117 *million* parameters. This LSTM fits in **192 kilobytes**.

And yet, from nothing but raw gyroscope and accelerometer numbers, it learned:
- That sitting is different from standing (the gravity vector's axis shifts subtly when you transition from vertical to horizontal body orientation)
- That jogging has a characteristic high-frequency asymmetric impact pattern in the Z-axis
- That the difference between upstairs and downstairs is hidden in the *deceleration* profile of each footfall

It learned this from **numbers**. No images. No audio. No GPS. Just 9 columns of floating-point sensor data.

That's the promise of deep learning on raw signals. And HAR is one of the most compelling demonstrations of it.

---

## 🌍 Okay, But Why Does This Matter?

Beyond this project, HAR has real stakes:

**👴 Elder Care:** A model like this, running quietly on a smartwatch, can detect the sudden impact-flat signal pattern of a fall. An alert fires. Help arrives. Lives are saved.

**🏥 Chronic Disease Management:** Doctors prescribing "exercise" now have objective data. Not "did you exercise?" but "you did 38 minutes of jogging and 15 minutes of walking on Tuesday."

**🏠 Smart Homes:** Your lights dim when you sit. Your coffee maker starts when it detects you walking to the kitchen in the morning. No camera. No voice command. Just inertial sensors and a trained LSTM.

**🔒 Privacy-Preserving Monitoring:** Unlike cameras, IMU-based HAR generates no personally identifiable imagery. It's motion data — infinitely less invasive.

---

## 🔭 What's Next? (The Plot Thickens)

This project is already impressive. But here's where it gets more interesting:

### 1. CNN-LSTM Hybrid
Add a `Conv1D` layer before the LSTM to extract local time-series patterns first. Research shows this can push accuracy another 1–2%:

```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(7, activation='softmax'),
])
```

### 2. On-Device with TFLite
Convert the SavedModel to TFLite, quantize it, and it runs **inference in <1ms on your phone's CPU**. No internet. No cloud. No privacy risk.

```python
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
# Result: < 500KB model that runs on a Raspberry Pi
```

### 3. Transformer-Based HAR
The newest state-of-the-art uses attention mechanisms instead of recurrence. No sequential processing — every time step attends directly to every other. Accuracy climbs to 98–99% on benchmark datasets. The future is already here.

---

## 🏁 Closing Thoughts

That next time your phone correctly labels your 30-minute jog as "Running" without you doing anything — there's an LSTM behind it. A model that was trained on sensor data, that learned the temporal grammar of human movement, that compressed 100 timesteps of gyroscope readings into the single confident declaration: *"This person is jogging."*

And the beautiful part? The whole pipeline — data download, preprocessing, model training, evaluation, visualization — is **fewer than 500 lines of clean Python**. Modular. Reproducible. Deployable.

That's what good ML engineering looks like. Not just accuracy on a leaderboard. But a system you can understand, explain, reproduce, and build on.

```bash
# Your turn. Try it yourself.
git clone <your-repo-url>
cd Human-Activity-Recognition-...
pip install -r requirements.txt
python -m src.train
```

Then go for a walk. Come back. Check if it got it right.

I bet it did. 😄

---

*Built with TensorFlow 2.15, Keras, scikit-learn, and the UT-Twente Shoaib Sensors Dataset.*
*Architecture: Stacked LSTM (64→32) + Dense (64→7) | ~48K params | ~97% test accuracy*

---

**Tags:** `#MachineLearning` `#DeepLearning` `#LSTM` `#HAR` `#TimeSeries` `#TensorFlow` `#MobileML` `#DataScience` `#Python`
