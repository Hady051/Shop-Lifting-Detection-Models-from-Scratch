# 🛒 Shoplifting Detection using CNN + LSTM from Scratch

A deep learning model for **video-based shoplifting detection**, built entirely **from scratch** using **TensorFlow and Keras**.  
This project combines **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Recurrent Neural Networks (LSTM)** for temporal sequence modeling — allowing the model to learn movement patterns over time.

**The new additions in the second file**

* Improved `apply_combined_augmentation()`

* Improved `extract_frames_from_video()`

* Added Regularization and modified the `build_cnn_rnn_model` to reduce **overfitting**.

* Added `Receiver Operating Characteristic (ROC)` and `AUC` for extended evaluation.

* Added a **function** to load a video from the `test set`, add it to a folder for **download** and **making the model predict the outcome**

* Added another **function** to **load the video from the dataset itself** for **better visualization** `(128 x 128)` instead of `(64 x 64)` which the **model trained on**, **preprocessing the video for the model to train on it**, **previewing the video itself (full video) on the notebook** and **the prediction of the model**

---

## Important Note for improving the Model's performance

### The best way to handle the dataset correctly and make the model train right, is to manually crop all videos to 75 frames (the video with the least amount of frames is 75 so that all videos have equal frames) if fps=15, we are talking about 5 seconds (preferrably less than 5 seconds to be sure eg. 4.x) or if NUM_FRAMES didn't work with 75 go for 60 or 50(worked in this code) and calculate the amount of seconds for 60 or 50 frames.

### Also, make sure to crop (leave) the important part of the person going near the objects in both folders (non shoplifter, shoplifter)

---

## 📹 Overview

This project classifies surveillance videos into two categories:
- 🧍‍♂️ **Non-Shoplifters** — Normal customer behavior  
- 🕵️‍♀️ **Shoplifters** — Suspicious or stealing behavior  

Each video is processed as a sequence of frames, with the CNN extracting per-frame features and the LSTM capturing temporal context.

---

## 🧠 Model Architecture

**Frames → TimeDistributed CNN → Flatten → LSTM Layers → Dense Layers → Softmax Output**

**Architecture Highlights:**
- `Conv2D` layers (feature extraction)
- `BatchNormalization` + `Dropout` (regularization)
- `LSTM(256)` and `LSTM(128)` (temporal modeling)
- `Dense` classification head with `softmax` output
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`

---

## 🧩 Key Features

✅ Trains on raw video clips  
✅ Frame-wise CNN feature extraction using `TimeDistributed` layers  
✅ Temporal sequence learning with stacked LSTMs  
✅ ROC–AUC evaluation for model performance  
✅ Easily extendable to multi-class action recognition  

---

## Evaluation Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.95  |
| Precision | 0.96  |
| Recall    | 0.96  |
| F1-Score  | 0.96 |
| AUC       | 1.00  |

---
### Different frames from different videos just for lower usage of memory for visualization
#### the Pictures aren't really visible, because I lowered the image dimensions to `64 x 64` for lower memory usage.
<img width="799" height="536" alt="Screenshot (1640)" src="https://github.com/user-attachments/assets/32bea404-59a5-4258-862e-8b346b3b8a08" /> 

### The Video on the Notebook
[▶️ Watch the video here](./videos/shop_lifter_123.mp4)
