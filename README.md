# 🛒 Shoplifting Detection using CNN + LSTM from Scratch

A deep learning model for **video-based shoplifting detection**, built entirely **from scratch** using **TensorFlow and Keras**.  
This project combines **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Recurrent Neural Networks (LSTM)** for temporal sequence modeling — allowing the model to learn movement patterns over time.

**The new additions in the second file**

* Improved `apply_combined_augmentation()`

* Improved `extract_frames_from_video()`

* Added Regularization and modified the `build_cnn_rnn_model` to reduce **overfitting**.

* Added `Receiver Operating Characteristic (ROC)` and `AUC` for extended evaluation.

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
| Accuracy  | 0.94  |
| Precision | 0.95  |
| Recall    | 0.94  |
| F1-Score  | 0.94 |
| AUC       | 1.00  |


<img width="799" height="536" alt="Screenshot (1640)" src="https://github.com/user-attachments/assets/32bea404-59a5-4258-862e-8b346b3b8a08" />

### the Pictures aren't really visible, because I lowered the image dimensions to `64 x 64` for lower memory usage. 
