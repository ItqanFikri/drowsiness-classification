# 🚗 Driver Drowsiness Detection System using IndRNN on Jetson Nano

A real-time driver drowsiness detection system using **Independent Recurrent Neural Network (IndRNN)** based on facial behavior analysis.
The system monitors driver alertness using **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** and provides warning alerts when drowsiness is detected.

This project was developed as an undergraduate thesis focusing on real-time embedded AI deployment for road safety applications.

---

# 📌 Overview

Driver drowsiness is one of the major causes of traffic accidents. This system continuously monitors the driver's facial condition using a camera and classifies the driver's state into different alert levels in real time.

The system is designed to run on an embedded device (**Jetson Nano**) and operate efficiently in real-world conditions.

Performance summary:

* **Training Accuracy:** 99.1%
* **Validation Accuracy:** 88.6%
* **Inference Speed:** ~25 FPS (varied 13–30 FPS)
* **Platform:** Jetson Nano
* **Real-time detection:** Yes

---

# 🎯 Objectives

The objectives of this project are:

* Detect driver drowsiness in real-time
* Classify driver alertness level
* Provide warning alerts to prevent accidents
* Evaluate different model strategies for classification performance
* Deploy deep learning models on embedded hardware

---

# 🧠 System Workflow

```
Camera
   ↓
Face Detection
   ↓
Facial Landmark Detection
   ↓
Feature Extraction (EAR & MAR)
   ↓
Data Preprocessing
   ↓
Inference (IndRNN Model)
   ↓
Prediction
   ↓
Alert System (Buzzer)
```

Classification output:

* Alert
* Low Alert
* Drowsy

---

# 🧩 Key Features

* Real-time driver drowsiness detection
* IndRNN-based time-series classification
* Sliding window temporal modeling
* Mediapipe-based facial landmark detection
* Deployment on Jetson Nano
* Real-time alert system
* Multiple model strategy evaluation
* Embedded AI system

---

# 🧠 Model Strategies

This project implements **two different classification strategies** to evaluate performance and robustness.

---

## 1) Single Classification Model

A single model is trained to classify all driver conditions simultaneously.

Characteristics:

* One model handles all classes
* Simpler deployment
* Faster inference pipeline
* Easier system integration
* Best-performing single model included in this repository

File used:

```
test_modelMediapipe.py
```

---

## 2) Multiple Classification Models

Multiple models are trained using pairwise class combinations.

Example:

Model 1:

Class 1 vs Class 2

Model 2:

Class 1 vs Class 3

Model 3:

Class 2 vs Class 3

Characteristics:

* Each model specializes in specific class pairs
* Improves classification robustness
* Reduces class confusion
* Ensemble-style decision making
* Best-performing multiple models included in this repository

File used:

```
test_2_modelMediapipe.py
```

---

# 📊 Best Models Included in Repository

The repository contains:

* Best-performing **single classification model**
* Best-performing **multiple classification models**
* Training history files
* Sliding window model

All models included are the final selected models based on performance evaluation.

---

# 📂 Dataset

The dataset was collected from recorded driver videos representing different driver conditions.

Each video frame is processed to extract:

* Eye Aspect Ratio (EAR)
* Mouth Aspect Ratio (MAR)

These values are used as sequential input features for training the IndRNN model.

Dataset split:

* Train
* Validation
* Test

---

# ▶️ Usage

This repository supports two different execution modes depending on the model strategy.

---

## Run Single Classification Model

```
python test_modelMediapipe.py
```

This mode:

* Uses one classification model
* Predicts all classes directly
* Simpler inference workflow

---

## Run Multiple Classification Models

```
python test_2_modelMediapipe.py
```

This mode:

* Uses multiple pairwise classification models
* Combines predictions
* Improves classification robustness

---

# 🛠️ Dependencies

This project uses standard Python libraries commonly available in computer vision and machine learning environments.

Main libraries:

* Python
* OpenCV
* Mediapipe
* NumPy
* TensorFlow / Keras
* Scikit-learn

Install dependencies manually if needed, for example:

```
pip install opencv-python mediapipe numpy tensorflow scikit-learn
```

---

# 🏗️ Hardware Setup

Hardware used:

* Jetson Nano
* Camera
* Buzzer

The system runs locally on Jetson Nano and processes video input in real time.

---

# 📁 Project Structure

```
drowsiness-classification/
|
├── 2_model/
|   ├── IndRNNAug5400_1&2.h5
|   ├── IndRNNAug5400_1&3.h5
|   ├── IndRNNAug5400_2&3.h5
|   ├── training_history_5400_1&2.npy
|   ├── training_history_5400_1&3.npy
│   └── training_history_5400_2&3.npy
|
├── Split/
|   ├── train/
|   ├── val/
|   └── test/
|
├── Split_revisi_3-1/
|   ├── train/
|   ├── val/
|   └── test/
|
├── model_slide_window/
|   ├── IndRNNAug5400_Slide_revisi80.h5
|   └── training_history_5400_Slide_revisi80.npy
|
├── ind_rnn.py
├── test_modelMediapipe.py
├── test_2_modelMediapipe.py
├── processed 2 kelas.ipynb
├── processed_revisi_3.ipynb
├── training_2_model.ipynb
├── training_aug_revisi3 copy.ipynb
├── data_dummy_bukaan.csv
├── data_dummy_ngantuk.csv
└── README.md
```

---

# 🚀 System Capability

The system is capable of:

* Real-time driver monitoring
* Multi-class drowsiness detection
* Embedded deployment
* Continuous alert generation
* Experimental comparison between model strategies

---

# 📌 Future Improvements

* Model optimization for higher FPS
* Night-time detection improvement
* Model compression (TensorRT)
* Real-time logging system
* Mobile notification integration

---

# 👤 Author

Itqan Fikri

Bachelor of Computer Engineering
Institut Teknologi Sepuluh Nopember (ITS)

Research Interests:

* Computer Vision
* Machine Learning
* Embedded Systems

---

# 📄 License

This project is developed for academic and research purposes.
