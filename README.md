# 🚨 AI Criminal Detection System

**3rd Place Winner** | AI/ML Hackathon 2024

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Multi-stage AI pipeline combining real-time object detection and facial recognition for automated criminal identification.

---

## 🎯 Overview

An intelligent surveillance system that automatically detects and identifies individuals in real time camera feeds using a five stage AI pipeline.

**Pipeline**  
YOLOv5 (Person Detection) → MTCNN (Face Detection) → RetinaFace (Alignment) → ArcFace (Feature Extraction) → Cosine Similarity (Matching)

**Achievement**  
3rd place in a competitive AI hackathon for technical depth and real world applicability.

---

## ✨ Key Features

- Real time webcam monitoring with YOLOv5 person detection  
- Multi face processing with simultaneous identification  
- High accuracy with low false positives  
- Sub second GPU inference  
- Visual output with bounding boxes and confidence scores  
- Google Colab compatible webcam support  

---

## 🏗️ Architecture

Live Camera Feed  
↓  
YOLOv5 Person Detection  
↓  
MTCNN Face Detection  
↓  
RetinaFace Alignment  
↓  
ArcFace Embeddings  
↓  
Cosine Similarity Matching  
↓  
Annotated Output  

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-----------|
| Detection | YOLOv5 |
| Face | MTCNN |
| Alignment | RetinaFace |
| Embeddings | ArcFace |
| Matching | Cosine Similarity |
| Frameworks | PyTorch, TensorFlow, OpenCV |

---

## 📦 Installation

```bash
git clone https://github.com/[your-username]/criminal-detection-ai-pipeline.git
cd criminal-detection-ai-pipeline
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
jupyter notebook notebooks/CrimeDetection.ipynb
```

---

## 📁 Project Structure

```
criminal-detection-ai-pipeline/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── CrimeDetection.ipynb
├── data/
│   ├── criminal_db/
│   └── test_images/
└── results/
    └── output_samples/
```
---
## QUICK SETUP CHECKLIST

```bash
git init
git add .
git commit -m "Initial commit: AI Criminal Detection System"
git remote add origin https://github.com/[your-username]/criminal-detection-ai-pipeline.git
git push -u origin main
```

Replace placeholders and add demo images to results/output_samples.
