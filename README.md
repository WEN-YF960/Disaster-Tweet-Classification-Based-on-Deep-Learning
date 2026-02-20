#  Disaster Tweets Classification (Deep Learning Version)

A deep learning implementation of the **Disaster Tweets classification task** using a Transformer-based model (BERT).  

This project is an upgraded version of my previous classical machine learning solution, moving from traditional feature engineering to pretrained language models.

---

##  Project Overview

The goal is to classify whether a tweet describes a **real disaster event** or not.

Compared with the earlier ML pipeline (TF-IDF + Logistic Regression / Embedding + MLP), this version:

- Uses pretrained Transformer encoder (**BERT**)  
- Applies minimal but effective text cleaning  
- Implements a full PyTorch training pipeline  
- Supports evaluation metrics and model checkpoint saving  

This reflects a transition from traditional NLP to modern deep learning approaches.

---

##  Model Architecture

**Encoder:**

- `bert-base-uncased` (HuggingFace Transformers)
- Pretrained contextual embeddings

**Classification Head:**

- Dropout layer (regularization)
- Linear classification layer (binary classification)

### Pipeline

```Tweet Text
↓
Tokenizer (BERT)
↓
Transformer Encoder
↓
Pooling ([CLS] / pooler_output)
↓
Dropout
↓
Linear Layer
↓
Prediction
```

---

## 🧹 Data Processing

### Minimal text cleaning:

- Replace URLs → `URL`
- Replace user mentions → `USER`
- Normalize whitespace

This keeps semantic information while reducing noise.

---

## ⚙️ Training Setup

**Framework:**

- PyTorch  
- HuggingFace Transformers  

**Key configurations:**

- Batch size:
  - Train: 16  
  - Validation: 32  
- Max sequence length: 128  
- Optimizer: AdamW  
- Learning rate: `2e-5`  
- Scheduler: Linear warmup + decay  
- Loss function: CrossEntropyLoss  
- Epochs: 3  

### Additional training practices:

- Gradient clipping  
- Stratified train/validation split  
- Best model checkpoint saving (based on F1 score)

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy  
- F1 Score (primary metric)  
- Validation loss  

Example output format:

Epoch 1/3
Train loss: ...
Val loss: ...
Val acc: ...
Val f1: ...


The best model is saved as: **best_model.pt**

---

## 📂 Project Structure

data/
└── raw/
├── train.csv
└── test.csv

src/
└── training scripts

best_model.pt
README.md

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch transformers scikit-learn pandas numpy

### 2. Prepare dataset

Place dataset:

