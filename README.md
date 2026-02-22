#  Disaster Tweets Classification (Deep Learning Version)

A deep learning implementation of the **Disaster Tweets classification task** using a Transformer-based model (BERT).  

This project is an upgraded version of my previous classical machine learning solution, moving from traditional feature engineering to pretrained language models.

---

## 📝 Project Overview

The goal is to classify whether a tweet describes a **real disaster event** or not.

Compared with the earlier ML pipeline (TF-IDF + Logistic Regression / Embedding + MLP), this version:

- Uses pretrained Transformer encoder (**BERT**)  
- Applies minimal but effective text cleaning  
- Implements a full PyTorch training pipeline  
- Supports evaluation metrics and model checkpoint saving  


---

## 🖥️ Model Architecture

**Encoder:**

- `bert-base-uncased` (HuggingFace Transformers)
- Pretrained contextual embeddings

**Classification Head:**

- Dropout layer (regularization)
- Linear classification layer (binary classification)

### Pipeline

```
Tweet Text
↓
Tokenizer 
↓
Transformer Encoder
↓
Pooling 
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
  - Train: `16`
  - Validation: `32`  
- Max sequence length: `128`  
- Optimizer: AdamW  
- Learning rate: `2e-5`  
- Scheduler: Linear warmup + decay  
- Loss function: CrossEntropyLoss  
- Epochs: `3`  

### Additional training practices:

- Gradient clipping  
- Stratified train/validation split  
- Best model checkpoint saving (based on `F1 score`)

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy  
- F1 Score (primary metric)  
- Validation loss  

Example output format:

```
Epoch 1/3
Train loss: 0.4845
Val loss: 0.3791
Val acc: 0.8444
Val f1: 0.8121

Epoch 2/3
Train loss: 0.3344
Val loss: 0.3727
Val acc: 0.8536
Val f1: 0.8200

Epoch 3/3
Train loss: 0.2481
Val loss: 0.4566
Val acc: 0.8424
Val f1: 0.8148
```

The best model will be saved as:  `best_model.pt`

---

## 📂 Project Structure
```
data/
└── raw/
├── sample_submission.csv
├── train.csv
└── test.csv

src/
└── training scripts

best_model.pt
README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch transformers scikit-learn pandas numpy
```

### 2. Prepare dataset

Place dataset:
```
data/raw/train.csv
data/raw/test.csv
```
(Kaggle Disaster Tweets dataset)

### 3. Train model
```
python train.py
```

---

## 🔎 Key Learning Outcomes
Through this project I:
- Implemented Transformer fine-tuning in PyTorch
- Learned tokenizer–encoder integration
- Understood training dynamics of pretrained LMs
- Practiced evaluation and checkpointing
- Compared classical ML vs deep learning NLP pipelines

---

## 📈 Future Improvements
Potential next steps:
- Hyperparameter tuning
- Larger pretrained models (RoBERTa, DeBERTa)
- Data augmentation
- LoRA / parameter-efficient fine-tuning
- Error analysis & explainability

---

## 📚 Dataset Source
Kaggle:

Natural Language Processing with Disaster Tweets

https://www.kaggle.com/competitions/nlp-getting-started/data

---

## ✨ Notes
This project is part of my ongoing NLP/LLM learning path, moving from:

`Traditional ML NLP → Deep Learning NLP → LLM Fine-tuning`