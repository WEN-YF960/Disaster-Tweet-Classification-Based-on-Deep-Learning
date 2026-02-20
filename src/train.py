import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)


# ===== 1. Load Data =====

train = pd.read_csv("data/raw/train.csv")
test  = pd.read_csv("data/raw/test.csv")



# ===== 2. Minimal Text Cleaning =====

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "URL", text)       # 1) URL -> URL
    text = re.sub(r"@\w+", "USER", text)                  # 2) @user -> USER
    text = re.sub(r"\s+", " ", text).strip()              # 3) compress spaces
    return text

train["text_clean"] = train["text"].apply(clean_text)    # Apply text cleaning



# ===== 3. Train/Validation Split =====

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train["text_clean"],
    train["target"],
    test_size=0.2,
    stratify=train["target"],      # Keep class distribution in train and val similar to original data
    random_state=42
)



# ===== 4. Tokenization & Feature Encoding =====

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer)



# ===== 5. Data Loading =====

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)



# ===== 6. Model  =====

class TransformerClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)           # Load pretrained Transformer encoder
        hidden_size = self.encoder.config.hidden_size                  # Get hidden size from encoder config
        
        self.dropout = nn.Dropout(0.1)                                 # Define dropout layer
        self.classifier = nn.Linear(hidden_size, num_labels)           # Define final classification layer

    def forward(self, input_ids, attention_mask):                      # Define forward pass
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) 

        # Use pooled output if available, otherwise CLS token cases
        pooled = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0]
        )

        logits = self.classifier(self.dropout(pooled))
        return logits               
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier().to(device)    



# ===== 7. Model Training =====

criterion = nn.CrossEntropyLoss()                             # Define loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)    # Initialize AdamW optimizer to update model parameters based on gradients

num_epochs = 3
total_steps = num_epochs * len(train_loader)                  # Compute total number of training steps

scheduler = get_linear_schedule_with_warmup(                  # Learning rate scheduler
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def train_one_epoch():     # Define function to train one epoch
    model.train()          # Enable training mode (dropout active)
    running_loss = 0.0

    for step, batch in enumerate(train_loader):                 # Iterate through dataloader, processing one batch at a time
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()  # 1.Clear gradients from previous iteration

        logits = model(input_ids, attention_mask)    # 2.Forward pass
        loss = criterion(logits, labels)             # 3.Compute loss

        loss.backward()                                          # 4.Backward pass (compute gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 5.Gradient clipping to prevent exploding gradients
        
        optimizer.step()   # 6.Update model parameters
        scheduler.step()   # 7.Update learning rate

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"[train] step={step} loss={loss.item():.4f}")

    return running_loss / len(train_loader)



# ===== 8. Validation / Metrics / Save Best Model =====

def evaluate():
    model.eval()            # Switch to evaluation mode (dropout disabled)
    
    total_loss = 0.0        # Disable autograd to save memory and speed up evaluation (required for eval)
    preds_all, labels_all = [], []

    with torch.no_grad():   # No gradient computation
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all)

    return total_loss / len(val_loader), acc, f1

best_f1 = 0.0

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

    train_loss = train_one_epoch()

    val_loss, val_acc, val_f1 = evaluate()

    print(
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.4f} | "
        f"Val f1: {val_f1:.4f}"
    )

    # Save best model based on validation performance
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        print(f"Saved new best model (f1={best_f1:.4f})")