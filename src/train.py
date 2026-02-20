import torch
import transformers
import sklearn
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# ===== 步骤一    数据集读取 =====

train_df = pd.read_csv("data/raw/train.csv")
test_df  = pd.read_csv("data/raw/test.csv")



# ===== 步骤二    数据轻清洗 =====

def minimal_clean(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "URL", text)       # 1) URL -> URL
    text = re.sub(r"@\w+", "USER", text)                  # 2) @user -> USER
    text = re.sub(r"\s+", " ", text).strip()              # 3) compress spaces
    return text

train_df["text_clean"] = train_df["text"].apply(minimal_clean)    # 执行数据清洗



# ===== 步骤三    划分训练集/测试集 =====

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text_clean"].tolist(),
    train_df["target"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=train_df["target"]      # 保证 train 和 val 中正负样本比例 ≈ 原始数据
)



# ===== 步骤四    向量化 + 特征输入 =====

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TweetDataset(
    train_texts,
    train_labels,
    tokenizer,
    max_len=128
)

val_dataset = TweetDataset(
    val_texts,
    val_labels,
    tokenizer,
    max_len=128
)



# ===== 步骤五    数据加载 =====

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

batch = next(iter(train_loader))



# ===== 步骤六    模型（Encoder + 分类头）=====

class TransformerClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)           # 加载预训练 Transformer 编码器
        hidden_size = self.encoder.config.hidden_size                  # 读取 encoder 的隐藏层维度 H
        self.dropout = nn.Dropout(dropout)                             # 定义 dropout 层
        self.classifier = nn.Linear(hidden_size, num_labels)           # 定义最终分类层（线性层）

    def forward(self, input_ids, attention_mask):                      # 定义前向传播
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)         #把输入送进 Transformer 编码器

        # BERT 有 pooler_output；有些模型没有，所以做兼容写法
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]  # [CLS]

        x = self.dropout(pooled)
        logits = self.classifier(x)            # 线性层输出 logits，形状 [B, 2]
        return logits 
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "bert-base-uncased"
model = TransformerClassifier(MODEL_NAME).to(device)                   # 创建模型实例，并把模型参数搬到 GPU/CPU

batch = next(iter(train_loader))                                       # 从 train_loader 里拿到第一个 batch ，创建一个迭代器，取出第一批数据
input_ids = batch["input_ids"].to(device)                              # 把 input_ids 搬到 GPU
attention_mask = batch["attention_mask"].to(device)                    # 把 mask 搬到 GPU
logits = model(input_ids=input_ids, attention_mask=attention_mask)     # 调看到 forward，执行整套 encoder + 分类头



# ===== 步骤七    模型训练=====

criterion = nn.CrossEntropyLoss()                             # 创建损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)    # 创建优化器 AdamW，用来根据梯度更新模型参数

num_epochs = 3
total_steps = num_epochs * len(train_loader)                  # 计算训练总 step 数

scheduler = get_linear_schedule_with_warmup(                  # 学习率调度
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):     # 定义“训练一个 epoch”的函数
    model.train()   # 开启训练模式（dropout 生效）
    total_loss = 0.0

    for step, batch in enumerate(dataloader):                 # 遍历 dataloader，每次拿一个 batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()  # 1️⃣ 清空上一轮梯度

        logits = model(input_ids=input_ids, attention_mask=attention_mask)    # 2️⃣ 前向传播
        loss = criterion(logits, labels)                                      # 3️⃣ 算 loss

        loss.backward()              # 4️⃣ 反向传播（算梯度）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 5️⃣ 防梯度爆炸

        optimizer.step()   # 6️⃣ 更新参数
        scheduler.step()   # 7️⃣ 更新学习率

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"step {step}, loss {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss



# ===== 步骤八    验证 / 指标 / 保存 best 模型=====

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()            # ① 切到评估模式（dropout 关闭）
    total_loss = 0.0        # 禁止 autograd 构建计算图，节省显存、速度更快，Eval 必须写

    all_preds = []
    all_labels = []

    with torch.no_grad():   # ② 不计算梯度
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)

    return avg_loss, acc, f1

num_epochs = 3
best_f1 = 0.0

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        device
    )

    val_loss, val_acc, val_f1 = eval_one_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    print(
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.4f} | "
        f"Val f1: {val_f1:.4f}"
    )

    # 保存验证集上最好的模型
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        print(f"✅ Saved new best model (f1={best_f1:.4f})")