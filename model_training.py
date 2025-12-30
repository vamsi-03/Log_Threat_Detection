import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import os

# --- Pure Training Configuration ---
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256  
BATCH_SIZE = 32
EPOCHS = 1     

class LogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def run_training():
    if not os.path.exists("cyber_logs.csv"):
        print("Error: cyber_logs.csv not found. Please run real_data_loader.py first.")
        return
        
    # Standard Loading
    df = pd.read_csv("cyber_logs.csv")
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label']) 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = LogDataset(train_df.text.to_list(), train_df.label.to_list(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    
    # We still use a small weight (5.0) to prioritize catching threats (Recall)
    weights = torch.tensor([1.0, 5.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    model.train()
    print(f"Starting training on {device}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
    model.save_pretrained("./cyber_bert_model")
    tokenizer.save_pretrained("./cyber_bert_model")
    print("Model fine-tuning complete and saved.")

if __name__ == "__main__":
    run_training()