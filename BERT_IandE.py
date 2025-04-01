import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import time
import torch
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc

df = pd.read_csv('cleaned_data.csv', encoding='ISO-8859-1')
print(df.head())

# See the distribution of dataset
# Create figure and axis
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

dimensions = {
    "I/E": ["I", "E"],
    "S/N": ["S", "N"],
    "F/T": ["F", "T"],
    "P/J": ["P", "J"],
}

# Draw the plot
for ax, (col, labels) in zip(axes.flatten(), dimensions.items()):
    counts = df[col].value_counts()
    ax.bar(labels, [counts.get(1, 0), counts.get(0, 0)], color='skyblue')
    ax.set_title(f"{col} Proportion")
    ax.set_ylabel("Count")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)

plt.tight_layout()
plt.show()

# Split dataset into training set, validation set and test set
X = df['clean_posts']
y = df['I/E']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# X_train: 60%
# X_val: 20%
# X_test: 20%

print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

# Bert Model
# 1. Bert Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_data(texts, tokenizer, max_len=256):
    return tokenizer(texts.tolist(), return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)

train_encodings = tokenize_data(X_train, tokenizer)
val_encodings = tokenize_data(X_val, tokenizer)
test_encodings = tokenize_data(X_test, tokenizer)

# 2. Create data loader
class MBTIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = MBTIDataset(train_encodings, y_train.tolist())
val_dataset = MBTIDataset(val_encodings, y_val.tolist())
test_dataset = MBTIDataset(test_encodings, y_test.tolist())

# 3. WeightedRandomSampler
class_counts = y_train.value_counts().to_dict()
weights = [1.0 / class_counts[label] for label in y_train]
sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=sampler)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

# 4. Load and fine-tune pre-trained Bert models
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(DEVICE)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)

def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
                
    return correct_pred.float() / num_examples * 100

start_time = time.time()
NUM_EPOCHS = 3

for epoch in range(NUM_EPOCHS):
    
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        
        ### Prepare data
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        ### Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        
        ### Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        ### Logging
        if not batch_idx % 250:
            print(f'Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d}'
                  f' | Batch'
                  f'{batch_idx:04d}/'
                  f'{len(train_loader):04d} | '
                  f'Loss: {loss:.4f}')
            
    model.eval()
    
    with torch.set_grad_enabled(False):
        print(f'Training accuracy: '
              f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
              f'\nValid accuracy: '
              f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')
        
    print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
    
print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')