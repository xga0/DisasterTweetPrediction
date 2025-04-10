#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:20:44 2020
Modified for PyTorch implementation

@author: seangao
"""

import pandas as pd
import re
import contractions
from emoticon_fix import emoticon_fix
from nltk.corpus import stopwords
import en_core_web_sm
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df_raw = pd.read_csv('train.csv')
df = df_raw[['keyword', 'text', 'target']].copy()  # Create a copy to avoid SettingWithCopyWarning

# KEYWORD CHECK
df.loc[:, 'keyword'] = df['keyword'].fillna('no keyword')
df.loc[:, 'keyword'] = df['keyword'].str.replace('%20', ' ')

# TEXT PREPROCESS
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # remove urls
    text = re.sub(r'@\w+', '', text)  # remove @s
    text = contractions.fix(text)  # fix contractions
    text = emoticon_fix.emoticon_fix(text)  # fix emoticons
    text = re.sub(r'(\d),(\d)', r'\1\2', text)  # fix thousand separator
    text = re.sub('[^A-Za-z0-9]+', ' ', text)  # remove punctuations
    text = text.lower()
    return text

# Lemmatization and stopwords removal
sp = en_core_web_sm.load()
stopwords = set(stopwords.words('english'))

def lemma(input_str):
    s = sp(input_str)
    input_list = [word.lemma_ for word in s]
    output = ' '.join(word for word in input_list if word not in stopwords)
    return output

# Process all texts
lst_text = df['text'].to_list()
lst_text = [preprocess_text(x) for x in tqdm(lst_text, desc="Preprocessing texts")]
lst_text = [lemma(x) for x in tqdm(lst_text, desc="Lemmatizing texts")]
lst_text = [re.sub(' +', ' ', x).strip() for x in lst_text]

# Create vocabulary and tokenize
from collections import Counter
words = ' '.join(lst_text).split()
word_counts = Counter(words)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}
vocab['<pad>'] = 0

# Convert texts to sequences
maxlen = 100
X = [[vocab[word] for word in text.split()[:maxlen]] for text in lst_text]
X = [seq + [0] * (maxlen - len(seq)) for seq in X]
X = np.array(X, dtype=np.int64)  # Convert to numpy array first
X = torch.tensor(X, dtype=torch.long)

y = df['target'].values.astype(np.float32)  # Convert to numpy array first
y = torch.tensor(y, dtype=torch.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42
)
X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TweetDataset(X_train, y_train)
test_dataset = TweetDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load GloVe embeddings
embedding_dim = 50
embedding_path = 'glove.6B.50d.txt'

embeddings_index = {}
with open(embedding_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define model
class TweetClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super(TweetClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, 64, bidirectional=True, batch_first=True, 
                           num_layers=2, dropout=0.1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = torch.mean(lstm_out, dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = TweetClassifier(len(vocab), embedding_dim, embedding_matrix).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=5):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, train_accs, val_accs

# Train the model
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, test_loader, criterion, optimizer
)

# Plot training results
def plot_results(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()  # Close the figure to free memory

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Evaluating")
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with current batch metrics
            batch_auroc = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
            val_pbar.set_postfix({'batch_auroc': f'{batch_auroc:.3f}'})
    
    # Calculate metrics
    auroc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_curves.png')
    plt.close()  # Close the figure to free memory
    
    # Save metrics
    metrics = {
        'auroc': float(auroc),
        'average_precision': float(ap),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }
    
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

# Evaluate the model
metrics = evaluate_model(model, test_loader)
print(f"\nFinal Evaluation Metrics:")
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Average Precision: {metrics['average_precision']:.4f}")

print("\nEvaluation completed successfully!") 