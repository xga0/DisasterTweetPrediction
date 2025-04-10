#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation with Multi-Head Attention for Tweet Disaster Classification

Model Architecture:
- GloVe Embeddings (50d)
- LSTM (64d, bidirectional, 2 layers)
- Multi-Head Attention (4 heads)
- Layer Normalization
- Dropout (0.1)

Training Features:
- Early Stopping with:
  * Moving Average (3 epochs)
  * Multiple Metrics (Loss & Accuracy)
  * Improvement Thresholds (0.001)
  * Patience (10 epochs)
- Learning Rate Scheduling (ReduceLROnPlateau)
- Gradient Clipping (max_norm=1.0)
- Adam Optimizer (lr=0.001)

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
df = df_raw[['keyword', 'text', 'target']].copy()

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
X = np.array(X, dtype=np.int64)
X = torch.tensor(X, dtype=torch.long)

y = df['target'].values.astype(np.float32)
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

# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final linear layer
        output = self.fc_out(context)
        return output, attention_weights

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Define enhanced model with multi-head attention
class EnhancedTweetClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super(EnhancedTweetClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, 64, bidirectional=True, batch_first=True, 
                           num_layers=2, dropout=0.1)
        
        self.attention = MultiHeadAttention(128) 
        self.layer_norm = nn.LayerNorm(128)
        
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Multi-head attention
        attn_out, attention_weights = self.attention(lstm_out)
        attn_out = self.layer_norm(attn_out)
        
        # Mean pooling
        x = torch.mean(attn_out, dim=1)
        
        # Feed-forward network
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x, attention_weights

# Update model initialization
model = EnhancedTweetClassifier(len(vocab), embedding_dim, embedding_matrix).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add learning rate scheduler without verbose parameter
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training loop with improved early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=1000, patience=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    # For moving average
    val_loss_window = []
    window_size = 3
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                outputs, _ = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update moving average window
        val_loss_window.append(val_loss)
        if len(val_loss_window) > window_size:
            val_loss_window.pop(0)
        
        # Calculate moving average
        moving_avg = sum(val_loss_window) / len(val_loss_window)
        
        # Improved early stopping criteria
        improvement_threshold = 0.001  # Minimum improvement to consider
        acc_improvement_threshold = 0.001  # Minimum accuracy improvement
        
        # Check if we have a significant improvement in either loss or accuracy
        loss_improved = val_loss < (best_val_loss - improvement_threshold)
        acc_improved = val_acc > (best_val_acc + acc_improvement_threshold)
        
        if loss_improved or acc_improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_enhanced.pth')
            print(f"New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                print(f'Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}')
                break
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Moving Avg: {moving_avg:.4f}')
    
    return train_losses, val_losses, train_accs, val_accs

# Update training call
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler
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
    plt.savefig('training_results_enhanced.png')
    plt.close()

# Load best model
model.load_state_dict(torch.load('best_model_enhanced.pth'))

# Evaluation
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Evaluating")
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, attention_weights = model(inputs)
            outputs = outputs.squeeze()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
            
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
    plt.savefig('evaluation_curves_enhanced.png')
    plt.close()
    
    # Save metrics and attention weights
    metrics = {
        'auroc': float(auroc),
        'average_precision': float(ap),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'attention_weights': np.array(all_attention_weights).tolist()
    }
    
    with open('evaluation_metrics_enhanced.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

# Evaluate the model
metrics = evaluate_model(model, test_loader)
print(f"\nFinal Evaluation Metrics:")
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Average Precision: {metrics['average_precision']:.4f}")

print("\nEvaluation completed successfully!") 