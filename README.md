# Disaster Tweet Prediction
Kaggle Competition: [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview).

## Features
- Bi-LSTM architecture
- GloVe Embeddings:
  - Original: 100-dimensional
  - PyTorch: 50-dimensional
- Text preprocessing:
  - Emoticon fix (e.g., ":D" to "Laugh") using [emoticon_fix](https://github.com/xga0/emoticon_fix)
  - Thousand separators fix (e.g., "1,000,000" to "1000000")
  - URL and @ mentions removal
  - Lemmatization and stopwords removal
- Early stopping with patience
- Comprehensive evaluation metrics (AUROC, Average Precision)

## Implementations
### Original Implementation (`tweetdisaster.py`)
![performance](https://raw.githubusercontent.com/xga0/DisasterTweetPrediction/master/img/disaster%20tweet.png)

### PyTorch Implementations
1. **Basic Implementation** (`tweetdisaster_pt.py`)
   - Standard Bi-LSTM architecture
   - GloVe embeddings (50d)
   - Basic early stopping

2. **Enhanced Implementation with Multi-Head Attention** (`tweetdisaster_attention.py`)
   ![evaluation_curves_enhanced](https://raw.githubusercontent.com/xga0/DisasterTweetPrediction/refs/heads/master/img/evaluation_curves_enhanced.png)
   
   **Enhanced Features**:
   - Multi-Head Attention (4 heads)
   - Layer Normalization
   - Improved Early Stopping:
     - Moving Average (3 epochs)
     - Multiple Metrics (Loss & Accuracy)
     - Improvement Thresholds (0.001)
     - Patience (10 epochs)
   - Learning Rate Scheduling (ReduceLROnPlateau)
   - Gradient Clipping (max_norm=1.0)
   - Adam Optimizer (lr=0.001)

## Requirements
See `requirements.txt` for detailed package versions.

## Usage
1. Ensure all required files are in the same directory:
   - `train.csv`
   - `glove.6B.50d.txt` (for PyTorch implementation)
2. Run the script:
   ```bash
   # Original implementation
   python tweetdisaster.py
   
   # PyTorch implementation
   python tweetdisaster_pt.py
   
   # Enhanced implementation with attention
   python tweetdisaster_attention.py
   ```
3. Results will be saved as:
   - `best_model.pth`: Best model weights (original)
   - `best_model_enhanced.pth`: Best model weights (attention)
   - `training_results.png`: Training/validation metrics (original)
   - `training_results_enhanced.png`: Training/validation metrics (attention)
   - `evaluation_curves.png`: ROC and PR curves (original)
   - `evaluation_curves_enhanced.png`: ROC and PR curves (attention)
   - `evaluation_metrics.json`: Detailed evaluation metrics (original)
   - `evaluation_metrics_enhanced.json`: Detailed evaluation metrics (attention)
