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
### Original Implementation
![performance](https://raw.githubusercontent.com/xga0/DisasterTweetPrediction/master/img/disaster%20tweet.png)

### PyTorch Implementation
The PyTorch version includes additional features:
- Progress bars for training and evaluation
- Training and validation metrics visualization
- ROC and Precision-Recall curves
- Class-weighted loss function for handling imbalanced data

#### Evaluation Curves
![evaluation_curves](https://raw.githubusercontent.com/xga0/DisasterTweetPrediction/refs/heads/master/img/evaluation_curves.png)

## Requirements
See `requirements.txt` for detailed package versions.

## Usage
1. Ensure all required files are in the same directory:
   - `train.csv`
   - `glove.6B.50d.txt` (for PyTorch implementation)
2. Run the script:
   ```bash
   python tweetdisaster_pt.py
   ```
3. Results will be saved as:
   - `best_model.pth`: Best model weights
   - `training_results.png`: Training/validation metrics
   - `evaluation_curves.png`: ROC and PR curves
   - `evaluation_metrics.json`: Detailed evaluation metrics
