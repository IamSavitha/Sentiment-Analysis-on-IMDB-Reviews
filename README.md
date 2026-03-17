# Sentiment Analysis on IMDB Reviews 

This project implements two deep learning models for **binary sentiment classification** on the IMDB movie review dataset:

- **Model 1:** Bidirectional LSTM (Baseline)  
- **Model 2:** CNN + Bidirectional LSTM (Hybrid Model)  

The goal is to classify reviews as **positive or negative** while comparing sequential and hybrid architectures.

---

## Project Overview

Sentiment analysis is a core NLP task that involves understanding the emotional tone of text. In this project, we:

- Built models **from scratch (no pre-trained embeddings)**
- Compared **pure sequence modeling vs hybrid architectures**
- Evaluated performance using multiple metrics

---

## Objectives

- Implement **BiLSTM for sequence classification**
- Extend with **CNN layers for n-gram feature extraction**
- Compare model performance (accuracy, precision, recall, F1)
- Understand trade-offs between architectures

---

##  Dataset

- **Stanford IMDB Movie Review Dataset**
- 50,000 total reviews:
  - Train: 25,000  
  - Test: 25,000  
- Balanced dataset:
  - 50% positive  
  - 50% negative  

---

## Data Preprocessing

- Lowercasing text  
- Removing HTML tags  
- Removing punctuation and non-alphabetic characters  
- Stopword removal  
- Lemmatization  
- Word-level tokenization  

### Vocabulary

- Top **20,000 words**
- Special tokens:
  - `<PAD>` for padding  
  - `<UNK>` for unknown words  

### Sequence Length

- Max length: **256 tokens**

---

##  Model Architectures

---

### Model 1: Bidirectional LSTM (Baseline)

Pipeline:
Raw Review → Preprocess → Tokenize → Pad →
Embedding (20K × 128) →
BiLSTM (2 layers, hidden=256, bidirectional) →
Concat (forward + backward) →
Dropout →
Linear (512 → 1) →
Logit → BCEWithLogitsLoss



#### Key Features

- Captures **long-range dependencies**
- Handles contextual relationships (e.g., negation)
- Strong baseline for sequence classification

---

### Model 2: CNN + BiLSTM (Hybrid)

Pipeline:
Embedding → Permute →
[Conv1D k=2, k=3, k=4] →
Concatenate →
BiLSTM →
Dropout →
Linear → Output

#### Key Features

- CNN captures **local n-gram patterns**:
  - "not good"
  - "highly recommend"
- BiLSTM captures **global context**
- Combines **short-term + long-term features**

---

## Results

| Metric     | BiLSTM | CNN + BiLSTM |
|------------|--------|-------------|
| Accuracy   | 86.48% | 86.42%      |
| Precision  | 0.8605 | 0.8872      |
| Recall     | 0.8708 | 0.8346      |
| F1 Score   | 0.8656 | 0.8601      |
| Test Loss  | 0.3723 | 0.3282      |

---

## Key Observations

- Both models achieve **~86% accuracy**
- **CNN + BiLSTM → higher precision**
- **BiLSTM → higher recall and F1**
- Hybrid model captures **n-gram patterns better**

---

## Training Setup

- Optimizer: **Adam**
- Learning Rate: `1e-3`
- Weight Decay: `1e-5`
- Loss: **BCEWithLogitsLoss**
- Gradient Clipping: `1.0`
- Early Stopping: patience = 3

### Scheduler

- ReduceLROnPlateau (×0.5)

---

## Limitations

- Struggles with very long reviews (>300 words)  
- Stopword removal may remove important words like "not"  
- Limited embedding dimension (128)  
- Sequence truncation at 256 tokens  

---

## Strengths

- Strong performance without pre-trained embeddings  
- Hybrid architecture improves feature extraction  
- Robust preprocessing pipeline  
- Balanced dataset simplifies training  

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- Scikit-learn  

---

##  Project Structure
Sentiment-Analysis/
│
├── data/
├── models/
│ ├── bilstm.py
│ ├── cnn_bilstm.py
│
├── train.py
├── evaluate.py
├── utils.py
└── README.md


---

##  How to Run

### 1. Install Dependencies

pip install torch numpy scikit-learn


### 2. Train Model

python train.py


### 3. Evaluate Model

python evaluate.py


---

## References

- Maas et al., *Learning Word Vectors for Sentiment Analysis* (ACL 2011)  
- Hochreiter & Schmidhuber, *LSTM Networks*  
- Kim, *CNN for Sentence Classification*  

---

## Key Learnings

- Sequence models vs hybrid architectures  
- Importance of preprocessing in NLP  
- Precision vs recall trade-offs  
- Combining CNN + RNN for better performance  

---

##  Future Improvements

- Use pre-trained embeddings (GloVe, Word2Vec)  
- Add attention mechanisms  
- Increase vocabulary size  
- Use transformer-based models (BERT)  

---

##  Author

**Savitha Vijayarangan** 
 
**Keith Rajesh Gonsalves**