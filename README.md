# 🎬 SentimentScope

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.0%2B-FFD21E.svg?style=for-the-badge)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **CineScope Recommendation Enhancement**: A transformer-based binary sentiment classifier for movie reviews, built from scratch in PyTorch.

---

## 📌 Overview

**SentimentScope** is a sentiment analysis project built for the fictional **CineScope** recommendation engine. It classifies IMDB movie reviews as positive or negative using a GPT-style transformer that is adapted for classification and trained from scratch in PyTorch. Text is tokenized with the pretrained `bert-base-uncased` tokenizer from Hugging Face.

The model reaches **75.51% accuracy on the 25,000-review IMDB test set**, meeting the project target of >75%.

### Highlights

* **Custom transformer**: Attention heads, multi-head attention, feed-forward layers and transformer blocks implemented in PyTorch.
* **Classification adaptation**: Mean pooling over token embeddings plus a linear head with two outputs.
* **Custom data pipeline**: A PyTorch `Dataset` and `DataLoader` with subword tokenization, truncation and padding.
* **Exploratory analysis**: Class balance, review-length distributions and sample reviews.

---

## 🏗️ Architecture & Pipeline

```text
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ Raw IMDB Review Text       │ ──► │ BERT Tokenizer             │ ──► │ Token + Position           │
│ (25k train / 25k test)     │     │ (max length 128, padded)   │     │ Embeddings                 │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
                                                                                     │
                                                                                     ▼
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ Prediction                 │ ◄── │ Mean Pooling + Linear Head │ ◄── │ 4 Transformer Blocks       │
│ (Positive / Negative)      │     │ (128 → 2 logits)           │     │ + Final LayerNorm          │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
```

### Model Configuration

| Parameter            | Value                          |
|----------------------|--------------------------------|
| Vocabulary size      | 30,522 (`bert-base-uncased`)   |
| Embedding dimension  | 128                            |
| Transformer layers   | 4                              |
| Attention heads      | 4 (head size 32)               |
| Context length       | 128 tokens                     |
| Dropout              | 0.1                            |
| Classes              | 2                              |

### Training Setup

| Setting        | Value                  |
|----------------|------------------------|
| Optimizer      | AdamW, learning rate 3e-4 |
| Loss           | Cross-entropy          |
| Batch size     | 32                     |
| Epochs         | 3                      |
| Train / val split | 22,500 / 2,500 (90/10, shuffled, seed 42) |
| Test set       | 25,000 (official IMDB test split) |

---

## 📂 Project Structure

```text
SentimentScope/
├── SentimentScope.ipynb     # Full pipeline: data, model, training, evaluation
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

The IMDB dataset (`aclImdb/`) is not included in the repository. See setup below.

---

## 🛠️ Quickstart

### Prerequisites

* **Python**: 3.10 or higher
* **Hardware**: CUDA-enabled GPU recommended for training (the notebook falls back to CPU, but it will be slow)

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/franz-ob/SentimentScope.git
cd SentimentScope

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download and unpack the IMDB dataset in the project root:

```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

This creates an `aclImdb/` folder with `train/` and `test/` subfolders, which the notebook reads from.

### 3. Run the Notebook

```bash
jupyter notebook SentimentScope.ipynb
```

Run the cells in order. The notebook covers:

1. Loading and exploring the dataset
2. Building the `IMDBDataset` class and `DataLoader`s
3. Defining the transformer model
4. Training for 3 epochs with validation after each epoch
5. Evaluating on the test set

---

## 📊 Results

| Stage                         | Accuracy |
|-------------------------------|----------|
| Validation, before training   | 49.92%   |
| Validation, after epoch 2     | 74.08%   |
| Validation, after epoch 3     | 76.16%   |
| **Test set (25,000 reviews)** | **75.51%** |

The project target was >75% test accuracy, which the model meets. Validation and test accuracy are close, which suggests the model is not overfitting at this scale.

---

## ⚠️ Limitations & Future Work

* **Undertrained**: Training loss was still decreasing at epoch 3. More epochs, a larger embedding size or more layers should improve accuracy.
* **Truncation**: Reviews are cut to 128 tokens, while the average review is about 234 words, so information in longer reviews is lost.
* **Attention mask**: The blocks keep the causal (left-to-right) mask from the original generative architecture. Bidirectional attention is more standard for classification and may help.
* **No pretrained weights**: The model learns language from scratch. Fine-tuning a pretrained encoder such as BERT would likely score considerably higher.
* **Metrics**: Only accuracy is reported. Precision, recall, F1 and a confusion matrix could be added.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
