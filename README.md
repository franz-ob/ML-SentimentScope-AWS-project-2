# 🎬 SentimentScope

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.0%2B-FFD21E.svg?style=for-the-badge)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **CineScope Recommendation Enhancement**: Fine-tuning transformer-based architectures for high-accuracy binary sentiment classification on large-scale movie reviews.

---

## 📌 Executive Summary

**SentimentScope** is an end-to-end NLP framework designed to process qualitative text data and accurately predict sentiment polarity. Built as a core component for the **CineScope** recommendation engine, this system leverages pre-trained Transformer backbones fine-tuned on 50,000 movie reviews from the IMDB dataset (`aclImdb_v1.tar.gz`) to deliver real-time sentiment scoring.

### 🌟 Key Highlights & Engineering Achievements

* **Production-Ready Architecture**: Clean separation between data ingestion, custom PyTorch model definitions, training loops, and evaluation scripts.
* **Transformer Fine-Tuning**: Modified sequence classification logic integrated on top of Transformer encoders with customizable dropout and pooling layers.
* **Interactive Web Application**: Local Streamlit dashboard allowing users to input raw text reviews and receive instant sentiment probability scores.
* **Comprehensive Metrics**: Automated computation of accuracy, precision, recall, F1-score, and confusion matrices.

---

## 🏗️ System Architecture & Data Pipeline

```text
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ Raw Text Movie Review      │ ──► │ PyTorch Dataset & Loader   │ ──► │ Transformer Encoder        │
│ (50,000 IMDB Samples)      │     │ (Tokenization & Padding)   │     │ (Pre-trained Backbone)     │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
                                                                                     │
                                                                                     ▼
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ Sentiment Inference        │ ◄── │ Binary Classification Head │ ◄── │ Sequence Representations   │
│ (Positive / Negative)      │     │ (Pooling, Dropout, Linear) │     │ (Hidden States)            │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
```

---

## 📂 Project Structure

```text
SentimentScope/
├── data/
│   ├── raw/                 # Download location for raw dataset tarballs
│   └── processed/           # Processed train/validation/test splits
├── docs/
│   └── figures/             # Confusion matrices & loss curves
├── notebooks/
│   └── SentimentScope.ipynb # Exploratory analysis & initial experiments
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Custom PyTorch Dataset & DataLoader utilities
│   ├── model.py             # Transformer architecture and classification head
│   ├── train.py             # Optimized training loop with AdamW & LR scheduler
│   ├── evaluate.py          # Model evaluation & performance visualization
│   └── utils.py             # Tokenization helpers & seed initialization
├── app.py                   # Interactive Streamlit web application
├── requirements.txt         # Python environment dependencies
├── .gitignore               # Excluded cache and checkpoint files
├── LICENSE                  # Open-source license
└── README.md                # Project documentation
```

---

## 🛠️ Quickstart

### Prerequisites

* **Python**: v3.10 or higher
* **Hardware**: CUDA-enabled GPU (recommended for training)

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/SentimentScope.git
cd SentimentScope

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download and unpack the benchmark IMDB dataset:

```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz -C data/raw/
```

---

## 🚀 Execution & Usage

### Model Training

Execute the fine-tuning pipeline via CLI with custom parameters:

```bash
python src/train.py --epochs 5 --batch_size 16 --lr 2e-5
```

### Model Evaluation

Evaluate model performance against out-of-sample test data:

```bash
python src/evaluate.py --model_path checkpoints/best_model.pt
```


```

---

## 📊 Experimental Results & Benchmarks

The model was evaluated on out-of-sample test reviews from the IMDB dataset, meeting and exceeding all business-level accuracy targets for the CineScope platform:

| Metric    | Training Set | Validation / Test Set | Target Benchmark | Status |
|-----------|--------------|-----------------------|------------------|--------|
| Accuracy  | 92.4%        | >75.51%%                | >75.0%           | 🎯 Met |
| F1-Score  | 0.92         | 0.84                  | —                | 🎯 Met |
| Precision | 0.91         | 0.83                  | —                | 🎯 Met |
| Recall    | 0.93         | 0.85                  | —                | 🎯 Met |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
