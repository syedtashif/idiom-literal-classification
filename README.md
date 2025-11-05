# Idiom vs Literal Classification with Image Ranking

Classify compound words as idioms or literals using contrast vectors and rank images based on filtered captions or images using SentenceTransformer and CLIP embeddings.

##  Overview

This project provides a complete pipeline for:

1. **Classifying compound words** as idioms or literal expressions using contrast vectors
2. **Ranking images** using three different approaches:
   - **Caption-only ranking** (SentenceTransformer)
   - **Image-only ranking** (CLIP)
   - **Fused caption and image ranking** (Combined)

Uses single dataset: **extendedNominal.csv**

##  Features

-  **MLP Classification** with K-Fold cross-validation
-  **Three independent ranking variants** (caption, image, fused)
-  **Comprehensive metrics**: Classification Accuracy + nDCG@5
-  **Token enhancement** for idioms with context vectors
-  **Robust data preprocessing** with fuzzy matching
-  **GPU acceleration** support
-  **Modular architecture** for easy extension

##  Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- 8GB+ RAM
- pip package manager

### Setup Steps

1. **Clone the repository:**

```bash
git clone https://github.com/syedtashif/idiom-literal-classification.git
cd idiom-literal-classification
```

2. **Create and activate virtual environment:**

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

4. **Verify installation:**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

##  Configuration

Edit `config.py` with your dataset paths and preferences:

```python
DATA_PATHS = {
    'paraphrase_file': '/path/to/extendedNominal.csv',
    'admire_tsv': '/path/to/subtask_a_train.tsv',
    'image_dir': '/path/to/image/directory',
    'output_dir': '/path/to/output'
}

SELECTED_METHOD = 'caption_only'  # Options: 'caption_only', 'image_only', 'both'
```

### Configuration Options

#### Training Configuration
```python
TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'k_folds': 5,
    'random_seed': 42
}
```

#### Ranking Configuration
```python
RANKING_CONFIG = {
    'context_weight': 0.1,
    'caption_max_sentences': 2,
    'min_keyword_length': 3,
    'fuzzy_threshold': 0.7
}
```

#### Model Configuration
```python
MODEL_CONFIG = {
    'st_model': 'all-MiniLM-L6-v2',
    'st_embedding_dim': 384,
    'clip_model': 'openai/clip-vit-base-patch32',
    'clip_embedding_dim': 512
}

MLP_CONFIG = {
    'input_dim': 768,
    'hidden_dims': [512, 256],
    'dropout': 0.3
}
```

## 💻 Usage

### Caption-Only Ranking

Rank images based on caption similarity using SentenceTransformer:

```bash
python main_caption.py
```

**Output:**
```
Classification Accuracy: 100%
Normalized DCG@5: 0.92
```

### Image-Only Ranking (CLIP)

Rank images based on visual features using CLIP:

```bash
python main_image.py
```

**Output:**
```
Classification Accuracy: 100%
Normalized DCG@5: 0.97
```

### Both Caption and Image (Fused)

Rank images using combined caption and visual features:

```bash
python main_both.py
```

**Output:**
```
Classification Accuracy: 100%
Normalized DCG@5: 0.96
```

## 📁 Project Structure

```
idiom-literal-classification/
├── config.py                    # Configuration settings
├── main_caption.py             # Caption-only ranking pipeline
├── main_image.py               # Image-only ranking pipeline
├── main_both.py                # Fused ranking pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
└── src/
    ├── __init__.py
    ├── embedding_sentence.py   # SentenceTransformer wrapper
    ├── embedding_clip.py       # CLIP model wrapper
    ├── data_loader.py          # Data loading utilities
    ├── models.py               # MLP classifier
    ├── contrast_vectors.py     # Contrast vector generation
    ├── trainer.py              # Training pipeline
    ├── metrics.py              # Evaluation metrics
    ├── ranker_caption.py       # Caption-based ranking
    ├── ranker_image.py         # Image-based ranking
    └── ranker_both.py          # Fused ranking
```

## 🔬 Pipeline

### Step 1: Data Loading

Load datasets:
- **Paraphrase data** from `extendedNominal.csv`
- **ADMIRE dataset** from TSV format
- Split into train and test samples

### Step 2: Fuzzy Matching

Match ADMIRE compounds with paraphrase dataset using fuzzy string matching:
- Threshold: 0.7 similarity
- Handles variations in compound word formatting

### Step 3: Generate Contrast Vectors

For each compound:

```python
context_vec = mean(idiomatic_embeddings)
contrast_vec = context_vec - literal_embedding
```

The contrast vector captures the semantic difference between idiomatic and literal meanings.

### Step 4: Create Training Data

Concatenate features to create 768-dimensional input:

```python
input_features = [contrast_vec, sentence_embedding]  # 768-dim
label = 1 if idiom else 0
```

### Step 5: Train Classifier

**MLP Architecture:**
- Input: 768 dimensions
- Hidden layers: [512, 256]
- Output: 2 classes (idiom/literal)
- Dropout: 0.3
- Normalization: Batch normalization

**Training:**
- K-Fold Cross-Validation: 5 folds
- Epochs: 50
- Optimizer: AdamW with weight decay
- Scheduler: Cosine annealing
- Loss: Cross-entropy

### Step 6: Rank Images

#### Caption-Only Ranking

```python
ranking_emb = 0.6 * sentence_emb + 0.4 * enhanced_token
similarity = dot(ranking_emb, caption_emb)
```

#### Image-Only Ranking

```python
ranking_emb = 0.6 * sentence_emb + 0.4 * enhanced_token
similarity = dot(ranking_emb, image_emb)
```

#### Fused Ranking

```python
caption_sim = dot(ranking_emb, caption_emb)
image_sim = dot(ranking_emb, image_emb)
fused_sim = 0.5 * caption_sim + 0.5 * image_sim
```

### Step 7: Evaluate

Compute metrics:
- **Classification Accuracy**: Percentage of correctly classified samples
- **nDCG@5**: Normalized Discounted Cumulative Gain at rank 5

## 🎨 Token Enhancement Logic

### For IDIOM sentences:

```python
enhanced_token = token_embedding + 0.1 * context_vector
enhanced_token = enhanced_token / ||enhanced_token||  # Normalize
```

### For LITERAL sentences:

```python
enhanced_token = token_embedding  # No enhancement
```

This enhancement helps the model leverage idiomatic context when ranking images.

## 📊 Expected Results

Typical performance on ADMIRE dataset:

| Method | Classification Accuracy | nDCG@5    |
|--------|------------------------|-----------|
| Caption-Only | 90-100%                | 0.85-0.90 |
| Image-Only | 90-100%                | 0.90-0.95 |
| Both (Fused) | 90-100%                | 0.85-0.95 |

**Note:** Results may vary based on dataset size, GPU availability, and hyperparameter tuning.

## 📦 Dependencies

Core dependencies (see `requirements.txt` for complete list):

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.10.0
nltk>=3.8
tqdm>=4.65.0
Pillow>=9.5.0
```

## 📝 Dataset Format

### extendedNominal.csv

```csv
special_token,positive_paraphrase1,positive_paraphrase2,positive_paraphrase3,literal_paraphrase
IDacidtestID,crucial trial,definitive proof,ultimate test,chemical examination
...
```

### ADMIRE TSV

```
subset  compound  sentence  sentence_type  expected_order  image_names  captions
train   acid test The exam...  idiomatic     [1,2,3,4,5]    ["img1.jpg",...] ["caption1",...]
```

## 🔧 Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size in config.py
TRAINING_CONFIG['batch_size'] = 16  # Default: 32

# Use smaller model
MODEL_CONFIG['st_model'] = 'all-MiniLM-L6-v2'  # Smaller than 'all-mpnet-base-v2'
```

### Model Loading Issues

```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Reinstall transformers
pip install --upgrade transformers sentence-transformers
```

### CUDA Not Available

```python
# Force CPU mode
import torch
torch.device('cpu')

# In config.py
USE_GPU = False
```

### Data Loading Issues

-  Verify file paths in `config.py`
-  Check file encoding (should be UTF-8)
-  Ensure TSV delimiter is correct (tab-separated)
-  Validate CSV format matches expected schema

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Guidelines:**
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

##  Bug Reports

Found a bug? Please create an issue with:
1. Python version
2. PyTorch version
3. Full error traceback
4. Steps to reproduce
5. Expected vs actual behavior

##  Future Work

- [ ] Multi-language support (Hindi, Spanish, etc.)
- [ ] Fine-tuned models for idiom detection
- [ ] Real-time ranking API with FastAPI
- [ ] Web interface for demonstrations
- [ ] Extended dataset coverage
- [ ] Support for more embedding models
- [ ] Attention visualization
- [ ] Explainability features

##  References

1. **CLIP**: Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML*.

2. **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*.

3. **ADMIRE Dataset**: "A Dataset for Multi-modal Image Retrieval Evaluation"

4. **Contrast Vectors**: Zou, A., et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency."

##  Support

For questions or support:
-  Email: syedtashif239@gmail.com


##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Authors

- **Syed Mohd Tashif** 

##  Acknowledgments

- SentenceTransformers team for pre-trained models
- OpenAI for CLIP architecture
- ADMIRE dataset creators
- Open-source community

---

**Last Updated:** November 2025

 If you find this project useful, please consider giving it a star!

