import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_PATHS = {
    'paraphrase_file': '<PLACEHOLDER_PARAPHRASE_FILE>',
    'admire_tsv': '<PLACEHOLDER_ADMIRE_TSV>',
    'image_dir': '<PLACEHOLDER_IMAGE_DIR>',
    'output_dir': '<PLACEHOLDER_OUTPUT_DIR>'
}

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

TRAINING_CONFIG = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'k_folds': 5,
    'random_seed': 42
}

RANKING_CONFIG = {
    'context_weight': 0.1,
    'caption_max_sentences': 2,
    'min_keyword_length': 3,
    'fuzzy_threshold': 0.7
}

SELECTED_METHOD = 'caption_only'
