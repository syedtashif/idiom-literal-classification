import torch
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from config import *
from src.embedding_clip import CLIPEmbeddingModel
from src.data_loader import (
    load_paraphrase_data, load_admire_data, split_train_test,
    filter_valid_samples, build_image_path_mapping
)
from src.contrast_vectors import generate_contrast_vectors
from src.utils import normalize_compound, fuzzy_match
from src.trainer import train_classifier
from src.ranker_image import ImageOnlyRanker
from src.metrics import compute_ndcg


def main():
    print("=" * 80)
    print("IMAGE-ONLY RANKING (CLIP)")
    print("=" * 80)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\nLoading data...")
    paraphrase_data, token_to_compound = load_paraphrase_data(PARAPHRASE_FILE)
    dataset_df = load_admire_data(ADMIRE_TSV)
    train_df_raw, test_df_raw = split_train_test(dataset_df)
    train_samples = filter_valid_samples(train_df_raw)
    test_samples = filter_valid_samples(test_df_raw)

    print("Fuzzy matching...")
    admire_compounds = set([normalize_compound(row['compound']) for row in train_samples + test_samples])
    compound_to_token = {}

    for admire_comp in admire_compounds:
        best_match, best_score = None, 0.0
        for token, token_comp in token_to_compound.items():
            score = fuzzy_match(admire_comp, token_comp, threshold=0.7)
            if score > best_score:
                best_score, best_match = score, token
        if best_match and best_score >= 0.7:
            compound_to_token[admire_comp] = (best_match, best_score)

    print("Loading CLIP model...")
    embedding_model = CLIPEmbeddingModel(DEVICE)

    print("Generating contrast vectors...")
    contrast_vecs, context_vecs = generate_contrast_vectors(
        paraphrase_data,
        lambda x: embedding_model.get_embedding(x, embedding_type='text')
    )

    print("Creating training data...")

    def create_data(samples):
        data = []
        for row in samples:
            try:
                norm_compound = normalize_compound(row['compound'])
                if norm_compound not in compound_to_token:
                    continue

                token, _ = compound_to_token[norm_compound]
                if token not in contrast_vecs:
                    continue

                sent_emb = embedding_model.get_embedding(row['sentence'], embedding_type='text')
                if sent_emb is None:
                    continue

                expected_order = row['expected_order']
                if isinstance(expected_order, str):
                    expected_order = eval(expected_order)
                expected_order = [img for img in expected_order if img]
                if len(expected_order) == 0:
                    continue

                data.append({
                    'compound': row['compound'],
                    'token': token,
                    'sentence': row['sentence'],
                    'true_type': row['sentence_type'],
                    'label': 1 if row['sentence_type'] == 'idiomatic' else 0,
                    'compound_contrast': contrast_vecs[token],
                    'st_context': context_vecs[token],
                    'sentence_embedding': sent_emb,
                    'expected_order': expected_order
                })
            except:
                continue
        return pd.DataFrame(data)

    train_data = create_data(train_samples)
    test_data = create_data(test_samples)

    print("\n" + "=" * 80)
    print("CLASSIFICATION")
    print("=" * 80)
    results = train_classifier(train_data, test_data)
    class_acc = results['test_acc']
    print(f"\nTEST ACCURACY: {class_acc * 100:.2f}%")

    print("\n" + "=" * 80)
    print("IMAGE RANKING (IMAGE-ONLY WITH CLIP)")
    print("=" * 80)
    image_to_path = build_image_path_mapping(IMAGE_DIR)

    id_tokens = set(token.replace('ID', '').replace('_', ' ') for token in paraphrase_data.keys())
    ranker = ImageOnlyRanker(image_to_path, embedding_model.get_embedding, id_tokens)
    ranking_results = ranker.rank_images(test_samples, compound_to_token, context_vecs)

    metrics = compute_ndcg(ranking_results)
    print(f"\nnDCG@5 SCORE: {metrics['ndcg']:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (IMAGE-ONLY)")
    print("=" * 80)
    print(f"\nClassification Accuracy: {class_acc * 100:7.2f}%")
    print(f"Normalized DCG@5:      {metrics['ndcg']:7.4f}\n")


if __name__ == "__main__":
    main()
