import numpy as np
from tqdm import tqdm


def generate_contrast_vectors(paraphrase_data, embedding_func):
    contrast_vectors = {}
    context_vectors = {}

    print("Generating contrast vectors...")

    for token, data in tqdm(paraphrase_data.items(), desc="Generating vectors"):
        try:
            pos_embs = [embedding_func(text) for text in data['positive']]
            pos_embs = [e for e in pos_embs if e is not None]

            if len(pos_embs) < 2:
                continue

            context_vec = np.mean(pos_embs, axis=0)
            context_vec = context_vec / np.linalg.norm(context_vec)

            lit_emb = embedding_func(data['literal'])
            if lit_emb is None:
                continue

            contrast_vec = context_vec - lit_emb
            contrast_vec = contrast_vec / np.linalg.norm(contrast_vec)

            contrast_vectors[token] = contrast_vec
            context_vectors[token] = context_vec

        except:
            continue

    print(f"Generated {len(contrast_vectors)} contrast vectors")
    return contrast_vectors, context_vectors
