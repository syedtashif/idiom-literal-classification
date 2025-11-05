import numpy as np
import os
from tqdm import tqdm

from config import CONTEXT_WEIGHT
from .utils import normalize_compound

class ImageOnlyRanker:
    def __init__(self, image_to_path, embedding_func, id_tokens=None):
        self.image_to_path = image_to_path
        self.embedding_func = embedding_func
        self.id_tokens = id_tokens or set()

    def rank_images(self, test_samples, token_to_token, context_vecs):
        ranking_results = []

        print("Ranking with IMAGE ONLY...")

        for test_sample in tqdm(test_samples, desc="Image Ranking"):
            try:
                compound = test_sample['compound']
                sentence = test_sample['sentence']
                true_type = test_sample['sentence_type']
                norm_compound = normalize_compound(compound)

                if norm_compound not in token_to_token:
                    continue

                token, _ = token_to_token[norm_compound]
                if token is None or token not in context_vecs:
                    continue

                compound_token_str = token.replace('ID', '').replace('_', ' ')
                compound_token_emb = self.embedding_func(compound_token_str, embedding_type='text')
                if compound_token_emb is None:
                    continue

                if true_type == 'idiomatic':
                    compound_token_enhanced = compound_token_emb + CONTEXT_WEIGHT * context_vecs[token]
                    compound_token_enhanced = compound_token_enhanced / np.linalg.norm(compound_token_enhanced)
                else:
                    compound_token_enhanced = compound_token_emb

                sentence_emb = self.embedding_func(sentence, embedding_type='text')
                if sentence_emb is None:
                    continue

                ranking_emb = 0.6 * sentence_emb + 0.4 * compound_token_enhanced
                ranking_emb = ranking_emb / np.linalg.norm(ranking_emb)

                try:
                    if isinstance(test_sample['expected_order'], str):
                        expected_order = eval(test_sample['expected_order'])
                    else:
                        expected_order = test_sample['expected_order']
                    expected_order = [img for img in expected_order if img]
                except:
                    continue

                if len(expected_order) == 0:
                    continue

                image_sims = []

                for img_name in expected_order:
                    img_name_lower = str(img_name).lower().strip()

                    if img_name_lower in self.image_to_path:
                        img_path = self.image_to_path[img_name_lower]

                        img_emb = self.embedding_func(img_path, embedding_type='image')
                        if img_emb is not None:
                            sim = float(np.dot(ranking_emb, img_emb))
                            image_sims.append((os.path.basename(img_name), sim))

                if len(image_sims) == 0:
                    continue

                image_sims.sort(key=lambda x: x[1], reverse=True)
                predicted_order = [img for img, sim in image_sims]

                ranking_results.append({
                    'compound': compound,
                    'expected_order': expected_order,
                    'predicted_order': predicted_order
                })

            except:
                continue

        return ranking_results
