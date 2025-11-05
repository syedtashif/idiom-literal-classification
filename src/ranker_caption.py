import numpy as np
import re
import os
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from config import CONTEXT_WEIGHT, CAPTION_MAX_SENTENCES, MIN_KEYWORD_LENGTH
from .utils import normalize_compound

class CaptionOnlyRanker:
    def __init__(self, image_to_caption, embedding_func, id_tokens=None):
        self.image_to_caption = image_to_caption
        self.embedding_func = embedding_func
        self.id_tokens = id_tokens or set()

    def extract_keywords(self, text):
        stopwords = {
            'the', 'and', 'is', 'in', 'it', 'to', 'that', 'of', 'for', 'on',
            'with', 'as', 'this', 'by', 'are', 'at', 'be', 'or', 'an', 'was',
            'but', 'not', 'you', 'from', 'have', 'has', 'had', 'a'
        }

        words = set()
        for word in re.findall(r'\b\w+\b', text.lower()):
            if len(word) >= MIN_KEYWORD_LENGTH and word not in stopwords:
                words.add(word)

        return words.union(self.id_tokens)

    def filter_caption(self, caption, anchor_keywords):
        if not caption or not anchor_keywords:
            return caption

        try:
            sentences = sent_tokenize(caption)
        except:
            sentences = [s.strip() for s in re.split(r'[.!?]+', caption) if s.strip()]

        if not sentences:
            return caption

        sentence_scores = []
        for sentence in sentences:
            sentence_keywords = self.extract_keywords(sentence)
            overlap = len(anchor_keywords.intersection(sentence_keywords))
            overlap_length = sum(len(w) for w in anchor_keywords.intersection(sentence_keywords))
            score = overlap + (overlap_length / 100)
            sentence_scores.append(score)

        if max(sentence_scores) == 0:
            return sentences[0]

        top_indices = np.argsort(sentence_scores)[-CAPTION_MAX_SENTENCES:]
        top_indices = sorted(top_indices)
        filtered_caption = '. '.join([sentences[i] for i in top_indices])

        return filtered_caption

    def rank_images(self, test_samples, token_to_token, context_vecs):
        ranking_results = []

        print("Ranking with CAPTION ONLY...")

        for test_sample in tqdm(test_samples, desc="Caption Ranking"):
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
                compound_token_emb = self.embedding_func(compound_token_str)
                if compound_token_emb is None:
                    continue

                if true_type == 'idiomatic':
                    compound_token_enhanced = compound_token_emb + CONTEXT_WEIGHT * context_vecs[token]
                    compound_token_enhanced = compound_token_enhanced / np.linalg.norm(compound_token_enhanced)
                else:
                    compound_token_enhanced = compound_token_emb

                sentence_emb = self.embedding_func(sentence)
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

                    if img_name_lower in self.image_to_caption:
                        caption = self.image_to_caption[img_name_lower]

                        anchor_keywords = self.extract_keywords(sentence)
                        filtered_caption = self.filter_caption(caption, anchor_keywords)

                        caption_emb = self.embedding_func(filtered_caption)
                        if caption_emb is not None:
                            sim = float(np.dot(ranking_emb, caption_emb))
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
