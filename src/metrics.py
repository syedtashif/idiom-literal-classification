import numpy as np


def compute_ndcg(ranking_results):
    if len(ranking_results) == 0:
        return {'ndcg': 0.0, 'samples': 0}

    ndcg_scores = []

    for result in ranking_results:
        predicted_order = result['predicted_order']
        expected_order = result['expected_order']

        predicted_rel = []
        for pred_img in predicted_order[:5]:
            rel = 0.0
            for pos, exp_img in enumerate(expected_order):
                if pred_img == exp_img:
                    rel = 1.0 if pos < 2 else 0.0
                    break
            predicted_rel.append(rel)

        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(predicted_rel))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(2, len(expected_order))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    avg_ndcg = np.mean(ndcg_scores)

    return {'ndcg': avg_ndcg, 'samples': len(ranking_results)}
