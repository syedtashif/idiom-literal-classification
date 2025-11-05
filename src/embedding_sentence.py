import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        print(f"Loaded SentenceTransformer: {model_name} ({device})")

    def get_embedding(self, text):
        try:
            if text is None or len(str(text).strip()) == 0:
                return None
            emb = self.model.encode(str(text), convert_to_numpy=True, show_progress_bar=False)
            return emb / np.linalg.norm(emb)
        except:
            return None
