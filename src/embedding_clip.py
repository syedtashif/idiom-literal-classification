import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


class CLIPEmbeddingModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("CLIP model loaded (512-dim)")

    def get_embedding(self, input_data, embedding_type='text'):
        try:
            with torch.no_grad():
                if embedding_type == 'text':
                    inputs = self.processor(text=str(input_data), return_tensors="pt", padding=True).to(self.device)
                    embeddings = self.model.get_text_features(**inputs)

                elif embedding_type == 'image':
                    if isinstance(input_data, str):
                        image = Image.open(input_data).convert('RGB')
                    else:
                        image = input_data
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    embeddings = self.model.get_image_features(**inputs)

                else:
                    return None

                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                return embeddings.cpu().numpy()[0]

        except:
            return None
