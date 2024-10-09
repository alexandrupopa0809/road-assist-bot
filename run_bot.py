import logging
import pickle

import openai
import torch

from utils import Utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RoadAssistBot(Utils):
    def __init__(self, model_name, dataset):
        super().__init__(model_name, dataset)
        self.use_mps = torch.backends.mps.is_available()
        self.device = "mps" if self.use_mps else "cpu"
        self.model.to(self.device)

        self.embeddings_file = f"embeddings/text_{model_name}.pkl"
        self.embeddings = self._load_embeddings()
        self.texts = self._load_texts()

    def _load_embeddings(self):
        logging.info(f"Loading embeddings from {self.embeddings_file}")
        with open(self.embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        embeddings = torch.from_numpy(embeddings)
        return embeddings.to(self.device)

    def _compute_similarity(self, query_embeddings):
        return self.embeddings @ query_embeddings

    def find_most_similar(self, query, top_n=5):
        query_embedding = self._compute_embeddings(query, self.device)
        similarities = self._compute_similarity(query_embedding)

        sorted_indices = torch.argsort(similarities, descending=True).tolist()
        top_n_sorted_indices = sorted_indices[:top_n]
        most_similar_texts = [self.texts[i] for i in top_n_sorted_indices]
        return most_similar_texts


if __name__ == "__main__":
    bot = RoadAssistBot(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        dataset="data/paragraphs.json",
    )

    query = "Pioritate de dreapta"
    similar_texts = bot.find_most_similar(query)
    for text in similar_texts:
        print(text)
        print("\n")
