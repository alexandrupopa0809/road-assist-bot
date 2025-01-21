import logging
import pickle
import torch

from utils import Utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EmbeddingSaver(Utils):
    def __init__(self, model_name, dataset, evaluation_dataset):
        super().__init__(model_name, dataset, evaluation_dataset)
        self.emb_output_file = f"embeddings/text_{model_name}.pkl"
        self.use_mps = torch.backends.mps.is_available()
        self.device = "mps" if self.use_mps else "cpu"

    def _save_embeddings(self, embeddings):
        with open(self.emb_output_file, "wb") as f:
            pickle.dump(embeddings, f)
        logging.info(f"Saved embeddings to {self.emb_output_file}")

    def process_and_save(self):
        texts = self._load_texts()
        embeddings = self._compute_embeddings(texts, self.device)
        self._save_embeddings(embeddings)


if __name__ == "__main__":
    saver = EmbeddingSaver(
        model_name="readerbench/RoBERT-base",
        dataset="data/paragraphs.json",
        evaluation_dataset="data/benchmark.json"
    )
    saver.process_and_save()
