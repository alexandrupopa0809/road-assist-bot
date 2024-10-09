import logging
import pickle

from utils import Utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EmbeddingSaver(Utils):
    def __init__(self, model_name, dataset):
        super().__init__(model_name, dataset)
        self.emb_output_file = f"embeddings/111text_{model_name}.pkl"

    def _save_embeddings(self, embeddings):
        with open(self.emb_output_file, "wb") as f:
            pickle.dump(embeddings, f)
        logging.info(f"Saved embeddings to {self.emb_output_file}")

    def process_and_save(self):
        texts = self._load_texts()
        embeddings = self._compute_embeddings(texts)
        self._save_embeddings(embeddings)


if __name__ == "__main__":
    saver = EmbeddingSaver(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        dataset="data/paragraphs.json",
    )
    saver.process_and_save()
