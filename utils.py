import json
import logging

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Utils:
    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dataset = dataset

    @staticmethod
    def write_json(output_file, data):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def read_json(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_texts(self):
        paragraphs = self.read_json(self.dataset)
        texts = [par_object["text"] for par_object in paragraphs]
        logging.info(f"Loaded {len(texts)} texts.")
        return texts

    def _compute_embeddings(self, texts, device):
        logging.info("Computing embeddings...")
        embeddings = self.model.encode(
            texts, convert_to_numpy=False, device=device, show_progress_bar=True
        )
        return embeddings
