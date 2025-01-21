import json
import logging
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Utils:
    def __init__(self, model_name, dataset, evaluation_dataset):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dataset = dataset
        self.evaluation_dataset = evaluation_dataset

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
    
    def _load_evaluation_data(self):
        evaluation_data = self.read_json(self.evaluation_dataset)
        logging.info(f"Loaded {len(evaluation_data)} evaluation questions from {self.evaluation_dataset}")
        return evaluation_data

    def _compute_embeddings(self, texts, device):
        logging.info("Computing embeddings...")
        embeddings = self.model.encode(
            texts, convert_to_numpy=False, device=device, show_progress_bar=True
        )
        return embeddings


SYSTEM_DESCRIPTION = """
    Esti un asistent specializat pe legislatia rutiera din Romania si
    vei raspunde la intrebarile utilizatorilor despre legislatia rutiera
    ajutandu-te de contextul primit. Foloseste maxim 50 de cuvinte pentru
    a raspunde la fiecare intrebare.
"""
