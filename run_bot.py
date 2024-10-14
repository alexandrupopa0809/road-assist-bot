import logging
import pickle

import torch
from openai import OpenAI

from utils import SYSTEM_DESCRIPTION, Utils

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
        self.conversation_history = [{"role": "system", "content": SYSTEM_DESCRIPTION}]
        self.client = OpenAI()

    def _load_embeddings(self):
        logging.info(f"Loading embeddings from {self.embeddings_file}")
        with open(self.embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        embeddings = torch.from_numpy(embeddings)
        return embeddings.to(self.device)

    def _compute_similarity(self, query_embeddings):
        return self.embeddings @ query_embeddings

    def _find_most_similar_paragraphs(self, query, top_n=5):
        query_embedding = self._compute_embeddings(query, self.device)
        similarities = self._compute_similarity(query_embedding)

        sorted_indices = torch.argsort(similarities, descending=True).tolist()
        top_n_sorted_indices = sorted_indices[:top_n]
        most_similar_texts = [self.texts[i] for i in top_n_sorted_indices]
        return "\n\n".join(most_similar_texts)

    def _create_prompt(self, query):
        context = self._find_most_similar_paragraphs(query)
        prompt = f"""
            Ai primit intrebarea urmatoare: {query}

            Raspunde la intrebare folosind informatiile din codul
            rutier delimitate de trei backticks. De asemenea, daca
            este relevant, te poti folosi si de informatiile primite
            pe parcursul conversatiei.

            ```{context}```
        """
        return prompt

    def generate_response(self, query):
        user_prompt = self._create_prompt(query)
        self.conversation_history.append({"role": "user", "content": user_prompt})
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=self.conversation_history,
            )
            chat_response = completion.choices[0].message
        except Exception as e:
            logging.warning(f"Message generation failed. Error: {e}")
            chat_response = "System is not available. Try again later."
        return chat_response


if __name__ == "__main__":
    bot = RoadAssistBot(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        dataset="data/paragraphs.json",
    )

    print("Pune o intrebare despre legislatia rutiera din Romania:")
    logger.setLevel(logging.INFO)

    while True:
        user_input = input()
        response_text = bot.generate_response(user_input)
        print(response_text)
