import logging
import pickle

import numpy as np
import torch
from openai import OpenAI

from utils import SYSTEM_DESCRIPTION, Utils

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RoadAssistBot(Utils):
    def __init__(self, model_name, dataset, evaluation_dataset):
        super().__init__(model_name, dataset, evaluation_dataset)
        self.use_mps = torch.backends.mps.is_available()
        self.device = "mps" if self.use_mps else "cpu"
        self.model.to(self.device)

        self.embeddings_file = f"embeddings/text_{model_name}.pkl"
        self.embeddings = self._load_embeddings()
        self.texts = self._load_texts()
        self.conversation_history = [{"role": "system", "content": SYSTEM_DESCRIPTION}]
        self.client = OpenAI()

        self.evaluation_dataset = self._load_evaluation_data()

    def _load_embeddings(self):
        logging.info(f"Loading embeddings from {self.embeddings_file}")
        with open(self.embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        
        if isinstance(embeddings, list): embeddings = np.array(embeddings)   
        embeddings = torch.from_numpy(embeddings)
        
        return embeddings.to(self.device)

    def _compute_similarity(self, query_embeddings):
        return self.embeddings @ query_embeddings
    
    def _rank_all_paragraph_indices_by_similarity(self, query):
        query_embedding = self._compute_embeddings(query, self.device)
        similarities = self._compute_similarity(query_embedding)

        sorted_indices = torch.argsort(similarities, descending=True).tolist()
        logging.info(f"Sorted indeces {sorted_indices}")
        return sorted_indices

    def _find_most_similar_paragraphs_indeces(self, query, top_n=5):
        sorted_indices = self._rank_all_paragraph_indices_by_similarity(query)
        top_n_sorted_indices = sorted_indices[:top_n]
        logging.info(f"Top n sorted indeces {top_n_sorted_indices}")
        return top_n_sorted_indices

    def _find_most_similar_paragraphs(self, query, top_n=5):
        top_n_sorted_indices = self._find_most_similar_paragraphs_indeces(query, top_n)
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
    
    def _create_prompt_without_context(self, query):
        prompt = f"""
            Ai primit intrebarea urmatoare: {query}

            Raspunde la intrebare folosind informațiile pe care le ai. De asemenea, daca
            este relevant, te poti folosi si de informatiile primite pe parcursul conversatiei.
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
    
    def generate_response_without_context(self, query):
        user_prompt = self._create_prompt_without_context(query)
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

    def evaluate_retrieval_and_generation(self, top_k=5):
        precision_scores = []
        recall_scores = []
        highest_positions = []
        bleu_scores = []
        rouge_l_scores = []

        rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        for entry in self.evaluation_dataset:
            query = entry["question"]
            golden_answer = entry["answer"]
            relevant_indices = entry["paragraphs"]

            retrieved_indices = self._find_most_similar_paragraphs_indeces(query, top_k)
            all_sorted_indices = self._rank_all_paragraph_indices_by_similarity(query)

            retrieved_relevant = set(retrieved_indices).intersection(set(relevant_indices))
            precision = len(retrieved_relevant) / top_k
            recall = len(retrieved_relevant) / len(relevant_indices) if relevant_indices else 0

            highest_position = min(
                (all_sorted_indices.index(ri) for ri in relevant_indices),
                default=None,
            )

            generated_response = self.generate_response(query)
            generated_response_text = generated_response.content

            smooth_fn = SmoothingFunction().method1
            bleu = sentence_bleu(
                [golden_answer.split()], generated_response_text.split(), smoothing_function=smooth_fn
            )
            bleu_scores.append(bleu)

            rouge_score = rouge_scorer_obj.score(golden_answer, generated_response_text)["rougeL"].fmeasure
            rouge_l_scores.append(rouge_score)

            precision_scores.append(precision)
            recall_scores.append(recall)
            highest_positions.append(highest_position)

        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_highest_position = (
            sum(pos for pos in highest_positions if pos is not None) / len(highest_positions)
            if highest_positions else None
        )
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0

        evaluation_results = {
            "precision": avg_precision,
            "recall": avg_recall,
            "avg_highest_position": avg_highest_position,
            "bleu": avg_bleu,
            "rougeL": avg_rouge_l,
            "highest_positions": highest_positions,
        }

        log_dir = f"logs/evaluation"
        log_file = f"{log_dir}/{self.model_name}@{top_k}.log"
        Utils.write_json(log_file, evaluation_results)

        logging.info(f"Evaluation results saved to {log_file}")

        return evaluation_results
    
    def evaluate_generation_without_context(self):
        bleu_scores = []
        rouge_l_scores = []

        rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        for entry in self.evaluation_dataset:
            query = entry["question"]
            golden_answer = entry["answer"]

            generated_response = self.generate_response_without_context(query)
            generated_response_text = generated_response.content

            smooth_fn = SmoothingFunction().method1
            bleu = sentence_bleu(
                [golden_answer.split()], generated_response_text.split(), smoothing_function=smooth_fn
            )
            bleu_scores.append(bleu)

            rouge_score = rouge_scorer_obj.score(golden_answer, generated_response_text)["rougeL"].fmeasure
            rouge_l_scores.append(rouge_score)

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0

        evaluation_results = {
            "bleu": avg_bleu,
            "bleu_scores": bleu_scores,
            "rougeL": avg_rouge_l,
            "rougeL_scores": rouge_l_scores
        }

        log_dir = f"logs/evaluation"
        log_file = f"{log_dir}/no_context.log"
        Utils.write_json(log_file, evaluation_results)

        logging.info(f"Evaluation results saved to {log_file}")

        return evaluation_results


if __name__ == "__main__":
    bot = RoadAssistBot(
        model_name="readerbench/RoBERT-base",
        dataset="data/paragraphs.json",
        evaluation_dataset="data/benchmark.json"
    )

    # print("Pune o intrebare despre legislatia rutiera din Romania:")
    # logger.setLevel(logging.INFO)

    # while True:
    #     user_input = input()
    #     response_text = bot.generate_response(user_input)
    #     print(response_text)

    # results = bot.evaluate_retrieval_and_generation(top_k=5)
    results = bot.evaluate_generation_without_context()
    print("Evaluation Results:")
    print(results)