import logging
import pickle

import numpy as np
import torch
from openai import OpenAI
import json
import time

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
        self.device = "cpu"
        self.model.to(self.device)

        self.embeddings_file = f"embeddings_v1/text_{model_name}.pkl"
        self.embeddings = self._load_embeddings()
        self.texts = self._load_texts()
        self.conversation_history = [{"role": "system", "content": SYSTEM_DESCRIPTION}]
        self.client = OpenAI()

        self.evaluation_dataset = self._load_evaluation_data()

    def _load_embeddings(self):
        logging.info(f"Loading embeddings from {self.embeddings_file}")
    
        with open(self.embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        return embeddings

    def _compute_similarity(self, query_embeddings):
        return np.dot(self.embeddings, query_embeddings)
    
    def _rank_all_paragraph_indices_by_similarity(self, query):
        query_embedding = self._compute_embeddings(query, self.device)
        similarities = self._compute_similarity(query_embedding)
        sorted_indices = np.argsort(similarities)[::-1].tolist()
        return sorted_indices

    def _find_most_similar_paragraphs_indeces(self, query, top_n=5):
        sorted_indices = self._rank_all_paragraph_indices_by_similarity(query)
        top_n_sorted_indices = sorted_indices[:top_n]
        return top_n_sorted_indices

    def _find_most_similar_paragraphs(self, query, top_n=10):
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

    def evaluate_with_llm(self, query, gen_answer, actual_answer, actual_context):
        system_prompt = """
            Ești un evaluator expert în legislația rutieră din România.
        """
        user_prompt = f"""
            Ai primit o întrebare, un răspuns generat de un chatbot, răspunsul corect de referință si paragraful
            din legislatia rutiera pe care il poti folosi ca adevar.
            
            Evaluează răspunsul generat pe baza următoarelor criterii:
            Corectitudine: Răspunsul conține informații corecte și conforme cu legislația rutieră din România?
            Completitudine: Răspunsul oferă toate detaliile esențiale sau omite informații importante?
            Claritate: Răspunsul este formulat clar și ușor de înțeles?
            Relevanță: Răspunsul este relevant pentru întrebare sau conține informații inutile?
            
            Intrare:
            Întrebare: {query}
            Răspuns generat: {gen_answer}
            Răspuns corect de referință: {actual_answer}
            Context: {actual_context}

            Atribuie un scor de la 0 la 10 pentru raspunsul generat.
            Trebuie sa returnezi un singur obiect JSON care sa aiba doar cheia "score".
            Nu folosi elemente de markdown!
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            llm_response = completion.choices[0].message
            score = json.loads(llm_response.content).get("score", None)
        except Exception as e:
            logging.warning(f"Message generation failed. Error: {e}")
            score = None
        time.sleep(3)
        return score

    @staticmethod
    def _compute_keywords_score(keywords, text):
        text_words = set(text.lower().split())
        keywords_set = set(map(str.lower, keywords))
        matched_keywords = keywords_set.intersection(text_words)
        return len(matched_keywords) / len(keywords_set) if keywords_set else 0.0

    @staticmethod
    def _compute_paragraph_acc(top_sorted_indices, par_index):
        return 1 if par_index in top_sorted_indices else 0

    @staticmethod
    def _check_retrieved_page(paragraph, index_list, range_size=2):
        page_paragraphs = list(range(paragraph - range_size, paragraph + range_size + 1))
        return 1 if any(p in index_list for p in page_paragraphs) else 0

    @staticmethod
    def _add_stats(**score_lists):
        averages = {}
        for key, values in score_lists.items():
            if values:
                averages[f"avg_{key}"] = sum(v for v in values if v is not None) / len(values)
            else:
                averages[f"avg_{key}"] = None if key == "highest_positions" else 0
        return averages

    def evaluate_retrieval_and_generation(self, top_k=10):
        retrieval_accuracy_scores = []
        highest_positions = []
        bleu_scores = []
        rouge_l_scores = []
        keywords_scores = []
        llm_eval_scores = []
        page_acc_scores = []

        rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        results = []

        for entry in self.evaluation_dataset:
            query = entry["question"]
            golden_answer = entry["answer"]
            actual_paragraph_index = entry["paragraph"]
            actual_paragraph_text = self.texts[actual_paragraph_index]

            # retrieval evaluation
            all_sorted_indices = self._rank_all_paragraph_indices_by_similarity(query)
            top_1_par_retrieved = self.texts[all_sorted_indices[0]]

            highest_position = all_sorted_indices.index(actual_paragraph_index)
            highest_positions.append(highest_position)

            retrieval_acc = self._compute_paragraph_acc(all_sorted_indices[:top_k], actual_paragraph_index)
            retrieval_accuracy_scores.append(retrieval_acc)

            page_acc = self._check_retrieved_page(actual_paragraph_index, all_sorted_indices[:top_k])
            page_acc_scores.append(page_acc)

            # answer generation evaluation
            generated_response = self.generate_response(query)
            generated_response_text = generated_response.content

            keywords_score = self._compute_keywords_score(entry["keywords"], generated_response_text)
            keywords_scores.append(keywords_score)

            llm_score = self.evaluate_with_llm(query, generated_response_text, golden_answer, actual_paragraph_text)
            llm_eval_scores.append(llm_score)

            smooth_fn = SmoothingFunction().method1
            bleu = sentence_bleu(
                [golden_answer.split()], generated_response_text.split(), smoothing_function=smooth_fn
            )
            bleu_scores.append(bleu)

            rouge_score = rouge_scorer_obj.score(golden_answer, generated_response_text)["rougeL"].fmeasure
            rouge_l_scores.append(rouge_score)

            results.append({
                "query": query,
                "actual_answer": golden_answer,
                "gen_answer": generated_response_text,
                "first_par_retrieved": top_1_par_retrieved,
                "ret_acc": retrieval_acc,
                "page_acc": page_acc,
                "highest_position": highest_position,
                "keywords_score": keywords_score,
                "llm_eval_score": llm_score,
                "bleu_score": bleu,
                "rouge_score" : rouge_score 
            })

        stats = self._add_stats(
            ret_acc=retrieval_accuracy_scores,
            page_acc=page_acc_scores,
            highest_position=highest_positions,
            keywords_score=keywords_scores,
            llm_score=llm_eval_scores,
            bleu_score=bleu_scores,
            rouge_score=rouge_l_scores
        ) | {"min_highest_pos": min(highest_positions), "max_highest_pos": max(highest_positions)}
        results.insert(0, stats)

        res_file = f"results/res_mpnet_top10.json"
        Utils.write_json(res_file, results)

    def evaluate_generation_without_context(self):
        bleu_scores = []
        rouge_l_scores = []
        keywords_scores = []
        llm_eval_scores = []
        results = []
        rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        for entry in self.evaluation_dataset:
            query = entry["question"]
            golden_answer = entry["answer"]
            actual_paragraph_index = entry["paragraph"]
            actual_paragraph_text = self.texts[actual_paragraph_index]

            generated_response = self.generate_response_without_context(query)
            generated_response_text = generated_response.content
            
            keywords_score = self._compute_keywords_score(entry["keywords"], generated_response_text)
            keywords_scores.append(keywords_score)

            llm_score = self.evaluate_with_llm(query, generated_response_text, golden_answer, actual_paragraph_text)
            llm_eval_scores.append(llm_score)
            
            smooth_fn = SmoothingFunction().method1
            bleu = sentence_bleu(
                [golden_answer.split()], generated_response_text.split(), smoothing_function=smooth_fn
            )
            bleu_scores.append(bleu)

            rouge_score = rouge_scorer_obj.score(golden_answer, generated_response_text)["rougeL"].fmeasure
            rouge_l_scores.append(rouge_score)
            
            results.append({
                "query": query,
                "actual_answer": golden_answer,
                "gen_answer": generated_response_text,
                "keywords_score": keywords_score,
                "llm_eval_score": llm_score,
                "bleu_score": bleu,
                "rouge_score" : rouge_score 
            })

        stats = self._add_stats(
            keywords_score=keywords_scores,
            llm_score=llm_eval_scores,
            bleu_score=bleu_scores,
            rouge_score=rouge_l_scores
        )
        results.insert(0, stats)
        res_file = f"results/res_without_context.json"
        Utils.write_json(res_file, results)


if __name__ == "__main__":
    bot = RoadAssistBot(        
        model_name="paraphrase-multilingual-mpnet-base-v2",
        dataset="data/paragraphs_v1.json",
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
