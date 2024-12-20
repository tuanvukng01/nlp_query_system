import torch
from models.categorization_model import CategorizationModel
from models.sentiment_model import SentimentModel
from inference.response_templates import RESPONSE_TEMPLATES

from langdetect import detect
import shap
from transformers import AutoTokenizer, AutoModel
# import faiss
import numpy as np
from elasticsearch import Elasticsearch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os

category_map = {0: "Billing", 1: "Technical Support", 2: "General Inquiry"}
sentiment_map = {0: "Positive", 1: "Neutral", 2: "Negative"}

ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)


class InferencePipeline:
    def __init__(self, category_model_path="../models/category_model.pt", sentiment_model_path="../models/sentiment_model.pt"):
        self.cat_model = CategorizationModel()
        self.cat_model.load(category_model_path)
        self.sent_model = SentimentModel()
        self.sent_model.load(sentiment_model_path)

        self.es = Elasticsearch("http://localhost:9200")

        self.retrieval_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.retrieval_model = AutoModel.from_pretrained("distilbert-base-uncased").eval()

        self.explainer = shap.Explainer(self.shap_model_predict)

    def shap_model_predict(self, texts):
        inputs = self.cat_model.tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.cat_model.model(**inputs)
        return outputs.logits.softmax(dim=1).cpu().numpy()

    def named_entity_recognition(self, text):
        tokens = ner_tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            output = ner_model(tokens)
        label_ids = torch.argmax(output.logits, dim=2)
        tokens_decoded = ner_tokenizer.convert_ids_to_tokens(tokens[0])
        entities = []
        for token, label_id in zip(tokens_decoded, label_ids[0].tolist()):

            pass
        return ["PERSON: John Doe"]

    def retrieve_supporting_docs(self, query_text):
        res = self.es.search(index="support_docs", query={"match": {"content": query_text}})
        hits = res.get("hits", {}).get("hits", [])
        top_content = hits[0]["_source"]["content"] if hits else ""
        return top_content

    def run(self, query_text):
        language = detect(query_text)

        cat_label = self.cat_model.predict(query_text)
        sent_label = self.sent_model.predict(query_text)
        cat = category_map[cat_label]
        sent = sentiment_map[sent_label]

        supporting_docs = self.retrieve_supporting_docs(query_text)

        entities = self.named_entity_recognition(query_text)

        shap_values = self.explainer([query_text])
        explanation = shap_values[0]

        response = RESPONSE_TEMPLATES[cat][sent]

        return {
            "query": query_text,
            "language": language,
            "category": cat,
            "sentiment": sent,
            "entities": entities,
            "supporting_docs": supporting_docs,
            "explanation": explanation.values.tolist(),
            "response": response
        }