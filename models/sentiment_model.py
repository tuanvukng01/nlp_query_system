import torch
import os
from transformers import AutoTokenizer
from models.transformer_finetune import train_model
from models.utils import get_device

class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        self.device = get_device()
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train(self, train_loader, val_loader, epochs=3, lr=2e-5):
        self.model = train_model(
            self.model_name,
            train_loader,
            val_loader,
            num_labels=self.num_labels,
            epochs=epochs,
            lr=float(lr),
            device=self.device
        )

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
        return preds.item()

    def save(self, path="sentiment_model.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="sentiment_model.pt"):
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))