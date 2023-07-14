import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Sentiment:
    def __init__(self, model_name="textattack/bert-base-uncased-imdb"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def evaluate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        return probabilities[0, 1].item()
