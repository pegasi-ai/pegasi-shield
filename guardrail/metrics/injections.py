import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PromptInjection:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("JasperLS/gelectra-base-injection")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "JasperLS/gelectra-base-injection"
        )

    def evaluate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        return probabilities[0, 1].item()
