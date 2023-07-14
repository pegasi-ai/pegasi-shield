import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Toxicity:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "nicholasKluge/ToxicityModel"
        )
        self.model.eval()

    def evaluate(self, response, prompt):
        inputs = self.tokenizer(
            prompt,
            response,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        score = self.model(**inputs)[0]
        # probabilities = torch.softmax(score.logits, dim=0)
        return score.item()
