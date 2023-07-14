from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline


class Bias:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            "d4data/bias-detection-model"
        )

    def evaluate(self, text):
        classifier = pipeline(
            "text-classification", model=self.model, tokenizer=self.tokenizer
        )  # cuda = 0,1 based on gpu availability
        return classifier(text)
