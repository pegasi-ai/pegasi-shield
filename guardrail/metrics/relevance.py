from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.spatial.distance import cosine


class Relevance:
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def evaluate(self, text1, text2):
        inputs1 = self.tokenizer(text1, return_tensors="pt")
        inputs2 = self.tokenizer(text2, return_tensors="pt")
        outputs1 = self.model(**inputs1)
        outputs2 = self.model(**inputs2)
        similarity = 1 - cosine(
            outputs1.logits.flatten().detach().numpy(),
            outputs2.logits.flatten().detach().numpy(),
        )
        return similarity
