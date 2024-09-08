

class TextGenerationMetrics():
    def __init__(self):
        pass

    def get_metrics(self, references, predictions):
        return {
            "BLEU-1": self.bleu_score(references, predictions, 1),
            "BLEU-2": self.bleu_score(references, predictions, 2),
            "BLEU-4": self.bleu_score(references, predictions, 4),
            "Perplexity": self.perplexity_score(references, predictions),
            "METEOR": self.meteor_score(references, predictions),
            "ROUGE-L": self.rouge_score(references, predictions),
            "CIDER": self.cider_score(references, predictions)
        }

    def bleu_score(self, references, candidates, n=4):
        return 0.5

    def perplexity_score(self, references, predictions):
        return 0.5

    def meteor_score(self, references, predictions):
        return 0.5

    def rouge_score(self, references, predictions):
        return 0.5

    def cider_score(self, references, predictions):
        return 0.5

