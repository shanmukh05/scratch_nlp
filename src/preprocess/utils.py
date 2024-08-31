import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# nltk.download("stopwords")


def preprocess_text(text, operations):
    if "lcase" in operations or operations is None:
        text = text.lower()
    if "remalpha" in operations or operations is None:
        text = re.sub(r"\W+", " ", text) 
    if "stopwords" in operations or operations is None:
        swords = stopwords.words("english")
        text = " ".join([word for word in text.split() if word not in swords])
    if "stemming" in operations or operations is None:
        snowball = SnowballStemmer(language="english")
        text = " ".join([snowball.stem(word) for word in text.split()])
    return text
