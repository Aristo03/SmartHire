import re
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text) -> str:
    """Lowercase, remove URLs/emails/symbols/garbled characters."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)            # emails
    text = re.sub(r"[^a-z0-9\s]", " ", text)        # symbols + garbled chars like â€¢
    return re.sub(r"\s+", " ", text).strip()


def build_tfidf(**overrides) -> TfidfVectorizer:
    params = dict(
        preprocessor=clean_text,   # cleaning happens INSIDE the vectorizer
        stop_words="english",      # drop 'the', 'and', 'is'...
        ngram_range=(1, 2),        # single words + word pairs ('machine learning')
        sublinear_tf=True,         # a word used 20x isn't 20x more important
        min_df=2,                  # ignore words appearing in only 1 document
        max_df=0.9,                # ignore words in >90% of documents
    )
    params.update(overrides)
    return TfidfVectorizer(**params)