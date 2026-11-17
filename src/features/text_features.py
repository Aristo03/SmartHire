import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Words that appear in almost every resume/job because of the template, not the role
DOMAIN_STOP_WORDS = {
    "details", "detail", "company", "description", "education", "skill", "skills",
    "exprience", "experience", "months", "month", "year", "years", "less",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "college", "university", "school", "institute", "board", "secondary", "higher",
    "ssc", "hsc", "sslc", "cbse", "percentage", "cgpa",
    "pvt", "ltd", "limited", "job", "jobs", "candidate", "candidates",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


def clean_text(text) -> str:
    """Lowercase, remove URLs/emails/numbers/symbols/garbled characters."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)            # emails
    text = re.sub(r"[^a-z\s]", " ", text)           # numbers, symbols, garbled chars like â€¢
    return re.sub(r"\s+", " ", text).strip()


def build_tfidf(**overrides) -> TfidfVectorizer:
    params = dict(
        preprocessor=clean_text,   # cleaning happens INSIDE the vectorizer
        stop_words=STOP_WORDS,     # English + resume/job template words
        ngram_range=(1, 2),        # single words + word pairs ('machine learning')
        sublinear_tf=True,         # a word used 20x isn't 20x more important
        min_df=2,                  # ignore words appearing in only 1 document
        max_df=0.9,                # ignore words in >90% of documents
    )
    params.update(overrides)
    return TfidfVectorizer(**params)