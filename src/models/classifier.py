import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src import config
from src.features.text_features import build_tfidf

CLASSIFIER_PATH = config.MODELS_DIR / "resume_classifier.pkl"


def build_classifier():
    return make_pipeline(build_tfidf(), LinearSVC(C=1.0))


def train_and_save(X, y, path=CLASSIFIER_PATH):
    """Train on ALL data (after evaluation is done) and save to disk."""
    model = build_classifier().fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def load_classifier(path=CLASSIFIER_PATH):
    return joblib.load(path)


def predict_top_k(model, text: str, k: int = 3):
    """Best category, plus up to k-1 alternatives the model also says 'yes' to (score > 0)."""
    scores = model.decision_function([text])[0]
    top = np.argsort(scores)[::-1][:k]
    ranked = [(model.classes_[i], round(float(scores[i]), 3)) for i in top]
    return [ranked[0]] + [r for r in ranked[1:] if r[1] > 0]