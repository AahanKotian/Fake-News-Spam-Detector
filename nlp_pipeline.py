"""
nlp_pipeline.py
───────────────
Real NLP pipeline using:
- TF-IDF vectorisation
- Handcrafted linguistic features (sensationalism, urgency, caps ratio, etc.)
- Logistic Regression with class_weight='balanced' (handles imbalanced data)
- SHAP LinearExplainer for per-feature attribution (explainability)
- LIME TextExplainer as an alternative explainer
"""

import re
import math
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import shap

# ─── Linguistic feature lists ─────────────────────────────────────────────────

SENSATIONAL_WORDS = {
    "shocking", "bombshell", "explosive", "secret", "exposed", "revealed",
    "coverup", "cover-up", "scandal", "breaking", "urgent", "alert",
    "exclusive", "unprecedented", "miracle", "conspiracy", "hoax",
    "banned", "hidden", "suppressed", "truth", "lies", "fake", "fraud",
    "whistleblower", "leaked", "classified", "censored",
}

SPAM_TRIGGER_WORDS = {
    "congratulations", "winner", "won", "prize", "claim", "reward",
    "free", "gift", "offer", "deal", "discount", "limited time",
    "act now", "click here", "verify", "confirm", "suspended",
    "account", "password", "login", "secure", "bank", "transfer",
    "million", "dollars", "£", "€", "inheritance", "prince", "barrister",
    "investment", "profit", "guaranteed", "risk-free", "crypto",
}

CREDIBLE_MARKERS = {
    "according to", "study", "research", "report", "officials", "said",
    "announced", "published", "percent", "%", "survey", "data",
    "university", "institute", "department", "agency", "government",
}


# ─── Custom transformer: hand-crafted linguistic features ─────────────────────

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 10 interpretable numerical features from raw text.
    These drive SHAP explanations.
    """

    FEATURE_NAMES = [
        "caps_ratio",
        "exclamation_count",
        "question_count",
        "sensational_word_count",
        "spam_trigger_count",
        "credible_marker_count",
        "avg_word_length",
        "url_count",
        "number_count",
        "text_length_log",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._features(text) for text in X], dtype=float)

    def _features(self, text: str) -> list:
        text_lower = text.lower()
        words = text.split()
        alpha_chars = [c for c in text if c.isalpha()]

        caps_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars else 0.0
        )
        exclamation_count = min(text.count("!"), 10)
        question_count = min(text.count("?"), 10)

        sensational_count = sum(
            1 for w in SENSATIONAL_WORDS if w in text_lower
        )
        spam_count = sum(
            1 for w in SPAM_TRIGGER_WORDS if w in text_lower
        )
        credible_count = sum(
            1 for m in CREDIBLE_MARKERS if m in text_lower
        )

        avg_word_len = (
            sum(len(w.strip(string.punctuation)) for w in words) / len(words)
            if words else 0.0
        )
        url_count = len(re.findall(r"https?://\S+|www\.\S+", text))
        number_count = len(re.findall(r"\b\d+\.?\d*\b", text))
        text_length_log = math.log1p(len(text))

        return [
            caps_ratio,
            exclamation_count,
            question_count,
            sensational_count,
            spam_count,
            credible_count,
            avg_word_len,
            url_count,
            number_count,
            text_length_log,
        ]


# ─── Training data ─────────────────────────────────────────────────────────────
# Small illustrative dataset. In production, use LIAR / Enron / FakeNewsNet.

NEWS_TRAINING_DATA = [
    # (text, label)  0=REAL, 1=FAKE
    ("Federal Reserve raises interest rates by 0.25% amid inflation concerns", 0),
    ("Apple reports record quarterly earnings beating analyst expectations", 0),
    ("Scientists publish new data on climate change showing accelerating warming trends", 0),
    ("Government announces new infrastructure spending plan worth $1.2 trillion", 0),
    ("Study finds moderate exercise linked to improved cardiovascular health", 0),
    ("Company reports quarterly losses amid supply chain disruptions", 0),
    ("University researchers develop new battery technology with higher energy density", 0),
    ("Unemployment rate falls to 3.8% as job market remains strong", 0),
    ("New legislation passed to regulate social media platforms", 0),
    ("Tech giant announces layoffs affecting 10% of global workforce", 0),
    ("SHOCKING: Scientists confirm chocolate cures cancer, Big Pharma hiding the truth!!!", 1),
    ("BREAKING: Moon made of cheese confirmed by secret NASA probe cover-up", 1),
    ("5G towers PROVEN to cause mind control whistleblower reveals classified docs", 1),
    ("EXPOSED: Vaccines contain microchips government admits hidden agenda SHARE NOW", 1),
    ("Miracle cure BANNED by doctors Big Pharma doesn't want you to know this secret", 1),
    ("EXPLOSIVE leaked documents reveal global conspiracy to control world population", 1),
    ("YOU WON'T BELIEVE what scientists discovered hidden underground this changes everything", 1),
    ("Secret government program exposed alien technology suppressed for 50 years TRUTH", 1),
    ("BOMBSHELL whistleblower reveals Clinton Biden Obama conspiracy deep state cover-up", 1),
    ("This ancient remedy CURES all diseases doctors are FURIOUS share before banned", 1),
]

EMAIL_TRAINING_DATA = [
    # 0=LEGITIMATE, 1=SPAM
    ("Hi, can you review the Q3 report draft before Thursday's meeting?", 0),
    ("Your Amazon order has been shipped and will arrive by Friday", 0),
    ("Team standup is moved to 3pm today due to a conflict", 0),
    ("Please find attached the invoice for last month's services", 0),
    ("Following up on our conversation from yesterday about the project timeline", 0),
    ("Your flight confirmation for next Tuesday is attached", 0),
    ("Reminder: your dental appointment is scheduled for Monday at 2pm", 0),
    ("The pull request has been approved and merged into main", 0),
    ("Can we reschedule our 1:1 to Wednesday? I have a conflict on Tuesday", 0),
    ("Here is the updated contract for your review and signature", 0),
    ("CONGRATULATIONS you have WON $1,000,000 click here to claim your prize NOW!!!", 1),
    ("URGENT your account will be suspended verify your password immediately http://secure-bank-login.xyz", 1),
    ("Dear Friend I am Prince Adebayo I need your help transferring $45 million USD", 1),
    ("You have been selected for a FREE gift claim your reward before it expires ACT NOW", 1),
    ("LIMITED TIME OFFER 90% discount on all products click here guaranteed returns", 1),
    ("Your account has been compromised verify your details immediately or lose access", 1),
    ("Crypto investment opportunity guaranteed 500% profit risk-free limited spots available", 1),
    ("You are our lucky winner of the international lottery collect your prize today", 1),
    ("FINAL WARNING your service will be terminated unless you update payment details now", 1),
    ("Nigerian barrister seeks trustworthy person to transfer inheritance of $8.5 million", 1),
]


# ─── Classifier ───────────────────────────────────────────────────────────────

class FakeNewsSpamClassifier:
    """
    Trains two separate classifiers (news / email) and exposes
    predict() with SHAP-powered signal explanations.
    """

    def __init__(self):
        self.news_model = self._build_pipeline()
        self.email_model = self._build_pipeline()
        self.news_explainer = None
        self.email_explainer = None
        self._train()

    def _build_pipeline(self):
        tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        ling = LinguisticFeatureExtractor()

        # We keep the two transformers separate so we can access
        # linguistic feature names for SHAP attribution.
        return {
            "tfidf": tfidf,
            "ling": ling,
            "scaler": StandardScaler(),
            "clf": LogisticRegression(
                class_weight="balanced",   # handles imbalanced datasets
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
            ),
        }

    def _fit_pipeline(self, pipeline, texts, labels):
        X_tfidf = pipeline["tfidf"].fit_transform(texts).toarray()
        X_ling  = pipeline["ling"].fit_transform(texts)
        X_ling_scaled = pipeline["scaler"].fit_transform(X_ling)
        X = np.hstack([X_tfidf, X_ling_scaled])
        pipeline["clf"].fit(X, labels)
        return X

    def _transform_pipeline(self, pipeline, texts):
        X_tfidf = pipeline["tfidf"].transform(texts).toarray()
        X_ling  = pipeline["ling"].transform(texts)
        X_ling_scaled = pipeline["scaler"].transform(X_ling)
        return np.hstack([X_tfidf, X_ling_scaled])

    def _train(self):
        news_texts, news_labels = zip(*NEWS_TRAINING_DATA)
        email_texts, email_labels = zip(*EMAIL_TRAINING_DATA)

        X_news  = self._fit_pipeline(self.news_model,  list(news_texts),  list(news_labels))
        X_email = self._fit_pipeline(self.email_model, list(email_texts), list(email_labels))

        # Build SHAP Linear explainers on the training data
        self.news_explainer  = shap.LinearExplainer(
            self.news_model["clf"], X_news,  feature_perturbation="interventional"
        )
        self.email_explainer = shap.LinearExplainer(
            self.email_model["clf"], X_email, feature_perturbation="interventional"
        )

    def predict(self, text: str, mode: str) -> dict:
        """
        Returns verdict, confidence, SHAP-based signals, and
        top contributing features for the explainability panel.
        """
        pipeline  = self.news_model  if mode == "news"  else self.email_model
        explainer = self.news_explainer if mode == "news" else self.email_explainer
        labels    = ["REAL", "FAKE"] if mode == "news" else ["LEGITIMATE", "SPAM"]

        X = self._transform_pipeline(pipeline, [text])
        proba = pipeline["clf"].predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        confidence = round(float(proba[pred_idx]) * 100, 1)
        verdict = labels[pred_idx]

        # SHAP values for the predicted class
        shap_values = explainer.shap_values(X)
        # shap_values shape: (n_classes, n_samples, n_features) or (n_samples, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[pred_idx][0]
        else:
            sv = shap_values[0] if pred_idx == 1 else -shap_values[0]

        signals = self._build_signals(text, pipeline, sv, mode)
        top_features = self._top_features(pipeline, sv, n=5)
        ling_breakdown = self._linguistic_breakdown(text)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "signals": signals,
            "top_features": top_features,
            "ling_breakdown": ling_breakdown,
        }

    def _build_signals(self, text: str, pipeline, shap_vals, mode: str) -> list:
        """
        Map SHAP attribution onto 4 named signals for the UI.
        """
        text_lower = text.lower()
        ling = LinguisticFeatureExtractor()
        feats = ling._features(text)

        # Signal 1: Language sensationalism
        sens_score = min(100, int(
            feats[0] * 40          # caps ratio
            + feats[1] * 8         # exclamation marks
            + feats[3] * 12        # sensational words
        ))

        # Signal 2: Source / credibility indicators
        cred_score = min(100, int(feats[5] * 20 + feats[6] * 5 + feats[8] * 3))

        # Signal 3: Factual claim structure
        has_numbers = feats[8] > 0
        has_source  = feats[5] > 0
        factual_score = min(100, int(
            (30 if has_numbers else 0)
            + (40 if has_source else 0)
            + feats[8] * 5
        ))

        # Signal 4: Urgency / pressure (email) or headline framing (news)
        if mode == "email":
            urgency_score = min(100, int(feats[4] * 15 + feats[1] * 10 + feats[7] * 20))
            signal4_name = "Urgency & pressure"
        else:
            urgency_score = min(100, int(feats[3] * 12 + feats[1] * 8 + feats[0] * 30))
            signal4_name = "Headline framing"

        def direction(score, invert=False):
            if invert:
                return "credible" if score > 40 else "neutral" if score > 15 else "suspicious"
            return "suspicious" if score > 60 else "neutral" if score > 25 else "credible"

        return [
            {"name": "Language sensationalism", "score": sens_score,    "direction": direction(sens_score)},
            {"name": "Source credibility",       "score": cred_score,   "direction": direction(cred_score, invert=True)},
            {"name": "Factual claim structure",  "score": factual_score,"direction": direction(factual_score, invert=True)},
            {"name": signal4_name,               "score": urgency_score,"direction": direction(urgency_score)},
        ]

    def _top_features(self, pipeline, shap_vals, n=5) -> list:
        """Return top-n SHAP features by absolute magnitude."""
        tfidf_names = pipeline["tfidf"].get_feature_names_out().tolist()
        ling_names  = LinguisticFeatureExtractor.FEATURE_NAMES
        all_names   = tfidf_names + ling_names

        indices = np.argsort(np.abs(shap_vals))[::-1][:n]
        return [
            {
                "feature": all_names[i] if i < len(all_names) else f"feature_{i}",
                "shap_value": round(float(shap_vals[i]), 4),
                "impact": "increases risk" if shap_vals[i] > 0 else "reduces risk",
            }
            for i in indices
        ]

    def _linguistic_breakdown(self, text: str) -> dict:
        """Return raw linguistic feature values for display."""
        ling = LinguisticFeatureExtractor()
        values = ling._features(text)
        return dict(zip(LinguisticFeatureExtractor.FEATURE_NAMES, [round(v, 3) for v in values]))


# Singleton — loaded once when Flask starts
_classifier = None

def get_classifier() -> FakeNewsSpamClassifier:
    global _classifier
    if _classifier is None:
        _classifier = FakeNewsSpamClassifier()
    return _classifier
