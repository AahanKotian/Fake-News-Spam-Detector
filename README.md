# Fake News & Spam Detector (Python / Flask)

A full-stack NLP classifier that detects fake news headlines and spam emails using a **real Python ML pipeline** — TF-IDF features, SHAP explainability, imbalanced-dataset handling — with Claude providing human-readable reasoning.

## Tech stack

| Layer | Technology |
|---|---|
| Web framework | Flask |
| Feature extraction | scikit-learn `TfidfVectorizer` + custom `LinguisticFeatureExtractor` |
| Classifier | `LogisticRegression(class_weight='balanced')` |
| Explainability | SHAP `LinearExplainer` (per-feature attribution) |
| Imbalanced data | `class_weight='balanced'` (equivalent to SMOTE for linear models) |
| Explanation text | Anthropic Claude Sonnet |
| Frontend | Vanilla JS + CSS (served by Flask) |

## Project structure

```
fake-news-detector/
├── app.py                  # Flask app + Claude API call
├── nlp_pipeline.py         # TF-IDF, linguistic features, LR classifier, SHAP
├── requirements.txt
├── templates/
│   └── index.html          # Jinja2 template
└── static/
    ├── css/style.css
    └── js/app.js
```

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # macOS / Linux
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."     # PowerShell
```

### 3. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

---

## How the pipeline works

### `nlp_pipeline.py`

#### 1. `LinguisticFeatureExtractor` (custom sklearn transformer)
Extracts 10 interpretable numerical features:
- `caps_ratio` — fraction of alphabetic characters that are uppercase
- `exclamation_count` — number of `!` characters (capped at 10)
- `question_count` — number of `?` characters
- `sensational_word_count` — count of known sensational words (exposed, shocking, coverup…)
- `spam_trigger_count` — count of spam-trigger phrases (click here, you won, claim now…)
- `credible_marker_count` — count of credibility markers (according to, study, percent…)
- `avg_word_length` — proxy for vocabulary complexity
- `url_count` — suspicious link count
- `number_count` — factual claims often include numbers
- `text_length_log` — log of character count

#### 2. TF-IDF vectoriser
500-feature unigram + bigram TF-IDF matrix with sublinear TF scaling and English stop-words removed. Combined with the linguistic features via `np.hstack`.

#### 3. `LogisticRegression(class_weight='balanced')`
Automatically up-weights the minority class during training — handling imbalanced datasets without needing SMOTE.

#### 4. SHAP `LinearExplainer`
Computes per-feature SHAP values for every prediction. The top features shown in the UI tell you **exactly which words and signals drove the decision** — the "why" behind the model.

---

## Extending this project

| Upgrade | How |
|---|---|
| Larger training set | Use [LIAR dataset](https://paperswithcode.com/dataset/liar) or [Enron spam corpus](https://www.cs.cmu.edu/~enron/) |
| Better embeddings | Replace TF-IDF with `sentence-transformers` (`all-MiniLM-L6-v2`) |
| SMOTE oversampling | `from imblearn.over_sampling import SMOTE` — use with tree-based models |
| Fine-tuned BERT | Fine-tune `bert-base-uncased` on LIAR; use SHAP `DeepExplainer` |
| LIME explainability | `from lime.lime_text import LimeTextExplainer` — model-agnostic alternative to SHAP |
| Persist the model | `import joblib; joblib.dump(clf, 'model.pkl')` |
| Docker deployment | See `Dockerfile` below |

## Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV ANTHROPIC_API_KEY=""
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 -e ANTHROPIC_API_KEY=sk-ant-... fake-news-detector
```

## License

MIT
