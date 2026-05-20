# Fake News & Spam Detector

A browser-based NLP classifier that detects fake news headlines and spam emails with a confidence score and signal-level explainability — built with the Anthropic Claude API.

![screenshot](screenshot.png)

## Features

- **Two modes** — News headline analysis and Email / message analysis
- **Verdict + confidence** — FAKE / REAL / SPAM / LEGITIMATE with a 0–100% confidence score
- **Signal analysis** — 4 explainability signals per prediction (sensationalism, credibility, factual structure, urgency) — mirrors SHAP/LIME model explainability
- **NLP techniques flagged** — lists real ML methods (TF-IDF, NER, BERT, word2vec, etc.) that would catch each pattern in a production pipeline
- **Dark mode** — auto-adapts to system preference
- **No build step** — pure HTML / CSS / JS, open `index.html` directly

## Quick start

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Add your API key**

   Open `app.js` and replace the placeholder:
   ```js
   const ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE";
   ```

   > ⚠️ Never commit a real API key. For production, route requests through a server-side proxy (see below).

3. **Open in browser**
   ```bash
   open index.html
   # or serve with any static server:
   npx serve .
   ```

## Project structure

```
fake-news-detector/
├── index.html   # Markup & layout
├── style.css    # All styles (light + dark mode)
├── app.js       # Classifier logic + Anthropic API calls
└── README.md
```

## Production / server-side proxy (recommended)

Calling the Anthropic API directly from the browser exposes your API key. For a real deployment, add a thin server proxy:

```
Browser  →  POST /api/analyze  →  Your server  →  Anthropic API
```

Example with Express:
```js
// server.js
const express = require("express");
const app = express();
app.use(express.json());

app.post("/api/analyze", async (req, res) => {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(req.body),
  });
  const data = await response.json();
  res.json(data);
});

app.listen(3000);
```

Then in `app.js`, change the fetch URL from `https://api.anthropic.com/v1/messages` to `/api/analyze` and remove the `x-api-key` header.

## Extending this project (resume-worthy upgrades)

| Technique | How to add |
|---|---|
| **TF-IDF features** | Pre-compute a vocabulary from a labeled dataset; score incoming text against known fake/spam n-grams |
| **SHAP / LIME explainability** | Train a scikit-learn pipeline (LogisticRegression or XGBoost), then use `shap.Explainer` or `lime.lime_text.LimeTextExplainer` to get per-token importance scores |
| **Imbalanced dataset handling** | Use `imbalanced-learn` SMOTE oversampling or `class_weight='balanced'` in sklearn classifiers |
| **Word embeddings** | Swap TF-IDF for `sentence-transformers` embeddings for semantic similarity |
| **Fine-tuned BERT** | Fine-tune `bert-base-uncased` on [LIAR dataset](https://paperswithcode.com/dataset/liar) or [Enron spam corpus](https://www.cs.cmu.edu/~enron/) |

## Datasets

- **Fake news**: [LIAR dataset](https://paperswithcode.com/dataset/liar), [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- **Spam**: [Enron email dataset](https://www.cs.cmu.edu/~enron/), [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## License

MIT
