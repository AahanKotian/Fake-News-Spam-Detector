"""
app.py
──────
Flask backend for the Fake News & Spam Detector.

Routes:
  GET  /                  → Serves the UI
  POST /api/analyze       → Runs NLP pipeline + Claude explanation
  GET  /api/health        → Health check
"""

import os
import json
import anthropic
from flask import Flask, render_template, request, jsonify
from nlp_pipeline import get_classifier

app = Flask(__name__)

# ─── Anthropic client ─────────────────────────────────────────────────────────
# Set ANTHROPIC_API_KEY in your environment:
#   export ANTHROPIC_API_KEY="sk-ant-..."
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    mode = body.get("mode", "news")  # "news" | "email"

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if mode not in ("news", "email"):
        return jsonify({"error": "mode must be 'news' or 'email'"}), 400

    # ── Step 1: Local NLP pipeline (TF-IDF + linguistic features + SHAP) ──────
    clf = get_classifier()
    nlp_result = clf.predict(text, mode)

    # ── Step 2: Claude — plain-English explanation & refined reasoning ─────────
    explanation = get_claude_explanation(text, mode, nlp_result)

    return jsonify({
        "verdict":          nlp_result["verdict"],
        "confidence":       nlp_result["confidence"],
        "signals":          nlp_result["signals"],
        "top_features":     nlp_result["top_features"],
        "ling_breakdown":   nlp_result["ling_breakdown"],
        "explanation":      explanation,
        "mode":             mode,
    })


# ─── Claude explanation ───────────────────────────────────────────────────────

def get_claude_explanation(text: str, mode: str, nlp_result: dict) -> str:
    """
    Calls Claude to produce a concise, human-readable explanation
    of the NLP pipeline's verdict, grounded in the SHAP signals.
    """
    mode_label = "news headline / article" if mode == "news" else "email / message"
    top = nlp_result["top_features"]
    signals = nlp_result["signals"]
    verdict = nlp_result["verdict"]
    confidence = nlp_result["confidence"]

    feature_summary = ", ".join(
        f'"{f["feature"]}" ({f["impact"]})' for f in top[:3]
    )
    signal_summary = "; ".join(
        f'{s["name"]}: {s["score"]}% ({s["direction"]})' for s in signals
    )

    prompt = f"""A machine learning NLP classifier analysed the following {mode_label}:

"{text}"

Verdict: {verdict} (confidence {confidence}%)
Top SHAP features driving this prediction: {feature_summary}
Signal breakdown: {signal_summary}

Write a 2-3 sentence plain-English explanation of WHY the classifier made this decision.
Be specific — reference the actual linguistic cues in the text.
Do NOT use bullet points. Do NOT start with "The classifier".
"""

    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load the classifier on startup
    print("Loading NLP pipeline...")
    get_classifier()
    print("Ready.")
    app.run(debug=True, port=5000)
