// ── Config ──────────────────────────────────────────────────────────────────
// Set your Anthropic API key here, or use an env var via a build tool / proxy.
// WARNING: Never expose real API keys in public repos.
// Use a server-side proxy or environment variable in production.
const ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE";
const MODEL = "claude-sonnet-4-20250514";

// ── Example data ─────────────────────────────────────────────────────────────
const EXAMPLES = {
  news: [
    "Scientists confirm that eating chocolate daily cures cancer, Big Pharma hiding truth",
    "Federal Reserve raises interest rates by 0.25% amid inflation concerns",
    "BREAKING: Moon found to be made of cheese by NASA probe, government covers up",
    "Apple reports record quarterly earnings, beats analyst expectations",
    "5G towers proven to cause mind control, whistleblower reveals secret documents",
  ],
  email: [
    "Congratulations! You've won $1,000,000. Click here to claim your prize NOW!!!",
    "Hi Sarah, can you review the Q3 report draft before Thursday's meeting?",
    "URGENT: Your account will be suspended. Verify your password immediately at: http://secure-bank-login.xyz",
    "Your Amazon order #114-8892341 has been shipped and will arrive Friday.",
    "Dear Friend, I am Prince Adebayo and I need your help transferring $45 million USD",
  ],
};

// ── State ─────────────────────────────────────────────────────────────────────
let currentMode = "news";

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(mode) {
  currentMode = mode;

  document.getElementById("tab-news").classList.toggle("active", mode === "news");
  document.getElementById("tab-news").setAttribute("aria-selected", mode === "news");
  document.getElementById("tab-email").classList.toggle("active", mode === "email");
  document.getElementById("tab-email").setAttribute("aria-selected", mode === "email");

  document.getElementById("input-text").placeholder =
    mode === "news"
      ? "Paste a news headline or article snippet to analyze…"
      : "Paste an email subject line or message body to analyze…";

  renderExamples();
  document.getElementById("result-area").innerHTML = "";
}

// ── Render example pills ──────────────────────────────────────────────────────
function renderExamples() {
  const row = document.getElementById("examples-row");
  row.innerHTML = EXAMPLES[currentMode]
    .map(
      (e) =>
        `<button class="ex-pill" title="${e}" role="listitem" onclick="loadExample(this)" aria-label="Load example: ${e}">${e}</button>`
    )
    .join("");
}

function loadExample(el) {
  document.getElementById("input-text").value = el.title;
  document.getElementById("input-text").focus();
}

function clearInput() {
  document.getElementById("input-text").value = "";
  document.getElementById("result-area").innerHTML = "";
  document.getElementById("input-text").focus();
}

// ── Analyze ───────────────────────────────────────────────────────────────────
async function analyze() {
  const text = document.getElementById("input-text").value.trim();
  if (!text) {
    document.getElementById("input-text").focus();
    return;
  }

  const btn = document.getElementById("analyze-btn");
  setLoading(btn, true);
  document.getElementById("result-area").innerHTML = "";

  const systemPrompt = buildSystemPrompt();
  const userMessage = buildUserMessage(text);

  try {
    const result = await callClaude(systemPrompt, userMessage);
    renderResult(result);
  } catch (err) {
    renderError(err.message || "Analysis failed. Please try again.");
  } finally {
    setLoading(btn, false);
  }
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  btn.innerHTML = loading
    ? `<span class="spinner" aria-hidden="true"></span><span>Analyzing…</span>`
    : `<i class="ti ti-shield-search" aria-hidden="true"></i><span>Analyze</span>`;
}

// ── Anthropic API call ────────────────────────────────────────────────────────
async function callClaude(system, userContent) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 1024,
      system,
      messages: [{ role: "user", content: userContent }],
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error?.message || `API error ${res.status}`);
  }

  const data = await res.json();
  const rawText = (data.content || []).map((b) => b.text || "").join("");

  try {
    const clean = rawText.replace(/```json|```/g, "").trim();
    return JSON.parse(clean);
  } catch {
    throw new Error("Could not parse classifier response. Please try again.");
  }
}

// ── Prompts ───────────────────────────────────────────────────────────────────
function buildSystemPrompt() {
  return `You are an expert NLP classifier specializing in detecting fake news and spam.
Analyze the given text and return ONLY a valid JSON object — no markdown fences, no extra text.

Required schema:
{
  "verdict": "FAKE" | "REAL" | "SPAM" | "LEGITIMATE",
  "confidence": <integer 0–100>,
  "summary": "<2-3 sentence plain-English explanation of your classification reasoning>",
  "signals": [
    {
      "name": "<signal name, max 4 words>",
      "score": <integer 0–100>,
      "direction": "suspicious" | "neutral" | "credible"
    }
  ],
  "techniques": ["<NLP technique name>", ...]
}

Rules:
- signals must contain exactly 4 items covering: language sensationalism, source credibility indicators, factual claim structure, and urgency/pressure tactics (for email) or headline framing (for news).
- techniques should list 3–6 real NLP/ML methods that would help detect this pattern (e.g. TF-IDF, Named Entity Recognition, BERT sentiment, word2vec similarity, LIME, class-weight balancing).
- Be precise, evidence-based, and cite specific linguistic cues in your summary.`;
}

function buildUserMessage(text) {
  const modeLabel =
    currentMode === "news" ? "NEWS HEADLINE / ARTICLE" : "EMAIL / MESSAGE";
  return `Mode: ${modeLabel}\n\nText to classify:\n"${text}"`;
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(r) {
  const verdict = (r.verdict || "UNKNOWN").toUpperCase();
  const isBad = verdict === "FAKE" || verdict === "SPAM";
  const conf = clamp(Math.round(r.confidence ?? 50), 0, 100);

  const badgeClass = {
    FAKE: "badge-fake",
    REAL: "badge-real",
    SPAM: "badge-spam",
    LEGITIMATE: "badge-legit",
  }[verdict] ?? "badge-real";

  const verdictIcon = {
    FAKE: "ti-circle-x",
    REAL: "ti-circle-check",
    SPAM: "ti-ban",
    LEGITIMATE: "ti-circle-check",
  }[verdict] ?? "ti-help-circle";

  const barClass =
    conf >= 75 ? (isBad ? "bar-danger" : "bar-success") : "bar-warning";

  const signalsHtml = buildSignalsHtml(r.signals || []);
  const techHtml = (r.techniques || [])
    .map((t) => `<span class="tech-tag">${t}</span>`)
    .join("");

  document.getElementById("result-area").innerHTML = `
    <div class="result-card" role="region" aria-label="Analysis result">
      <div class="verdict-row">
        <span class="verdict-badge ${badgeClass}">
          <i class="ti ${verdictIcon}" aria-hidden="true"></i>
          ${verdict}
        </span>
        <div class="conf-group">
          <span class="conf-label">Confidence</span>
          <span class="conf-val">${conf}%</span>
        </div>
      </div>

      <div class="conf-bar-wrap" role="progressbar" aria-valuenow="${conf}" aria-valuemin="0" aria-valuemax="100">
        <div class="conf-bar ${barClass}" style="width: ${conf}%"></div>
      </div>

      <p class="section-label">Explanation</p>
      <p class="explanation">${escapeHtml(r.summary || "No explanation provided.")}</p>

      <p class="section-label">Signal analysis</p>
      <div class="signals-grid">${signalsHtml}</div>

      <p class="section-label" style="margin-top: 1rem;">NLP techniques</p>
      <div class="tech-list">${techHtml}</div>
    </div>`;
}

function buildSignalsHtml(signals) {
  const colorMap = {
    suspicious: { bar: "var(--red-bar)",   score: "var(--red-text)"   },
    credible:   { bar: "var(--green-bar)", score: "var(--green-text)" },
    neutral:    { bar: "var(--amber-bar)", score: "var(--amber-text)" },
  };

  return signals
    .map((s) => {
      const sc = clamp(Math.round(s.score ?? 50), 0, 100);
      const dir = s.direction || "neutral";
      const { bar, score } = colorMap[dir] ?? colorMap.neutral;
      return `<div class="signal-item">
        <div class="signal-top">
          <span class="signal-name">${escapeHtml(s.name)}</span>
          <span class="signal-score" style="color:${score}">${sc}%</span>
        </div>
        <div class="signal-direction">${dir}</div>
        <div class="signal-bar-wrap">
          <div class="signal-bar" style="width:${sc}%; background:${bar}"></div>
        </div>
      </div>`;
    })
    .join("");
}

function renderError(message) {
  document.getElementById("result-area").innerHTML = `
    <div class="error-msg" role="alert">
      <i class="ti ti-alert-circle" aria-hidden="true"></i>
      ${escapeHtml(message)}
    </div>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Init ──────────────────────────────────────────────────────────────────────
renderExamples();
