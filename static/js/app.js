"use strict";

// ── Examples ──────────────────────────────────────────────────────────────────
const EXAMPLES = {
  news: [
    "Scientists confirm that eating chocolate daily cures cancer, Big Pharma hiding the truth!!!",
    "Federal Reserve raises interest rates by 0.25% amid inflation concerns",
    "BREAKING: Moon found to be made of cheese by NASA probe, government covers up",
    "Apple reports record quarterly earnings, beats analyst expectations",
    "5G towers PROVEN to cause mind control, whistleblower reveals classified documents",
  ],
  email: [
    "Congratulations! You've won $1,000,000. Click here to claim your prize NOW!!!",
    "Hi Sarah, can you review the Q3 report draft before Thursday's meeting?",
    "URGENT: Your account will be suspended. Verify your password immediately at: http://secure-bank-login.xyz",
    "Your Amazon order #114-8892341 has been shipped and will arrive Friday.",
    "Dear Friend, I am Prince Adebayo and I need your help transferring $45 million USD",
  ],
};

let currentMode = "news";

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(mode) {
  currentMode = mode;
  document.getElementById("tab-news").classList.toggle("active", mode === "news");
  document.getElementById("tab-news").setAttribute("aria-selected", String(mode === "news"));
  document.getElementById("tab-email").classList.toggle("active", mode === "email");
  document.getElementById("tab-email").setAttribute("aria-selected", String(mode === "email"));
  document.getElementById("input-text").placeholder =
    mode === "news"
      ? "Paste a news headline or article snippet…"
      : "Paste an email subject line or message body…";
  renderExamples();
  document.getElementById("result-area").innerHTML = "";
}

function renderExamples() {
  const row = document.getElementById("examples-row");
  row.innerHTML = EXAMPLES[currentMode]
    .map(
      (e) =>
        `<button class="pill" title="${e}" onclick="loadExample(this)">${e}</button>`
    )
    .join("");
}

function loadExample(el) {
  document.getElementById("input-text").value = el.title;
}

function clearAll() {
  document.getElementById("input-text").value = "";
  document.getElementById("result-area").innerHTML = "";
}

// ── Analyze (calls Flask backend) ────────────────────────────────────────────
async function analyze() {
  const text = document.getElementById("input-text").value.trim();
  if (!text) return;

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Analyzing…`;
  document.getElementById("result-area").innerHTML = "";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode: currentMode }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    document.getElementById("result-area").innerHTML = `
      <div class="error-box">
        <i class="ti ti-alert-circle"></i> ${escHtml(err.message || "Analysis failed.")}
      </div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="ti ti-shield-search"></i> Analyze`;
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(d) {
  const conf = clamp(Math.round(d.confidence), 0, 100);
  const isBad = d.verdict === "FAKE" || d.verdict === "SPAM";
  const barColor = conf >= 75
    ? (isBad ? "var(--bar-red)" : "var(--bar-grn)")
    : "var(--bar-amb)";

  const verdictIcon = {
    FAKE: "ti-circle-x", REAL: "ti-circle-check",
    SPAM: "ti-ban",      LEGITIMATE: "ti-circle-check",
  }[d.verdict] ?? "ti-help-circle";

  document.getElementById("result-area").innerHTML = `
    <div class="result-card">

      <!-- Verdict + confidence -->
      <div class="verdict-row">
        <span class="verdict-badge badge-${d.verdict}">
          <i class="ti ${verdictIcon}"></i> ${d.verdict}
        </span>
        <div class="conf-group">
          <span class="conf-label">Confidence</span>
          <span class="conf-val">${conf}%</span>
        </div>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${conf}%; background:${barColor}"></div>
      </div>

      <!-- Claude explanation -->
      <p class="section-label">Explanation</p>
      <p class="explanation">${escHtml(d.explanation || "No explanation available.")}</p>

      <!-- Signals -->
      <p class="section-label">Signal analysis (SHAP-informed)</p>
      <div class="signals-grid">${buildSignalsHtml(d.signals || [])}</div>

      <!-- SHAP top features -->
      <p class="section-label" style="margin-top:1rem">Top SHAP features</p>
      <div class="shap-list">${buildShapHtml(d.top_features || [])}</div>

      <!-- Linguistic features -->
      <p class="section-label" style="margin-top:1rem">Linguistic feature values</p>
      <div class="ling-grid">${buildLingHtml(d.ling_breakdown || {})}</div>

    </div>`;
}

function buildSignalsHtml(signals) {
  const colorMap = {
    suspicious: { bar: "var(--bar-red)",  score: "var(--red-fg)"  },
    credible:   { bar: "var(--bar-grn)",  score: "var(--grn-fg)"  },
    neutral:    { bar: "var(--bar-amb)",  score: "var(--amb-fg)"  },
  };
  return signals.map((s) => {
    const sc  = clamp(Math.round(s.score), 0, 100);
    const dir = s.direction || "neutral";
    const { bar, score } = colorMap[dir] ?? colorMap.neutral;
    return `<div class="signal-item">
      <div class="signal-top">
        <span class="signal-name">${escHtml(s.name)}</span>
        <span class="signal-score" style="color:${score}">${sc}%</span>
      </div>
      <div class="signal-dir">${escHtml(dir)}</div>
      <div class="signal-bar-wrap">
        <div class="signal-bar" style="width:${sc}%; background:${bar}"></div>
      </div>
    </div>`;
  }).join("");
}

function buildShapHtml(features) {
  return features.map((f) => {
    const isPos = f.shap_value > 0;
    const color = isPos ? "var(--red-fg)" : "var(--grn-fg)";
    return `<div class="shap-item">
      <span class="shap-name">${escHtml(f.feature)}</span>
      <div class="shap-right">
        <span class="shap-val" style="color:${color}">${f.shap_value > 0 ? "+" : ""}${f.shap_value}</span>
        <span class="shap-impact">${escHtml(f.impact)}</span>
      </div>
    </div>`;
  }).join("");
}

function buildLingHtml(breakdown) {
  return Object.entries(breakdown).map(([k, v]) =>
    `<div class="ling-item">
      <span class="ling-name">${escHtml(k.replace(/_/g, " "))}</span>
      <span class="ling-val">${v}</span>
    </div>`
  ).join("");
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Init ──────────────────────────────────────────────────────────────────────
renderExamples();
