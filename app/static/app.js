/**
 * CreditRiskML — Interactive Dashboard Application Logic
 */

// Presets configuration
const PRESETS = {
  prime: {
    status_checking: "A14",
    duration_months: 12,
    credit_history: "A34",
    purpose: "A40",
    credit_amount: 1500,
    savings_account: "A64",
    employment_since: "A75",
    installment_rate_pct: 2,
    personal_status_sex: "A93",
    other_debtors: "A101",
    residence_since_years: 4,
    property: "A121",
    age_years: 45,
    other_installment_plans: "A143",
    housing: "A152",
    existing_credits: 1,
    job: "A173",
    people_liable: 1,
    telephone: "A192",
    foreign_worker: "A201"
  },
  borderline: {
    status_checking: "A12",
    duration_months: 24,
    credit_history: "A32",
    purpose: "A42",
    credit_amount: 3800,
    savings_account: "A62",
    employment_since: "A73",
    installment_rate_pct: 4,
    personal_status_sex: "A92",
    other_debtors: "A101",
    residence_since_years: 2,
    property: "A122",
    age_years: 29,
    other_installment_plans: "A143",
    housing: "A151",
    existing_credits: 1,
    job: "A173",
    people_liable: 1,
    telephone: "A191",
    foreign_worker: "A201"
  },
  default: {
    status_checking: "A11",
    duration_months: 48,
    credit_history: "A30",
    purpose: "A49",
    credit_amount: 9800,
    savings_account: "A61",
    employment_since: "A71",
    installment_rate_pct: 4,
    personal_status_sex: "A94",
    other_debtors: "A101",
    residence_since_years: 1,
    property: "A124",
    age_years: 22,
    other_installment_plans: "A141",
    housing: "A151",
    existing_credits: 2,
    job: "A172",
    people_liable: 2,
    telephone: "A191",
    foreign_worker: "A201"
  }
};

let recentPredictions = [];

// Toast notification helper
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Tab Switching
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");

      if (targetId === "tab-model") loadModelMetadata();
      if (targetId === "tab-monitoring") loadMonitoringData();
    });
  });
}

// Populate form from preset
function applyPreset(presetKey) {
  const data = PRESETS[presetKey];
  if (!data) return;

  const form = document.getElementById("scoring-form");
  for (const [key, value] of Object.entries(data)) {
    const input = form.elements[key];
    if (input) {
      input.value = value;
    }
  }
  showToast(`Loaded ${presetKey.toUpperCase()} borrower profile preset`, "info");
}

// Collect form data
function getFormData() {
  const form = document.getElementById("scoring-form");
  const formData = new FormData(form);
  const data = {};

  const numFields = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "residence_since_years", "age_years", "existing_credits", "people_liable"
  ];

  formData.forEach((value, key) => {
    if (numFields.includes(key)) {
      data[key] = parseFloat(value);
    } else {
      data[key] = value;
    }
  });

  return data;
}

// Render Risk Gauge and Metrics
function updateScoreResults(result, inputData) {
  const prob = typeof result.default_probability === "number" ? result.default_probability : 0;
  const pct = (prob * 100).toFixed(1);
  const isDefault = result.prediction === 1;

  // 1. Gauge value text
  document.getElementById("gauge-val").textContent = `${pct}%`;

  // 2. Gauge Arc Animation
  const gaugeFill = document.getElementById("gauge-fill-path");
  const totalLength = 283; // approx semi-circle stroke-dasharray
  const offset = totalLength * (1 - prob);
  gaugeFill.style.strokeDashoffset = offset;

  // 3. Gauge Color & Risk Badge
  const badge = document.getElementById("risk-badge");
  badge.className = "risk-badge";

  if (prob < 0.30) {
    gaugeFill.style.stroke = "var(--risk-low)";
    badge.classList.add("low");
    badge.innerHTML = `🛡️ Low Risk Tier (Approved)`;
  } else if (prob < 0.47) {
    gaugeFill.style.stroke = "var(--risk-medium)";
    badge.classList.add("medium");
    badge.innerHTML = `⚠️ Medium Risk Tier (Review)`;
  } else {
    gaugeFill.style.stroke = "var(--risk-high)";
    badge.classList.add("high");
    badge.innerHTML = `🚨 High Default Risk (Declined)`;
  }

  // 4. Metric pills
  document.getElementById("res-decision").textContent = isDefault ? "REJECT / DEFAULT RISK" : "APPROVE / LOW RISK";
  document.getElementById("res-decision").style.color = isDefault ? "var(--accent-rose)" : "var(--accent-emerald)";

  const threshold = result.decision_threshold ?? result.threshold_used ?? 0.47;
  document.getElementById("res-threshold").textContent = typeof threshold === "number" ? threshold.toFixed(2) : String(threshold);

  const latency = result.latency_ms ?? 0;
  document.getElementById("res-latency").textContent = `${typeof latency === "number" ? latency.toFixed(1) : latency} ms`;

  const reqId = result.request_id || result.prediction_id || "—";
  document.getElementById("res-id").textContent = reqId.slice(0, 8);

  // 5. Contributing risk factors calculation
  renderKeyFactors(inputData, prob, result.top_risk_factors);

  // 6. Add to Recent Predictions Table
  addRecentPrediction(result, inputData);
}

// Synthesize Key Risk Indicators
function renderKeyFactors(input, prob, topRiskFactors = []) {
  const container = document.getElementById("risk-factors-list");
  container.innerHTML = "";

  const factors = [];

  // 1. Model-based explanations (if returned by backend)
  if (Array.isArray(topRiskFactors) && topRiskFactors.length > 0) {
    topRiskFactors.forEach(f => {
      const isPositive = f.direction === "decreases_risk";
      const tagText = isPositive ? "Protective" : "Elevated Risk";
      const tagType = isPositive ? "positive" : "negative";
      const cleanName = (f.feature || "").replace(/^(num__|cat__)/, "").replace(/_/g, " ");
      const displayFeature = cleanName ? cleanName.charAt(0).toUpperCase() + cleanName.slice(1) : f.feature;
      const contribStr = typeof f.contribution === "number" ? ` (${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(3)})` : "";
      factors.push({
        label: `${displayFeature}${contribStr}`,
        tag: tagText,
        type: tagType
      });
    });
  }

  // 2. Domain heuristics fallback
  if (factors.length === 0) {
    // Checking status
    if (input.status_checking === "A11") {
      factors.push({ label: "Checking account balance < 0 DM (Overdrawn)", tag: "Negative", type: "negative" });
    } else if (input.status_checking === "A14") {
      factors.push({ label: "No checking account / balance >= 200 DM", tag: "Positive", type: "positive" });
    }

    // Duration
    if (input.duration_months >= 36) {
      factors.push({ label: `Long loan term (${input.duration_months} mo)`, tag: "Elevated Risk", type: "negative" });
    } else if (input.duration_months <= 12) {
      factors.push({ label: `Short credit duration (${input.duration_months} mo)`, tag: "Low Exposure", type: "positive" });
    }

    // Savings
    if (input.savings_account === "A61") {
      factors.push({ label: "Low savings buffer (< 100 DM)", tag: "Negative", type: "negative" });
    } else if (input.savings_account === "A64" || input.savings_account === "A63") {
      factors.push({ label: "Substantial savings buffer (>= 500 DM)", tag: "Positive", type: "positive" });
    }

    // Installment rate
    if (input.installment_rate_pct >= 4) {
      factors.push({ label: `High debt burden (${input.installment_rate_pct}% of income)`, tag: "Burden", type: "negative" });
    }

    // Credit amount
    if (input.credit_amount > 5000) {
      factors.push({ label: `High credit amount (${input.credit_amount} DM)`, tag: "Exposure", type: "negative" });
    }

    if (factors.length === 0) {
      factors.push({ label: "Standard applicant profile within normal credit thresholds", tag: "Neutral", type: "positive" });
    }
  }

  factors.slice(0, 4).forEach(f => {
    const item = document.createElement("div");
    item.className = "driver-item";
    item.innerHTML = `
      <span>${f.label}</span>
      <span class="driver-tag ${f.type}">${f.tag}</span>
    `;
    container.appendChild(item);
  });
}

function addRecentPrediction(result, input) {
  const reqId = result.request_id || result.prediction_id || String(Date.now());
  const probVal = typeof result.default_probability === "number" ? result.default_probability : 0;
  const latencyVal = result.latency_ms ?? 0;
  const ratingVal = result.risk_level || result.risk_tier || (probVal >= 0.47 ? "High" : (probVal >= 0.30 ? "Medium" : "Low"));

  const item = {
    id: reqId.slice(0, 8),
    time: new Date().toLocaleTimeString(),
    amount: `${input.credit_amount} DM`,
    duration: `${input.duration_months} mo`,
    probability: `${(probVal * 100).toFixed(1)}%`,
    rating: ratingVal,
    status: result.prediction === 1 ? "Bad / Default" : "Good",
    latency: `${typeof latencyVal === "number" ? latencyVal.toFixed(1) : latencyVal}ms`
  };

  recentPredictions.unshift(item);
  if (recentPredictions.length > 8) recentPredictions.pop();

  const tbody = document.getElementById("recent-predictions-tbody");
  if (!tbody) return;

  tbody.innerHTML = "";
  recentPredictions.forEach(row => {
    const tr = document.createElement("tr");
    const isBad = row.status === "Bad / Default";
    tr.innerHTML = `
      <td style="font-family: monospace;">${row.id}</td>
      <td>${row.time}</td>
      <td>${row.amount}</td>
      <td>${row.duration}</td>
      <td><span style="font-weight:700; color:${isBad ? 'var(--accent-rose)' : 'var(--accent-emerald)'}">${row.probability}</span></td>
      <td><span class="driver-tag ${isBad ? 'negative' : 'positive'}">${row.rating}</span></td>
      <td>${row.latency}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Model Metadata & Architecture Loader
async function loadModelMetadata() {
  try {
    const res = await fetch("/model");
    if (!res.ok) throw new Error("Failed to fetch model metadata");
    const data = await res.json();

    document.getElementById("meta-name").textContent = data.model_name || "Champion";
    document.getElementById("meta-version").textContent = data.model_version || "v1";
    const threshold = data.decision_threshold ?? 0.47;
    document.getElementById("meta-threshold").textContent = typeof threshold === "number" ? threshold.toFixed(2) : String(threshold);
    document.getElementById("meta-status").textContent = (data.status || "Champion").toUpperCase();

    // Test metrics
    const test = data.test_metrics;
    if (test) {
      if (test.roc_auc != null) document.getElementById("metric-roc-auc").textContent = Number(test.roc_auc).toFixed(3);
      if (test.recall != null) document.getElementById("metric-recall").textContent = `${(Number(test.recall) * 100).toFixed(1)}%`;
      if (test.precision != null) document.getElementById("metric-precision").textContent = `${(Number(test.precision) * 100).toFixed(1)}%`;
      if (test.f1 != null) document.getElementById("metric-f1").textContent = Number(test.f1).toFixed(3);
      if (test.pr_auc != null) document.getElementById("metric-prauc").textContent = Number(test.pr_auc).toFixed(3);
      if (test.brier_score != null) document.getElementById("metric-brier").textContent = Number(test.brier_score).toFixed(3);
    }
  } catch (err) {
    console.error(err);
    showToast("Could not load active model metadata", "error");
  }
}

// MLOps Drift & Performance Checker
async function loadMonitoringData() {
  // Load drift
  try {
    const res = await fetch("/monitoring/drift");
    if (res.ok) {
      const data = await res.json();
      renderDriftTable(data);
    }
  } catch (e) {
    console.warn("Drift check:", e);
  }

  // Load performance
  try {
    const res = await fetch("/monitoring/performance");
    if (res.ok) {
      const data = await res.json();
      const perfStatus = document.getElementById("perf-status-desc");
      if (perfStatus) {
        perfStatus.textContent = data.status === "insufficient ground truth"
          ? `Monitoring active (${data.samples_count} labeled records in buffer). Minimum required: 20.`
          : `Observed Performance Evaluated. Degradation detected: ${data.degradation_detected}`;
      }
    }
  } catch (e) {
    console.warn("Performance check:", e);
  }
}

function renderDriftTable(data) {
  const tbody = document.getElementById("drift-table-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const featureList = Array.isArray(data.features)
    ? data.features
    : Object.entries(data.features || {}).map(([feat, info]) => ({ feature: feat, ...info }));

  featureList.forEach(info => {
    const tr = document.createElement("tr");
    const isDrifted = Boolean(info.drift_detected);
    const testType = info.type ? info.type.toUpperCase() : (info.test_type ? info.test_type.toUpperCase() : "PSI");
    const statVal = info.psi != null ? info.psi : (info.statistic != null ? info.statistic : 0);
    const pVal = info.ks_pvalue != null ? info.ks_pvalue : (info.p_value != null ? info.p_value : null);
    const thresholdVal = info.threshold ?? data.psi_threshold ?? 0.10;

    tr.innerHTML = `
      <td style="font-weight: 600;">${info.feature || "—"}</td>
      <td>${testType}</td>
      <td>${typeof statVal === "number" ? statVal.toFixed(4) : statVal}</td>
      <td>${pVal !== null && pVal !== undefined ? (typeof pVal === "number" ? pVal.toFixed(4) : pVal) : "—"}</td>
      <td>${thresholdVal}</td>
      <td><span class="driver-tag ${isDrifted ? 'negative' : 'positive'}">${isDrifted ? 'DRIFT ALERT' : 'STABLE'}</span></td>
    `;
    tbody.appendChild(tr);
  });

  const badge = document.getElementById("drift-summary-badge");
  if (badge) {
    const hasDrift = Boolean(data.dataset_drift_detected || data.drift_detected);
    badge.textContent = hasDrift ? "Drift Warning" : "Zero Drift";
    badge.className = `driver-tag ${hasDrift ? 'negative' : 'positive'}`;
  }
}

// Retrain Pipeline Trigger
async function triggerRetraining() {
  const btn = document.getElementById("btn-trigger-retrain");
  btn.disabled = true;
  btn.innerHTML = "Retraining in progress...";
  showToast("Triggered automated retraining pipeline...", "info");

  try {
    const res = await fetch("/retrain", { method: "POST" });
    const result = await res.json();

    if (res.ok) {
      showToast(`Retraining Complete: ${result.message}`, "success");
      loadModelMetadata();
      loadMonitoringData();
    } else {
      showToast(`Retraining Error: ${result.detail || 'Failed'}`, "error");
    }
  } catch (err) {
    showToast(`Retraining failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Trigger Automated Retraining";
  }
}

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  initTabs();

  // Preset buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const p = btn.getAttribute("data-preset");
      applyPreset(p);
    });
  });

  // Scoring Form Submit
  const form = document.getElementById("scoring-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn-score-app");
    btn.disabled = true;
    btn.innerHTML = "Evaluating Credit Risk...";

    try {
      const payload = getFormData();
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Scoring failed");
      }

      const result = await res.json();
      updateScoreResults(result, payload);
      showToast("Credit risk assessment generated successfully!", "success");
    } catch (err) {
      console.error(err);
      showToast(`Error: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.innerHTML = "⚡ Evaluate Borrower Application";
    }
  });

  // Retrain button
  const retrainBtn = document.getElementById("btn-trigger-retrain");
  if (retrainBtn) {
    retrainBtn.addEventListener("click", triggerRetraining);
  }

  // Load Prime preset initially and evaluate
  applyPreset("prime");
  loadModelMetadata();
});
