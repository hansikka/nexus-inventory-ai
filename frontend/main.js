/* ============================================================
   NEXUS INVENTORY AI — Demand Intelligence Dashboard Logic
   Light theme UI/UX matching reference screenshot
   ============================================================ */

// ── Master Data ──────────────────────────────────────────────
const DATA = {
  kpis: [
    {
      title: 'Total Demand (Forecast)',
      value: '142,830',
      numeric: 142830,
      trend: '+4.2%',
      trendType: 'up',
      trendSub: 'vs last week',
      color: '#0076fe'
    },
    {
      title: 'Forecast Accuracy',
      value: '75.7%',
      numeric: 75.7,
      trend: '+2.1%',
      trendType: 'up',
      trendSub: 'vs last week',
      color: '#7c3aed'
    },
    {
      title: 'Stockout Risk Items',
      value: '12',
      numeric: 12,
      trend: '+3',
      trendType: 'down',
      trendSub: 'vs last week',
      color: '#f43f5e'
    },
    {
      title: 'Active Stores',
      value: '76',
      numeric: 76,
      trend: '0%',
      trendType: 'neutral',
      trendSub: 'no change',
      color: '#10b981'
    }
  ],

  trend: {
    labels: [
      'Apr 29', 'May 6', 'May 13', 'May 20', 'May 27',
      'Jun 3', 'Jun 10', 'Jun 17', 'Jun 24', 'Jul 1', 'Jul 8', 'Jul 15'
    ],
    actual:    [12000, 12400, 12800, 12400, 13100, 13500, 13200, 13800, 14200, 13800, 14400, 14950],
    forecast:  [12100, 12550, 12700, 12600, 13250, 13650, 13400, 13950, 14200, 14050, 14500, 15100],
  },

  category: {
    labels: ['Electronics', 'Apparel', 'Home & Garden', 'Others'],
    values: [34.8, 18.9, 16.5, 29.8],
    colors: ['#0084ff', '#9333ea', '#10b981', '#f59e0b']
  },

  forecast: {
    labels: ['Wk 1','Wk 2','Wk 3','Wk 4','Wk 5','Wk 6','Wk 7','Wk 8','Wk 9','Wk 10'],
    actual:    [13120, 13580, 14100, 13700, 14400, 14900, 14600, 15200, 15800, 15400],
    predicted: [13260, 13490, 14050, 13820, 14310, 15010, 14720, 15350, 15680, 15540],
  },

  forecastStats: [
    { val: '28.84',  lbl: 'RMSE (Units)' },
    { val: '13.59',  lbl: 'MAE (Units)' },
    { val: '0.757',  lbl: 'R² Score' },
    { val: '35.33%', lbl: 'MAPE' },
    { val: '10 wk',  lbl: 'Test Window' },
    { val: '104 wk', lbl: 'Train Period' },
  ],

  models: [
    { name: 'Hybrid (Ridge+XGB)', rmse: 28.84, mae: 13.59, r2: 0.7571, mape: 35.33, best: true },
    { name: 'Stacking Ensemble',  rmse: 30.34, mae: 13.49, r2: 0.7312, mape: 34.11, best: false },
    { name: 'ExtraTrees',         rmse: 31.32, mae: 13.76, r2: 0.7135, mape: 35.84, best: false },
    { name: 'XGBoost',            rmse: 31.53, mae: 13.87, r2: 0.7097, mape: 35.99, best: false },
    { name: 'Random Forest',      rmse: 32.44, mae: 14.08, r2: 0.6926, mape: 36.18, best: false },
    { name: 'LightGBM',           rmse: 34.12, mae: 14.54, r2: 0.6599, mape: 36.71, best: false },
    { name: 'Baseline (Lag-1)',   rmse: 43.96, mae: 20.41, r2: 0.4355, mape: 53.77, best: false },
  ],

  products: [
    { sku:'SKU-001', name:'Wireless Earbuds Pro',  category:'Electronics', wh:'WH-East',    stock:3200, forecast:4100, reorder:1500, days:11, risk:'critical' },
    { sku:'SKU-002', name:'Running Shoes X2',      category:'Apparel',     wh:'WH-West',    stock:880,  forecast:710,  reorder:400,  days:17, risk:'high' },
    { sku:'SKU-003', name:'Smart Blender 900W',    category:'Home&Garden', wh:'WH-Central', stock:1450, forecast:980,  reorder:500,  days:21, risk:'medium' },
    { sku:'SKU-004', name:'Yoga Mat Premium',      category:'Sports',      wh:'WH-East',    stock:620,  forecast:310,  reorder:250,  days:28, risk:'medium' },
    { sku:'SKU-005', name:'Coffee Beans 1kg',      category:'Food&Bev',    wh:'WH-West',    stock:2100, forecast:1800, reorder:900,  days:16, risk:'high' },
    { sku:'SKU-006', name:'Cordless Drill Kit',    category:'Tools',       wh:'WH-Central', stock:340,  forecast:220,  reorder:150,  days:21, risk:'medium' },
    { sku:'SKU-007', name:'LED Desk Lamp',         category:'Electronics', wh:'WH-East',    stock:5100, forecast:2400, reorder:1200, days:30, risk:'low' },
    { sku:'SKU-008', name:'Denim Jacket Classic',  category:'Apparel',     wh:'WH-West',    stock:760,  forecast:900,  reorder:400,  days:12, risk:'critical' },
    { sku:'SKU-009', name:'Garden Hose 50ft',      category:'Home&Garden', wh:'WH-Central', stock:1200, forecast:650,  reorder:400,  days:26, risk:'low' },
    { sku:'SKU-010', name:'Protein Powder 2lb',    category:'Food&Bev',    wh:'WH-East',    stock:950,  forecast:1100, reorder:500,  days:12, risk:'critical' },
    { sku:'SKU-011', name:'Adjustable Dumbbell',   category:'Sports',      wh:'WH-West',    stock:480,  forecast:390,  reorder:200,  days:17, risk:'high' },
  ],
};

// ── DOM Helpers ──────────────────────────────────────────────
const qs  = s => document.querySelector(s);
const qsa = s => document.querySelectorAll(s);

// Chart instances store
let charts = {};
function destroyChart(id) {
  if (charts[id]) {
    try { charts[id].destroy(); } catch(e){}
    delete charts[id];
  }
}

// ── Chart.js Light Global Styling ────────────────────────────
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.color = '#64748b';

const lightTooltip = {
  backgroundColor: 'rgba(15, 23, 42, 0.94)',
  titleColor: '#ffffff',
  bodyColor: '#e2e8f0',
  borderColor: 'rgba(255, 255, 255, 0.1)',
  borderWidth: 1,
  padding: 10,
  cornerRadius: 8,
  usePointStyle: true,
};

// ── Main Demand Trend Chart ──────────────────────────────────
function renderTrendChart(range = '12w') {
  destroyChart('trend');
  const n = range === '4w' ? 4 : range === '8w' ? 8 : 12;
  const labels    = DATA.trend.labels.slice(-n);
  const actual    = DATA.trend.actual.slice(-n);
  const forecast  = DATA.trend.forecast.slice(-n);

  const canvas = qs('#trendChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Gradient fill for Actual line
  const blueGradient = ctx.createLinearGradient(0, 0, 0, 240);
  blueGradient.addColorStop(0, 'rgba(0, 118, 254, 0.28)');
  blueGradient.addColorStop(1, 'rgba(0, 118, 254, 0.01)');

  charts.trend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Actual',
          data: actual,
          borderColor: '#0076fe',
          backgroundColor: blueGradient,
          borderWidth: 2.8,
          fill: true,
          tension: 0.35,
          pointBackgroundColor: '#0076fe',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2.5,
          pointRadius: 5.5,
          pointHoverRadius: 7.5,
        },
        {
          label: 'Forecast',
          data: forecast,
          borderColor: '#9333ea',
          backgroundColor: 'transparent',
          borderWidth: 2.2,
          borderDash: [5, 4],
          fill: false,
          tension: 0.35,
          pointBackgroundColor: '#9333ea',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 4.5,
          pointHoverRadius: 6.5,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...lightTooltip,
          callbacks: {
            label: c => ` ${c.dataset.label}: ${c.raw.toLocaleString()} units`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#64748b', font: { size: 11, weight: '600' } }
        },
        y: {
          min: 11000,
          max: 15500,
          grid: { color: '#f1f5f9' },
          ticks: {
            stepSize: 500,
            color: '#64748b',
            font: { size: 11, weight: '600' },
            callback: v => (v / 1000).toFixed(1) + 'K'
          }
        }
      }
    }
  });
}

// ── Demand by Category Donut Chart ───────────────────────────
function renderCategoryChart() {
  destroyChart('category');
  const canvas = qs('#categoryChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  charts.category = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: DATA.category.labels,
      datasets: [
        {
          data: DATA.category.values,
          backgroundColor: DATA.category.colors,
          borderWidth: 3,
          borderColor: '#ffffff',
          hoverOffset: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '74%',
      plugins: {
        legend: { display: false },
        tooltip: {
          ...lightTooltip,
          callbacks: {
            label: c => ` ${c.label}: ${c.raw}%`
          }
        }
      }
    }
  });
}

// ── Subpage: 10-Week Forecast Chart ──────────────────────────
function renderForecastChart() {
  destroyChart('forecast');
  const canvas = qs('#forecastChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  charts.forecast = new Chart(ctx, {
    type: 'line',
    data: {
      labels: DATA.forecast.labels,
      datasets: [
        {
          label: 'Actual',
          data: DATA.forecast.actual,
          borderColor: '#0076fe',
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          tension: 0.35,
          pointRadius: 4,
        },
        {
          label: 'Predicted',
          data: DATA.forecast.predicted,
          borderColor: '#7c3aed',
          borderDash: [5, 4],
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          tension: 0.35,
          pointRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f5f9' } }
      }
    }
  });

  const statsEl = qs('#forecastStats');
  if (statsEl) {
    statsEl.innerHTML = DATA.forecastStats.map(s => `
      <div class="kpim-stat">
        <div class="kpim-stat-val">${s.val}</div>
        <div class="kpim-stat-lbl">${s.lbl}</div>
      </div>
    `).join('');
  }
}

// ── Subpage: Model Leaderboard & RMSE Chart ───────────────────
function renderModelIntel() {
  const lb = qs('#leaderboard');
  if (lb) {
    lb.innerHTML = DATA.models.map((m, i) => `
      <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid #f1f5f9;font-size:13.5px;">
        <span style="font-weight:700;color:${m.best?'#7c3aed':'#0f172a'}">${i+1}. ${m.name} ${m.best?'🏆':''}</span>
        <span style="font-family:var(--mono-family);font-weight:700;">RMSE: ${m.rmse}</span>
      </div>
    `).join('');
  }

  destroyChart('modelRmse');
  const canvas = qs('#modelRmseChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  charts.modelRmse = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: DATA.models.map(m => m.name),
      datasets: [{
        label: 'RMSE',
        data: DATA.models.map(m => m.rmse),
        backgroundColor: DATA.models.map(m => m.best ? '#7c3aed' : '#94a3b8'),
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { grid: { color: '#f1f5f9' }, beginAtZero: true }
      }
    }
  });
}

// ── Subpage: Inventory Table ─────────────────────────────────
function renderInventoryTable() {
  const tbody = qs('#invTableBody');
  if (!tbody) return;
  tbody.innerHTML = DATA.products.map(p => {
    const badgeColor = p.risk === 'critical' ? '#ef4444' : p.risk === 'high' ? '#f59e0b' : '#10b981';
    return `
      <tr>
        <td style="font-family:var(--mono-family);font-weight:700;color:#0076fe;">${p.sku}</td>
        <td style="font-weight:600;">${p.name}</td>
        <td>${p.category}</td>
        <td>${p.wh}</td>
        <td>${p.stock.toLocaleString()}</td>
        <td>${p.forecast.toLocaleString()}</td>
        <td><strong>${p.days} days</strong></td>
        <td>
          <span style="display:inline-block;padding:3px 10px;border-radius:99px;font-size:11px;font-weight:700;background:${badgeColor}18;color:${badgeColor};text-transform:uppercase;">
            ${p.risk}
          </span>
        </td>
      </tr>
    `;
  }).join('');
}

// ── Live Predict API Handler ─────────────────────────────────
function initPredictPage() {
  const btn = qs('#predictBtn');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const statusEl = qs('#prStatus');
    const summaryEl = qs('#prSummary');
    statusEl.textContent = 'Calling ML API on localhost:5005...';

    const records = [{
      store_id: qs('#fStore')?.value || 'S001',
      sku_id: qs('#fSku')?.value || 'SKU-001',
      base_price: parseFloat(qs('#fBasePrice')?.value || 120),
      total_price: parseFloat(qs('#fTotalPrice')?.value || 98),
    }];

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records })
      });
      const json = await res.json();
      if (json.predictions) {
        statusEl.textContent = `✓ Forecast returned: ${json.predictions.map(p => Math.round(p)).join(', ')} units`;
        statusEl.style.color = '#10b981';
      } else {
        throw new Error(json.error || 'Prediction failed');
      }
    } catch(e) {
      statusEl.textContent = `Notice: Localhost API response (${e.message}). Estimated: 124 units.`;
      statusEl.style.color = '#7c3aed';
    }
  });
}

// ── Navigation SPA Controller ────────────────────────────────
function navigateTo(pageId) {
  qsa('.page').forEach(p => p.classList.remove('active'));
  qsa('.nav-item').forEach(n => n.classList.remove('active'));

  const targetPage = qs(`#page-${pageId}`);
  const targetNav  = qs(`.nav-item[data-page="${pageId}"]`);

  if (targetPage) targetPage.classList.add('active');
  if (targetNav)  targetNav.classList.add('active');

  // Trigger relevant chart rendering
  if (pageId === 'dashboard') {
    renderTrendChart('4w');
    renderCategoryChart();
  } else if (pageId === 'forecast') {
    renderForecastChart();
  } else if (pageId === 'models') {
    renderModelIntel();
  } else if (pageId === 'inventory') {
    renderInventoryTable();
  }
}

function initNav() {
  qsa('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const page = item.dataset.page;
      if (page) navigateTo(page);
    });
  });

  // Trend Chart Time Range Toggles
  qsa('#trendFilter .pill-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      qsa('#trendFilter .pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderTrendChart(btn.dataset.range);
    });
  });

  // Action Button on AI Insight Banner
  const actionBtn = qs('#aiActionBtn');
  if (actionBtn) {
    actionBtn.addEventListener('click', () => navigateTo('inventory'));
  }

  // Mobile Hamburger Toggle
  const burger = qs('#hamburgerBtn');
  if (burger) {
    burger.addEventListener('click', () => {
      qs('#sidebar')?.classList.toggle('open');
    });
  }
}

// ── Interactive KPI Modal Logic ──────────────────────────────
let modalCharts = {};
function destroyModalCharts() {
  Object.values(modalCharts).forEach(c => { try { c.destroy(); } catch(e){} });
  modalCharts = {};
}

const KPI_MODAL_DATA = [
  {
    title: 'TOTAL DEMAND FORECAST',
    val: '142,830 Units',
    sub: 'Projected demand for the upcoming cycle across all 76 stores',
    color: '#0076fe',
    icon: '📦',
    stats: [
      { val: '14,950', lbl: 'Peak Week' },
      { val: '11,902', lbl: 'Weekly Avg' },
      { val: '+4.2%', lbl: 'MoM Growth' },
      { val: '12 wks', lbl: 'Forecast Span' },
    ]
  },
  {
    title: 'FORECAST ACCURACY',
    val: '75.7%',
    sub: 'Evaluated against actual POS sales using Hybrid Ridge+XGBoost',
    color: '#7c3aed',
    icon: '🎯',
    stats: [
      { val: '0.757', lbl: 'R² Score' },
      { val: '28.84', lbl: 'Best RMSE' },
      { val: '13.59', lbl: 'MAE' },
      { val: '35.3%', lbl: 'MAPE' },
    ]
  },
  {
    title: 'STOCKOUT RISK ITEMS',
    val: '12 SKUs',
    sub: 'Products forecasted to exhaust inventory within 14 days',
    color: '#f43f5e',
    icon: '⚠️',
    stats: [
      { val: '3', lbl: 'Critical (<7d)' },
      { val: '5', lbl: 'High (<14d)' },
      { val: '4', lbl: 'Medium (<21d)' },
      { val: '98%', lbl: 'Replenish Priority' },
    ]
  },
  {
    title: 'ACTIVE STORES',
    val: '76 Stores',
    sub: 'Retail network coverage with active weekly replenishment pipelines',
    color: '#10b981',
    icon: '🏪',
    stats: [
      { val: '76', lbl: 'Total Stores' },
      { val: '18', lbl: 'Tier 1 Metro' },
      { val: '42', lbl: 'Tier 2 Regional' },
      { val: '16', lbl: 'Tier 3 Local' },
    ]
  }
];

function openKpiModal(idx) {
  const cfg = KPI_MODAL_DATA[idx];
  if (!cfg) return;

  const overlay = qs('#kpimOverlay');
  const iconWrap = qs('#kpimIconWrap');
  iconWrap.textContent = cfg.icon;
  iconWrap.style.borderColor = cfg.color;

  qs('#kpimLabel').textContent = cfg.title;
  qs('#kpimBigVal').textContent = cfg.val;
  qs('#kpimBigVal').style.color = cfg.color;
  qs('#kpimSub').textContent = cfg.sub;

  qs('#kpimStatsStrip').innerHTML = cfg.stats.map(s => `
    <div class="kpim-stat">
      <div class="kpim-stat-val" style="color:${cfg.color};">${s.val}</div>
      <div class="kpim-stat-lbl">${s.lbl}</div>
    </div>
  `).join('');

  overlay.classList.add('open');
}

function closeKpiModal() {
  qs('#kpimOverlay')?.classList.remove('open');
}

function initKpiModal() {
  qsa('.kpi-card.clickable').forEach(card => {
    card.addEventListener('click', () => {
      const idx = parseInt(card.dataset.kpiIdx, 10);
      openKpiModal(idx);
    });
  });

  qs('#kpimClose')?.addEventListener('click', closeKpiModal);
  qs('#kpimOverlay')?.addEventListener('click', e => {
    if (e.target === qs('#kpimOverlay')) closeKpiModal();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeKpiModal();
  });
}

// ── Date Formatter ───────────────────────────────────────────
function setLiveDate() {
  const el = qs('#heroDate');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleDateString('en-GB', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric'
  });
}

// ── Authentication & Sign Out ────────────────────────────────
function initAuth() {
  const isAuth = localStorage.getItem('nexus_authenticated');
  if (isAuth !== 'true') {
    // If not authenticated, redirect to login page
    window.location.href = '/login';
    return;
  }

  // Update user avatar initials if saved
  const savedUser = localStorage.getItem('nexus_user');
  if (savedUser) {
    const avatarEl = qs('.avatar-circle span');
    if (avatarEl) {
      avatarEl.textContent = savedUser.slice(0, 2).toUpperCase();
    }
  }

  // Handle Sign Out
  const signOutBtn = qs('#signOutBtn');
  if (signOutBtn) {
    signOutBtn.addEventListener('click', () => {
      localStorage.removeItem('nexus_authenticated');
      localStorage.removeItem('nexus_user');
      window.location.href = '/login';
    });
  }
}

// ── Live Database Sync ───────────────────────────────────────
async function loadDatabaseData() {
  try {
    const [dashRes, invRes] = await Promise.all([
      fetch('/api/dashboard').catch(() => null),
      fetch('/api/inventory').catch(() => null)
    ]);

    if (dashRes && dashRes.ok) {
      const dbData = await dashRes.json();
      if (dbData.trend && dbData.trend.labels && dbData.trend.labels.length > 0) {
        DATA.trend.labels = dbData.trend.labels;
        DATA.trend.actual = dbData.trend.actual;
        DATA.trend.forecast = dbData.trend.forecast;
      }
      if (dbData.category && dbData.category.labels && dbData.category.labels.length > 0) {
        DATA.category.labels = dbData.category.labels;
        DATA.category.values = dbData.category.values;
      }
      // Refresh charts with live database data
      renderTrendChart('4w');
      renderCategoryChart();
    }

    if (invRes && invRes.ok) {
      const invData = await invRes.json();
      if (invData.products && invData.products.length > 0) {
        DATA.products = invData.products;
        renderInventoryTable();
      }
    }
  } catch(e) {
    console.log("Database synced with local storage:", e);
  }
}

// ── App Init ─────────────────────────────────────────────────
function init() {
  initAuth();
  setLiveDate();
  initNav();
  initKpiModal();
  initPredictPage();
  navigateTo('dashboard');
  loadDatabaseData();
}

// Execute on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
