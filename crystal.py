"""
crystal.py — Crystal Hub v14.4 convertido a FastAPI + HTMX
============================================================
Ruta: http://localhost:7860/crystal

Características:
  - Day Trading & Scalping con señales en vivo
  - Backtesting estratégico (6 meses)
  - Noticias del ticker
  - Diseño oscuro consistente con el resto del proyecto
"""

import os, json, datetime as dt, threading
import pathlib as _pl

import pandas as pd
import numpy as np
import yfinance as yf

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse

# ── Cargar .env ──────────────────────────────────────────────────────────────
for _ep in [_pl.Path(__file__).parent / ".env", _pl.Path(".env")]:
    if _ep.exists():
        for _line in _ep.read_text(encoding="utf-8-sig").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
        break

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE TRADING (igual que Crystal Hub original)
# ─────────────────────────────────────────────────────────────────────────────

def calc_trading_logic(ticker: str) -> dict:
    data = yf.download(ticker, period="3d", interval="5m", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        raise ValueError(f"Sin datos para {ticker}")

    p_act  = float(data["Close"].iloc[-1])
    high_d = float(data["High"].max())
    low_d  = float(data["Low"].min())
    atr    = (high_d - low_d) * 0.12
    mid    = (high_d + low_d) / 2
    ts     = dt.datetime.now().strftime("%d/%m/%Y | %H:%M:%S")

    bias           = "CALL" if p_act > mid else "PUT"
    trend_strength = abs(p_act - mid) / (high_d - low_d) if (high_d - low_d) > 0 else 0
    prob_scalp     = min(88, 60 + (trend_strength * 100))
    prob_day       = min(82, 55 + (trend_strength * 80))

    # Scalping
    s_e  = p_act
    s_sl = s_e - atr * 0.7 if bias == "CALL" else s_e + atr * 0.7
    s_tp = s_e + atr * 1.1 if bias == "CALL" else s_e - atr * 1.1

    # Day trading
    d_e  = p_act + atr * 0.1 if bias == "CALL" else p_act - atr * 0.1
    d_sl = d_e - atr * 1.8  if bias == "CALL" else d_e + atr * 1.8
    d_tp = d_e + atr * 4.0  if bias == "CALL" else d_e - atr * 4.0

    return {
        "ticker": ticker, "ts": ts, "bias": bias,
        "price": round(p_act, 2), "high": round(high_d, 2), "low": round(low_d, 2),
        "atr": round(atr, 2),
        "scalp": {
            "entry": round(s_e, 2), "sl": round(s_sl, 2),
            "tp": round(s_tp, 2), "prob": round(prob_scalp, 1),
            "rr": round((s_tp - s_e) / (s_e - s_sl), 2) if abs(s_e - s_sl) > 0 else 0,
        },
        "day": {
            "entry": round(d_e, 2), "sl": round(d_sl, 2),
            "tp": round(d_tp, 2), "prob": round(prob_day, 1),
            "rr": round((d_tp - d_e) / (d_e - d_sl), 2) if abs(d_e - d_sl) > 0 else 0,
        },
    }


def calc_backtest(ticker: str) -> dict:
    hist = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    if hist.empty:
        raise ValueError(f"Sin datos para {ticker}")

    wins, total = 0, 0
    results = []
    for i in range(1, len(hist)):
        o = float(hist["Open"].iloc[i])
        h = float(hist["High"].iloc[i])
        l = float(hist["Low"].iloc[i])
        c = float(hist["Close"].iloc[i])
        c_prev = float(hist["Close"].iloc[i - 1])
        total += 1
        rng = h - l
        won = False
        if c_prev < c and h > (o + rng * 0.2):
            won = True
        elif c_prev > c and l < (o - rng * 0.2):
            won = True
        if won:
            wins += 1
        results.append({
            "date": str(hist.index[i])[:10],
            "result": "WIN" if won else "LOSS",
            "change": round((c - c_prev) / c_prev * 100, 2),
        })

    pct = round(wins / total * 100, 1) if total > 0 else 0
    ts  = dt.datetime.now().strftime("%d/%m/%Y | %H:%M:%S")

    # Racha máxima ganadora
    max_streak = cur_streak = 0
    for r in results:
        if r["result"] == "WIN":
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    return {
        "ticker": ticker, "ts": ts,
        "wins": wins, "losses": total - wins, "total": total,
        "pct": pct, "max_streak": max_streak,
        "recent": results[-20:],
    }


def fetch_news(ticker: str) -> list:
    import urllib.request, re
    try:
        url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
               f"?s={ticker}&region=US&lang=en-US")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            xml = r.read().decode("utf-8", "ignore")
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml)
        dates  = re.findall(r"<pubDate>(.*?)</pubDate>", xml)
        return [{"title": t.strip(), "date": dates[i][:22] if i < len(dates) else ""}
                for i, t in enumerate(titles[1:8])]
    except Exception as e:
        return [{"title": f"Error: {e}", "date": ""}]


# ─────────────────────────────────────────────────────────────────────────────
# HTML BASE
# ─────────────────────────────────────────────────────────────────────────────

CRYSTAL_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crystal Hub v14.4</title>
<script src="https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0f;color:#e8ecf2;font-family:-apple-system,BlinkMacSystemFont,'Inter',monospace;min-height:100vh;display:flex;flex-direction:column}

/* TOPBAR */
.topbar{background:#111118;border-bottom:1px solid #1e1e2e;padding:12px 24px;
        display:flex;align-items:center;gap:14px;flex-shrink:0;position:sticky;top:0;z-index:100}
.logo-text{font-size:22px;font-weight:800;letter-spacing:.1em;
           background:linear-gradient(90deg,#00d4ff,#00ff88);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo-sub{font-size:10px;color:#444;letter-spacing:.15em;text-transform:uppercase;margin-top:2px}
.nav-tabs{display:flex;gap:4px;margin-left:28px}
.nav-tab{padding:7px 18px;border-radius:7px;font-size:12px;font-weight:600;
         cursor:pointer;border:1px solid transparent;transition:all .2s;text-decoration:none;
         display:inline-flex;align-items:center;gap:6px;letter-spacing:.04em}
.nav-tab:hover{background:#1a1a2e;border-color:#2a2a3e;color:#fff}
.nav-tab.active{background:#1a1a2e;border-color:#00d4ff44;color:#00d4ff}
.back-link{margin-left:auto;font-size:11px;color:#444;text-decoration:none;
           padding:6px 12px;border-radius:6px;border:1px solid #1e1e2e;transition:all .2s}
.back-link:hover{color:#fff;border-color:#2a2a3e}

/* CONTENT */
.content{flex:1;padding:24px;max-width:1400px;margin:0 auto;width:100%}

/* SEARCH BAR */
.search-bar{background:#111118;border:1px solid #1e1e2e;border-radius:12px;
            padding:16px 20px;display:flex;align-items:center;gap:12px;
            margin-bottom:24px;flex-wrap:wrap}
.search-bar input{background:#0a0a0f;border:1px solid #2a2a3e;color:#fff;
                  border-radius:8px;padding:9px 14px;font-size:14px;
                  font-family:monospace;text-transform:uppercase;outline:none;
                  width:120px;transition:border-color .2s;letter-spacing:.08em}
.search-bar input:focus{border-color:#00d4ff}
.btn{padding:9px 20px;border-radius:8px;font-size:12px;font-weight:700;
     border:none;cursor:pointer;transition:all .15s;letter-spacing:.06em}
.btn-cyan{background:#00d4ff;color:#000}
.btn-cyan:hover{background:#00bfe6}
.btn-gold{background:#ffd700;color:#000}
.btn-gold:hover{background:#e6c200}
.btn-refresh{background:#1f538d;color:#fff;width:38px;padding:9px 0;text-align:center}
.btn-refresh:hover{background:#2563a8}
.btn:disabled{opacity:.5;cursor:not-allowed}

/* RESULTS */
#results{min-height:200px}
.loading-msg{display:flex;align-items:center;justify-content:center;
             height:200px;color:#444;gap:10px;font-size:13px}
.spinner{width:18px;height:18px;border:2px solid #1e1e2e;
         border-top-color:#00d4ff;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ANALYSIS GRID */
.analysis-header{background:#111118;border:1px solid #1e1e2e;border-radius:10px;
                 padding:14px 20px;margin-bottom:18px;display:flex;align-items:center;gap:10px}
.analysis-ts{font-size:12px;color:#ffd700;font-weight:700;letter-spacing:.06em}
.price-tag{font-family:monospace;font-size:20px;font-weight:800;margin-left:auto}

.cards-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
@media(max-width:800px){.cards-grid{grid-template-columns:1fr}}

.card{background:#0d0d14;border-radius:12px;padding:24px;border:1px solid}
.card-cyan{border-color:#00d4ff33}
.card-gold{border-color:#ffd70033}
.card-title{font-size:16px;font-weight:800;letter-spacing:.08em;margin-bottom:20px;
            display:flex;align-items:center;gap:8px}
.card-title.cyan{color:#00d4ff}
.card-title.gold{color:#ffd700}

.stat-row{display:flex;justify-content:space-between;align-items:center;
          padding:10px 0;border-bottom:1px solid #1a1a2a}
.stat-row:last-of-type{border:none}
.stat-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.08em}
.stat-val{font-family:monospace;font-size:16px;font-weight:700}
.val-white{color:#fff}
.val-green{color:#00ff88}
.val-red{color:#ff4d4d}
.val-cyan{color:#00d4ff}
.val-gold{color:#ffd700}

.direction-badge{display:inline-flex;align-items:center;gap:6px;
                 padding:6px 16px;border-radius:8px;font-size:18px;font-weight:800;
                 letter-spacing:.1em}
.badge-call{background:rgba(0,255,136,.1);color:#00ff88;border:1px solid rgba(0,255,136,.3)}
.badge-put{background:rgba(255,77,77,.1);color:#ff4d4d;border:1px solid rgba(255,77,77,.3)}

/* PROB BAR */
.prob-wrap{background:#111118;border-radius:10px;padding:14px;margin-top:16px}
.prob-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.prob-value{font-size:22px;font-weight:800;margin-bottom:10px}
.prob-bar-bg{height:8px;background:#1a1a2a;border-radius:4px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:4px;transition:width .5s}
.rr-tag{display:inline-flex;align-items:center;gap:4px;margin-top:10px;
        padding:4px 10px;border-radius:5px;font-size:11px;font-weight:700;
        background:#1a1a2a;color:#888}
.rr-tag span{color:#ffd700}

/* BACKTESTING */
.bt-container{background:#0d0d14;border:2px solid #ffd70033;border-radius:14px;padding:28px}
.bt-title{font-size:20px;font-weight:800;color:#00d4ff;letter-spacing:.08em;margin-bottom:6px}
.bt-ts{font-size:11px;color:#555;margin-bottom:28px}
.bt-pct{font-size:56px;font-weight:900;line-height:1;margin:16px 0}
.bt-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}
.bt-stat{background:#111118;border-radius:8px;padding:14px;text-align:center}
.bt-stat-label{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.bt-stat-val{font-size:20px;font-weight:800;font-family:monospace}
.recent-title{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.recent-grid{display:flex;flex-direction:column;gap:4px;max-height:300px;overflow-y:auto}
.recent-row{display:flex;align-items:center;gap:10px;padding:6px 10px;
            border-radius:5px;font-size:12px}
.recent-win{background:rgba(0,255,136,.05);border-left:2px solid #00ff88}
.recent-loss{background:rgba(255,77,77,.05);border-left:2px solid #ff4d4d}
.recent-date{color:#555;font-family:monospace;min-width:90px}
.recent-tag{font-weight:700;min-width:36px}
.recent-chg{font-family:monospace;margin-left:auto}

/* NEWS */
.news-container{display:flex;flex-direction:column;gap:12px}
.news-item{background:#0d0d14;border:1px solid #1e1e2e;border-radius:10px;padding:16px 20px}
.news-title{font-size:13px;color:#d1d5db;line-height:1.6;margin-bottom:6px}
.news-date{font-size:11px;color:#444}
.news-empty{text-align:center;padding:40px;color:#444;font-size:13px}

/* HTMX indicator */
.htmx-indicator{display:none}
.htmx-request .htmx-indicator{display:flex}
.htmx-request .hide-on-load{display:none}
</style>
</head>
<body>

<div class="topbar">
  <div>
    <div class="logo-text">CRYSTAL HUB</div>
    <div class="logo-sub">v14.4 · Day Trading & Backtesting</div>
  </div>
  <nav class="nav-tabs">
    <a class="nav-tab active" href="/crystal">🎯 DAY TRADING</a>
    <a class="nav-tab" href="/crystal/backtest-page">📊 BACKTESTING</a>
    <a class="nav-tab" href="/crystal/news-page">📰 NOTICIAS</a>
  </nav>
  <a class="back-link" href="/">← ORB Chat</a>
</div>

<div class="content">

  <!-- SEARCH BAR -->
  <div class="search-bar">
    <span style="font-size:12px;color:#555;letter-spacing:.1em;text-transform:uppercase">Ticker</span>
    <input type="text" id="ticker-input" placeholder="SPY" maxlength="8"
           value="" autocomplete="off"
           onkeydown="if(event.key==='Enter') analyze()">
    <button class="btn btn-cyan" onclick="analyze()">ANALIZAR</button>
    <button class="btn btn-refresh" onclick="analyze()">🔄</button>
    <button class="btn btn-gold" onclick="backtest()">BACKTESTING</button>
    <span style="margin-left:auto;font-size:11px;color:#333">Crystal Hub Day Trading & Scalping</span>
  </div>

  <!-- RESULTS -->
  <div id="results">
    <div class="loading-msg" style="color:#222">
      Ingresa un ticker y presiona ANALIZAR
    </div>
  </div>

</div>

<script>
function getTicker(){
  return document.getElementById('ticker-input').value.trim().toUpperCase();
}

function analyze(){
  const t = getTicker();
  if(!t) return;
  document.getElementById('results').innerHTML = `
    <div class="loading-msg">
      <div class="spinner"></div> Analizando ${t}...
    </div>`;
  fetch('/crystal/analyze', {
    method:'POST',
    headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:'ticker='+encodeURIComponent(t)
  }).then(r=>r.text()).then(html=>{
    document.getElementById('results').innerHTML = html;
  }).catch(e=>{
    document.getElementById('results').innerHTML =
      `<div class="loading-msg" style="color:#ff4d4d">Error: ${e}</div>`;
  });
}

function backtest(){
  const t = getTicker();
  if(!t) return;
  document.getElementById('results').innerHTML = `
    <div class="loading-msg">
      <div class="spinner"></div> Ejecutando backtesting de ${t} (6 meses)...
    </div>`;
  fetch('/crystal/backtest', {
    method:'POST',
    headers:{'Content-Type':'application/x-www-form-urlencoded'},
    body:'ticker='+encodeURIComponent(t)
  }).then(r=>r.text()).then(html=>{
    document.getElementById('results').innerHTML = html;
  }).catch(e=>{
    document.getElementById('results').innerHTML =
      `<div class="loading-msg" style="color:#ff4d4d">Error: ${e}</div>`;
  });
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def render_analysis(d: dict) -> str:
    bias = d["bias"]
    badge_cls = "badge-call" if bias == "CALL" else "badge-put"
    arrow = "▲" if bias == "CALL" else "▼"
    bias_color = "#00ff88" if bias == "CALL" else "#ff4d4d"
    sc = d["scalp"]
    dy = d["day"]

    def stat(label, val, color="val-white"):
        return f"""<div class="stat-row">
          <span class="stat-label">{label}</span>
          <span class="stat-val {color}">{val}</span>
        </div>"""

    def prob_bar(label, pct, color_hex, bar_color):
        return f"""<div class="prob-wrap">
          <div class="prob-label">{label}</div>
          <div class="prob-value" style="color:{color_hex}">{pct}%</div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%;background:{color_hex}"></div>
          </div>
        </div>"""

    return f"""
    <div class="analysis-header">
      <span class="analysis-ts">⚡ ANÁLISIS DE {d["ticker"]} — {d["ts"]}</span>
      <span class="price-tag" style="color:{bias_color}">${d["price"]}</span>
      <span style="font-size:11px;color:#444;margin-left:8px">ATR ${d["atr"]}</span>
    </div>

    <div class="cards-grid">

      <!-- SCALPING -->
      <div class="card card-cyan">
        <div class="card-title cyan">⚡ SCALPING (1M–5M)</div>
        <div style="margin-bottom:16px">
          <div class="stat-label">DIRECCIÓN</div>
          <div style="margin-top:8px">
            <span class="direction-badge {badge_cls}">{arrow} {bias}</span>
          </div>
        </div>
        {stat("ENTRADA", f"${sc['entry']}", "val-white")}
        {stat("STOP LOSS", f"${sc['sl']}", "val-red")}
        {stat("TAKE PROFIT", f"${sc['tp']}", "val-green")}
        <div class="rr-tag">R:R <span>{sc['rr']}x</span></div>
        {prob_bar("PROBABILIDAD DE ÉXITO", sc['prob'], "#00d4ff", "#00d4ff")}
      </div>

      <!-- DAY TRADING -->
      <div class="card card-gold">
        <div class="card-title gold">📈 DAY TRADING (15M–1H)</div>
        <div style="margin-bottom:16px">
          <div class="stat-label">DIRECCIÓN</div>
          <div style="margin-top:8px">
            <span class="direction-badge {badge_cls}">{arrow} {bias}</span>
          </div>
        </div>
        {stat("ENTRADA", f"${dy['entry']}", "val-white")}
        {stat("STOP LOSS", f"${dy['sl']}", "val-red")}
        {stat("TAKE PROFIT", f"${dy['tp']}", "val-green")}
        <div class="rr-tag">R:R <span>{dy['rr']}x</span></div>
        {prob_bar("PROBABILIDAD DE TENDENCIA", dy['prob'], "#ffd700", "#ffd700")}
      </div>

    </div>"""


def render_backtest(d: dict) -> str:
    pct = d["pct"]
    pct_color = "#00ff88" if pct >= 60 else "#ffd700" if pct >= 45 else "#ff4d4d"

    recent_rows = ""
    for r in reversed(d["recent"]):
        cls  = "recent-win" if r["result"] == "WIN" else "recent-loss"
        tag_color = "#00ff88" if r["result"] == "WIN" else "#ff4d4d"
        chg_color = "#00ff88" if r["change"] >= 0 else "#ff4d4d"
        chg_sign  = "+" if r["change"] >= 0 else ""
        recent_rows += f"""<div class="recent-row {cls}">
          <span class="recent-date">{r["date"]}</span>
          <span class="recent-tag" style="color:{tag_color}">{r["result"]}</span>
          <span class="recent-chg" style="color:{chg_color}">{chg_sign}{r["change"]}%</span>
        </div>"""

    return f"""
    <div class="bt-container">
      <div class="bt-title">📊 BACKTESTING ESTRATÉGICO: {d["ticker"]}</div>
      <div class="bt-ts">Reporte generado: {d["ts"]}</div>

      <div style="margin-bottom:8px;font-size:12px;color:#555;text-transform:uppercase;letter-spacing:.08em">
        Precisión estimada
      </div>
      <div class="bt-pct" style="color:{pct_color}">{pct}%</div>

      <div class="bt-stats">
        <div class="bt-stat">
          <div class="bt-stat-label">Total días</div>
          <div class="bt-stat-val" style="color:#fff">{d["total"]}</div>
        </div>
        <div class="bt-stat">
          <div class="bt-stat-label">Ganados</div>
          <div class="bt-stat-val" style="color:#00ff88">{d["wins"]}</div>
        </div>
        <div class="bt-stat">
          <div class="bt-stat-label">Perdidos</div>
          <div class="bt-stat-val" style="color:#ff4d4d">{d["losses"]}</div>
        </div>
        <div class="bt-stat">
          <div class="bt-stat-label">Racha máx.</div>
          <div class="bt-stat-val" style="color:#ffd700">{d["max_streak"]}</div>
        </div>
      </div>

      <div class="recent-title">Últimos 20 días analizados</div>
      <div class="recent-grid">{recent_rows}</div>

      <button onclick="document.getElementById('results').innerHTML=''"
              style="margin-top:20px;background:#1a1a2a;color:#888;border:1px solid #2a2a3e;
                     border-radius:8px;padding:10px 20px;cursor:pointer;font-size:12px">
        ← Volver al análisis
      </button>
    </div>"""


def render_news(news: list, ticker: str) -> str:
    if not news:
        return f'<div class="news-empty">Sin noticias recientes para {ticker}</div>'
    html = f'<div style="font-size:12px;color:#555;margin-bottom:14px;text-transform:uppercase;letter-spacing:.1em">Noticias recientes — {ticker}</div>'
    html += '<div class="news-container">'
    for n in news:
        html += f"""<div class="news-item">
          <div class="news-title">{n["title"]}</div>
          <div class="news-date">{n["date"]}</div>
        </div>"""
    html += "</div>"
    return html


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/crystal", response_class=HTMLResponse)
async def crystal_home():
    return HTMLResponse(CRYSTAL_HTML)


@router.post("/crystal/analyze", response_class=HTMLResponse)
async def crystal_analyze(ticker: str = Form(...)):
    ticker = ticker.upper().strip()
    try:
        d = calc_trading_logic(ticker)
        return HTMLResponse(render_analysis(d))
    except Exception as e:
        return HTMLResponse(
            f'<div class="loading-msg" style="color:#ff4d4d">❌ Error: {e}</div>'
        )


@router.post("/crystal/backtest", response_class=HTMLResponse)
async def crystal_backtest(ticker: str = Form(...)):
    ticker = ticker.upper().strip()
    try:
        d = calc_backtest(ticker)
        return HTMLResponse(render_backtest(d))
    except Exception as e:
        return HTMLResponse(
            f'<div class="loading-msg" style="color:#ff4d4d">❌ Error: {e}</div>'
        )


@router.get("/crystal/news-page", response_class=HTMLResponse)
async def crystal_news_page(ticker: str = "SPY"):
    html = CRYSTAL_HTML.replace(
        'class="nav-tab active" href="/crystal"',
        'class="nav-tab" href="/crystal"'
    ).replace(
        'class="nav-tab" href="/crystal/news-page"',
        'class="nav-tab active" href="/crystal/news-page"'
    )
    news = fetch_news(ticker)
    news_html = render_news(news, ticker)
    # Inject news into results
    html = html.replace(
        '<div id="results">\n    <div class="loading-msg" style="color:#222">\n      Ingresa un ticker y presiona ANALIZAR\n    </div>\n  </div>',
        f'<div id="results">{news_html}</div>'
    )
    # Change button to fetch news
    html = html.replace(
        'onclick="analyze()">ANALIZAR',
        'onclick="loadNews()">VER NOTICIAS'
    )
    html = html.replace(
        '</script>',
        """
function loadNews(){
  const t = getTicker();
  if(!t) return;
  fetch('/crystal/news?ticker='+encodeURIComponent(t))
    .then(r=>r.text()).then(html=>{
      document.getElementById('results').innerHTML = html;
    });
}
</script>"""
    )
    return HTMLResponse(html)


@router.get("/crystal/news", response_class=HTMLResponse)
async def crystal_news(ticker: str = "SPY"):
    ticker = ticker.upper().strip()
    news = fetch_news(ticker)
    return HTMLResponse(render_news(news, ticker))


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()
    app.include_router(router)

    port = int(os.getenv("PORT", 7862))
    print(f"\n{'='*50}")
    print(f"  Crystal Hub v14.4")
    print(f"  http://localhost:{port}/crystal")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")