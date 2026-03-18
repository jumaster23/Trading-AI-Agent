"""
live.py — Panel de Trading en Vivo
====================================
Ruta: http://localhost:7860/live

Características:
  - Gráfico de velas 1m en tiempo real (lightweight-charts via CDN)
  - Indicador CALL/PUT generado al minuto 1 de apertura y cada 5 min
  - Niveles S/R de la última semana (precio más respetado)
  - Noticias y earnings del ticker
  - Volumen en vivo con comparación vs promedio
  - Gamma Exposure estimado por strikes

Uso:
  Integrar en server.py: from live import router; app.include_router(router)
  O correr solo: python live.py
"""

import os, json, datetime as dt, re, urllib.request
import pathlib as _pl
from typing import Optional

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

# ── Intentar importar el agente para señales LLM ────────────────────────────
try:
    from agent import crear_agente, fetch_daily, fetch_5m, get_market_context, get_today_gap
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

router = APIRouter()

MARKET_OPEN  = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)

# ─────────────────────────────────────────────────────────────────────────────
# CAPA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("America/New_York")
    else:
        raw.index = raw.index.tz_convert("America/New_York")
    return raw


def fetch_1m(ticker: str, days: int = 1) -> pd.DataFrame:
    """Datos 1m del día actual."""
    end   = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=max(days, 1))
    try:
        raw = yf.download(ticker,
                          start=start.strftime("%Y-%m-%d"),
                          end=(end + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                          interval="1m", auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        raw = _clean(raw)
        mask = (raw.index.time >= MARKET_OPEN) & (raw.index.time <= MARKET_CLOSE)
        return raw[mask].copy()
    except:
        return pd.DataFrame()


def fetch_daily_week(ticker: str) -> pd.DataFrame:
    """Últimos 10 días de datos diarios."""
    try:
        raw = yf.download(ticker, period="10d", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        raw = _clean(raw)
        raw.index = pd.to_datetime([str(d)[:10] for d in raw.index])
        return raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    except:
        return pd.DataFrame()


def get_sr_levels(daily: pd.DataFrame, current_price: float) -> list:
    """
    Calcula niveles S/R de la última semana:
    - Máximos y mínimos diarios
    - Niveles más tocados (precio que más veces fue respetado)
    - Números redondos cercanos
    """
    if daily.empty or len(daily) < 2:
        return []

    levels = []
    w = daily.tail(7)

    # Máximos y mínimos de cada día
    for i, (date, row) in enumerate(w.iterrows()):
        recency = (i + 1) / len(w)  # más reciente = mayor peso
        levels.append({
            "price": round(float(row["High"]), 2),
            "type": "R",
            "label": f"High {str(date)[:10]}",
            "recency": recency,
            "touches": 1,
        })
        levels.append({
            "price": round(float(row["Low"]), 2),
            "type": "S",
            "label": f"Low {str(date)[:10]}",
            "recency": recency,
            "touches": 1,
        })

    # Prev close (nivel más importante)
    if len(daily) >= 2:
        pc = round(float(daily["Close"].iloc[-2]), 2)
        levels.append({"price": pc, "type": "SR", "label": "Prev Close", "recency": 1.0, "touches": 3})

    # Números redondos dentro de ±3%
    step = 5 if current_price >= 100 else 1
    lo = int(current_price * 0.97 / step) * step
    hi = int(current_price * 1.03 / step + 1) * step
    for p in range(lo, hi + step, step):
        if abs(p - current_price) / current_price > 0.03:
            continue
        # Contar cuántas veces las velas tocaron este nivel
        zone = p * 0.002
        touches = sum(
            1 for _, row in w.iterrows()
            if float(row["High"]) >= p - zone and float(row["Low"]) <= p + zone
        )
        if touches >= 2:
            levels.append({
                "price": float(p),
                "type": "SR",
                "label": f"${p} (redondo)",
                "recency": 0.8,
                "touches": touches,
            })

    # Merge niveles muy cercanos (< 0.15%)
    levels.sort(key=lambda x: x["price"])
    merged = []
    for lv in levels:
        if (merged and
                abs(lv["price"] - merged[-1]["price"]) / merged[-1]["price"] < 0.0015):
            if lv["touches"] > merged[-1]["touches"]:
                merged[-1] = lv
            else:
                merged[-1]["touches"] = max(merged[-1]["touches"], lv["touches"])
        else:
            merged.append(lv)

    # Solo niveles dentro de ±3% del precio actual
    nearby = [lv for lv in merged
              if abs(lv["price"] - current_price) / current_price <= 0.03]
    nearby.sort(key=lambda x: (-x["touches"], abs(x["price"] - current_price)))
    return nearby[:12]


def get_volume_context(df1m: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """
    Análisis de volumen en vivo:
    - Volumen acumulado hoy vs promedio histórico
    - Pace (ritmo) — si sigue este ritmo, ¿cuánto habrá al cierre?
    - Barras de alto volumen (posibles institucionales)
    """
    if df1m.empty:
        return {}

    today_vol = int(df1m["Volume"].astype(float).sum())

    # Promedio diario histórico
    avg_daily_vol = 0
    if not daily.empty and "Volume" in daily.columns:
        avg_daily_vol = int(daily["Volume"].astype(float).tail(20).mean())

    # Minutos transcurridos desde apertura
    now = df1m.index[-1]
    try:
        import pytz
        ny = pytz.timezone("America/New_York")
        open_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
        mins_elapsed = max(1, int((now - open_dt).total_seconds() / 60))
    except:
        mins_elapsed = len(df1m)

    total_mins = 390  # minutos en una sesión completa
    pace_full_day = int(today_vol / mins_elapsed * total_mins) if mins_elapsed > 0 else 0
    vol_pct_avg   = round(today_vol / avg_daily_vol * 100, 1) if avg_daily_vol > 0 else 0

    # Barras de alto volumen (>2x promedio del día)
    avg_bar_vol = float(df1m["Volume"].astype(float).mean()) + 1e-9
    high_vol_bars = []
    for ts, row in df1m.tail(30).iterrows():
        v = float(row.get("Volume", 0))
        if v >= avg_bar_vol * 2.0:
            direction = "buy" if float(row["Close"]) >= float(row["Open"]) else "sell"
            high_vol_bars.append({
                "time": str(ts)[-8:-3],
                "volume": int(v),
                "ratio": round(v / avg_bar_vol, 1),
                "direction": direction,
                "price": round(float(row["Close"]), 2),
            })

    return {
        "today_vol":    today_vol,
        "avg_daily":    avg_daily_vol,
        "vol_pct_avg":  vol_pct_avg,
        "pace_eod":     pace_full_day,
        "mins_elapsed": mins_elapsed,
        "high_vol_bars": high_vol_bars[-5:],
        "above_avg":    vol_pct_avg > 110,
    }


def get_gamma_exposure(ticker: str, current_price: float, atr: float) -> list:
    """
    Estimación de Gamma Exposure por strikes.
    Sin API de opciones real, estimamos OI por:
    - Strikes redondos ± 2% (donde se concentran contratos)
    - Peso mayor a ATM (at-the-money)
    - Patrones de precio histórico (donde el precio ha rebotado más)
    """
    step = 5 if current_price >= 100 else 1
    lo = int((current_price - atr * 2) / step) * step
    hi = int((current_price + atr * 2) / step + 1) * step

    strikes = []
    for s in range(lo, hi + step, step):
        dist = abs(s - current_price)
        dist_pct = dist / current_price

        # OI estimado: mayor concentración ATM, decae con distancia
        # Aproximación gaussiana centrada en precio actual
        oi_weight = round(100 * np.exp(-0.5 * (dist / (atr * 0.8)) ** 2), 1)

        if oi_weight < 5:
            continue

        # Gamma: más alto ATM, decae en OTM
        gamma = round(oi_weight * (1 - dist_pct * 2), 1)
        gamma = max(0, gamma)

        # Tipo: call wall (resistencia) o put wall (soporte)
        if s > current_price:
            gtype = "call_wall"
            color = "#f97316"
        elif s < current_price:
            gtype = "put_wall"
            color = "#3b82f6"
        else:
            gtype = "atm"
            color = "#8b5cf6"

        strikes.append({
            "strike":   float(s),
            "oi_est":   oi_weight,
            "gamma":    gamma,
            "type":     gtype,
            "color":    color,
            "label":    f"${s}",
        })

    # Ordenar por gamma descendente
    strikes.sort(key=lambda x: x["gamma"], reverse=True)
    return strikes[:10]


def fetch_news_ticker(ticker: str) -> list:
    """Headlines de Yahoo Finance RSS."""
    try:
        url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
               f"?s={ticker}&region=US&lang=en-US")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            xml = r.read().decode("utf-8", "ignore")
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml)
        dates  = re.findall(r"<pubDate>(.*?)</pubDate>", xml)
        return [{"title": t.strip(), "date": dates[i][:16] if i < len(dates) else ""}
                for i, t in enumerate(titles[1:6])]
    except:
        return []


def build_signal(ticker: str, df1m: pd.DataFrame, daily: pd.DataFrame,
                 sr_levels: list, vol_ctx: dict, gamma: list) -> dict:
    """
    Genera señal CALL/PUT/WAIT basada en:
    1. Posición del precio vs S/R clave
    2. Momentum de velas (últimas 5 velas 1m)
    3. Volumen confirmando dirección
    4. Gamma exposure (donde están los muros)
    5. Tendencia del día anterior

    Solo genera señal después del primer minuto de apertura.
    """
    if df1m.empty or len(df1m) < 2:
        return {"signal": "WAIT", "confidence": 0, "reason": "Esperando datos..."}

    last_close = float(df1m["Close"].iloc[-1])
    last_open  = float(df1m["Open"].iloc[-1])
    prev_close = float(df1m["Close"].iloc[-2]) if len(df1m) >= 2 else last_close

    # ── 1. Momentum de velas (últimas 5 barras 1m) ───────────────────────────
    w5 = df1m.tail(5)
    bull_bars  = sum(1 for _, r in w5.iterrows() if float(r["Close"]) > float(r["Open"]))
    bear_bars  = 5 - bull_bars
    last_move  = round((last_close - prev_close) / prev_close * 100, 3)

    # ── 2. Volumen confirmando dirección ─────────────────────────────────────
    vol_confirm = vol_ctx.get("above_avg", False)
    last_bar_up = last_close >= last_open

    # ── 3. Posición vs S/R ───────────────────────────────────────────────────
    resistances = [lv for lv in sr_levels if lv["price"] > last_close]
    supports    = [lv for lv in sr_levels if lv["price"] < last_close]
    nearest_res = resistances[0]["price"] if resistances else last_close * 1.01
    nearest_sup = supports[0]["price"]    if supports    else last_close * 0.99
    dist_to_res = round((nearest_res - last_close) / last_close * 100, 3)
    dist_to_sup = round((last_close - nearest_sup) / last_close * 100, 3)

    # ── 4. Gamma walls ───────────────────────────────────────────────────────
    call_walls = [g for g in gamma if g["type"] == "call_wall"]
    put_walls  = [g for g in gamma if g["type"] == "put_wall"]
    nearest_cw = call_walls[0]["strike"] if call_walls else nearest_res
    nearest_pw = put_walls[0]["strike"]  if put_walls  else nearest_sup

    # ── 5. Tendencia diaria ──────────────────────────────────────────────────
    day_trend = "up"
    if not daily.empty and len(daily) >= 3:
        closes = daily["Close"].astype(float).tail(3).values
        day_trend = "up" if closes[-1] > closes[-3] else "down"

    # ── SCORING ──────────────────────────────────────────────────────────────
    bull_score = 0
    bear_score = 0
    reasons_bull = []
    reasons_bear = []

    # Momentum de velas
    if bull_bars >= 4:
        bull_score += 25
        reasons_bull.append(f"{bull_bars}/5 velas alcistas")
    elif bull_bars >= 3:
        bull_score += 12
    if bear_bars >= 4:
        bear_score += 25
        reasons_bear.append(f"{bear_bars}/5 velas bajistas")
    elif bear_bars >= 3:
        bear_score += 12

    # Dirección del último movimiento
    if last_move > 0.05:
        bull_score += 15
        reasons_bull.append(f"Momentum +{last_move}%")
    elif last_move < -0.05:
        bear_score += 15
        reasons_bear.append(f"Momentum {last_move}%")

    # Volumen
    if vol_confirm and last_bar_up:
        bull_score += 20
        reasons_bull.append("Volumen alto confirmando subida")
    elif vol_confirm and not last_bar_up:
        bear_score += 20
        reasons_bear.append("Volumen alto confirmando bajada")

    # Distancia a resistencia vs soporte (más cerca del soporte = alcista)
    if dist_to_sup < dist_to_res * 0.5:
        bull_score += 15
        reasons_bull.append(f"Precio cerca de soporte ${nearest_sup}")
    elif dist_to_res < dist_to_sup * 0.5:
        bear_score += 15
        reasons_bear.append(f"Precio cerca de resistencia ${nearest_res}")

    # Gamma walls — precio atrapado bajo call wall = bajista
    if nearest_cw and last_close > nearest_cw * 0.998:
        bear_score += 18
        reasons_bear.append(f"Call wall en ${nearest_cw} actuando como techo")
    if nearest_pw and last_close < nearest_pw * 1.002:
        bull_score += 18
        reasons_bull.append(f"Put wall en ${nearest_pw} actuando como piso")

    # Tendencia del día
    if day_trend == "up":
        bull_score += 10
        reasons_bull.append("Tendencia diaria alcista")
    else:
        bear_score += 10
        reasons_bear.append("Tendencia diaria bajista")

    # ── DECISIÓN ─────────────────────────────────────────────────────────────
    total = bull_score + bear_score
    if total == 0:
        return {"signal": "WAIT", "confidence": 0, "reason": "Sin datos suficientes"}

    bull_pct = round(bull_score / total * 100)
    bear_pct = 100 - bull_pct

    if bull_score >= bear_score * 1.4:
        signal     = "CALL"
        confidence = min(92, bull_pct)
        reasons    = reasons_bull[:3]
        entry      = round(last_close, 2)
        target     = round(nearest_res - 0.01, 2)
        stop       = round(nearest_sup + 0.01, 2)
    elif bear_score >= bull_score * 1.4:
        signal     = "PUT"
        confidence = min(92, bear_pct)
        reasons    = reasons_bear[:3]
        entry      = round(last_close, 2)
        target     = round(nearest_sup + 0.01, 2)
        stop       = round(nearest_res - 0.01, 2)
    else:
        signal     = "WAIT"
        confidence = 50
        reasons    = ["Señales mixtas — esperar confirmación"]
        entry      = round(last_close, 2)
        target     = 0
        stop       = 0

    return {
        "signal":     signal,
        "confidence": confidence,
        "reason":     " · ".join(reasons),
        "entry":      entry,
        "target":     target,
        "stop":       stop,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "price":      round(last_close, 2),
        "last_move":  last_move,
        "generated":  dt.datetime.now().strftime("%H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML DEL PANEL
# ─────────────────────────────────────────────────────────────────────────────

LIVE_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORB Live Panel</title>
<script src="https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js"></script>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e8ecf2;font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;min-height:100vh}

.topbar{background:#1a1d27;border-bottom:1px solid #2a2d3a;padding:12px 20px;
        display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:100}
.logo{width:30px;height:30px;border-radius:7px;
      background:linear-gradient(135deg,#f97316,#ea580c);
      display:flex;align-items:center;justify-content:center;flex-shrink:0}
.brand{font-size:14px;font-weight:600;color:#fff}
.sub{font-size:11px;color:#6b7280;margin-top:1px}
.badge{font-size:11px;padding:3px 9px;border-radius:20px;font-weight:500;flex-shrink:0}
.live-dot{background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3)}
.ticker-form{display:flex;gap:8px;align-items:center;margin-left:auto}
.ticker-input{background:#2a2d3a;border:1px solid #3a3d4a;color:#fff;
              border-radius:7px;padding:7px 12px;font-size:13px;
              font-family:monospace;font-weight:600;text-transform:uppercase;
              width:100px;outline:none;transition:border-color .2s}
.ticker-input:focus{border-color:#f97316}
.go-btn{background:linear-gradient(135deg,#f97316,#ea580c);color:#fff;
        border:none;border-radius:7px;padding:7px 16px;font-weight:600;
        font-size:12px;cursor:pointer;transition:opacity .15s}
.go-btn:hover{opacity:.85}

.grid{display:grid;grid-template-columns:1fr 320px;gap:0;min-height:calc(100vh - 57px)}
.main{display:flex;flex-direction:column;border-right:1px solid #2a2d3a}
.sidebar{display:flex;flex-direction:column;overflow-y:auto}

/* Signal card */
.signal-card{padding:16px 20px;border-bottom:1px solid #2a2d3a;flex-shrink:0}
.sig-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.sig-badge{font-size:18px;font-weight:800;padding:6px 18px;border-radius:8px;letter-spacing:.05em}
.sig-call{background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3)}
.sig-put{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.3)}
.sig-wait{background:rgba(245,158,11,.12);color:#f59e0b;border:1px solid rgba(245,158,11,.25)}
.sig-reason{font-size:12px;color:#9ca3af;line-height:1.6}
.sig-levels{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}
.sig-lv{background:#1a1d27;border-radius:6px;padding:8px 10px;text-align:center}
.sig-lv-label{font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.sig-lv-val{font-size:14px;font-weight:700;font-family:monospace}
.conf-bar{height:5px;border-radius:3px;background:#2a2d3a;margin-top:8px;overflow:hidden}
.conf-fill{height:100%;border-radius:3px;transition:width .5s}
.next-update{font-size:10px;color:#6b7280;margin-top:6px;text-align:right}

/* Chart */
.chart-wrap{flex:1;min-height:360px;position:relative}
#chart{width:100%;height:100%}
.chart-toolbar{padding:8px 16px;border-bottom:1px solid #2a2d3a;
               display:flex;align-items:center;gap:10px;flex-shrink:0}
.tf-btn{font-size:11px;padding:4px 10px;border-radius:5px;border:1px solid #3a3d4a;
        background:transparent;color:#9ca3af;cursor:pointer;transition:all .15s}
.tf-btn.active,.tf-btn:hover{border-color:#f97316;color:#f97316;background:rgba(249,115,22,.08)}
.price-tag{font-family:monospace;font-size:16px;font-weight:700;color:#fff}
.price-chg{font-size:12px;font-weight:500}
.price-up{color:#34d399}
.price-dn{color:#ef4444}

/* Sidebar sections */
.s-section{padding:14px 16px;border-bottom:1px solid #2a2d3a}
.s-title{font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;
         letter-spacing:.1em;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.s-title::before{content:'';display:inline-block;width:3px;height:12px;
                 background:linear-gradient(#f97316,#ea580c);border-radius:2px}

/* S/R Levels */
.sr-list{display:flex;flex-direction:column;gap:5px}
.sr-item{display:flex;align-items:center;gap:8px;padding:6px 8px;
         border-radius:5px;font-size:12px}
.sr-item.resistance{background:rgba(239,68,68,.06);border-left:2px solid #ef4444}
.sr-item.support{background:rgba(52,211,153,.06);border-left:2px solid #34d399}
.sr-item.sr{background:rgba(245,158,11,.06);border-left:2px solid #f59e0b}
.sr-price{font-family:monospace;font-weight:700;min-width:60px}
.sr-label{color:#9ca3af;font-size:11px;flex:1}
.sr-touches{font-size:10px;color:#6b7280;padding:2px 6px;background:#2a2d3a;border-radius:4px}
.current-price-line{background:rgba(139,92,246,.15);border-left:2px solid #8b5cf6;
                    display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:5px}

/* Volume */
.vol-stat{display:flex;justify-content:space-between;align-items:center;
          padding:5px 0;font-size:12px;border-bottom:1px solid #2a2d3a}
.vol-stat:last-child{border:none}
.vol-label{color:#9ca3af}
.vol-val{font-family:monospace;font-weight:600}
.vol-bars{margin-top:8px;display:flex;flex-direction:column;gap:4px}
.vol-bar-item{display:flex;align-items:center;gap:6px;font-size:11px}
.vol-bar-time{color:#6b7280;min-width:40px;font-family:monospace}
.vol-bar-fill{height:6px;border-radius:3px;min-width:4px}
.vol-bar-num{color:#9ca3af}

/* Gamma */
.gamma-list{display:flex;flex-direction:column;gap:4px}
.gamma-item{display:flex;align-items:center;gap:8px;font-size:11px;padding:4px 0}
.gamma-strike{font-family:monospace;font-weight:600;min-width:52px}
.gamma-bar-wrap{flex:1;height:5px;background:#2a2d3a;border-radius:3px;overflow:hidden}
.gamma-bar-fill{height:100%;border-radius:3px}
.gamma-type{font-size:10px;padding:1px 6px;border-radius:3px;font-weight:500}
.gamma-call{background:rgba(249,115,22,.15);color:#f97316}
.gamma-put{background:rgba(59,130,246,.15);color:#60a5fa}
.gamma-atm{background:rgba(139,92,246,.15);color:#a78bfa}

/* News */
.news-item{padding:7px 0;border-bottom:1px solid #2a2d3a;font-size:11px;line-height:1.5}
.news-item:last-child{border:none}
.news-title{color:#d1d5db}
.news-date{color:#6b7280;margin-top:2px;font-size:10px}

/* Status bar */
.status-bar{padding:6px 16px;background:#1a1d27;border-top:1px solid #2a2d3a;
            display:flex;align-items:center;gap:8px;font-size:11px;color:#6b7280;
            flex-shrink:0}
.status-dot{width:6px;height:6px;border-radius:50%;background:#34d399;
            animation:pulse-dot 2s infinite}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.3}}

/* Loading */
.loading{display:flex;align-items:center;justify-content:center;
         height:200px;color:#6b7280;font-size:13px;gap:8px}
.spinner{width:18px;height:18px;border:2px solid #3a3d4a;
         border-top-color:#f97316;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

@media(max-width:900px){.grid{grid-template-columns:1fr}.sidebar{border-top:1px solid #2a2d3a}}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="white" stroke-width="2.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div>
    <div class="brand">ORB Live Panel</div>
    <div class="sub">Señales en tiempo real · actualiza cada 5 min</div>
  </div>
  <span class="badge live-dot">● LIVE</span>
  <div class="ticker-form">
    <input class="ticker-input" id="ticker-inp" type="text" placeholder="SPY"
           maxlength="8" value="SPY" autocomplete="off"
           onkeydown="if(event.key==='Enter') loadTicker()">
    <button class="go-btn" onclick="loadTicker()">Analizar ▶</button>
    <a href="/" style="font-size:11px;color:#6b7280;text-decoration:none;margin-left:4px">← Chat</a>
  </div>
</div>

<div class="grid">

  <!-- MAIN: signal + chart -->
  <div class="main">

    <!-- Signal -->
    <div class="signal-card" id="signal-panel">
      <div class="loading"><div class="spinner"></div> Ingresa un ticker para comenzar</div>
    </div>

    <!-- Chart toolbar -->
    <div class="chart-toolbar">
      <span id="price-display" class="price-tag">—</span>
      <span id="price-chg" class="price-chg">—</span>
      <span style="margin-left:auto;font-size:11px;color:#6b7280">Timeframe:</span>
      <button class="tf-btn active" onclick="setTF('1m',this)">1m</button>
      <button class="tf-btn" onclick="setTF('5m',this)">5m</button>
    </div>

    <!-- Chart -->
    <div class="chart-wrap">
      <div id="chart"></div>
    </div>

    <div class="status-bar">
      <div class="status-dot"></div>
      <span id="status-txt">Esperando ticker...</span>
      <span style="margin-left:auto" id="last-update">—</span>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- S/R Levels -->
    <div class="s-section" id="sr-panel">
      <div class="s-title">Niveles S/R — última semana</div>
      <div class="loading" style="height:80px"><div class="spinner"></div></div>
    </div>

    <!-- Volume -->
    <div class="s-section" id="vol-panel">
      <div class="s-title">Volumen en vivo</div>
      <div class="loading" style="height:80px"><div class="spinner"></div></div>
    </div>

    <!-- Gamma -->
    <div class="s-section" id="gamma-panel">
      <div class="s-title">Gamma Exposure estimado</div>
      <div class="loading" style="height:80px"><div class="spinner"></div></div>
    </div>

    <!-- News -->
    <div class="s-section" id="news-panel">
      <div class="s-title">Noticias recientes</div>
      <div class="loading" style="height:80px"><div class="spinner"></div></div>
    </div>

  </div>
</div>

<script>
let chart, candleSeries, volumeSeries;
let currentTicker = '';
let currentTF = '1m';
let refreshTimer = null;
let minuteCount = 0;

// ── Chart init ───────────────────────────────────────────────────────────────
function initChart() {
  const el = document.getElementById('chart');
  if (chart) { chart.remove(); chart = null; }
  chart = LightweightCharts.createChart(el, {
    width: el.parentElement.offsetWidth,
    height: el.parentElement.offsetHeight || 360,
    layout: { background: { color: '#0f1117' }, textColor: '#9ca3af' },
    grid: { vertLines: { color: '#1e2130' }, horzLines: { color: '#1e2130' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a2d3a' },
    timeScale: { borderColor: '#2a2d3a', timeVisible: true, secondsVisible: false },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#34d399', downColor: '#ef4444',
    borderUpColor: '#34d399', borderDownColor: '#ef4444',
    wickUpColor: '#34d399', wickDownColor: '#ef4444',
  });
  volumeSeries = chart.addHistogramSeries({
    color: '#3a3d4a', priceFormat: { type: 'volume' },
    priceScaleId: '', scaleMargins: { top: 0.8, bottom: 0 },
  });
  new ResizeObserver(() => {
    if (chart) chart.applyOptions({
      width: el.parentElement.offsetWidth,
      height: el.parentElement.offsetHeight,
    });
  }).observe(el.parentElement);
}

// ── Load ticker ──────────────────────────────────────────────────────────────
function loadTicker() {
  const t = document.getElementById('ticker-inp').value.trim().toUpperCase();
  if (!t) return;
  currentTicker = t;
  minuteCount = 0;
  document.getElementById('status-txt').textContent = `Cargando ${t}...`;

  initChart();
  refreshAll();

  // Refresh cada 60 segundos
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(() => {
    minuteCount++;
    refreshAll();
  }, 60000);
}

function setTF(tf, btn) {
  currentTF = tf;
  document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (currentTicker) fetchChart();
}

// ── Refresh all panels ───────────────────────────────────────────────────────
function refreshAll() {
  if (!currentTicker) return;
  fetchChart();
  fetchSignal();
  fetchSidebar();
}

// ── Chart data ───────────────────────────────────────────────────────────────
function fetchChart() {
  fetch(`/live/chart?ticker=${currentTicker}&tf=${currentTF}`)
    .then(r => r.json())
    .then(d => {
      if (!d.candles || !d.candles.length) return;
      candleSeries.setData(d.candles);
      volumeSeries.setData(d.volumes);
      chart.timeScale().fitContent();

      // Price display
      const last = d.candles[d.candles.length - 1];
      const prev = d.candles.length > 1 ? d.candles[d.candles.length - 2] : last;
      const chg = ((last.close - prev.close) / prev.close * 100).toFixed(2);
      document.getElementById('price-display').textContent = `$${last.close.toFixed(2)}`;
      const chgEl = document.getElementById('price-chg');
      chgEl.textContent = `${chg > 0 ? '+' : ''}${chg}%`;
      chgEl.className = 'price-chg ' + (chg >= 0 ? 'price-up' : 'price-dn');

      // Draw S/R lines on chart
      if (d.sr_lines) {
        d.sr_lines.forEach(lv => {
          candleSeries.createPriceLine({
            price: lv.price,
            color: lv.type === 'R' ? '#ef444455' : lv.type === 'S' ? '#34d39955' : '#f59e0b55',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            title: lv.label,
          });
        });
      }

      document.getElementById('last-update').textContent =
        'Actualizado: ' + new Date().toLocaleTimeString('es');
      document.getElementById('status-txt').textContent =
        `${currentTicker} · ${d.candles.length} barras ${currentTF}`;
    })
    .catch(e => console.error('chart error', e));
}

// ── Signal panel ─────────────────────────────────────────────────────────────
function fetchSignal() {
  fetch(`/live/signal?ticker=${currentTicker}`)
    .then(r => r.text())
    .then(html => { document.getElementById('signal-panel').innerHTML = html; })
    .catch(e => console.error('signal error', e));
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function fetchSidebar() {
  fetch(`/live/sidebar?ticker=${currentTicker}`)
    .then(r => r.json())
    .then(d => {
      document.getElementById('sr-panel').innerHTML    = d.sr;
      document.getElementById('vol-panel').innerHTML   = d.vol;
      document.getElementById('gamma-panel').innerHTML = d.gamma;
      document.getElementById('news-panel').innerHTML  = d.news;
    })
    .catch(e => console.error('sidebar error', e));
}

// Auto-load SPY on start
window.addEventListener('load', () => {
  initChart();
  // Don't auto-load — wait for user
});
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# RENDERERS HTML
# ─────────────────────────────────────────────────────────────────────────────

def render_signal_html(sig: dict) -> str:
    s = sig["signal"]
    conf = sig["confidence"]
    cls = {"CALL": "sig-call", "PUT": "sig-put", "WAIT": "sig-wait"}.get(s, "sig-wait")
    icon = {"CALL": "▲", "PUT": "▼", "WAIT": "◆"}.get(s, "◆")
    bar_color = "#34d399" if s == "CALL" else "#ef4444" if s == "PUT" else "#f59e0b"

    levels_html = ""
    if s != "WAIT" and sig.get("entry"):
        def lv(label, val, color):
            return f"""<div class="sig-lv">
              <div class="sig-lv-label">{label}</div>
              <div class="sig-lv-val" style="color:{color}">${val}</div>
            </div>"""
        levels_html = f"""<div class="sig-levels">
          {lv("Entrada", sig["entry"], "#fff")}
          {lv("Target", sig["target"], "#34d399")}
          {lv("Stop", sig["stop"], "#ef4444")}
        </div>"""

    return f"""
    <div class="sig-header">
      <span class="sig-badge {cls}">{icon} {s}</span>
      <div>
        <div style="font-size:11px;color:#9ca3af">Confianza</div>
        <div style="font-family:monospace;font-size:16px;font-weight:700;
                    color:{bar_color}">{conf}%</div>
      </div>
      <div style="margin-left:auto;font-size:11px;color:#6b7280;text-align:right">
        <div>${sig.get("price", 0)}</div>
        <div>{sig.get("generated", "")}</div>
      </div>
    </div>
    <div class="sig-reason">{sig["reason"]}</div>
    {levels_html}
    <div class="conf-bar">
      <div class="conf-fill" style="width:{conf}%;background:{bar_color}"></div>
    </div>
    <div class="next-update">↺ Actualiza cada 5 min · próxima señal en 1 min</div>
    """


def render_sr_html(sr_levels: list, current_price: float) -> str:
    html = '<div class="s-title">Niveles S/R — última semana</div><div class="sr-list">'

    # Insertar precio actual
    above = [lv for lv in sr_levels if lv["price"] > current_price]
    below = [lv for lv in sr_levels if lv["price"] <= current_price]

    for lv in sorted(above, key=lambda x: x["price"])[:4]:
        dist = round((lv["price"] - current_price) / current_price * 100, 2)
        type_cls = "resistance" if lv["type"] == "R" else "sr"
        type_lbl = "R" if lv["type"] == "R" else "SR"
        clr = "#ef4444" if lv["type"] == "R" else "#f59e0b"
        html += f"""<div class="sr-item {type_cls}">
          <span class="sr-price" style="color:{clr}">${lv["price"]}</span>
          <span class="sr-label">{lv["label"][:20]}</span>
          <span class="sr-touches">{lv["touches"]}x</span>
          <span style="font-size:10px;color:{clr}">+{dist}%</span>
        </div>"""

    html += f"""<div class="current-price-line">
      <span style="font-size:11px;color:#a78bfa;font-weight:600">▶ Precio: ${current_price}</span>
    </div>"""

    for lv in sorted(below, key=lambda x: x["price"], reverse=True)[:4]:
        dist = round((current_price - lv["price"]) / current_price * 100, 2)
        type_cls = "support" if lv["type"] == "S" else "sr"
        clr = "#34d399" if lv["type"] == "S" else "#f59e0b"
        html += f"""<div class="sr-item {type_cls}">
          <span class="sr-price" style="color:{clr}">${lv["price"]}</span>
          <span class="sr-label">{lv["label"][:20]}</span>
          <span class="sr-touches">{lv["touches"]}x</span>
          <span style="font-size:10px;color:{clr}">-{dist}%</span>
        </div>"""

    html += "</div>"
    return html


def render_vol_html(vol: dict) -> str:
    if not vol:
        return '<div class="s-title">Volumen en vivo</div><div style="color:#6b7280;font-size:12px">Sin datos</div>'

    def fmt_vol(v):
        if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
        if v >= 1_000:     return f"{v/1_000:.0f}K"
        return str(v)

    above = vol.get("above_avg", False)
    color = "#34d399" if above else "#9ca3af"
    pct   = vol.get("vol_pct_avg", 0)

    html = f'<div class="s-title">Volumen en vivo</div>'
    html += f"""
    <div class="vol-stat"><span class="vol-label">Acumulado hoy</span>
      <span class="vol-val" style="color:{color}">{fmt_vol(vol.get("today_vol",0))}</span></div>
    <div class="vol-stat"><span class="vol-label">Promedio diario</span>
      <span class="vol-val">{fmt_vol(vol.get("avg_daily",0))}</span></div>
    <div class="vol-stat"><span class="vol-label">% del promedio</span>
      <span class="vol-val" style="color:{color}">{pct}%</span></div>
    <div class="vol-stat"><span class="vol-label">Pace EOD estimado</span>
      <span class="vol-val">{fmt_vol(vol.get("pace_eod",0))}</span></div>
    """

    bars = vol.get("high_vol_bars", [])
    if bars:
        html += '<div style="font-size:10px;color:#6b7280;margin-top:8px;margin-bottom:4px">BARRAS DE ALTO VOLUMEN</div>'
        html += '<div class="vol-bars">'
        max_v = max(b["volume"] for b in bars) if bars else 1
        for b in bars:
            w = int(b["volume"] / max_v * 100)
            clr = "#34d399" if b["direction"] == "buy" else "#ef4444"
            html += f"""<div class="vol-bar-item">
              <span class="vol-bar-time">{b["time"]}</span>
              <div style="flex:1;height:6px;background:#2a2d3a;border-radius:3px;overflow:hidden">
                <div class="vol-bar-fill" style="width:{w}%;background:{clr}"></div>
              </div>
              <span class="vol-bar-num" style="color:{clr}">{b["ratio"]}x</span>
            </div>"""
        html += '</div>'

    return html


def render_gamma_html(gamma: list, current_price: float) -> str:
    if not gamma:
        return '<div class="s-title">Gamma Exposure estimado</div><div style="color:#6b7280;font-size:12px">Sin datos</div>'

    html = '<div class="s-title">Gamma Exposure estimado</div>'
    html += '<div style="font-size:10px;color:#6b7280;margin-bottom:8px">Estimación por strikes y OI histórico</div>'
    html += '<div class="gamma-list">'

    sorted_gamma = sorted(gamma, key=lambda x: x["strike"])
    max_g = max(g["gamma"] for g in sorted_gamma) if sorted_gamma else 1

    for g in sorted_gamma:
        w   = int(g["gamma"] / max_g * 100) if max_g > 0 else 0
        clr = g["color"]
        type_cls = {"call_wall": "gamma-call", "put_wall": "gamma-put",
                    "atm": "gamma-atm"}.get(g["type"], "gamma-call")
        type_lbl = {"call_wall": "CALL", "put_wall": "PUT",
                    "atm": "ATM"}.get(g["type"], "")
        is_current = abs(g["strike"] - current_price) < 0.5
        style = "background:#2a2d3a;border-radius:4px;padding:3px 4px;" if is_current else ""

        html += f"""<div class="gamma-item" style="{style}">
          <span class="gamma-strike" style="color:{clr}">${g["strike"]:.0f}</span>
          <div class="gamma-bar-wrap">
            <div class="gamma-bar-fill" style="width:{w}%;background:{clr}"></div>
          </div>
          <span class="gamma-type {type_cls}">{type_lbl}</span>
        </div>"""

    html += '</div>'
    html += '<div style="font-size:10px;color:#4b5563;margin-top:8px">⚠ Estimación — sin API de opciones real</div>'
    return html


def render_news_html(news: list, ticker: str) -> str:
    html = f'<div class="s-title">Noticias — {ticker}</div>'
    if not news:
        html += '<div style="color:#6b7280;font-size:12px">Sin noticias recientes</div>'
        return html
    for n in news:
        html += f"""<div class="news-item">
          <div class="news-title">{n["title"][:80]}{"..." if len(n["title"]) > 80 else ""}</div>
          <div class="news-date">{n["date"]}</div>
        </div>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/live", response_class=HTMLResponse)
async def live_panel():
    return HTMLResponse(LIVE_HTML)


@router.get("/live/chart")
async def live_chart(ticker: str = "SPY", tf: str = "1m"):
    """Devuelve datos de velas para el gráfico."""
    ticker = ticker.upper().strip()
    try:
        if tf == "1m":
            df = fetch_1m(ticker, days=1)
        else:
            # 5m: últimos 5 días
            end   = dt.datetime.now(dt.timezone.utc)
            start = end - dt.timedelta(days=5)
            raw = yf.download(ticker,
                              start=start.strftime("%Y-%m-%d"),
                              end=(end + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                              interval="5m", auto_adjust=True, progress=False)
            if raw.empty:
                return {"candles": [], "volumes": [], "sr_lines": []}
            df = _clean(raw)
            mask = (df.index.time >= MARKET_OPEN) & (df.index.time <= MARKET_CLOSE)
            df = df[mask].copy()

        if df.empty:
            return {"candles": [], "volumes": [], "sr_lines": []}

        # Convertir a formato lightweight-charts
        candles = []
        volumes = []
        for ts, row in df.iterrows():
            try:
                t = int(pd.Timestamp(ts).timestamp())
                o = round(float(row["Open"]),  2)
                h = round(float(row["High"]),  2)
                l = round(float(row["Low"]),   2)
                c = round(float(row["Close"]), 2)
                v = int(float(row.get("Volume", 0)))
                if any(x == 0 for x in [o, h, l, c]): continue
                candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
                volumes.append({"time": t, "value": v,
                                "color": "#34d39955" if c >= o else "#ef444455"})
            except:
                continue

        # S/R lines para el gráfico
        daily = fetch_daily_week(ticker)
        cur   = candles[-1]["close"] if candles else 100
        sr    = get_sr_levels(daily, cur)
        sr_lines = [{"price": lv["price"], "type": lv["type"], "label": lv["label"][:12]}
                    for lv in sr[:8]]

        return {"candles": candles, "volumes": volumes, "sr_lines": sr_lines}

    except Exception as e:
        return {"candles": [], "volumes": [], "sr_lines": [], "error": str(e)}


@router.get("/live/signal", response_class=HTMLResponse)
async def live_signal(ticker: str = "SPY"):
    """Genera señal CALL/PUT/WAIT."""
    ticker = ticker.upper().strip()
    try:
        df1m  = fetch_1m(ticker, days=1)
        daily = fetch_daily_week(ticker)
        cur   = float(df1m["Close"].iloc[-1]) if not df1m.empty else 100.0
        atr   = 2.0

        if not daily.empty:
            hi = daily["High"].astype(float)
            lo = daily["Low"].astype(float)
            cl = daily["Close"].astype(float)
            tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
            atr = max(0.5, round(float(tr.tail(5).mean()), 2))

        sr     = get_sr_levels(daily, cur)
        vol    = get_volume_context(df1m, daily)
        gamma  = get_gamma_exposure(ticker, cur, atr)
        signal = build_signal(ticker, df1m, daily, sr, vol, gamma)

        return HTMLResponse(render_signal_html(signal))
    except Exception as e:
        return HTMLResponse(f'<div style="color:#ef4444;font-size:12px">Error: {e}</div>')


@router.get("/live/sidebar")
async def live_sidebar(ticker: str = "SPY"):
    """Devuelve todos los paneles del sidebar como JSON de HTML."""
    ticker = ticker.upper().strip()
    try:
        df1m  = fetch_1m(ticker, days=1)
        daily = fetch_daily_week(ticker)
        cur   = float(df1m["Close"].iloc[-1]) if not df1m.empty else 100.0
        atr   = 2.0

        if not daily.empty:
            hi = daily["High"].astype(float)
            lo = daily["Low"].astype(float)
            cl = daily["Close"].astype(float)
            tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
            atr = max(0.5, round(float(tr.tail(5).mean()), 2))

        sr    = get_sr_levels(daily, cur)
        vol   = get_volume_context(df1m, daily)
        gamma = get_gamma_exposure(ticker, cur, atr)
        news  = fetch_news_ticker(ticker)

        return {
            "sr":    render_sr_html(sr, cur),
            "vol":   render_vol_html(vol),
            "gamma": render_gamma_html(gamma, cur),
            "news":  render_news_html(news, ticker),
        }
    except Exception as e:
        err = f'<div style="color:#ef4444;font-size:12px">Error: {e}</div>'
        return {"sr": err, "vol": err, "gamma": err, "news": err}


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE — python live.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()
    app.include_router(router)

    port = int(os.getenv("PORT", 7861))
    print(f"\n{'='*50}")
    print(f"  ORB Live Panel")
    print(f"  http://localhost:{port}/live")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")