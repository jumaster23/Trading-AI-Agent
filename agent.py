"""
ORB Trading AI Agent — phidata + ml_engine + trading_db + HTMX
===============================================================
Coloca este archivo en la misma carpeta que ml_engine.py y trading_db.py.

Uso:
    python agent.py                      # chat interactivo en terminal
    python agent.py --ticker SPY         # análisis rápido
    python agent.py --ticker SPY --plan  # plan completo del día
    python agent.py --ticker QQQ --ml   # análisis ML
    python agent.py --server             # UI web HTMX en localhost:7860

Setup:
    pip install phidata openai yfinance fastapi uvicorn python-dotenv
    Crea .env con: GROQ_API_KEY=sk-tu-key
"""

import os, json, datetime as dt, sys
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import pathlib as _pl
# Lee .env manualmente — soporta BOM y cualquier encoding de Windows
for _ep in [_pl.Path(__file__).parent/".env", _pl.Path(".env")]:
    if _ep.exists():
        for _line in _ep.read_text(encoding="utf-8-sig").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
        break

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools import tool

# ── Importar tus módulos existentes ─────────────────────────────────────────
try:
    from ml_engine import run_ml_analysis, REGIME_NAMES
    ML_AVAILABLE = True
    print("[✓] ml_engine.py cargado")
except ImportError:
    ML_AVAILABLE = False
    print("[✗] ml_engine.py no encontrado — funciones ML deshabilitadas")
    def run_ml_analysis(*a, **k): return {"error": "ml_engine.py no disponible"}
    REGIME_NAMES = {}

try:
    from trading_db import get_stats, pattern_score, load_db, save_day, atr_quality_check
    DB_AVAILABLE = True
    print("[✓] trading_db.py cargado")
except ImportError:
    DB_AVAILABLE = False
    print("[✗] trading_db.py no encontrado — base de patrones deshabilitada")
    def get_stats(t): return {"n": 0, "message": "trading_db.py no disponible"}
    def pattern_score(*a, **k): return {"similar_n": 0, "quality": 0, "recommendation": "TRAIN"}
    def load_db(t): return {"days": {}}
    def save_day(*a, **k): pass
    def atr_quality_check(*a, **k): return {"valid": True, "notes": [], "quality_score": 50}

# ─────────────────────────────────────────────────────────────────────────────
# CAPA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
MARKET_OPEN  = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
SIZE_BUCKETS = {
    "tiny":   (0.00, 0.25),
    "small":  (0.25, 0.50),
    "medium": (0.50, 1.00),
    "large":  (1.00, 2.00),
    "huge":   (2.00, 999.0),
}

def classify_size(v: float) -> str:
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= v < hi:
            return name
    return "huge"

def clean_df(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("America/New_York")
    else:
        raw.index = raw.index.tz_convert("America/New_York")
    return raw

def fetch_daily(ticker: str, years: int = 2) -> pd.DataFrame:
    raw = yf.download(ticker, period=f"{years}y", interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty: return pd.DataFrame()
    raw = clean_df(raw)
    raw.index = pd.to_datetime([str(d)[:10] for d in raw.index])
    return raw[["Open", "High", "Low", "Close", "Volume"]].copy()

def fetch_5m(ticker: str, days: int = 59) -> pd.DataFrame:
    end   = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    raw = yf.download(ticker,
                      start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      interval="5m", auto_adjust=True, progress=False)
    if raw.empty: return pd.DataFrame()
    raw = clean_df(raw)
    mask = (raw.index.time >= MARKET_OPEN) & (raw.index.time < MARKET_CLOSE)
    return raw[mask].copy()

def get_market_context(daily: pd.DataFrame) -> dict:
    if daily.empty or len(daily) < 20: return {}
    cl   = daily["Close"].astype(float)
    last = float(cl.iloc[-1])
    ema20  = float(cl.ewm(span=20).mean().iloc[-1])
    ema50  = float(cl.ewm(span=50).mean().iloc[-1])
    ema200 = float(cl.ewm(span=min(200, len(cl)-1)).mean().iloc[-1])
    hi = daily["High"].astype(float)
    lo = daily["Low"].astype(float)
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    atr = round(float(tr.tail(14).mean()), 2)
    if   last > ema20 > ema50 > ema200: trend = "STRONG_UPTREND"
    elif last > ema20 > ema50:           trend = "UPTREND"
    elif last < ema20 < ema50 < ema200: trend = "STRONG_DOWNTREND"
    elif last < ema20 < ema50:           trend = "DOWNTREND"
    else:                                trend = "RANGING"
    streak, streak_dir = 0, ""
    for i in range(len(daily)-1, max(len(daily)-11, -1), -1):
        d = "up" if float(daily["Close"].iloc[i]) > float(daily["Open"].iloc[i]) else "down"
        if streak == 0: streak = 1; streak_dir = d
        elif d == streak_dir: streak += 1
        else: break
    return {
        "last_price":   round(last, 2),
        "ema20":        round(ema20, 2),
        "ema50":        round(ema50, 2),
        "ema200":       round(ema200, 2),
        "atr":          atr,
        "trend":        trend,
        "streak":       streak,
        "streak_dir":   streak_dir,
        "high_60d":     round(float(daily["High"].tail(60).max()), 2),
        "low_60d":      round(float(daily["Low"].tail(60).min()),  2),
        "above_ema20":  last > ema20,
        "above_ema50":  last > ema50,
        "above_ema200": last > ema200,
    }

def get_today_gap(daily: pd.DataFrame) -> dict:
    if len(daily) < 2: return {}
    pc  = float(daily["Close"].iloc[-2])
    op  = float(daily["Open"].iloc[-1])
    hi  = float(daily["High"].iloc[-1])
    lo  = float(daily["Low"].iloc[-1])
    cl  = float(daily["Close"].iloc[-1])
    gp  = (op - pc) / pc * 100
    if   gp >  0.05: direction, filled = "up",   lo <= pc
    elif gp < -0.05: direction, filled = "down",  hi >= pc
    else:            direction, filled = "flat",  True
    return {
        "date":       str(daily.index[-1])[:10],
        "prev_close": round(pc, 2),
        "open":       round(op, 2),
        "close":      round(cl, 2),
        "gap_pct":    round(gp, 3),
        "direction":  direction,
        "size_cat":   classify_size(abs(gp)),
        "filled":     filled,
    }

def compute_gap_stats(daily: pd.DataFrame) -> dict:
    rows = []
    dates = daily.index.tolist()
    for i in range(1, len(dates)):
        pc = float(daily.loc[dates[i-1], "Close"])
        op = float(daily.loc[dates[i],   "Open"])
        hi = float(daily.loc[dates[i],   "High"])
        lo = float(daily.loc[dates[i],   "Low"])
        gp = (op - pc) / pc * 100
        if   gp >  0.05: direction, filled = "up",   lo <= pc
        elif gp < -0.05: direction, filled = "down",  hi >= pc
        else: continue
        rows.append({"gap_pct": gp, "direction": direction,
                     "size_cat": classify_size(abs(gp)), "filled": filled})
    if not rows: return {}
    df = pd.DataFrame(rows)
    stats = {}
    for size in SIZE_BUCKETS:
        sub = df[df["size_cat"] == size]
        if sub.empty: continue
        up = sub[sub["direction"] == "up"]
        dn = sub[sub["direction"] == "down"]
        stats[size] = {
            "count":       len(sub),
            "fill_pct":    round(sub["filled"].mean() * 100, 1),
            "up_fill_pct": round(up["filled"].mean() * 100, 1) if len(up) else 0,
            "dn_fill_pct": round(dn["filled"].mean() * 100, 1) if len(dn) else 0,
            "avg_gap":     round(sub["gap_pct"].abs().mean(), 3),
        }
    return stats

def compute_orb_stats(df5m: pd.DataFrame, minutes: int = 15) -> dict:
    if df5m.empty: return {}
    results = []
    for date, bars in df5m.groupby(df5m.index.date):
        bars = bars.sort_index()
        ot  = bars.index[0]
        orb = bars[bars.index <  ot + pd.Timedelta(minutes=minutes)]
        rst = bars[bars.index >= ot + pd.Timedelta(minutes=minutes)]
        if orb.empty or rst.empty: continue
        oh = float(orb["High"].max())
        ol = float(orb["Low"].min())
        oo = float(orb["Open"].iloc[0])
        if oo == 0: continue
        rng = (oh - ol) / oo * 100
        bu  = float(rst["High"].max()) > oh
        bd  = float(rst["Low"].min())  < ol
        results.append({
            "broke_up":   bu,   "broke_down": bd,
            "ext_up":     round((float(rst["High"].max()) - oh) / oo * 100, 3) if bu else 0,
            "ext_down":   round((ol - float(rst["Low"].min())) / oo * 100, 3) if bd else 0,
            "range_pct":  round(rng, 3),
            "size_cat":   classify_size(rng),
        })
    if not results: return {}
    df = pd.DataFrame(results)
    up = df[df["broke_up"]]
    dn = df[df["broke_down"]]
    return {
        "n_days":     len(df),
        "up_pct":     round(df["broke_up"].mean()  * 100, 1),
        "dn_pct":     round(df["broke_down"].mean() * 100, 1),
        "both_pct":   round((df["broke_up"] & df["broke_down"]).mean() * 100, 1),
        "avg_ext_up": round(up["ext_up"].mean(),   3) if len(up) else 0,
        "avg_ext_dn": round(dn["ext_down"].mean(),  3) if len(dn) else 0,
        "avg_range":  round(df["range_pct"].mean(), 3),
    }

def fetch_news(ticker: str) -> list:
    import urllib.request, re
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            xml = r.read().decode("utf-8", "ignore")
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml)
        dates  = re.findall(r"<pubDate>(.*?)</pubDate>", xml)
        return [{"title": t.strip(), "date": dates[i][:22] if i < len(dates) else ""}
                for i, t in enumerate(titles[1:7])]
    except Exception as e:
        return [{"title": f"Error obteniendo noticias: {e}", "date": ""}]

# ─────────────────────────────────────────────────────────────────────────────
# TOOLS DE PHIDATA
# ─────────────────────────────────────────────────────────────────────────────

@tool
def analizar_ticker(ticker: str) -> str:
    """
    Obtiene datos reales de mercado para un ticker: precio, tendencia,
    EMAs (20/50/200), ATR, rango 60d, racha de días y contexto general.
    Llama esta tool PRIMERO antes de cualquier análisis.

    Args:
        ticker: Símbolo bursátil, ej: 'SPY', 'QQQ', 'AAPL'
    """
    ticker = ticker.upper().strip()
    daily  = fetch_daily(ticker)
    if daily.empty:
        return json.dumps({"error": f"Sin datos para {ticker}"})
    ctx = get_market_context(daily)
    ctx.update({
        "ticker":    ticker,
        "desde":     str(daily.index.min())[:10],
        "hasta":     str(daily.index.max())[:10],
        "total_dias": len(daily),
    })
    return json.dumps(ctx, indent=2)


@tool
def analisis_gap(ticker: str) -> str:
    """
    Analiza el gap de hoy y las tasas históricas de fill por tamaño
    (tiny/small/medium/large/huge). Úsala para preguntas sobre
    'se llenará el gap hoy?' o probabilidad de fill.

    Args:
        ticker: Símbolo bursátil, ej: 'SPY', 'QQQ'
    """
    ticker = ticker.upper().strip()
    daily  = fetch_daily(ticker)
    if daily.empty:
        return json.dumps({"error": f"Sin datos para {ticker}"})
    gap       = get_today_gap(daily)
    gap_stats = compute_gap_stats(daily)
    gs        = gap_stats.get(gap.get("size_cat", "medium"), {})
    fkey      = "up_fill_pct" if gap.get("direction") == "up" else "dn_fill_pct"
    fill_prob = gs.get(fkey, gs.get("fill_pct", 50))
    return json.dumps({
        "ticker":          ticker,
        "gap_hoy":         gap,
        "probabilidad_fill": f"{fill_prob}%",
        "tasas_historicas": gap_stats,
        "leyenda":         {"tiny":"0–0.25%","small":"0.25–0.5%",
                            "medium":"0.5–1%","large":"1–2%","huge":"2%+"},
    }, indent=2)


@tool
def analisis_orb(ticker: str) -> str:
    """
    Calcula estadísticas de Opening Range Breakout para ventanas de
    5m, 15m y 30m: % que rompe arriba/abajo, extensión promedio, rango típico.

    Args:
        ticker: Símbolo bursátil, ej: 'SPY', 'QQQ'
    """
    ticker = ticker.upper().strip()
    df5m   = fetch_5m(ticker)
    if df5m.empty:
        return json.dumps({"error": f"Sin datos 5m para {ticker}"})
    return json.dumps({
        "ticker":  ticker,
        "orb_5m":  compute_orb_stats(df5m, 5),
        "orb_15m": compute_orb_stats(df5m, 15),
        "orb_30m": compute_orb_stats(df5m, 30),
        "nota":    "Basado en últimos 59 días de datos 5m de Yahoo Finance",
    }, indent=2)


@tool
def prediccion_ml(ticker: str) -> str:
    """
    Ejecuta el ML Pattern Engine (ml_engine.py) sobre los últimos 30 días.
    Devuelve: régimen de mercado (KMeans), probabilidad de sesgo alcista/bajista
    (Random Forest + Gradient Boosting), días similares históricos,
    patrones repetibles y detección de anomalías (LOF).

    Args:
        ticker: Símbolo bursátil, ej: 'SPY', 'QQQ'
    """
    if not ML_AVAILABLE:
        return json.dumps({"error": "ml_engine.py no disponible en esta carpeta"})
    ticker = ticker.upper().strip()
    daily  = fetch_daily(ticker)
    df5m   = fetch_5m(ticker)
    if daily.empty:
        return json.dumps({"error": f"Sin datos para {ticker}"})
    try:
        result = run_ml_analysis(ticker, daily, df5m, levels=[], lookback=30)
    except Exception as e:
        return json.dumps({"error": str(e)})
    bm = result.get("bias_model", {}) or {}
    pb = bm.get("proba_bull", 0.5) or 0.5
    return json.dumps({
        "ticker":      ticker,
        "ml_activo":   True,
        "error":       result.get("error"),
        "regimen": {
            "actual":      result.get("regime", {}).get("current"),
            "nombre":      result.get("regime", {}).get("label"),
            "descripcion": result.get("regime", {}).get("description"),
        } if result.get("regime") else None,
        "sesgo_ml": {
            "prob_alcista": round(pb, 3),
            "prob_bajista": round(1 - pb, 3),
            "sesgo":        "ALCISTA" if pb > 0.55 else "BAJISTA" if pb < 0.45 else "NEUTRAL",
            "accuracy_cv":  bm.get("accuracy"),
            "n_entrenamiento": bm.get("n_train"),
            "top_features": bm.get("top_features", [])[:5],
            "prob_rf":      bm.get("proba_rf"),
            "prob_gb":      bm.get("proba_gb"),
        },
        "anomalia":    result.get("anomaly", {}),
        "dias_similares": result.get("similar_days", [])[:5],
        "estadisticas_outcome": result.get("outcome_stats", {}),
        "patrones": [
            {"nombre": p["name"], "bull_rate": p["bull_rate"],
             "ocurrencias": p["count"], "ret_siguiente": p.get("avg_next_ret", 0)}
            for p in result.get("patterns", [])[:4]
        ],
    }, indent=2)


@tool
def stats_base_datos(ticker: str) -> str:
    """
    Consulta la base de datos de patrones (trading_db.py).
    Devuelve: días registrados, win rate de CALL/PUT, tasa de días alcistas,
    estadísticas de ORB y fill de gap basadas en días históricos reales.

    Args:
        ticker: Símbolo bursátil, ej: 'SPY', 'QQQ'
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "trading_db.py no disponible"})
    ticker = ticker.upper().strip()
    stats  = get_stats(ticker)
    return json.dumps({"ticker": ticker, "stats_db": stats}, indent=2)


@tool
def plan_del_dia(ticker: str) -> str:
    """
    Genera un plan de trading completo para hoy combinando:
    gap + ORB + contexto de mercado + ML (si disponible) + niveles clave.
    Es la tool principal para preguntas del tipo 'dame un plan para X hoy'.

    Args:
        ticker: Símbolo bursátil, ej: 'SPY', 'QQQ', 'NVDA'
    """
    ticker = ticker.upper().strip()
    daily  = fetch_daily(ticker)
    if daily.empty:
        return json.dumps({"error": f"Sin datos para {ticker}"})

    ctx       = get_market_context(daily)
    gap       = get_today_gap(daily)
    gap_stats = compute_gap_stats(daily)
    df5m      = fetch_5m(ticker)
    orb_15    = compute_orb_stats(df5m, 15) if not df5m.empty else {}
    orb_5     = compute_orb_stats(df5m, 5)  if not df5m.empty else {}

    gs        = gap_stats.get(gap.get("size_cat", "medium"), {})
    fkey      = "up_fill_pct" if gap.get("direction") == "up" else "dn_fill_pct"
    fill_prob = gs.get(fkey, gs.get("fill_pct", 50))

    last  = ctx.get("last_price", 0)
    atr   = ctx.get("atr", 0)
    prev  = gap.get("prev_close", last)
    today_o = gap.get("open", last)

    niveles = {
        "prev_close":    round(prev, 2),
        "apertura_hoy":  round(today_o, 2),
        "max_dia_ant":   round(float(daily["High"].iloc[-2]), 2) if len(daily) >= 2 else 0,
        "min_dia_ant":   round(float(daily["Low"].iloc[-2]),  2) if len(daily) >= 2 else 0,
        "max_60d":       ctx.get("high_60d"),
        "min_60d":       ctx.get("low_60d"),
    }
    if atr and last:
        niveles["target_alcista_atr"] = round(last + atr * 0.6, 2)
        niveles["target_bajista_atr"] = round(last - atr * 0.6, 2)
        niveles["stop_ajustado"]      = round(atr * 0.25, 2)
        niveles["stop_normal"]        = round(atr * 0.40, 2)

    # ML regime
    ml_info = None
    if ML_AVAILABLE:
        try:
            ml = run_ml_analysis(ticker, daily, df5m, levels=[], lookback=30)
            if ml and not ml.get("error"):
                bm = ml.get("bias_model", {}) or {}
                pb = bm.get("proba_bull", 0.5) or 0.5
                ml_info = {
                    "regimen":     ml.get("regime", {}).get("label"),
                    "prob_alcista": round(pb, 3),
                    "sesgo":       "ALCISTA" if pb > 0.55 else "BAJISTA" if pb < 0.45 else "NEUTRAL",
                    "anomalia":    ml.get("anomaly", {}).get("is_anomaly", False),
                }
        except Exception:
            pass

    # Pattern DB score
    db_rec = None
    if DB_AVAILABLE:
        try:
            db_stats = get_stats(ticker)
            if db_stats.get("n", 0) > 0:
                db_rec = {
                    "dias_registrados": db_stats["n"],
                    "call_win_rate":    db_stats.get("calls", {}).get("win_rate", 0),
                    "put_win_rate":     db_stats.get("puts", {}).get("win_rate", 0),
                    "bull_rate":        db_stats.get("bull_rate", 0),
                }
        except Exception:
            pass

    return json.dumps({
        "ticker":        ticker,
        "fecha":         gap.get("date", str(dt.date.today())),
        "contexto":      ctx,
        "gap_hoy":       gap,
        "prob_fill_gap": f"{fill_prob}%",
        "orb_5m":        orb_5,
        "orb_15m":       orb_15,
        "niveles_clave": niveles,
        "ml":            ml_info,
        "base_datos":    db_rec,
        "resumen": {
            "tendencia":   ctx.get("trend"),
            "gap":         f"{gap.get('direction')} {gap.get('size_cat')} ({gap.get('gap_pct', 0):+.2f}%)",
            "fill_prob":   f"{fill_prob}%",
            "sesgo_orb":   f"↑ {orb_15.get('up_pct', 0)}%  ↓ {orb_15.get('dn_pct', 0)}%",
            "racha":       f"{ctx.get('streak', 0)}d {ctx.get('streak_dir', '')}",
            "sesgo_ml":    ml_info.get("sesgo") if ml_info else "N/A",
        },
    }, indent=2)


@tool
def noticias(ticker: str) -> str:
    """
    Obtiene los últimos titulares de noticias para un ticker desde Yahoo Finance.
    Úsala cuando pregunten sobre noticias, catalizadores o eventos recientes.

    Args:
        ticker: Símbolo bursátil, ej: 'SPY', 'AAPL', 'TSLA'
    """
    ticker = ticker.upper().strip()
    return json.dumps({
        "ticker":     ticker,
        "titulares":  fetch_news(ticker),
        "obtenido":   dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }, indent=2)


@tool
def comparar_tickers(ticker1: str, ticker2: str) -> str:
    """
    Compara dos tickers en paralelo: tendencia, ATR, gap de hoy,
    sesgo ORB 15m y posición respecto a EMAs.

    Args:
        ticker1: Primer símbolo, ej: 'SPY'
        ticker2: Segundo símbolo, ej: 'QQQ'
    """
    resultado = {}
    for t in [ticker1.upper(), ticker2.upper()]:
        daily = fetch_daily(t, years=1)
        if daily.empty:
            resultado[t] = {"error": "Sin datos"}
            continue
        ctx  = get_market_context(daily)
        gap  = get_today_gap(daily)
        df5m = fetch_5m(t, days=30)
        orb  = compute_orb_stats(df5m, 15) if not df5m.empty else {}
        resultado[t] = {
            "precio":        ctx.get("last_price"),
            "tendencia":     ctx.get("trend"),
            "atr":           ctx.get("atr"),
            "gap_hoy":       f"{gap.get('gap_pct', 0):+.2f}% {gap.get('direction')}",
            "orb_15m_arriba": orb.get("up_pct"),
            "orb_15m_abajo":  orb.get("dn_pct"),
            "sobre_ema200":  ctx.get("above_ema200"),
            "sobre_ema20":   ctx.get("above_ema20"),
        }
    return json.dumps(resultado, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE
# ─────────────────────────────────────────────────────────────────────────────

def crear_agente() -> Agent:
    return Agent(
        name="ORB Trading AI Agent",
        provider=Groq(id="llama-3.3-70b-versatile"),
        agent_id="orb-trading-agent",
        session_id="default",
        tools=[
            analizar_ticker,
            analisis_gap,
            analisis_orb,
            prediccion_ml,
            stats_base_datos,
            plan_del_dia,
            noticias,
            comparar_tickers,
        ],
        description=(
            "Eres un trader experto en estrategias de Opening Range Breakout (ORB), "
            "análisis de gaps y detección de regímenes de mercado con Machine Learning. "
            "Usas datos reales de Yahoo Finance, un ML engine propio (Random Forest + KMeans) "
            "y una base de datos de patrones históricos para generar planes de trading precisos y accionables."
        ),
        instructions=[
            "SIEMPRE busca datos reales antes de responder — nunca inventes precios o estadísticas.",
            "Para planes completos usa plan_del_dia primero, luego enriquece con prediccion_ml.",
            "Presenta siempre: dirección del gap + prob fill, sesgo ORB, niveles clave, targets ATR.",
            "Incluye el régimen ML y stats de la base de datos cuando estén disponibles.",
            "Sé conciso y accionable — da precios específicos de entrada/target/stop.",
            "Responde en español siempre.",
            "Termina siempre con: ⚠ Las estadísticas pasadas no garantizan resultados futuros.",
        ],
        add_chat_history_to_messages=True,
        markdown=True,
        show_tool_calls=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND HTMX
# ─────────────────────────────────────────────────────────────────────────────

HTML_UI = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORB AI Agent</title>
<script src="https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111318;color:#e8ecf2;font-family:'Inter',system-ui,sans-serif;min-height:100vh;display:flex;flex-direction:column}
.topbar{background:#1c1f27;border-bottom:1px solid #2e3340;padding:14px 28px;display:flex;align-items:center;gap:14px;flex-shrink:0}
.brand{font-family:monospace;font-size:1.05rem;font-weight:700;color:#5b8af5;letter-spacing:.05em}
.sub{font-size:.65rem;color:#636b7a;text-transform:uppercase;letter-spacing:.1em}
.badge{padding:3px 10px;border-radius:20px;font-size:.62rem;font-weight:700;
       background:rgba(52,209,122,.12);color:#34d17a;border:1px solid rgba(52,209,122,.25)}
.ml-badge{background:rgba(91,138,245,.12);color:#5b8af5;border-color:rgba(91,138,245,.25)}
.chat-wrap{flex:1;overflow-y:auto;padding:20px}
#chat-log{max-width:880px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.msg{padding:14px 18px;border-radius:12px;font-size:.875rem;line-height:1.75;max-width:88%}
.msg.user{background:#1c1f27;border:1px solid #2e3340;align-self:flex-end;color:#e8ecf2}
.msg.agent{background:#16191f;border:1px solid #2e3340;align-self:flex-start;color:#9ba3b2;white-space:pre-wrap}
.msg.agent b,.msg.agent strong{color:#e8ecf2}
.msg.agent code{background:#1c1f27;padding:1px 6px;border-radius:3px;
                font-family:monospace;font-size:.8rem;color:#5b8af5}
.thinking{color:#636b7a;font-style:italic}
.thinking::after{content:'...';animation:dots 1.2s steps(3,end) infinite}
@keyframes dots{0%{content:''}33%{content:'.'}66%{content:'..'}100%{content:'...'}}
.input-bar{background:#111318;border-top:1px solid #2e3340;padding:14px 20px;flex-shrink:0}
.quick{max-width:880px;margin:0 auto 10px;display:flex;gap:7px;flex-wrap:wrap}
.q{background:#1c1f27;border:1px solid #2e3340;color:#9ba3b2;border-radius:16px;
   padding:5px 13px;font-size:.7rem;cursor:pointer;transition:all .15s}
.q:hover{border-color:#5b8af5;color:#e8ecf2}
.input-row{max-width:880px;margin:0 auto;display:flex;gap:10px}
textarea{flex:1;background:#1c1f27;border:1px solid #363c4a;color:#e8ecf2;
         border-radius:8px;padding:11px 15px;font-size:.875rem;resize:none;
         font-family:inherit;outline:none;transition:border-color .2s}
textarea:focus{border-color:#5b8af5}
textarea::placeholder{color:#636b7a}
.send{background:#5b8af5;color:#fff;border:none;border-radius:8px;
      padding:11px 20px;font-weight:600;cursor:pointer;font-size:.8rem;transition:background .15s}
.send:hover{background:#4a79e8}
.send:disabled{opacity:.5;cursor:not-allowed}
</style>
</head>
<body>
<div class="topbar">
  <div>
    <div class="brand">ORB AI Agent</div>
    <div class="sub">phidata · GPT-4o-mini · ml_engine · trading_db</div>
  </div>
  <span class="badge">● Live</span>
  <span class="badge ml-badge" id="ml-status">🧠 ML</span>
  <span class="badge ml-badge" id="db-status">🗄 DB</span>
</div>

<div class="chat-wrap">
  <div id="chat-log">
    <div class="msg agent">👋 <b>ORB Trading AI Agent</b> — listo para analizar mercados en tiempo real.

Pregúntame sobre cualquier ticker:
  • <code>Dame un plan para SPY hoy</code>
  • <code>¿Cuál es la probabilidad de fill del gap de QQQ?</code>
  • <code>Análisis ML de AAPL</code>
  • <code>Compara SPY y QQQ</code>
  • <code>¿Qué noticias hay de NVDA?</code>
  • <code>Stats de la base de datos para SPY</code></div>
  </div>
</div>

<div class="input-bar">
  <div class="quick">
    <button class="q" onclick="ask('Dame un plan completo para SPY hoy')">📊 Plan SPY</button>
    <button class="q" onclick="ask('Análisis ML de QQQ')">🧠 ML QQQ</button>
    <button class="q" onclick="ask('¿Cuál es el gap de SPY hoy y se llenará?')">📈 Gap SPY</button>
    <button class="q" onclick="ask('Estadísticas ORB de QQQ')">⏱ ORB QQQ</button>
    <button class="q" onclick="ask('Compara SPY y QQQ')">⚖️ Comparar</button>
    <button class="q" onclick="ask('Noticias de AAPL')">📰 AAPL News</button>
    <button class="q" onclick="ask('Stats de la base de datos para SPY')">🗄 DB SPY</button>
  </div>
  <div class="input-row">
    <textarea id="inp" rows="2" placeholder="Escribe tu pregunta sobre cualquier ticker..."
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
    <button class="send" id="btn" onclick="send()">Enviar ↵</button>
  </div>
</div>

<script>
function ask(q){ document.getElementById('inp').value=q; send(); }

function send(){
  const inp = document.getElementById('inp');
  const msg = inp.value.trim();
  if(!msg) return;
  inp.value = '';

  const log = document.getElementById('chat-log');

  // Mensaje del usuario
  log.innerHTML += `<div class="msg user">${esc(msg)}</div>`;

  // Estado "pensando"
  const tid = 'think_' + Date.now();
  log.innerHTML += `<div class="msg agent thinking" id="${tid}">🤖 Analizando con datos reales</div>`;
  scroll();

  document.getElementById('btn').disabled = true;

  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'message=' + encodeURIComponent(msg)
  })
  .then(r => r.text())
  .then(html => {
    document.getElementById(tid)?.remove();
    log.innerHTML += html;
    scroll();
  })
  .catch(e => {
    if(document.getElementById(tid))
      document.getElementById(tid).textContent = '❌ Error: ' + e;
  })
  .finally(() => {
    document.getElementById('btn').disabled = false;
  });
}

function scroll(){ 
  const w = document.querySelector('.chat-wrap');
  w.scrollTop = w.scrollHeight;
}

function esc(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>
"""

def md_to_html(text: str) -> str:
    """Convierte markdown básico a HTML seguro."""
    import re
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`",        r"<code>\1</code>", text)
    text = re.sub(r"^#{1,3} (.+)$",  r"<b>\1</b>", text, flags=re.MULTILINE)
    return text

def iniciar_servidor(port: int = 7860):
    try:
        from fastapi import FastAPI, Form
        from fastapi.responses import HTMLResponse
        import uvicorn
    except ImportError:
        print("Instala: pip install fastapi uvicorn")
        return

    app   = FastAPI()
    agent = crear_agente()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(HTML_UI)

    @app.post("/chat", response_class=HTMLResponse)
    async def chat(message: str = Form(...)):
        try:
            run  = agent.run(message, stream=False)
            text = run.content if hasattr(run, "content") else str(run)
        except Exception as e:
            text = f"❌ Error: {e}"
        return HTMLResponse(f'<div class="msg agent">{md_to_html(text)}</div>')

    print(f"\n{'='*52}")
    print(f"  ORB AI Agent — Web UI HTMX")
    print(f"  http://localhost:{port}")
    print(f"  ml_engine.py : {'✅ activo' if ML_AVAILABLE else '❌ no encontrado'}")
    print(f"  trading_db.py: {'✅ activo' if DB_AVAILABLE else '❌ no encontrado'}")
    print(f"{'='*52}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="ORB Trading AI Agent")
    p.add_argument("--ticker", help="Ticker a analizar")
    p.add_argument("--plan",   action="store_true", help="Plan completo del día")
    p.add_argument("--ml",     action="store_true", help="Análisis ML")
    p.add_argument("--news",   action="store_true", help="Noticias")
    p.add_argument("--db",     action="store_true", help="Stats base de datos")
    p.add_argument("--server", action="store_true", help="Iniciar servidor HTMX")
    p.add_argument("--port",   type=int, default=7860)
    args = p.parse_args()

    if args.server:
        iniciar_servidor(args.port)
        return

    agent = crear_agente()

    if args.ticker and args.plan:
        agent.print_response(
            f"Dame un plan de trading completo para {args.ticker.upper()} hoy. "
            f"Incluye: gap, probabilidad de fill, sesgo ORB 15m, análisis ML si disponible, "
            f"y niveles específicos de entrada/target/stop.",
            stream=True)

    elif args.ticker and args.ml:
        agent.print_response(
            f"Ejecuta el análisis ML completo de {args.ticker.upper()}. "
            f"Muestra régimen, sesgo alcista/bajista, días similares y patrones.",
            stream=True)

    elif args.ticker and args.news:
        agent.print_response(
            f"Obtén las últimas noticias de {args.ticker.upper()} y explica el impacto en el precio.",
            stream=True)

    elif args.ticker and args.db:
        agent.print_response(
            f"Muestra las estadísticas de la base de datos de patrones para {args.ticker.upper()}.",
            stream=True)

    elif args.ticker:
        agent.print_response(
            f"Analiza {args.ticker.upper()}: tendencia, gap de hoy, sesgo ORB y dirección probable.",
            stream=True)

    else:
        # Chat interactivo
        print(f"\n{'='*55}")
        print(f"  ORB Trading AI Agent  |  escribe 'salir' para terminar")
        print(f"  ml_engine.py : {'✅' if ML_AVAILABLE else '❌ no encontrado'}")
        print(f"  trading_db.py: {'✅' if DB_AVAILABLE else '❌ no encontrado'}")
        print(f"{'='*55}")
        print("\nEjemplos:")
        print("  Dame un plan para SPY hoy")
        print("  Análisis ML de QQQ")
        print("  ¿Se llenará el gap de AAPL?")
        print("  Compara SPY y QQQ\n")

        while True:
            try:
                q = input("Tú: ").strip()
                if q.lower() in ("salir", "exit", "quit", "q"):
                    print("¡Hasta luego!"); break
                if not q: continue
                print()
                agent.print_response(q, stream=True)
                print()
            except KeyboardInterrupt:
                print("\n¡Hasta luego!"); break

if __name__ == "__main__":
    main()