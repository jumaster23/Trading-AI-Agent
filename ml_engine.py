"""
ml_engine.py — Professional ML Pattern Engine
==============================================
30-day rolling window analysis. Finds repeating patterns in price action,
classifies market regimes, predicts intraday bias, identifies high-probability
setups using real machine learning — not heuristics.

Architecture:
  1. Feature Engineering  — 40+ features from 30d daily + intraday data
  2. Regime Classification — KMeans clusters market states (4 regimes)
  3. Pattern Matching      — finds similar historical days via cosine similarity
  4. Bias Prediction       — Random Forest P(bull day) for next session
  5. Level Intelligence    — which S/R levels held/failed in current regime
  6. Anomaly Detection     — flags abnormal days (outlier factor)

All models are retrained on every call (30-day window = ~21 bars).
No external model files. No lookahead.
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ── sklearn imports ────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────


def _ema(series: pd.Series, span: int) -> float:
    """EMA of a series, returns last value."""
    if len(series) < 2:
        return float(series.iloc[-1])
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    """Average True Range."""
    hi = df["High"].astype(float)
    lo = df["Low"].astype(float)
    cl = df["Close"].astype(float)
    tr = pd.concat(
        [hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1
    ).max(axis=1)
    return float(tr.tail(n).mean())


def _rsi(series: pd.Series, n: int = 14) -> float:
    """RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def _bb_position(series: pd.Series, n: int = 20) -> float:
    """Where is price within Bollinger Bands? 0=lower, 0.5=mid, 1=upper."""
    m = series.rolling(n).mean().iloc[-1]
    s = series.rolling(n).std().iloc[-1]
    if s < 1e-9:
        return 0.5
    return float(np.clip((series.iloc[-1] - (m - 2 * s)) / (4 * s), 0, 1))


def _vol_zscore(df: pd.DataFrame, n: int = 20) -> float:
    """Volume z-score: how unusual is today's volume."""
    vol = df["Volume"].astype(float)
    if len(vol) < n + 1:
        return 0.0
    mu = vol.iloc[:-1].tail(n).mean()
    sig = vol.iloc[:-1].tail(n).std() + 1e-9
    return float((vol.iloc[-1] - mu) / sig)


def build_features_for_day(
    daily: pd.DataFrame,
    df5m: Optional[pd.DataFrame],
    target_date: pd.Timestamp,
    lookback: int = 30,
) -> Optional[dict]:
    """
    Build the full feature vector for ONE day.
    Uses only data BEFORE target_date (no lookahead).

    Returns dict of features, or None if insufficient data.
    """
    past = daily[daily.index < target_date].tail(lookback + 5)
    if len(past) < 10:
        return None

    cl = past["Close"].astype(float)
    op = past["Open"].astype(float)
    hi = past["High"].astype(float)
    lo = past["Low"].astype(float)
    vo = (
        past["Volume"].astype(float)
        if "Volume" in past.columns
        else pd.Series([1.0] * len(past))
    )

    last = float(cl.iloc[-1])
    atr = _atr(past) + 1e-9

    # ── Price momentum features ─────────────────────────────────────────────
    ret1 = (cl.iloc[-1] / cl.iloc[-2] - 1) if len(cl) >= 2 else 0.0
    ret3 = (cl.iloc[-1] / cl.iloc[-4] - 1) if len(cl) >= 4 else ret1
    ret5 = (cl.iloc[-1] / cl.iloc[-6] - 1) if len(cl) >= 6 else ret1
    ret10 = (cl.iloc[-1] / cl.iloc[-11] - 1) if len(cl) >= 11 else ret5
    ret20 = (cl.iloc[-1] / cl.iloc[-21] - 1) if len(cl) >= 21 else ret10

    # Normalize by ATR so QQQ@600 and SPY@500 are comparable
    ret1_atr = ret1 * last / atr
    ret5_atr = ret5 * last / atr
    ret10_atr = ret10 * last / atr

    # ── EMAs ────────────────────────────────────────────────────────────────
    ema8 = _ema(cl, 8)
    ema21 = _ema(cl, 21)
    ema50 = _ema(cl, min(50, len(cl) - 1))
    ema200 = _ema(cl, min(200, len(cl) - 1))

    ema8_dist = (last - ema8) / atr
    ema21_dist = (last - ema21) / atr
    ema50_dist = (last - ema50) / atr
    ema200_dist = (last - ema200) / atr

    ema_align_bull = 1.0 if (ema8 > ema21 > ema50) else 0.0
    ema_align_bear = 1.0 if (ema8 < ema21 < ema50) else 0.0

    # ── Oscillators ─────────────────────────────────────────────────────────
    rsi14 = _rsi(cl, 14)
    rsi5 = _rsi(cl, 5)
    bb_pos = _bb_position(cl, min(20, len(cl) - 1))

    # ── ATR / Volatility regime ─────────────────────────────────────────────
    atr5 = _atr(past.tail(5), 5)
    atr20 = _atr(past.tail(20), 14) if len(past) >= 20 else atr
    atr_ratio = atr5 / (atr20 + 1e-9)  # >1 = expanding, <1 = compressing

    # ── Candle structure ────────────────────────────────────────────────────
    # Last day's candle body vs range
    last_body = abs(float(cl.iloc[-1]) - float(op.iloc[-1])) / (atr + 1e-9)
    last_wick_up = (
        float(hi.iloc[-1]) - max(float(cl.iloc[-1]), float(op.iloc[-1]))
    ) / (atr + 1e-9)
    last_wick_down = (
        min(float(cl.iloc[-1]), float(op.iloc[-1])) - float(lo.iloc[-1])
    ) / (atr + 1e-9)
    last_bull = 1.0 if float(cl.iloc[-1]) > float(op.iloc[-1]) else 0.0

    # Consecutive direction
    if len(cl) >= 3:
        dirs = (cl.diff() > 0).astype(int).values
        consec = 0
        last_dir = dirs[-1]
        for d in reversed(dirs[:-1]):
            if d == last_dir:
                consec += 1
            else:
                break
        consec_signed = consec if last_dir else -consec
    else:
        consec_signed = 0.0

    # ── Gap analysis ────────────────────────────────────────────────────────
    gap_pct = (
        (float(op.iloc[-1]) - float(cl.iloc[-2])) / (float(cl.iloc[-2]) + 1e-9)
        if len(cl) >= 2
        else 0.0
    )
    gap_atr = gap_pct * last / atr

    # Historical gap fill rate (last 30 days)
    gaps_filled = 0
    gap_count = 0
    for i in range(1, min(len(past), 31)):
        g = float(op.iloc[-i]) - float(cl.iloc[-i - 1]) if (len(cl) > i) else 0.0
        if abs(g) / float(cl.iloc[-i - 1] + 1e-9) > 0.001:
            gap_count += 1
            if g > 0 and float(lo.iloc[-i]) <= float(cl.iloc[-i - 1]):
                gaps_filled += 1
            elif g < 0 and float(hi.iloc[-i]) >= float(cl.iloc[-i - 1]):
                gaps_filled += 1
    gap_fill_rate = gaps_filled / max(gap_count, 1)

    # ── Volume ──────────────────────────────────────────────────────────────
    vol_z = _vol_zscore(past)
    vol_trend = (
        (float(vo.tail(5).mean()) / (float(vo.tail(20).mean()) + 1e-9) - 1)
        if len(vo) >= 20
        else 0.0
    )

    # ── Structure: HH/HL or LH/LL ─────────────────────────────────────────
    if len(hi) >= 5:
        hh5 = 1.0 if (hi.iloc[-1] > hi.iloc[-2] > hi.iloc[-3]) else 0.0
        ll5 = 1.0 if (lo.iloc[-1] < lo.iloc[-2] < lo.iloc[-3]) else 0.0
        hl5 = 1.0 if (lo.iloc[-1] > lo.iloc[-3]) else 0.0
        lh5 = 1.0 if (hi.iloc[-1] < hi.iloc[-3]) else 0.0
    else:
        hh5 = ll5 = hl5 = lh5 = 0.5

    # ── Distance from 30d range extremes ──────────────────────────────────
    range_hi = float(hi.tail(30).max())
    range_lo = float(lo.tail(30).min())
    range_rng = range_hi - range_lo + 1e-9
    pos_in_range = (last - range_lo) / range_rng  # 0=bottom, 1=top

    # Proximity to 30d high/low (normalized by ATR)
    dist_30h = (range_hi - last) / atr
    dist_30l = (last - range_lo) / atr

    # ── Day-of-week effects ─────────────────────────────────────────────────
    dow = target_date.dayofweek  # 0=Mon, 4=Fri
    dow_mon = 1.0 if dow == 0 else 0.0
    dow_fri = 1.0 if dow == 4 else 0.0

    # ── Month position ─────────────────────────────────────────────────────
    month_pos = target_date.day / 31.0  # 0=start, 1=end

    # ── 5m intraday features (if available) ────────────────────────────────
    orb5_range = orb5_broke_up = orb5_broke_down = orb15_broke_up = orb15_broke_down = (
        0.0
    )
    intra_vol_z = intra_trend = 0.0

    if df5m is not None and not df5m.empty:
        prev_dates = sorted(
            [d for d in set(df5m.index.date) if pd.Timestamp(d) < target_date]
        )
        if len(prev_dates) >= 3:
            # Last 10 intraday sessions for pattern
            recent_dates = prev_dates[-10:]
            orb5_breaks_up = 0
            orb5_breaks_dn = 0
            orb15_breaks_up = 0
            orb15_breaks_dn = 0
            orb5_ranges = []
            for d in recent_dates:
                day_bars = df5m[df5m.index.date == d].sort_index()
                if len(day_bars) < 6:
                    continue
                oh5 = float(day_bars["High"].iloc[:1].max())
                ol5 = float(day_bars["Low"].iloc[:1].min())
                oh15 = float(day_bars["High"].iloc[:3].max())
                ol15 = float(day_bars["Low"].iloc[:3].min())
                rng_open = float(day_bars["Open"].iloc[0]) + 1e-9
                orb5_ranges.append((oh5 - ol5) / rng_open * 100)
                rest5 = day_bars.iloc[1:]
                rest15 = day_bars.iloc[3:]
                if not rest5.empty:
                    if float(rest5["High"].max()) > oh5:
                        orb5_breaks_up += 1
                    if float(rest5["Low"].min()) < ol5:
                        orb5_breaks_dn += 1
                if not rest15.empty:
                    if float(rest15["High"].max()) > oh15:
                        orb15_breaks_up += 1
                    if float(rest15["Low"].min()) < ol15:
                        orb15_breaks_dn += 1
            n_sess = len(recent_dates)
            orb5_broke_up = orb5_breaks_up / max(n_sess, 1)
            orb5_broke_down = orb5_breaks_dn / max(n_sess, 1)
            orb15_broke_up = orb15_breaks_up / max(n_sess, 1)
            orb15_broke_down = orb15_breaks_dn / max(n_sess, 1)
            if orb5_ranges:
                orb5_range = float(np.mean(orb5_ranges)) / (atr / last * 100 + 1e-9)

    return {
        # Momentum (ATR-normalized)
        "ret1_atr": float(ret1_atr),
        "ret5_atr": float(ret5_atr),
        "ret10_atr": float(ret10_atr),
        "ret20_raw": float(ret20),
        # EMA distances
        "ema8_dist": float(ema8_dist),
        "ema21_dist": float(ema21_dist),
        "ema50_dist": float(ema50_dist),
        "ema200_dist": float(ema200_dist),
        "ema_align_bull": float(ema_align_bull),
        "ema_align_bear": float(ema_align_bear),
        # Oscillators
        "rsi14": float(rsi14 / 100.0),
        "rsi5": float(rsi5 / 100.0),
        "bb_pos": float(bb_pos),
        # Volatility
        "atr_ratio": float(atr_ratio),
        # Candle structure
        "last_body": float(last_body),
        "last_wick_up": float(last_wick_up),
        "last_wick_dn": float(last_wick_down),
        "last_bull": float(last_bull),
        "consec_signed": float(np.clip(consec_signed, -6, 6) / 6.0),
        # Gap
        "gap_atr": float(np.clip(gap_atr, -3, 3)),
        "gap_fill_rate": float(gap_fill_rate),
        # Volume
        "vol_z": float(np.clip(vol_z, -3, 3)),
        "vol_trend": float(np.clip(vol_trend, -1, 1)),
        # Structure
        "hh5": float(hh5),
        "ll5": float(ll5),
        "hl5": float(hl5),
        "lh5": float(lh5),
        # Range position
        "pos_in_range": float(pos_in_range),
        "dist_30h": float(np.clip(dist_30h, 0, 10)),
        "dist_30l": float(np.clip(dist_30l, 0, 10)),
        # Calendar
        "dow_mon": float(dow_mon),
        "dow_fri": float(dow_fri),
        "month_pos": float(month_pos),
        # Intraday patterns
        "orb5_range": float(orb5_range),
        "orb5_broke_up": float(orb5_broke_up),
        "orb5_broke_down": float(orb5_broke_down),
        "orb15_broke_up": float(orb15_broke_up),
        "orb15_broke_down": float(orb15_broke_down),
    }


def build_feature_matrix(
    daily: pd.DataFrame, df5m: Optional[pd.DataFrame], lookback: int = 30
) -> pd.DataFrame:
    """
    Build feature matrix for ALL days in the last `lookback` days.
    Each row = one day's feature vector.
    """
    dates = daily.index[-lookback - 2 :]  # extra buffer
    rows = []
    for dt in dates:
        feats = build_features_for_day(daily, df5m, dt, lookback=lookback + 10)
        if feats is None:
            continue
        # Label: next day bull (1) or bear (0)
        fut_idx = daily.index.get_loc(dt)
        if fut_idx + 1 < len(daily):
            next_cl = float(daily["Close"].iloc[fut_idx + 1])
            next_op = float(daily["Open"].iloc[fut_idx + 1])
            label = 1 if next_cl > next_op else 0
        else:
            label = -1  # unknown (last row)
        feats["label"] = label
        feats["date"] = dt
        feats["close"] = float(daily.loc[dt, "Close"])
        feats["open"] = float(daily.loc[dt, "Open"])
        feats["high"] = float(daily.loc[dt, "High"])
        feats["low"] = float(daily.loc[dt, "Low"])
        rows.append(feats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION (KMeans)
# ─────────────────────────────────────────────────────────────────────────────

REGIME_NAMES = {
    0: (
        "TREND_BULL",
        "Tendencia Alcista",
        "var(--up)",
        "Días de momentum comprador sostenido",
    ),
    1: (
        "TREND_BEAR",
        "Tendencia Bajista",
        "var(--down)",
        "Días de momentum vendedor sostenido",
    ),
    2: (
        "RANGE_CHOP",
        "Rango / Chop",
        "var(--text3)",
        "Mercado lateral, sin dirección clara",
    ),
    3: (
        "VOL_SPIKE",
        "Spike Volátil",
        "var(--both)",
        "Días de expansión de volatilidad, movimientos bruscos",
    ),
}


def classify_regimes(feat_matrix: pd.DataFrame, n_regimes: int = 4) -> tuple:
    """
    KMeans clustering on feature matrix.
    Returns (labels array, cluster_centers, scaler, kmeans_model).
    """
    REGIME_FEATURES = [
        "ret1_atr",
        "ret5_atr",
        "ret10_atr",
        "ema8_dist",
        "ema21_dist",
        "rsi14",
        "bb_pos",
        "atr_ratio",
        "last_bull",
        "consec_signed",
        "vol_z",
        "pos_in_range",
    ]
    cols = [c for c in REGIME_FEATURES if c in feat_matrix.columns]
    X = feat_matrix[cols].fillna(0.0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10, max_iter=300)
    km.fit(Xs)
    labels = km.labels_

    # Identify regime semantics by cluster mean on ret1_atr and atr_ratio
    cluster_stats = {}
    ret_idx = cols.index("ret1_atr") if "ret1_atr" in cols else 0
    atr_idx = cols.index("atr_ratio") if "atr_ratio" in cols else 0
    bull_idx = cols.index("last_bull") if "last_bull" in cols else 0

    for c in range(n_regimes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mu_ret = float(Xs[mask, ret_idx].mean())
        mu_atr = float(Xs[mask, atr_idx].mean())
        mu_bull = float(X[mask, bull_idx].mean())
        cluster_stats[c] = {
            "ret": mu_ret,
            "atr": mu_atr,
            "bull": mu_bull,
            "count": int(mask.sum()),
        }

    # Map clusters to semantic regimes
    # Sort by return (most negative = TREND_BEAR, most positive = TREND_BULL)
    sorted_by_ret = sorted(cluster_stats.keys(), key=lambda c: cluster_stats[c]["ret"])
    mapping = {}
    if len(sorted_by_ret) >= 4:
        mapping[sorted_by_ret[0]] = 1  # TREND_BEAR
        mapping[sorted_by_ret[-1]] = 0  # TREND_BULL
        # Among middle two, higher atr = VOL_SPIKE
        mid = sorted_by_ret[1:-1]
        mid_sorted = sorted(mid, key=lambda c: cluster_stats[c]["atr"])
        mapping[mid_sorted[0]] = 2  # RANGE_CHOP
        mapping[mid_sorted[-1]] = 3  # VOL_SPIKE
    else:
        for i, c in enumerate(sorted_by_ret):
            mapping[c] = min(i, 3)

    semantic_labels = np.array([mapping.get(l, 2) for l in labels])
    return semantic_labels, cluster_stats, mapping, scaler, km, cols


# ─────────────────────────────────────────────────────────────────────────────
# BIAS PREDICTION (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────


def train_bias_model(feat_matrix: pd.DataFrame) -> dict:
    """
    Train Random Forest to predict next-day bull/bear.
    Uses rolling 30-day window features as training data.
    Returns model, accuracy, feature importances.
    """
    PRED_FEATURES = [
        "ret1_atr",
        "ret5_atr",
        "ret10_atr",
        "ema8_dist",
        "ema21_dist",
        "ema50_dist",
        "rsi14",
        "rsi5",
        "bb_pos",
        "atr_ratio",
        "last_body",
        "last_bull",
        "consec_signed",
        "gap_atr",
        "gap_fill_rate",
        "vol_z",
        "vol_trend",
        "pos_in_range",
        "dist_30h",
        "dist_30l",
        "hh5",
        "ll5",
        "orb5_broke_up",
        "orb5_broke_down",
        "dow_mon",
        "dow_fri",
    ]
    cols = [c for c in PRED_FEATURES if c in feat_matrix.columns]
    train = feat_matrix[feat_matrix["label"].isin([0, 1])].copy()

    if len(train) < 8:
        return {"model": None, "accuracy": None, "proba_bull": 0.5, "features": cols}

    X = train[cols].fillna(0.0).values
    y = train["label"].values

    # RandomForest with conservative settings (small dataset)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,  # avoid overfit on 20-30 bars
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X, y)

    # CV accuracy
    try:
        cv_scores = cross_val_score(
            rf, X, y, cv=min(5, len(train) // 4), scoring="accuracy"
        )
        accuracy = float(cv_scores.mean())
    except:
        accuracy = None

    # Predict on LAST row (today's features → tomorrow's probability)
    today_row = feat_matrix[feat_matrix["label"] == -1]
    if today_row.empty:
        today_row = feat_matrix.iloc[[-1]]
    X_today = today_row[cols].fillna(0.0).values
    try:
        proba = float(rf.predict_proba(X_today)[0][1])
    except:
        proba = 0.5

    # Feature importances
    importances = dict(zip(cols, rf.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]

    # Gradient Boosting for second opinion
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    try:
        gb.fit(X, y)
        proba_gb = float(gb.predict_proba(X_today)[0][1])
    except:
        proba_gb = proba

    # Ensemble: average RF + GB
    proba_ensemble = proba * 0.6 + proba_gb * 0.4

    return {
        "model": rf,
        "model_gb": gb,
        "accuracy": accuracy,
        "proba_bull": round(proba_ensemble, 3),
        "proba_rf": round(proba, 3),
        "proba_gb": round(proba_gb, 3),
        "top_features": top_features,
        "n_train": len(train),
        "feature_cols": cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN MATCHING (Cosine Similarity)
# ─────────────────────────────────────────────────────────────────────────────


def find_similar_days(feat_matrix: pd.DataFrame, top_k: int = 5) -> list:
    """
    Find the K most similar historical days to TODAY using cosine similarity.
    Returns list of dicts with date, similarity, outcome, and stats.
    """
    SIM_FEATURES = [
        "ret1_atr",
        "ret5_atr",
        "ret10_atr",
        "ema8_dist",
        "ema21_dist",
        "rsi14",
        "bb_pos",
        "atr_ratio",
        "consec_signed",
        "vol_z",
        "pos_in_range",
        "gap_atr",
        "hh5",
        "ll5",
        "last_bull",
        "orb5_broke_up",
        "orb5_broke_down",
    ]
    cols = [c for c in SIM_FEATURES if c in feat_matrix.columns]

    today_mask = feat_matrix["label"] == -1
    if not today_mask.any():
        today_mask = feat_matrix.index == feat_matrix.index[-1]

    hist_mask = feat_matrix["label"].isin([0, 1])
    if hist_mask.sum() < 3:
        return []

    today_vec = feat_matrix.loc[today_mask, cols].fillna(0.0).values
    hist_vecs = feat_matrix.loc[hist_mask, cols].fillna(0.0).values
    hist_dates = feat_matrix.loc[hist_mask].index

    scaler = StandardScaler()
    all_vecs = np.vstack([hist_vecs, today_vec])
    scaled = scaler.fit_transform(all_vecs)
    hist_sc = scaled[:-1]
    today_sc = scaled[[-1]]

    sims = cosine_similarity(today_sc, hist_sc)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        dt = hist_dates[idx]
        sim = float(sims[idx])
        lbl = int(feat_matrix.loc[dt, "label"])
        cl = float(feat_matrix.loc[dt, "close"])
        op = float(feat_matrix.loc[dt, "open"])
        hi = float(feat_matrix.loc[dt, "high"])
        lo = float(feat_matrix.loc[dt, "low"])
        day_ret = (cl - op) / op * 100
        day_rng = (hi - lo) / op * 100
        results.append(
            {
                "date": str(dt)[:10],
                "similarity": sim,
                "bull": lbl == 1,
                "day_ret": round(day_ret, 2),
                "day_range": round(day_rng, 2),
                "close": round(cl, 2),
                "open": round(op, 2),
            }
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────


def detect_anomaly(feat_matrix: pd.DataFrame) -> dict:
    """
    Local Outlier Factor to flag if today looks abnormally different
    from the recent 30-day distribution.
    """
    ANOMALY_FEATURES = [
        "ret1_atr",
        "ret5_atr",
        "atr_ratio",
        "vol_z",
        "rsi14",
        "bb_pos",
        "consec_signed",
        "pos_in_range",
    ]
    cols = [c for c in ANOMALY_FEATURES if c in feat_matrix.columns]
    X = feat_matrix[cols].fillna(0.0).values

    if len(X) < 8:
        return {"is_anomaly": False, "score": 0.0, "note": "Insuficiente historia"}

    # LOF: novelty=True allows predicting on new point
    lof = LocalOutlierFactor(
        n_neighbors=min(5, len(X) - 1), novelty=False, contamination=0.15
    )
    lof_scores = lof.fit_predict(X)
    lof_nfact = lof.negative_outlier_factor_

    # Today = last row
    today_score = float(lof_nfact[-1])
    today_label = int(lof_scores[-1])
    is_anomaly = today_label == -1
    percentile = float(np.mean(lof_nfact > today_score) * 100)

    return {
        "is_anomaly": is_anomaly,
        "score": round(today_score, 3),
        "percentile": round(percentile, 1),
        "note": (
            "⚠ Día anómalo — comportamiento fuera del rango normal de los últimos 30 días"
            if is_anomaly
            else "Comportamiento dentro del patrón normal"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL INTELLIGENCE: which S/R held/failed in current regime
# ─────────────────────────────────────────────────────────────────────────────


def analyze_level_behavior(
    daily: pd.DataFrame,
    levels: list,
    regime_labels: np.ndarray,
    current_regime: int,
    lookback: int = 30,
) -> list:
    """
    For each level, analyze: how did price behave at this level
    during days classified in the SAME regime as today?
    Returns enriched level list with regime_tested, held_pct, failed_pct.
    """
    if daily.empty or len(levels) == 0 or regime_labels is None:
        return levels

    past = daily.tail(lookback + 2)
    hi_arr = past["High"].astype(float).values
    lo_arr = past["Low"].astype(float).values
    cl_arr = past["Close"].astype(float).values
    op_arr = past["Open"].astype(float).values
    n_past = len(past)

    # Align regime labels (they cover past 30 days, match by position)
    n_labels = len(regime_labels)

    enriched = []
    for lv in levels:
        p = lv["price"]
        zone = p * 0.003  # 0.3% tolerance

        hits_same_regime = 0
        held = 0
        failed = 0

        for i in range(n_past - 1):
            # Was this level touched this day?
            touched = (hi_arr[i] >= p - zone) and (lo_arr[i] <= p + zone)
            if not touched:
                continue

            # What regime was this day?
            label_idx = i - (n_past - n_labels)
            if label_idx < 0 or label_idx >= n_labels:
                continue
            day_regime = int(regime_labels[label_idx])
            if day_regime != current_regime:
                continue

            hits_same_regime += 1
            # Did price hold the level (closed beyond it in the same direction)?
            if p > op_arr[i]:  # level is resistance
                if cl_arr[i] < p:
                    held += 1  # rejected → held as resistance
                else:
                    failed += 1  # broke above
            else:  # level is support
                if cl_arr[i] > p:
                    held += 1  # bounced → held as support
                else:
                    failed += 1  # broke below

        lv_new = dict(lv)
        lv_new["regime_hits"] = hits_same_regime
        if hits_same_regime > 0:
            lv_new["regime_held_pct"] = round(held / hits_same_regime * 100)
            lv_new["regime_failed_pct"] = round(failed / hits_same_regime * 100)
        else:
            lv_new["regime_held_pct"] = 50
            lv_new["regime_failed_pct"] = 50
        enriched.append(lv_new)

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN STATISTICS: what happened AFTER similar days
# ─────────────────────────────────────────────────────────────────────────────


def pattern_outcome_stats(similar_days: list, daily: pd.DataFrame) -> dict:
    """
    Given a list of similar days, look up what happened the NEXT day.
    Returns: bull_rate, avg_move, avg_range, best_case, worst_case.
    """
    if not similar_days or daily.empty:
        return {}

    next_rets = []
    next_ranges = []
    next_highs = []
    next_lows = []

    for sd in similar_days:
        try:
            dt = pd.Timestamp(sd["date"])
            idx = daily.index.get_loc(dt)
            if idx + 1 >= len(daily):
                continue
            next_ = daily.iloc[idx + 1]
            nop = float(next_["Open"])
            ncl = float(next_["Close"])
            nhi = float(next_["High"])
            nlo = float(next_["Low"])
            if nop <= 0:
                continue
            next_rets.append((ncl - nop) / nop * 100)
            next_ranges.append((nhi - nlo) / nop * 100)
            next_highs.append((nhi - nop) / nop * 100)  # max gain from open
            next_lows.append((nop - nlo) / nop * 100)  # max pain from open
        except:
            continue

    if not next_rets:
        return {}

    bull_rate = sum(1 for r in next_rets if r > 0) / len(next_rets)

    # Weight by similarity score
    weights = [sd["similarity"] for sd in similar_days if "similarity" in sd]
    if len(weights) == len(next_rets) and sum(weights) > 0:
        w_arr = np.array(weights[: len(next_rets)])
        w_arr = w_arr / w_arr.sum()
        avg_ret = float(np.average(next_rets, weights=w_arr))
    else:
        avg_ret = float(np.mean(next_rets))

    return {
        "n_similar": len(next_rets),
        "bull_rate": round(bull_rate * 100, 1),
        "avg_ret": round(avg_ret, 2),
        "avg_range": round(float(np.mean(next_ranges)), 2),
        "avg_high": round(float(np.mean(next_highs)), 2),
        "avg_low": round(float(np.mean(next_lows)), 2),
        "best_case": round(float(np.max(next_rets)), 2),
        "worst_case": round(float(np.min(next_rets)), 2),
        "median_ret": round(float(np.median(next_rets)), 2),
        "std_ret": round(float(np.std(next_rets)), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPEATING PATTERNS: find recurring setups in 30d window
# ─────────────────────────────────────────────────────────────────────────────


def find_repeating_patterns(feat_matrix: pd.DataFrame, daily: pd.DataFrame) -> list:
    """
    Cluster the 30-day window into micro-patterns.
    Find which micro-patterns have the highest predictive power.
    Returns list of pattern dicts with stats.
    """
    PATTERN_FEATURES = [
        "ret1_atr",
        "ret5_atr",
        "rsi14",
        "bb_pos",
        "atr_ratio",
        "consec_signed",
        "last_bull",
        "vol_z",
        "pos_in_range",
    ]
    cols = [c for c in PATTERN_FEATURES if c in feat_matrix.columns]
    train = feat_matrix[feat_matrix["label"].isin([0, 1])].copy()

    if len(train) < 10:
        return []

    X = train[cols].fillna(0.0).values
    y = train["label"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_patterns = min(4, len(train) // 4)
    if n_patterns < 2:
        return []

    km = KMeans(n_clusters=n_patterns, random_state=42, n_init=10)
    km.fit(Xs)
    labels = km.labels_

    patterns = []
    for c in range(n_patterns):
        mask = labels == c
        if mask.sum() < 2:
            continue
        y_c = y[mask]
        bull_rt = float(y_c.mean())
        count = int(mask.sum())

        # Feature summary for this pattern
        X_c = X[mask]
        feats_c = dict(zip(cols, X_c.mean(axis=0).tolist()))

        # Label the pattern
        ret = feats_c.get("ret1_atr", 0)
        rsi = feats_c.get("rsi14", 0.5)
        atr_r = feats_c.get("atr_ratio", 1.0)
        vol = feats_c.get("vol_z", 0)
        pos = feats_c.get("pos_in_range", 0.5)
        cons = feats_c.get("consec_signed", 0)

        if bull_rt >= 0.65:
            if ret > 0.5 and rsi > 0.55:
                name = "Momentum Comprador"
                desc = f"Tendencia alcista con RSI {rsi * 100:.0f}, precio en parte alta del rango"
            elif atr_r < 0.8:
                name = "Compresión Pre-Alcista"
                desc = "Volatilidad baja antes de movimiento alcista"
            else:
                name = "Sesgo Comprador"
                desc = (
                    f"Días similares cierran alcistas {bull_rt * 100:.0f}% de las veces"
                )
        elif bull_rt <= 0.35:
            if ret < -0.5 and rsi < 0.45:
                name = "Momentum Vendedor"
                desc = f"Tendencia bajista con RSI {rsi * 100:.0f}, precio en parte baja del rango"
            elif atr_r > 1.3:
                name = "Expansión Bajista"
                desc = "Volatilidad en expansión con sesgo vendedor"
            else:
                name = "Sesgo Vendedor"
                desc = f"Días similares cierran bajistas {(1 - bull_rt) * 100:.0f}% de las veces"
        else:
            if atr_r < 0.7:
                name = "Compresión / Chop"
                desc = "Volatilidad muy baja, mercado sin dirección"
            else:
                name = "Rango Equilibrado"
                desc = "Sin dirección clara, probabilidad 50/50"

        # Dates in this pattern
        pattern_dates = [
            str(train.index[i])[:10] for i in range(len(train)) if labels[i] == c
        ]

        # Next-day outcomes for days in this pattern
        next_rets = []
        for i in range(len(train) - 1):
            if labels[i] != c:
                continue
            dt = train.index[i]
            loc = daily.index.get_loc(dt)
            if loc + 1 < len(daily):
                nop = float(daily["Open"].iloc[loc + 1])
                ncl = float(daily["Close"].iloc[loc + 1])
                if nop > 0:
                    next_rets.append((ncl - nop) / nop * 100)

        patterns.append(
            {
                "id": c,
                "name": name,
                "description": desc,
                "count": count,
                "bull_rate": round(bull_rt * 100, 1),
                "bear_rate": round((1 - bull_rt) * 100, 1),
                "avg_next_ret": round(float(np.mean(next_rets)) if next_rets else 0, 2),
                "dates": pattern_dates[-5:],  # last 5 dates
                "features": {k: round(v, 3) for k, v in feats_c.items()},
                "rsi": round(rsi * 100, 1),
                "vol_z": round(float(feats_c.get("vol_z", 0)), 2),
                "ret1": round(float(ret), 2),
                "pos_range": round(float(pos), 2),
            }
        )

    # Sort by strength of signal (furthest from 50%)
    patterns.sort(key=lambda p: abs(p["bull_rate"] - 50), reverse=True)
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ML ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────────────────


def run_ml_analysis(
    ticker: str,
    daily: pd.DataFrame,
    df5m: Optional[pd.DataFrame],
    levels: list,
    lookback: int = 30,
) -> dict:
    """
    Full ML pipeline. Returns structured dict for rendering.

    Steps:
      1. Feature matrix (30d)
      2. Regime classification (KMeans)
      3. Bias prediction (RF + GB ensemble)
      4. Pattern matching (cosine sim)
      5. Outcome statistics
      6. Anomaly detection (LOF)
      7. Repeating patterns
      8. Level behavior in current regime
    """
    result = {
        "ticker": ticker,
        "lookback": lookback,
        "error": None,
        "feat_matrix": None,
        "regime": None,
        "bias_model": None,
        "similar_days": [],
        "outcome_stats": {},
        "anomaly": {},
        "patterns": [],
        "levels": levels,
    }

    if daily.empty or len(daily) < lookback // 2:
        result["error"] = "Insuficiente data diaria"
        return result

    # 1. Feature matrix
    try:
        fm = build_feature_matrix(daily, df5m, lookback=lookback)
        if fm.empty or len(fm) < 8:
            result["error"] = "Insuficientes features"
            return result
        result["feat_matrix"] = fm
    except Exception as e:
        result["error"] = f"Feature engineering: {e}"
        return result

    # 2. Regime classification
    try:
        regime_labels, cluster_stats, mapping, scaler, km, regime_cols = (
            classify_regimes(fm)
        )
        current_regime = int(regime_labels[-1])
        regime_info = REGIME_NAMES.get(
            current_regime, ("UNKNOWN", "Desconocido", "var(--text3)", "")
        )
        regime_history = []
        for i, (dt, lbl) in enumerate(zip(fm.index, regime_labels)):
            regime_history.append(
                {
                    "date": str(dt)[:10],
                    "regime": int(lbl),
                    "name": REGIME_NAMES.get(int(lbl), ("?", "?", "", ""))[0],
                }
            )
        result["regime"] = {
            "current": current_regime,
            "name": regime_info[0],
            "label": regime_info[1],
            "color": regime_info[2],
            "description": regime_info[3],
            "history": regime_history[-30:],
            "labels_array": regime_labels,
            "cluster_stats": cluster_stats,
            "mapping": mapping,
        }
    except Exception as e:
        result["regime"] = {
            "error": str(e),
            "current": 2,
            "label": "Error",
            "color": "var(--text3)",
        }
        regime_labels = None

    # 3. Bias prediction
    try:
        bm = train_bias_model(fm)
        result["bias_model"] = bm
    except Exception as e:
        result["bias_model"] = {"error": str(e), "proba_bull": 0.5, "accuracy": None}

    # 4. Pattern matching
    try:
        similar = find_similar_days(fm, top_k=5)
        result["similar_days"] = similar
    except Exception as e:
        result["similar_days"] = []

    # 5. Outcome statistics
    try:
        if result["similar_days"]:
            result["outcome_stats"] = pattern_outcome_stats(
                result["similar_days"], daily
            )
    except:
        pass

    # 6. Anomaly detection
    try:
        result["anomaly"] = detect_anomaly(fm)
    except Exception as e:
        result["anomaly"] = {"is_anomaly": False, "note": str(e)}

    # 7. Repeating patterns in 30d window
    try:
        result["patterns"] = find_repeating_patterns(fm, daily)
    except:
        result["patterns"] = []

    # 8. Level behavior in current regime
    try:
        if regime_labels is not None and levels:
            result["levels"] = analyze_level_behavior(
                daily,
                levels,
                regime_labels,
                current_regime=result["regime"]["current"],
                lookback=lookback,
            )
    except:
        pass

    return result