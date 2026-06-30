import os
import time
import hmac
import base64
import hashlib
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

# ============================================================
# CONFIG
# ============================================================
BASE_URL = "https://api.bitget.com"

BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE", "")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TIMEFRAMES = ["15m", "1H", "4H"]
CANDLE_LIMITS = {"15m": 240, "1H": 220, "4H": 180}

TOP_N = 5
MAX_SYMBOLS_PER_RUN = 140

MIN_QUOTE_VOLUME_USDT = 120_000
MIN_SCORE = 58
EARLY_SCORE = 46

SLEEP_BETWEEN_REQUESTS = 0.12
REQUEST_TIMEOUT = 20

ALPHA_DAYS_BACK = 3
ALPHA_NUM_RESULTS = 20

OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "scanner.log")
CSV_FILE = os.path.join(OUTPUT_DIR, "scan_results.csv")
BLACKLIST_FILE = os.path.join(OUTPUT_DIR, "blacklist.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def log(msg):
    print(msg)
    logging.info(msg)

def log_err(msg):
    print(msg)
    logging.error(msg)

# ============================================================
# HELPERS
# ============================================================
def normalize_text(s):
    return (
        str(s).upper()
        .replace("$", "")
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .replace(".", "")
        .strip()
    )

def similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def utc_ms():
    return str(int(datetime.now(timezone.utc).timestamp() * 1000))

def sign(secret: str, timestamp: str, method: str, path: str, query: str = "", body: str = ""):
    payload = f"{timestamp}{method.upper()}{path}"
    if query:
        payload += f"?{query}"
    payload += body
    digest = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def http_get(url, params=None, headers=None):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def bitget_public(path, params=None):
    return http_get(BASE_URL + path, params=params)

def safe_telegram_send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram disabled - token/chat_id not set.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text[:3900],
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            log_err(f"Telegram error: {resp.status_code} - {resp.text}")
            return False
        return True
    except Exception as e:
        log_err(f"Telegram exception: {e}")
        return False

# ============================================================
# BLACKLIST
# ============================================================
def load_blacklist():
    if not os.path.exists(BLACKLIST_FILE):
        return {}
    try:
        df = pd.read_csv(BLACKLIST_FILE)
        if df.empty:
            return {}
        return {normalize_text(r["symbol"]): int(r.get("fail_count", 0)) for _, r in df.iterrows()}
    except Exception as e:
        log_err(f"Blacklist load error: {e}")
        return {}

def save_blacklist(data):
    pd.DataFrame([{"symbol": k, "fail_count": v} for k, v in data.items()]).to_csv(BLACKLIST_FILE, index=False)

def update_blacklist(blacklist, symbol, success):
    key = normalize_text(symbol)
    if success:
        blacklist[key] = max(0, blacklist.get(key, 0) - 1)
    else:
        blacklist[key] = blacklist.get(key, 0) + 1
    if blacklist.get(key, 0) <= 0:
        blacklist.pop(key, None)
    save_blacklist(blacklist)

def is_blacklisted(blacklist, symbol):
    return blacklist.get(normalize_text(symbol), 0) >= 4

# ============================================================
# SIGNAL SOURCES
# ============================================================
def load_alpha_and_trending_candidates():
    candidates = set()
    meta = {"alpha_hits": [], "trending_hits": [], "new_listing_hits": []}

    try:
        from functions import retrieve_alpha
        alpha = retrieve_alpha(
            query="crypto listings launch partnership funding upgrade tokenomics positive catalyst",
            days_ago=ALPHA_DAYS_BACK,
            num_results=ALPHA_NUM_RESULTS,
            sentiment="positive",
            event_types=["listing", "launch", "partnership", "funding", "upgrade", "tokenomics"],
            market_segments=["layer1", "layer2", "crosschain", "defi", "stablecoins", "infrastructure", "developer tools", "ai agents", "payments wallets", "launchpads airdrops", "culture memecoins", "other"],
            engagement_levels=["medium", "high"],
        )
        for item in alpha.get("results", []):
            for key in ["project_name", "name", "ticker", "symbol"]:
                val = item.get(key)
                if val:
                    candidates.add(normalize_text(val))
                    meta["alpha_hits"].append(str(val))
    except Exception as e:
        log_err(f"Alpha error: {e}")

    try:
        from functions import find_trending_liquidity_pools
        pools = find_trending_liquidity_pools(duration="6h")
        for item in pools.get("pools", []):
            token = item.get("base_token_symbol", "")
            if token:
                candidates.add(normalize_text(token))
                meta["trending_hits"].append(token)
    except Exception as e:
        log_err(f"Trending pools error: {e}")

    try:
        from functions import get_recently_added_coins
        recent = get_recently_added_coins()
        for coin in recent.get("coins", []):
            sym = coin.get("symbol", "")
            name = coin.get("name", "")
            if sym:
                candidates.add(normalize_text(sym))
                meta["new_listing_hits"].append(sym)
            if name:
                candidates.add(normalize_text(name))
    except Exception as e:
        log_err(f"Recently added coins error: {e}")

    return candidates, meta

# ============================================================
# UNIVERSE
# ============================================================
def get_bitget_spot_usdt_symbols():
    data = bitget_public("/api/v2/spot/public/symbols")
    items = data.get("data", [])
    out = []
    for x in items:
        symbol = x.get("symbol")
        quote = x.get("quoteCoin")
        status = x.get("status")
        if symbol and quote == "USDT" and status == "online":
            out.append(symbol)
    return sorted(list(set(out)))

def build_symbol_maps(spot_symbols):
    norm_map = {}
    for s in spot_symbols:
        base = s.replace("_USDT", "")
        norm_map[normalize_text(base)] = s
        norm_map[normalize_text(s)] = s
    return norm_map

def match_candidates_to_bitget(candidates, spot_map, spot_symbols):
    matched = set()
    for cand in candidates:
        if cand in spot_map:
            matched.add(spot_map[cand])
            continue
        best_sym = None
        best_score = 0.0
        for s in spot_symbols:
            base = s.replace("_USDT", "")
            sc = max(similarity(cand, s), similarity(cand, base))
            if sc > best_score:
                best_score = sc
                best_sym = s
        if best_sym and best_score >= 0.72:
            matched.add(best_sym)
    return sorted(list(matched))

# ============================================================
# DATA
# ============================================================
def get_spot_candles(symbol, granularity="4H", limit=180):
    for g in [granularity, "1H", "15m"]:
        try:
            params = {"symbol": symbol, "granularity": g, "limit": str(limit)}
            data = bitget_public("/api/v2/spot/market/candles", params=params)
            rows = data.get("data", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "baseVol", "quoteVol"])
                for col in ["ts", "open", "high", "low", "close", "baseVol", "quoteVol"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna()
                if not df.empty:
                    return df.sort_values("ts").reset_index(drop=True)
        except Exception as e:
            log_err(f"{symbol} candles failed on {g}: {e}")
    return None

# ============================================================
# SCORING
# ============================================================
def compute_tf_metrics(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    last = float(close.iloc[-1])

    sma20 = SMAIndicator(close, window=20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)

    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    vol20 = float(df["quoteVol"].tail(20).mean())
    vol60 = float(df["quoteVol"].tail(60).mean())

    recent_high = float(high.tail(20).max())
    recent_low = float(low.tail(20).min())
    breakout = last > recent_high

    cross_count = ((close.tail(20) > sma20) != (close.tail(20).shift(1) > sma20)).sum() if pd.notna(sma20) else 0
    sma_gap = abs(float(sma20) - float(sma50)) / last if pd.notna(sma20) and pd.notna(sma50) and last > 0 else 0

    return {
        "last": last,
        "sma20": float(sma20) if pd.notna(sma20) else None,
        "sma50": float(sma50) if pd.notna(sma50) else None,
        "rsi": float(rsi),
        "macd_hist": float(macd.macd_diff().iloc[-1]),
        "macd_line": float(macd.macd().iloc[-1]),
        "macd_signal": float(macd.macd_signal().iloc[-1]),
        "atr_pct": float(atr) / last if last > 0 else 0,
        "vol20": vol20,
        "vol60": vol60,
        "breakout": breakout,
        "cross_count": int(cross_count),
        "sma_gap": float(sma_gap),
        "recent_high": recent_high,
        "recent_low": recent_low,
    }

def score_trend_and_flow(metrics_by_tf, narrative_hit=False):
    m15 = metrics_by_tf["15m"]
    m1h = metrics_by_tf["1H"]
    m4h = metrics_by_tf["4H"]

    score = 0
    notes = []

    if m15["vol20"] >= 250_000:
        score += 10
    elif m15["vol20"] >= 80_000:
        score += 6

    if m1h["vol20"] >= 400_000:
        score += 10
    elif m1h["vol20"] >= 120_000:
        score += 6

    if m4h["vol20"] >= 700_000:
        score += 8
    elif m4h["vol20"] >= 200_000:
        score += 5

    if m15["rsi"] >= 45 and m15["rsi"] <= 74:
        score += 5
    if m1h["rsi"] >= 42 and m1h["rsi"] <= 72:
        score += 6
    if m4h["rsi"] >= 40 and m4h["rsi"] <= 68:
        score += 7

    if m15["macd_hist"] > 0:
        score += 4
    if m1h["macd_hist"] > 0:
        score += 5
    if m4h["macd_hist"] > 0:
        score += 7

    if m15["breakout"]:
        score += 6
        notes.append("15m breakout")
    if m1h["breakout"]:
        score += 8
        notes.append("1H breakout")
    if m4h["breakout"]:
        score += 10
        notes.append("4H breakout")

    if m15["cross_count"] >= 6 and m15["sma_gap"] < 0.01:
        score -= 8
        notes.append("15m chop")
    if m1h["cross_count"] >= 6 and m1h["sma_gap"] < 0.01:
        score -= 10
        notes.append("1H chop")
    if m4h["cross_count"] >= 6 and m4h["sma_gap"] < 0.01:
        score -= 12
        notes.append("4H chop")

    if m15["atr_pct"] >= 0.02:
        score += 3
    if m1h["atr_pct"] >= 0.02:
        score += 3
    if m4h["atr_pct"] >= 0.02:
        score += 3

    if m15["vol20"] > m15["vol60"] * 1.15:
        score += 4
        notes.append("15m volume acceleration")
    if m1h["vol20"] > m1h["vol60"] * 1.15:
        score += 4
        notes.append("1H volume acceleration")
    if m4h["vol20"] > m4h["vol60"] * 1.10:
        score += 4
        notes.append("4H volume acceleration")

    if narrative_hit:
        score += 10
        notes.append("narrative hit")

    alignment = int((m15["macd_hist"] > 0) + (m1h["macd_hist"] > 0) + (m4h["macd_hist"] > 0))
    if alignment == 3:
        score += 8
        notes.append("3 TF aligned")
    elif alignment == 2:
        score += 4

    return round(score, 2), notes

def build_levels_from_4h(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    last = float(close.iloc[-1])

    recent_high_20 = float(high.tail(20).max())
    recent_low_20 = float(low.tail(20).min())
    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    breakout = last > recent_high_20
    if breakout:
        entry = recent_high_20 * 0.997
    else:
        fib_618 = recent_low_20 + (recent_high_20 - recent_low_20) * 0.618
        fib_50 = recent_low_20 + (recent_high_20 - recent_low_20) * 0.5
        entry = min(fib_50, fib_618, recent_high_20 * 0.997)

    stop = recent_low_20 * 0.982
    risk = max(entry - stop, float(atr) * 0.7)

    return {
        "entry": round(float(entry), 8),
        "stop": round(float(stop), 8),
        "tp1": round(float(entry + risk * 1.4), 8),
        "tp2": round(float(entry + risk * 2.2), 8),
        "tp3": round(float(entry + risk * 3.2), 8),
        "risk": risk,
    }

def rank_potential(score, metrics_by_tf, narrative_hit, fdv_score, buy_surge_score, vol_accel_score):
    m15 = metrics_by_tf["15m"]
    m1h = metrics_by_tf["1H"]
    m4h = metrics_by_tf["4H"]

    rank = score
    rank += fdv_score + buy_surge_score + vol_accel_score
    rank += min(m15["vol20"] / 50_000, 8)
    rank += min(m1h["vol20"] / 100_000, 8)
    rank += min(m4h["vol20"] / 150_000, 8)
    rank += 3 if m15["breakout"] else 0
    rank += 4 if m1h["breakout"] else 0
    rank += 5 if m4h["breakout"] else 0
    if narrative_hit:
        rank += 5
    return round(rank, 2)

def build_reason(notes, extra_bits):
    parts = []
    parts.extend(notes[:4])
    parts.extend(extra_bits[:4])
    return " - ".join([p for p in parts if p])

def build_signal_label(score, early=False):
    if score >= 66:
        return "🟢 BUY"
    if score >= 58:
        return "🟡 HOLD" if not early else "🟡 EARLY"
    return "🔴 WATCH"

# ============================================================
# EXTRA SIGNALS
# ============================================================
def estimate_buy_surge_from_trending(item):
    buys = safe_float(item.get("buys_24h", 0))
    sells = safe_float(item.get("sells_24h", 0))
    tx = safe_float(item.get("transactions_24h", buys + sells))
    ratio = buys / max(sells, 1.0)
    buy_share = buys / max(tx, 1.0)

    score = 0
    if ratio >= 1.6:
        score += 8
    elif ratio >= 1.25:
        score += 5

    if buy_share >= 0.58:
        score += 6
    elif buy_share >= 0.53:
        score += 3

    if tx >= 5000:
        score += 4
    elif tx >= 1000:
        score += 2

    return round(score, 2), ratio, buy_share, tx

def estimate_fdv_score(fdv, volume_24h, liquidity_usd):
    score = 0
    if fdv <= 10_000_000:
        score += 10
    elif fdv <= 30_000_000:
        score += 8
    elif fdv <= 75_000_000:
        score += 5

    if volume_24h >= 1_000_000:
        score += 6
    elif volume_24h >= 250_000:
        score += 4

    if liquidity_usd >= 250_000:
        score += 2
    elif liquidity_usd >= 100_000:
        score += 1

    return round(score, 2)

def estimate_volume_acceleration(volume_24h, tx_24h):
    score = 0
    if volume_24h >= 10_000_000:
        score += 8
    elif volume_24h >= 2_000_000:
        score += 6
    elif volume_24h >= 500_000:
        score += 4

    if tx_24h >= 50_000:
        score += 6
    elif tx_24h >= 10_000:
        score += 4
    elif tx_24h >= 2_000:
        score += 2

    return round(score, 2)

# ============================================================
# OUTPUT
# ============================================================
def save_csv(rows):
    if not rows:
        return None
    pd.DataFrame(rows).to_csv(CSV_FILE, index=False)
    return CSV_FILE

def telegram_report(results, meta):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "🔥 *BITGET MULTI-TF SCANNER*",
        f"🕒 {now_str}",
        f"Alpha hits: {len(meta['alpha_hits'])} - Trending hits: {len(meta['trending_hits'])} - New listings: {len(meta['new_listing_hits'])}",
        "",
    ]
    if not results:
        lines.append("Tidak ada setup valid hari ini.")
        return "\n".join(lines)

    for i, x in enumerate(results[:3], 1):
        lines.append(
            f"{i}. *{x['symbol']}* - {x['label']} - score {x['score']} - rank {x['rank']}\n"
            f"Entry: {x['entry']} | SL: {x['stop']}\n"
            f"TP1: {x['tp1']} | TP2: {x['tp2']} | TP3: {x['tp3']}\n"
            f"15m RSI {x['rsi_15m']} | 1H RSI {x['rsi_1h']} | 4H RSI {x['rsi_4h']}\n"
            f"FDV score: {x['fdv_score']} | Buy surge: {x['buy_surge_score']} | Vol accel: {x['vol_accel_score']}\n"
            f"Reason: {x['reason']}\n"
        )
    return "\n".join(lines)

# ============================================================
# MAIN
# ============================================================
def run_scan():
    blacklist = load_blacklist()
    spot_symbols = get_bitget_spot_usdt_symbols()
    spot_map = build_symbol_maps(spot_symbols)

    candidates, meta = load_alpha_and_trending_candidates()
    matched = match_candidates_to_bitget(candidates, spot_map, spot_symbols)

    if not matched:
        matched = spot_symbols

    matched = [s for s in matched if not is_blacklisted(blacklist, s)]
    matched = matched[:MAX_SYMBOLS_PER_RUN]

    results = []
    csv_rows = []

    for symbol in matched:
        try:
            metrics_by_tf = {}
            dfs = {}
            for tf in TIMEFRAMES:
                df = get_spot_candles(symbol, granularity=tf, limit=CANDLE_LIMITS[tf])
                if df is None or len(df) < 60:
                    raise ValueError(f"not enough candles for {tf}")
                dfs[tf] = df
                metrics_by_tf[tf] = compute_tf_metrics(df)

            base_symbol = normalize_text(symbol.replace("_USDT", ""))
            narrative_hit = base_symbol in candidates

            score, notes = score_trend_and_flow(metrics_by_tf, narrative_hit=narrative_hit)

            fdv_score = 0
            buy_surge_score = 0
            vol_accel_score = 0
            extra_bits = []

            try:
                from functions import find_trending_liquidity_pools
                pools = find_trending_liquidity_pools(duration="6h")
                for item in pools.get("pools", []):
                    token = normalize_text(item.get("base_token_symbol", ""))
                    if token and token == base_symbol:
                        fdv = safe_float(item.get("fdv", 0))
                        vol24 = safe_float(item.get("volume_24h", 0))
                        liq = safe_float(item.get("liquidity_usd", 0))
                        tx24 = safe_float(item.get("transactions_24h", 0))
                        buys24 = safe_float(item.get("buys_24h", 0))
 
