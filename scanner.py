import os
import time
import math
import hmac
import base64
import hashlib
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
BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE", "")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BASE_URL = "https://api.bitget.com"

TIMEFRAME = "4H"
CANDLE_LIMIT = 180
TOP_N = 5

MIN_QUOTE_VOLUME_USDT = 500_000
MIN_AVG_QUOTE_VOLUME_USDT = 300_000
MIN_SCORE = 62

MAX_SYMBOLS_PER_RUN = 150
SLEEP_BETWEEN_REQUESTS = 0.15
REQUEST_TIMEOUT = 20

ALPHA_DAYS_BACK = 2
ALPHA_NUM_RESULTS = 15

# score weights
W_VOL = 18
W_TREND = 18
W_MOMENTUM = 16
W_MACD = 12
W_RSI = 10
W_ATR = 6
W_NARRATIVE = 12
W_BREAKOUT = 8

# ============================================================
# BASIC HELPERS
# ============================================================
def utc_ms():
    return str(int(datetime.now(timezone.utc).timestamp() * 1000))


def normalize_text(s):
    return (
        str(s)
        .upper()
        .replace("$", "")
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .replace(".", "")
        .strip()
    )


def sign(secret: str, timestamp: str, method: str, path: str, query: str = "", body: str = ""):
    payload = f"{timestamp}{method.upper()}{path}"
    if query:
        payload += f"?{query}"
    payload += body
    digest = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def get_headers(method: str, path: str, query: str = "", body: str = ""):
    ts = utc_ms()
    signature = sign(BITGET_API_SECRET, ts, method, path, query, body)
    return {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US",
    }


def http_get(url, params=None, headers=None):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def bitget_public(path, params=None):
    return http_get(BASE_URL + path, params=params)


def bitget_private(path, params=None):
    query = ""
    if params:
        query = "&".join([f"{k}={v}" for k, v in params.items()])
    return http_get(BASE_URL + path, params=params or {}, headers=get_headers("GET", path, query=query))


def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3800):
        chunk = text[i:i + 3800]
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


# ============================================================
# NARRATIVE SOURCES
# ============================================================
def load_alpha_and_trending_candidates():
    candidates = set()
    meta = {
        "alpha_hits": [],
        "trending_hits": [],
    }

    # alpha
    try:
        from functions import retrieve_alpha
        alpha = retrieve_alpha(
            query="crypto listings launch partnership funding upgrade tokenomics positive catalyst",
            days_ago=ALPHA_DAYS_BACK,
            num_results=ALPHA_NUM_RESULTS,
            sentiment="positive",
            event_types=["listing", "launch", "partnership", "funding", "upgrade", "tokenomics"],
            market_segments=[
                "layer1", "layer2", "crosschain", "defi", "stablecoins",
                "infrastructure", "developer tools", "ai agents", "payments wallets",
                "launchpads airdrops", "culture memecoins", "other"
            ],
            engagement_levels=["medium", "high"],
        )

        for item in alpha.get("results", []):
            name = item.get("project_name") or item.get("name") or ""
            ticker = item.get("ticker") or item.get("symbol") or ""
            if name:
                meta["alpha_hits"].append(name)
                candidates.add(normalize_text(name))
            if ticker:
                meta["alpha_hits"].append(ticker)
                candidates.add(normalize_text(ticker))
    except Exception as e:
        print(f"Alpha error: {e}")

    # trending coins
    try:
        from functions import find_trending_coins
        trending = find_trending_coins()
        coins = trending.get("trending_coins", {}).get("trending_coins", [])

        for coin in coins:
            sym = coin.get("symbol") or ""
            name = coin.get("name") or ""
            if sym:
                meta["trending_hits"].append(sym)
                candidates.add(normalize_text(sym))
            if name:
                meta["trending_hits"].append(name)
                candidates.add(normalize_text(name))
    except Exception as e:
        print(f"Trending coins error: {e}")

    candidates.discard("")
    return candidates, meta


# ============================================================
# BITGET SPOT UNIVERSE
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

        # fuzzy match against all Bitget symbols
        best_sym = None
        best_score = 0.0
        for s in spot_symbols:
            base = s.replace("_USDT", "")
            score = max(
                similarity(cand, s),
                similarity(cand, base),
            )
            if score > best_score:
                best_score = score
                best_sym = s

        if best_sym and best_score >= 0.72:
            matched.add(best_sym)

    return sorted(list(matched))


# ============================================================
# MARKET DATA
# ============================================================
def get_spot_candles(symbol, granularity="4H", limit=180):
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": str(limit),
    }
    data = bitget_public("/api/v2/spot/market/candles", params=params)
    rows = data.get("data", [])
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "baseVol", "quoteVol"])
    for col in ["ts", "open", "high", "low", "close", "baseVol", "quoteVol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None

    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ============================================================
# TRADE LOGIC
# ============================================================
def calc_levels(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    last = float(close.iloc[-1])

    recent_high_20 = float(high.tail(20).max())
    recent_low_20 = float(low.tail(20).min())
    recent_high_50 = float(high.tail(50).max())
    recent_low_50 = float(low.tail(50).min())

    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    sma20 = SMAIndicator(close, window=20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]

    breakout = last > recent_high_20
    uptrend = sma20 > sma50 if pd.notna(sma20) and pd.notna(sma50) else False

    # Prefer pullback entry near structure, not chasing spot
    if breakout:
        entry = recent_high_20 * 0.995
    else:
        fib_618 = recent_low_20 + (recent_high_20 - recent_low_20) * 0.618
        fib_50 = recent_low_20 + (recent_high_20 - recent_low_20) * 0.5
        entry = min(fib_50, fib_618, recent_high_20 * 0.995)

    stop = recent_low_20 * 0.985
    risk = max(entry - stop, atr * 0.8)

    tp1 = entry + risk * 1.5
    tp2 = entry + risk * 2.5
    tp3 = entry + risk * 3.5

    return {
        "last": last,
        "entry": round(float(entry), 8),
        "stop": round(float(stop), 8),
        "tp1": round(float(tp1), 8),
        "tp2": round(float(tp2), 8),
        "tp3": round(float(tp3), 8),
        "atr": float(atr),
        "sma20": float(sma20) if pd.notna(sma20) else None,
        "sma50": float(sma50) if pd.notna(sma50) else None,
        "breakout": breakout,
        "uptrend": uptrend,
        "recent_high_20": recent_high_20,
        "recent_low_20": recent_low_20,
        "recent_high_50": recent_high_50,
        "recent_low_50": recent_low_50,
    }


def score_symbol(df, narrative_hit=False):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    quote_vol = df["quoteVol"]

    last = float(close.iloc[-1])
    vol20 = float(quote_vol.tail(20).mean())
    vol60 = float(quote_vol.tail(60).mean())

    rsi = RSIIndicator(close, window=14).rsi()
    rsi_last = float(rsi.iloc[-1])

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = float(macd.macd_diff().iloc[-1])
    macd_line = float(macd.macd().iloc[-1])
    signal_line = float(macd.macd_signal().iloc[-1])

    sma20 = SMAIndicator(close, window=20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]

    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    atr_pct = float(atr) / last if last > 0 else 0

    levels = calc_levels(df)

    score = 0

    # volume
    if vol20 >= 5_000_000:
        score += W_VOL
    elif vol20 >= 1_500_000:
        score += int(W_VOL * 0.75)
    elif vol20 >= 500_000:
        score += int(W_VOL * 0.5)

    # trend
    if levels["uptrend"]:
        score += W_TREND
    elif pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50 * 0.995:
        score += int(W_TREND * 0.7)

    # momentum / RSI
    if 45 <= rsi_last <= 68:
        score += W_RSI
    elif 35 <= rsi_last < 45:
        score += int(W_RSI * 0.7)
    elif rsi_last < 35:
        score += int(W_RSI * 0.4)

    # MACD
    if macd_hist > 0:
        score += W_MACD
    if macd_line > signal_line:
        score += 4

    # ATR healthy
    if 0.025 <= atr_pct <= 0.12:
        score += W_ATR
    elif atr_pct > 0.12:
        score += 3

    # breakout
    if levels["breakout"]:
        score += W_BREAKOUT

    # narrative
    if narrative_hit:
        score += W_NARRATIVE

    # extra momentum if volume is lifting
    if vol20 > vol60 * 1.2:
        score += 4

    return {
        "score": round(score, 2),
        "rsi": round(rsi_last, 2),
        "macd_hist": round(macd_hist, 6),
        "macd_line": round(macd_line, 6),
        "signal_line": round(signal_line, 6),
        "vol20": round(vol20, 2),
        "vol60": round(vol60, 2),
        "atr_pct": round(atr_pct, 4),
        "levels": levels,
    }


def build_reason(item):
    parts = []
    if item["levels"]["breakout"]:
        parts.append("breakout di atas swing high")
    else:
        parts.append("entry di area pullback struktur")

    if item["rsi"] >= 45 and item["rsi"] <= 68:
        parts.append("RSI sehat")
    if item["macd_hist"] > 0:
        parts.append("MACD mendukung trend naik")
    if item["atr_pct"] >= 0.025:
        parts.append("range masih layak buat follow-through")
    return " - ".join(parts)


def build_signal_label(score):
    if score >= 75:
        return "🟢 BUY"
    elif score >= 62:
        return "🟡 HOLD"
    else:
        return "🔴 SELL"


# ============================================================
# MAIN SCAN
# ============================================================
def run_scan():
    spot_symbols = get_bitget_spot_usdt_symbols()
    spot_map = build_symbol_maps(spot_symbols)

    candidates, meta = load_alpha_and_trending_candidates()
    matched = match_candidates_to_bitget(candidates, spot_map, spot_symbols)

    # fallback if alpha/trending misses
    if not matched:
        matched = spot_symbols

    matched = matched[:MAX_SYMBOLS_PER_RUN]

    results = []
    for symbol in matched:
        try:
            df = get_spot_candles(symbol, granularity=TIMEFRAME, limit=CANDLE_LIMIT)
            if df is None or len(df) < 60:
                continue

            narrative_hit = normalize_text(symbol.replace("_USDT", "")) in candidates
            metrics = score_symbol(df, narrative_hit=narrative_hit)
            levels = metrics["levels"]

            if metrics["vol20"] < MIN_QUOTE_VOLUME_USDT:
                continue
            if metrics["vol60"] < MIN_AVG_QUOTE_VOLUME_USDT:
                continue
            if metrics["score"] < MIN_SCORE:
                continue

            results.append({
                "symbol": symbol,
                "score": metrics["score"],
                "label": build_signal_label(metrics["score"]),
                "reason": build_reason(metrics),
                "price": round(float(df["close"].iloc[-1]), 8),
                "rsi": metrics["rsi"],
                "macd_hist": metrics["macd_hist"],
                "vol20": metrics["vol20"],
                "entry": levels["entry"],
                "stop": levels["stop"],
                "tp1": levels["tp1"],
                "tp2": levels["tp2"],
                "tp3": levels["tp3"],
            })

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"{symbol} error: {e}")
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:TOP_N]
    return results, meta


def telegram_report(results, meta):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("🔥 *BITGET SPOT NARRATIVE SCANNER*")
    lines.append(f"🕒 {now_str}")
    lines.append("")
    lines.append(f"Alpha hits: {len(meta['alpha_hits'])} - Trending hits: {len(meta['trending_hits'])}")
    lines.append("")

    if not results:
        lines.append("Tidak ada setup yang lolos filter hari ini.")
        lines.append("Coba longgarkan MIN_SCORE atau turunkan minimum volume.")
        return "\n".join(lines)

    for i, x in enumerate(results, 1):
        lines.append(
            f"{i}. *{x['symbol']}* - {x['label']} - score {x['score']}\n"
            f" Price: {x['price']}\n"
            f" Entry: {x['entry']} | SL: {x['stop']}\n"
            f" TP1: {x['tp1']} | TP2: {x['tp2']} | TP3: {x['tp3']}\n"
            f" RSI: {x['rsi']} | MACD hist: {x['macd_hist']} | Vol20: {x['vol20']}\n"
            f" Thesis: {x['reason']}\n"
        )

    return "\n".join(lines)


def main():
    results, meta = run_scan()
    report = telegram_report(results, meta)
    send_telegram(report)
    print(report)


if __name__ == "__main__":
    main()
                     
