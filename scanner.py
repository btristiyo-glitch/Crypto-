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

TIMEFRAME = "4H"
CANDLE_LIMIT = 180
TOP_N = 3
MAX_SYMBOLS_PER_RUN = 150

MIN_QUOTE_VOLUME_USDT = 500_000
MIN_AVG_QUOTE_VOLUME_USDT = 300_000
MIN_SCORE = 66

SLEEP_BETWEEN_REQUESTS = 0.15
REQUEST_TIMEOUT = 20

ALPHA_DAYS_BACK = 2
ALPHA_NUM_RESULTS = 15

OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "scanner.log")
CSV_FILE = os.path.join(OUTPUT_DIR, "scan_results.csv")
BLACKLIST_FILE = os.path.join(OUTPUT_DIR, "blacklist.csv")

# ============================================================
# SETUP
# ============================================================
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
# UTIL
# ============================================================
def utc_ms():
    return str(int(datetime.now(timezone.utc).timestamp() * 1000))

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

def safe_telegram_send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram disabled - token/chat_id not set.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [text[i:i + 3500] for i in range(0, len(text), 3500)]

    ok = True
    for chunk in chunks:
        try:
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
            if resp.status_code != 200:
                ok = False
                log_err(f"Telegram send failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            ok = False
            log_err(f"Telegram exception: {e}")
    return ok

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
        return {normalize_text(row["symbol"]): int(row["fail_count"]) for _, row in df.iterrows()}
    except Exception as e:
        log_err(f"Blacklist load error: {e}")
        return {}

def save_blacklist(data):
    rows = [{"symbol": k, "fail_count": v} for k, v in data.items()]
    pd.DataFrame(rows).to_csv(BLACKLIST_FILE, index=False)

def update_blacklist(blacklist, symbol, success, max_fails=3):
    key = normalize_text(symbol)
    if success:
        if key in blacklist:
            blacklist[key] = max(0, blacklist[key] - 1)
    else:
        blacklist[key] = blacklist.get(key, 0) + 1

    if blacklist.get(key, 0) < 1:
        blacklist.pop(key, None)

    save_blacklist(blacklist)

def is_blacklisted(blacklist, symbol, max_fails=3):
    return blacklist.get(normalize_text(symbol), 0) >= max_fails

# ============================================================
# NARRATIVE
# ============================================================
def load_alpha_and_trending_candidates():
    candidates = set()
    meta = {"alpha_hits": [], "trending_hits": []}

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
        from functions import find_trending_coins
        trending = find_trending_coins()
        coins = trending.get("trending_coins", {}).get("trending_coins", [])
        for coin in coins:
            for key in ["symbol", "name"]:
                val = coin.get(key)
                if val:
                    candidates.add(normalize_text(val))
                    meta["trending_hits"].append(str(val))
    except Exception as e:
        log_err(f"Trending error: {e}")

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
    trial_granularities = [granularity, "1H", "15m"]
    for g in trial_granularities:
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
# STRATEGY
# ============================================================
def calc_levels(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    last = float(close.iloc[-1])

    recent_high_20 = float(high.tail(20).max())
    recent_low_20 = float(low.tail(20).min())

    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    sma20 = SMAIndicator(close, window=20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]

    breakout = last > recent_high_20
    uptrend = bool(pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50)

    if breakout:
        entry = recent_high_20 * 0.995
    else:
        fib_618 = recent_low_20 + (recent_high_20 - recent_low_20) * 0.618
        fib_50 = recent_low_20 + (recent_high_20 - recent_low_20) * 0.5
        entry = min(fib_50, fib_618, recent_high_20 * 0.995)

    stop = recent_low_20 * 0.985
    risk = max(entry - stop, float(atr) * 0.8)

    return {
        "last": last,
        "entry": round(float(entry), 8),
        "stop": round(float(stop), 8),
        "tp1": round(float(entry + risk * 1.5), 8),
        "tp2": round(float(entry + risk * 2.5), 8),
        "tp3": round(float(entry + risk * 3.5), 8),
        "atr": float(atr),
        "sma20": float(sma20) if pd.notna(sma20) else None,
        "sma50": float(sma50) if pd.notna(sma50) else None,
        "breakout": breakout,
        "uptrend": uptrend,
    }

def score_symbol(df, narrative_hit=False):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    quote_vol = df["quoteVol"]

    last = float(close.iloc[-1])
    vol20 = float(quote_vol.tail(20).mean())
    vol60 = float(quote_vol.tail(60).mean())

    rsi_last = float(RSIIndicator(close, window=14).rsi().iloc[-1])
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = float(macd.macd_diff().iloc[-1])
    macd_line = float(macd.macd().iloc[-1])
    signal_line = float(macd.macd_signal().iloc[-1])

    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    atr_pct = float(atr) / last if last > 0 else 0

    levels = calc_levels(df)
    score = 0

    if vol20 >= 5_000_000:
        score += 18
    elif vol20 >= 1_500_000:
        score += 13
    elif vol20 >= 500_000:
        score += 9

    if levels["uptrend"]:
        score += 18
    elif levels["sma20"] is not None and levels["sma50"] is not None and levels["sma20"] > levels["sma50"] * 0.995:
        score += 12

    if 45 <= rsi_last <= 68:
        score += 10
    elif 35 <= rsi_last < 45:
        score += 7

    if macd_hist > 0:
        score += 12
    if macd_line > signal_line:
        score += 4

    if 0.025 <= atr_pct <= 0.12:
        score += 6
    elif atr_pct > 0.12:
        score += 3

    if levels["breakout"]:
        score += 8

    if narrative_hit:
        score += 12

    if vol20 > vol60 * 1.2:
        score += 4

    # Cooldown chop filter
    chop = False
    if levels["sma20"] is not None and levels["sma50"] is not None:
        close_crosses = ((close.tail(20) > levels["sma20"]) != (close.tail(20).shift(1) > levels["sma20"])).sum()
        sma_gap = abs(levels["sma20"] - levels["sma50"]) / last
        if close_crosses >= 6 and sma_gap < 0.01:
            chop = True
            score -= 12

    return {
        "score": round(score, 2),
        "rsi": round(rsi_last, 2),
        "macd_hist": round(macd_hist, 6),
        "vol20": round(vol20, 2),
        "vol60": round(vol60, 2),
        "atr_pct": round(atr_pct, 4),
        "levels": levels,
        "chop": chop,
    }

def build_reason(item):
    parts = []
    if item["levels"]["breakout"]:
        parts.append("breakout di atas swing high")
    else:
        parts.append("entry di area pullback struktur")
    if 45 <= item["rsi"] <= 68:
        parts.append("RSI sehat")
    if item["macd_hist"] > 0:
        parts.append("MACD mendukung trend naik")
    if item["atr_pct"] >= 0.025:
        parts.append("range masih enak buat follow-through")
    if item["chop"]:
        parts.append("hindari - pasar terlalu chop")
    return " - ".join(parts)

def build_signal_label(score):
    if score >= 75:
        return "🟢 BUY"
    if score >= 66:
        return "🟡 HOLD"
    return "🔴 SELL"

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
        "🔥 *BITGET SPOT NARRATIVE SCANNER*",
        f"🕒 {now_str}",
        f"Alpha hits: {len(meta['alpha_hits'])} - Trending hits: {len(meta['trending_hits'])}",
        "",
    ]

    if not results:
        lines.append("Tidak ada setup valid hari ini.")
        return "\n".join(lines)

    for i, x in enumerate(results[:3], 1):
        lines.append(
            f"{i}. *{x['symbol']}* - {x['label']} - score {x['score']}\n"
            f"Entry: {x['entry']} | SL: {x['stop']}\n"
            f"TP1: {x['tp1']} | TP2: {x['tp2']} | TP3: {x['tp3']}\n"
            f"RSI: {x['rsi']} | MACD: {x['macd_hist']} | Vol20: {x['vol20']}\n"
            f"Thesis: {x['reason']}\n"
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
        log("Tidak ada match alpha/trending - fallback ke full Bitget USDT scan.")
        matched = spot_symbols

    matched = [s for s in matched if not is_blacklisted(blacklist, s)]
    matched = matched[:MAX_SYMBOLS_PER_RUN]

    results = []
    csv_rows = []

    for symbol in matched:
        try:
            df = get_spot_candles(symbol, granularity=TIMEFRAME, limit=CANDLE_LIMIT)
            if df is None or len(df) < 60:
                update_blacklist(blacklist, symbol, success=False)
                continue

            base_symbol = normalize_text(symbol.replace("_USDT", ""))
            narrative_hit = base_symbol in candidates

            metrics = score_symbol(df, narrative_hit=narrative_hit)

            if metrics["vol20"] < MIN_QUOTE_VOLUME_USDT or metrics["vol60"] < MIN_AVG_QUOTE_VOLUME_USDT:
                update_blacklist(blacklist, symbol, success=False)
                continue

            if metrics["chop"]:
                update_blacklist(blacklist, symbol, success=False)
                continue

            if metrics["score"] < MIN_SCORE:
                update_blacklist(blacklist, symbol, success=False)
                continue

            levels = metrics["levels"]
            last_price = round(float(df["close"].iloc[-1]), 8)

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "symbol": symbol,
                "label": build_signal_label(metrics["score"]),
                "score": metrics["score"],
                "price": last_price,
                "entry": levels["entry"],
                "stop": levels["stop"],
                "tp1": levels["tp1"],
                "tp2": levels["tp2"],
                "tp3": levels["tp3"],
                "rsi": metrics["rsi"],
                "macd_hist": metrics["macd_hist"],
                "vol20": metrics["vol20"],
                "vol60": metrics["vol60"],
                "atr_pct": metrics["atr_pct"],
                "breakout": levels["breakout"],
                "narrative_hit": narrative_hit,
                "reason": build_reason(metrics),
            }

            results.append(row)
            csv_rows.append(row)
            update_blacklist(blacklist, symbol, success=True)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            log_err(f"{symbol} error: {e}")
            update_blacklist(blacklist, symbol, success=False)
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:TOP_N]
    return results, meta, csv_rows

def main():
    try:
        results, meta, csv_rows = run_scan()
        csv_path = save_csv(csv_rows)

        if csv_path:
            log(f"CSV saved to {csv_path}")

        if results:
            report = telegram_report(results, meta)
            safe_telegram_send(report)
            log(report)
        else:
            log("No valid setup - Telegram not sent.")

    except Exception as e:
        log_err(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
        
