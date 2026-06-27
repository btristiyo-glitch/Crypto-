import os
import time
import hmac
import base64
import hashlib
import requests
import pandas as pd
from datetime import datetime, timezone
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = "https://api.bitget.com"
TIMEFRAME = "4H"
LIMIT_TOP = 5

MIN_QUOTE_VOLUME_USDT = 500_000
MIN_LIQUIDITY_USDT = 300_000
MIN_SCORE = 60

def _utc_timestamp_ms():
    return str(int(datetime.now(timezone.utc).timestamp() * 1000))

def _sign(secret: str, timestamp: str, method: str, path: str, query: str = "", body: str = ""):
    payload = f"{timestamp}{method.upper()}{path}"
    if query:
        payload += f"?{query}"
    payload += body
    digest = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def _headers(method: str, path: str, query: str = "", body: str = ""):
    ts = _utc_timestamp_ms()
    sign = _sign(BITGET_API_SECRET, ts, method, path, query, body)
    return {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US",
    }

def bitget_public_get(path, params=None):
    url = BASE_URL + path
    r = requests.get(url, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def bitget_private_get(path, params=None):
    query = ""
    if params:
        query = "&".join([f"{k}={v}" for k, v in params.items()])
    url = BASE_URL + path
    r = requests.get(url, params=params or {}, headers=_headers("GET", path, query=query), timeout=20)
    r.raise_for_status()
    return r.json()

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3800):
        chunk = text[i:i+3800]
        requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            },
            timeout=20
        )

def load_trending_candidates():
    candidates = set()

    # 1) Alpha/news fresh
    try:
        alpha = retrieve_alpha(
            query="Bitget spot crypto listings partnership launch catalyst",
            days_ago=2,
            num_results=10,
            sentiment="positive",
            event_types=["listing", "launch", "partnership", "funding", "upgrade", "tokenomics"],
            market_segments=["layer1", "layer2", "defi", "infrastructure", "culture memecoins", "ai agents", "payments wallets"],
            engagement_levels=["medium", "high"]
        )
        for item in alpha.get("results", []):
            for key in ["ticker", "symbol", "project_name"]:
                if item.get(key):
                    candidates.add(str(item[key]).upper())
    except Exception:
        pass

    # 2) Trending pools
    try:
        pools = find_trending_liquidity_pools(duration="24h")
        for p in pools:
            sym = p.get("name")
            if sym:
                candidates.add(str(sym).upper())
    except Exception:
        pass

    return list(candidates)

def get_bitget_spot_symbols():
    data = bitget_public_get("/api/v2/spot/public/symbols")
    items = data.get("data", [])
    symbols = []
    for x in items:
        if x.get("quoteCoin") == "USDT" and x.get("status") == "online":
            symbols.append(x["symbol"])
    return symbols

def get_spot_kline(symbol: str, granularity="4H", limit=120):
    params = {"symbol": symbol, "granularity": granularity, "limit": str(limit)}
    data = bitget_public_get("/api/v2/spot/market/candles", params=params)
    rows = data.get("data", [])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "baseVol", "quoteVol"])
    for c in ["open", "high", "low", "close", "baseVol", "quoteVol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df

def analyze_symbol(symbol: str):
    df = get_spot_kline(symbol)
    if df is None or len(df) < 60:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = macd.macd_diff().iloc[-1]
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    last = float(close.iloc[-1])
    prev_20_high = float(high.tail(20).max())
    prev_20_low = float(low.tail(20).min())

    volume_usdt = float(df["quoteVol"].tail(20).mean())
    liquidity_proxy = float(df["quoteVol"].tail(60).mean())

    score = 0
    if volume_usdt >= 5_000_000:
        score += 25
    elif volume_usdt >= 1_000_000:
        score += 18
    elif volume_usdt >= 500_000:
        score += 10

    if liquidity_proxy >= 3_000_000:
        score += 15

    if 45 <= rsi <= 68:
        score += 15
    elif rsi < 35:
        score += 8

    if macd_hist > 0:
        score += 15
    if macd_line > signal_line:
        score += 10

    if last > prev_20_high:
        score += 10
    elif last > (prev_20_low + (prev_20_high - prev_20_low) * 0.618):
        score += 5

    if atr / last > 0.03:
        score += 5

    entry = round(prev_20_high * 0.995, 8)
    stop = round(prev_20_low * 0.985, 8)
    tp1 = round(entry + (entry - stop) * 1.5, 8)
    tp2 = round(entry + (entry - stop) * 2.5, 8)
    tp3 = round(entry + (entry - stop) * 3.5, 8)

    return {
        "symbol": symbol,
        "score": round(score, 1),
        "price": round(last, 8),
        "rsi": round(float(rsi), 2),
        "macd_hist": round(float(macd_hist), 6),
        "volume_usdt": round(volume_usdt, 2),
        "liquidity_proxy": round(liquidity_proxy, 2),
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
    }

def main():
    if not BITGET_API_KEY or not BITGET_API_SECRET or not BITGET_API_PASSPHRASE:
        raise RuntimeError("Bitget API credentials belum di-set.")

    candidates = load_trending_candidates()
    symbols = get_bitget_spot_symbols()

    ranked = []
    candidate_map = {x.replace("USDT", ""): x for x in candidates}

    for s in symbols:
        base = s.replace("_USDT", "").replace("USDT", "")
        if candidates and base not in candidate_map and base not in candidates:
            continue

        try:
            item = analyze_symbol(s)
            if not item:
                continue
            if item["volume_usdt"] < MIN_QUOTE_VOLUME_USDT:
                continue
            if item["score"] < MIN_SCORE:
                continue
            ranked.append(item)
            time.sleep(0.2)
        except Exception:
            continue

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)[:LIMIT_TOP]

    if not ranked:
        send_telegram("Tidak ada setup spot Bitget yang lolos filter hari ini.")
        return

    lines = []
    lines.append("🔥 *BITGET SPOT SCAN*")
    lines.append(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("*Top 5 setup:*")

    for i, x in enumerate(ranked, 1):
        lines.append(
            f"{i}. *{x['symbol']}* - score {x['score']}\n"
            f"   Entry {x['entry']} | SL {x['stop']} | TP1 {x['tp1']} | TP2 {x['tp2']} | TP3 {x['tp3']}\n"
            f"   Price {x['price']} | RSI {x['rsi']} | MACD hist {x['macd_hist']} | Vol {x['volume_usdt']}"
        )

    send_telegram("\n".join(lines))

if __name__ == "__main__":
    main()
              
