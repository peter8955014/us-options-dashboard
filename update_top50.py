import io
import json
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# 內建美股最大市值候選池（確保即使 Wikipedia 無法連線也能完整抓齊 Top 50 巨頭）
MEGA_CAP_POOL = [
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "LLY",
    "JPM", "WMT", "V", "XOM", "UNH", "ORCL", "MA", "COST", "PG", "HD",
    "NFLX", "JNJ", "BAC", "CRM", "ABBV", "CVX", "KO", "MRK", "WFC", "CSCO",
    "AMD", "PEP", "NOW", "ADBE", "LIN", "DIS", "MCD", "IBM", "PM", "ABT",
    "TMO", "GE", "CAT", "ISRG", "INTU", "TXN", "QCOM", "VZ", "BKNG", "SBUX",
    "GS", "MS", "AXP", "BLK", "RTX", "SPGI", "UBER", "AMAT", "PLTR", "T",
    "PFE", "LOW", "NEE", "HON", "UNP", "SYK", "ETN", "C", "COP", "LMT",
    "BA", "DE", "MDT", "PLD", "ANET", "PANW", "MU", "ADI", "KLAC", "LRCX"
]

BENCHMARK_ETFS = [
    {"sym": "SPY", "default_name": "SPDR S&P 500 ETF (標普大盤)", "default_pe": 26.5},
    {"sym": "QQQ", "default_name": "Invesco QQQ ETF (那斯達克)", "default_pe": 33.8},
    {"sym": "SOXL", "default_name": "3倍做多半導體 ETF (教學基準)", "default_pe": 38.0}
]


def load_existing_data(filepath="top50.json"):
    """讀取現有 top50.json，保留中文名稱與作為異常時的保底數值"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["sym"]: item for item in data if "sym" in item}
    except Exception:
        return {}


def fetch_sp500_tickers():
    """附加瀏覽器 User-Agent 讀取 Wikipedia S&P 500 清單，避免 403 Forbidden"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        tickers = [str(t).strip().replace(".", "-") for t in tables[0]["Symbol"].tolist()]
        print(f"成功從 Wikipedia 取得 {len(tickers)} 檔 S&P 500 成分股")
        return tickers
    except Exception as e:
        print(f"Wikipedia 讀取失敗 ({e})，改用內建市值巨頭候選池")
        return []


def format_mcap(num):
    if not num or num <= 0:
        return "N/A"
    if num >= 1e12:
        return f"{num / 1e12:.2f}T"
    if num >= 1e9:
        return f"{round(num / 1e9)}B"
    return str(int(num))


def extract_close_series(df, sym):
    """相容不同版本 yfinance 的 MultiIndex 欄位格式，安全取出收盤價序列"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if sym in df.columns.get_level_values(0):
            sub = df[sym]
            if "Close" in sub.columns:
                return sub["Close"].dropna()
        elif sym in df.columns.get_level_values(1):
            sub = df.xs(sym, axis=1, level=1)
            if "Close" in sub.columns:
                return sub["Close"].dropna()
    elif "Close" in df.columns:
        return df["Close"].dropna()
    return pd.Series(dtype=float)


def calc_price_and_vol(close_series, fallback_item=None):
    """從 1 年日收盤價計算最新股價與 21 日滾動年化波動率區間"""
    fb = fallback_item or {}
    if len(close_series) >= 25:
        price = round(float(close_series.iloc[-1]), 2)
        log_ret = np.log(close_series / close_series.shift(1))
        rolling_vol = (log_ret.rolling(window=21).std() * np.sqrt(252) * 100).dropna()
        if len(rolling_vol) > 0:
            iv = max(5, round(float(rolling_vol.iloc[-1])))
            iv_low = max(5, round(float(rolling_vol.min())))
            iv_high = max(iv_low + 1, round(float(rolling_vol.max())))
            return price, iv, iv_low, iv_high

    return (
        fb.get("price", 100.0),
        fb.get("iv", 30),
        fb.get("ivLow", 20),
        fb.get("ivHigh", 45),
    )


def generate_top50_json():
    existing_map = load_existing_data()
    wiki_tickers = fetch_sp500_tickers()

    # 合併內建 80 檔 mega-cap 與 Wikipedia 清單（去重並保持順序）
    seen = set()
    candidates = []
    for sym in MEGA_CAP_POOL + wiki_tickers:
        if sym and sym not in seen:
            seen.add(sym)
            candidates.append(sym)
        if len(candidates) >= 85:
            break

    etf_syms = [e["sym"] for e in BENCHMARK_ETFS]
    all_download_syms = candidates + [s for s in etf_syms if s not in seen]

    print(f"開始批次下載 {len(all_download_syms)} 檔標的過去 1 年歷史行情...")
    hist_df = yf.download(
        tickers=all_download_syms,
        period="1y",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    results = []
    for sym in candidates:
        fb = existing_map.get(sym, {})
        close_s = extract_close_series(hist_df, sym)
        price, iv, iv_low, iv_high = calc_price_and_vol(close_s, fb)

        mcap_val = 0
        pe_val = fb.get("pe", 25.0)
        name_val = fb.get("name", sym)

        try:
            tk = yf.Ticker(sym)
            f_info = tk.fast_info
            mcap_val = float(f_info.get("market_cap") or 0)

            info = tk.info or {}
            raw_pe = info.get("trailingPE") or info.get("forwardPE")
            if raw_pe and 0 < float(raw_pe) < 1000:
                pe_val = round(float(raw_pe), 1)
            if sym not in existing_map and info.get("shortName"):
                name_val = info["shortName"]
        except Exception as e:
            print(f"  [略過部分基本面] {sym}: {e}")

        # 若 Yahoo 暫時未回傳市值，從舊資料字串反推避免排名遺失
        if mcap_val <= 0 and isinstance(fb.get("mcap"), str):
            m_str = fb["mcap"]
            if m_str.endswith("T"):
                mcap_val = float(m_str[:-1]) * 1e12
            elif m_str.endswith("B"):
                mcap_val = float(m_str[:-1]) * 1e9

        if price > 0 and mcap_val > 0:
            results.append({
                "sym": sym,
                "name": name_val,
                "marketCap": mcap_val,
                "mcap": format_mcap(mcap_val),
                "price": price,
                "pe": pe_val,
                "iv": iv,
                "ivLow": iv_low,
                "ivHigh": iv_high,
            })

        time.sleep(0.1)

    # 依最新市值由大到小動態排序，取前 50 名
    results.sort(key=lambda x: x["marketCap"], reverse=True)
    top50 = results[:50]

    for idx, item in enumerate(top50):
        item["rank"] = idx + 1
        item.pop("marketCap", None)

    # 動態更新 SPY、QQQ、SOXL 最新股價與波動率
    for idx, etf in enumerate(BENCHMARK_ETFS):
        sym = etf["sym"]
        fb = existing_map.get(sym, {})
        close_s = extract_close_series(hist_df, sym)
        price, iv, iv_low, iv_high = calc_price_and_vol(close_s, fb)
        top50.append({
            "rank": 51 + idx,
            "sym": sym,
            "name": fb.get("name", etf["default_name"]),
            "mcap": "ETF",
            "price": price,
            "pe": fb.get("pe", etf["default_pe"]),
            "iv": iv,
            "ivLow": iv_low,
            "ivHigh": iv_high,
        })

    with open("top50.json", "w", encoding="utf-8") as f:
        json.dump(top50, f, ensure_ascii=False, indent=2)

    print(f"更新完成！共寫入 {len(top50)} 筆最新標的資料至 top50.json")


if __name__ == "__main__":
    generate_top50_json()
