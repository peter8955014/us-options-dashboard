import json
import yfinance as yf
import pandas as pd
import numpy as np

def generate_top50_json():
    # 1. 從 Wikipedia 讀取最新 S&P 500 成分股清單
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]
    tickers = [t.replace('.', '-') for t in sp500_table['Symbol'].tolist()]

    # 為加速篩選，可先用預篩候選池或批次抓取市值
    # 這裡示範對候選龍頭股動態抓取最新市值並自動重排 Top 50
    watch_pool = tickers[:100] # 也可放入全部 tickers
    results = []

    for sym in watch_pool:
        try:
            tk = yf.Ticker(sym)
            info = tk.fast_info
            full_info = tk.info
            mcap = info.get('market_cap', 0)
            price = info.get('last_price', 0)
            pe = full_info.get('trailingPE') or full_info.get('forwardPE') or 25.0
            name = full_info.get('shortName', sym)

            # 透過過去 1 年日K線滾動歷史波動率估算當前波動率與 52 週高低區間
            hist = tk.history(period="1y")
            if len(hist) > 30:
                log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
                rolling_vol = log_ret.rolling(window=21).std() * np.sqrt(252) * 100
                rolling_vol = rolling_vol.dropna()
                iv = round(float(rolling_vol.iloc[-1]))
                iv_low = round(float(rolling_vol.min()))
                iv_high = round(float(rolling_vol.max()))
            else:
                iv, iv_low, iv_high = 30, 20, 45

            results.append({
                "sym": sym,
                "name": name,
                "marketCap": mcap,
                "price": round(float(price), 2),
                "pe": round(float(pe), 1),
                "iv": iv,
                "ivLow": iv_low,
                "ivHigh": iv_high
            })
        except Exception:
            continue

    # 2. 依最新市值由大到小動態排序，取前 50 名
    results.sort(key=lambda x: x["marketCap"], reverse=True)
    top50 = results[:50]
    for idx, item in enumerate(top50):
        item["rank"] = idx + 1

    # 3. 附加常用 ETF 基準 (SPY, QQQ, SOXL)
    top50.extend([
        {"rank": 51, "sym": "SPY", "name": "SPDR S&P 500 ETF", "mcap": "ETF", "price": 595.0, "pe": 26.5, "iv": 14, "ivLow": 10, "ivHigh": 28},
        {"rank": 52, "sym": "QQQ", "name": "Invesco QQQ ETF", "mcap": "ETF", "price": 512.0, "pe": 33.8, "iv": 19, "ivLow": 13, "ivHigh": 34},
        {"rank": 53, "sym": "SOXL", "name": "3倍做多半導體 ETF (教學基準)", "mcap": "ETF", "price": 90.0, "pe": 38.0, "iv": 95, "ivLow": 65, "ivHigh": 145}
    ])

    with open("top50.json", "w", encoding="utf-8") as f:
        json.dump(top50, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_top50_json()
