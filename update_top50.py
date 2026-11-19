import io
import json
import math
import time
from datetime import datetime, timezone, date, timedelta
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# 美股最大市值候選池（確保穩定抓齊前 50 大權值股）
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

# 額外高流動性 ETF 與熱門期權觀察清單
EXTRA_WATCHLIST = [
    {"sym": "SPY", "default_name": "SPDR S&P 500 ETF (標普大盤)", "default_pe": 26.5, "is_etf": True},
    {"sym": "QQQ", "default_name": "Invesco QQQ ETF (那斯達克)", "default_pe": 33.8, "is_etf": True},
    {"sym": "IWM", "default_name": "iShares 羅素 2000 小型股 ETF", "default_pe": 22.0, "is_etf": True},
    {"sym": "TLT", "default_name": "iShares 20年期以上美國公債 ETF", "default_pe": 15.0, "is_etf": True},
    {"sym": "SOXL", "default_name": "3倍做多半導體 ETF (教學基準)", "default_pe": 38.0, "is_etf": True},
    {"sym": "TQQQ", "default_name": "3倍做多那斯達克 100 ETF", "default_pe": 34.0, "is_etf": True},
    {"sym": "SMH", "default_name": "VanEck 半導體 ETF", "default_pe": 35.5, "is_etf": True},
    {"sym": "PLTR", "default_name": "Palantir Technologies", "default_pe": 110.0, "is_etf": False},
    {"sym": "COIN", "default_name": "Coinbase Global", "default_pe": 32.0, "is_etf": False},
    {"sym": "MSTR", "default_name": "Strategy (MicroStrategy)", "default_pe": 45.0, "is_etf": False},
]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def calc_bsm_delta(S, K, dte_days, sigma, opt_type="put", r=0.045):
    T = max(0.001, dte_days / 365.0)
    sigma = max(0.02, sigma)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    nd1 = norm_cdf(d1)
    return nd1 if opt_type == "call" else (nd1 - 1.0)


def load_existing_data(filepath="top50.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["sym"]: item for item in data if "sym" in item}
    except Exception:
        return {}


def fetch_sp500_tickers():
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


def calc_hv_stats(close_series, fallback_item=None):
    fb = fallback_item or {}
    if len(close_series) >= 25:
        price = round(float(close_series.iloc[-1]), 2)
        log_ret = np.log(close_series / close_series.shift(1))
        rolling_vol = (log_ret.rolling(window=21).std() * np.sqrt(252) * 100).dropna()
        if len(rolling_vol) > 0:
            hv = max(5, round(float(rolling_vol.iloc[-1])))
            hv_low = max(5, round(float(rolling_vol.min())))
            hv_high = max(hv_low + 1, round(float(rolling_vol.max())))
            return price, hv, hv_low, hv_high

    return (
        fb.get("price", 100.0),
        fb.get("iv", 30),
        fb.get("ivLow", 20),
        fb.get("ivHigh", 45),
    )


def pick_best_otm_contract(df, spot_price, dte, opt_type="put", target_abs_delta=0.20):
    """從期權鏈挑選最接近 Delta 0.20 的價外合約並計算年化收租殖利率"""
    if df is None or df.empty or spot_price <= 0 or dte <= 0:
        return None

    best_row = None
    best_diff = 999.0

    for _, row in df.iterrows():
        strike = float(row.get("strike") or 0)
        if strike <= 0:
            continue
        if opt_type == "put" and strike >= spot_price * 0.99:
            continue
        if opt_type == "call" and strike <= spot_price * 1.01:
            continue

        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        last_p = float(row.get("lastPrice") or 0)
        prem = round((bid + ask) / 2.0, 2) if (bid > 0 and ask > 0) else round(last_p, 2)
        if prem <= 0.02:
            continue

        raw_iv = float(row.get("impliedVolatility") or 0)
        if not (0.05 <= raw_iv <= 3.0):
            continue

        delta = calc_bsm_delta(spot_price, strike, dte, raw_iv, opt_type=opt_type)
        abs_d = abs(delta)
        if not (0.08 <= abs_d <= 0.35):
            continue

        diff = abs(abs_d - target_abs_delta)
        if diff < best_diff:
            best_diff = diff
            capital_base = strike if opt_type == "put" else spot_price
            roc_pct = round((prem / capital_base) * 100.0, 2)
            ann_yield = round((prem / capital_base) * (365.0 / max(1, dte)) * 100.0, 1)
            best_row = {
                "strike": round(strike, 2),
                "prem": prem,
                "delta": round(delta, 2),
                "iv": round(raw_iv * 100.0),
                "rocPct": roc_pct,
                "annYield": ann_yield,
            }

    return best_row


def fetch_option_chain_snapshot(tk, spot_price):
    """一次讀取期權鏈，同時萃取 ATM IV 與最佳 30D CSP / Covered Call 合約"""
    try:
        expirations = tk.options
        if not expirations or spot_price <= 0:
            return None, None, None

        today = datetime.now(timezone.utc).date()
        best_exp = None
        best_dte = 30
        best_diff = 9999

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if 14 <= dte <= 65:
                diff = abs(dte - 30)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str
                    best_dte = dte

        if not best_exp:
            best_exp = expirations[0]
            best_dte = max(1, (datetime.strptime(best_exp, "%Y-%m-%d").date() - today).days)

        chain = tk.option_chain(best_exp)
        iv_candidates = []

        for side_df in (chain.calls, chain.puts):
            if side_df is None or side_df.empty:
                continue
            df_valid = side_df.copy()
            df_valid["dist"] = (df_valid["strike"] - spot_price).abs()
            near_atm = df_valid.nsmallest(2, "dist")
            for _, row in near_atm.iterrows():
                raw_iv = row.get("impliedVolatility")
                if pd.notna(raw_iv) and 0.05 <= float(raw_iv) <= 3.0:
                    iv_candidates.append(float(raw_iv) * 100.0)

        atm_iv = round(float(np.mean(iv_candidates))) if iv_candidates else None

        best_csp = pick_best_otm_contract(chain.puts, spot_price, best_dte, opt_type="put", target_abs_delta=0.20)
        if best_csp:
            best_csp["exp"] = best_exp
            best_csp["dte"] = best_dte

        best_cc = pick_best_otm_contract(chain.calls, spot_price, best_dte, opt_type="call", target_abs_delta=0.20)
        if best_cc:
            best_cc["exp"] = best_exp
            best_cc["dte"] = best_dte

        return atm_iv, best_csp, best_cc
    except Exception:
        return None, None, None


def fetch_earnings_info(tk, info_dict):
    today = datetime.now(timezone.utc).date()
    candidates = []

    try:
        cal = tk.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            ed_list = cal["Earnings Date"]
            if not isinstance(ed_list, (list, tuple)):
                ed_list = [ed_list]
            for item in ed_list:
                if isinstance(item, datetime):
                    candidates.append(item.date())
                elif isinstance(item, date):
                    candidates.append(item)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                for val in cal.loc["Earnings Date"].values:
                    ts = pd.to_datetime(val, errors="coerce")
                    if pd.notna(ts):
                        candidates.append(ts.date())
    except Exception:
        pass

    for key in ("earningsTimestamp", "earningsTimestampStart"):
        ts_val = info_dict.get(key)
        if ts_val:
            try:
                d = datetime.fromtimestamp(int(ts_val), tz=timezone.utc).date()
                candidates.append(d)
            except Exception:
                pass

    future_dates = sorted([d for d in candidates if d >= today])
    if future_dates:
        next_ed = future_dates[0]
        return next_ed.strftime("%Y-%m-%d"), (next_ed - today).days
    return None, None


def generate_top50_json():
    existing_map = load_existing_data()
    wiki_tickers = fetch_sp500_tickers()

    seen = set()
    candidates = []
    for sym in MEGA_CAP_POOL + wiki_tickers:
        if sym and sym not in seen:
            seen.add(sym)
            candidates.append(sym)
        if len(candidates) >= 85:
            break

    watch_syms = [w["sym"] for w in EXTRA_WATCHLIST]
    all_download_syms = candidates + [s for s in watch_syms if s not in seen]

    print(f"開始批次下載 {len(all_download_syms)} 檔標的過去 1 年歷史行情...")
    hist_df = yf.download(
        tickers=all_download_syms,
        period="1y",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    latest_trade_date = ""
    if hist_df is not None and not hist_df.empty:
        latest_trade_date = pd.to_datetime(hist_df.index[-1]).strftime("%Y-%m-%d")
    tw_now = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
    updated_at_label = f"{latest_trade_date} 美股收盤 (同步於 {tw_now})" if latest_trade_date else tw_now

    results = []
    for sym in candidates:
        fb = existing_map.get(sym, {})
        close_s = extract_close_series(hist_df, sym)
        price, hv, hv_low, hv_high = calc_hv_stats(close_s, fb)

        mcap_val = 0
        pe_val = fb.get("pe", 25.0)
        name_val = fb.get("name", sym)
        option_iv = None
        best_csp = fb.get("bestCsp")
        best_cc = fb.get("bestCc")
        earnings_date = fb.get("earningsDate")
        days_to_earnings = fb.get("daysToEarnings")

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

            opt_iv, csp_snap, cc_snap = fetch_option_chain_snapshot(tk, price)
            if opt_iv is not None:
                option_iv = opt_iv
            if csp_snap is not None:
                best_csp = csp_snap
            if cc_snap is not None:
                best_cc = cc_snap

            ed_str, ed_days = fetch_earnings_info(tk, info)
            if ed_str:
                earnings_date, days_to_earnings = ed_str, ed_days
        except Exception as e:
            print(f"  [略過部分基本面] {sym}: {e}")

        if mcap_val <= 0 and isinstance(fb.get("mcap"), str):
            m_str = fb["mcap"]
            if m_str.endswith("T"):
                mcap_val = float(m_str[:-1]) * 1e12
            elif m_str.endswith("B"):
                mcap_val = float(m_str[:-1]) * 1e9

        final_iv = option_iv if option_iv is not None else hv
        iv_source = "Option Chain" if option_iv is not None else "HV (21D)"
        iv_low = min(hv_low, final_iv, fb.get("ivLow", final_iv))
        iv_high = max(hv_high, final_iv, fb.get("ivHigh", final_iv))

        if price > 0 and mcap_val > 0:
            results.append({
                "sym": sym,
                "name": name_val,
                "category": "Top50",
                "marketCap": mcap_val,
                "mcap": format_mcap(mcap_val),
                "price": price,
                "pe": pe_val,
                "iv": final_iv,
                "ivLow": iv_low,
                "ivHigh": iv_high,
                "ivSource": iv_source,
                "bestCsp": best_csp,
                "bestCc": best_cc,
                "earningsDate": earnings_date,
                "daysToEarnings": days_to_earnings,
                "updatedAt": updated_at_label,
            })

        time.sleep(0.12)

    results.sort(key=lambda x: x["marketCap"], reverse=True)
    top50 = results[:50]
    top50_syms = set()

    for idx, item in enumerate(top50):
        item["rank"] = idx + 1
        item.pop("marketCap", None)
        top50_syms.add(item["sym"])

    extra_rank = 51
    for w in EXTRA_WATCHLIST:
        sym = w["sym"]
        if sym in top50_syms:
            continue

        fb = existing_map.get(sym, {})
        close_s = extract_close_series(hist_df, sym)
        price, hv, hv_low, hv_high = calc_hv_stats(close_s, fb)

        option_iv = None
        best_csp = fb.get("bestCsp")
        best_cc = fb.get("bestCc")
        mcap_str = "ETF" if w["is_etf"] else fb.get("mcap", "Watchlist")
        pe_val = fb.get("pe", w["default_pe"])
        earnings_date = None
        days_to_earnings = None

        try:
            tk = yf.Ticker(sym)
            opt_iv, csp_snap, cc_snap = fetch_option_chain_snapshot(tk, price)
            if opt_iv is not None:
                option_iv = opt_iv
            if csp_snap is not None:
                best_csp = csp_snap
            if cc_snap is not None:
                best_cc = cc_snap

            if not w["is_etf"]:
                f_info = tk.fast_info
                m_val = float(f_info.get("market_cap") or 0)
                if m_val > 0:
                    mcap_str = format_mcap(m_val)
                info = tk.info or {}
                raw_pe = info.get("trailingPE") or info.get("forwardPE")
                if raw_pe and 0 < float(raw_pe) < 1000:
                    pe_val = round(float(raw_pe), 1)
                ed_str, ed_days = fetch_earnings_info(tk, info)
                if ed_str:
                    earnings_date, days_to_earnings = ed_str, ed_days
        except Exception:
            pass

        final_iv = option_iv if option_iv is not None else hv
        iv_source = "Option Chain" if option_iv is not None else "HV (21D)"
        iv_low = min(hv_low, final_iv, fb.get("ivLow", final_iv))
        iv_high = max(hv_high, final_iv, fb.get("ivHigh", final_iv))

        top50.append({
            "rank": extra_rank,
            "sym": sym,
            "name": fb.get("name", w["default_name"]),
            "category": "ETF / 熱門期權",
            "mcap": mcap_str,
            "price": price,
            "pe": pe_val,
            "iv": final_iv,
            "ivLow": iv_low,
            "ivHigh": iv_high,
            "ivSource": iv_source,
            "bestCsp": best_csp,
            "bestCc": best_cc,
            "earningsDate": earnings_date,
            "daysToEarnings": days_to_earnings,
            "updatedAt": updated_at_label,
        })
        extra_rank += 1
        time.sleep(0.12)

    with open("top50.json", "w", encoding="utf-8") as f:
        json.dump(top50, f, ensure_ascii=False, indent=2)

    print(f"更新完成！共寫入 {len(top50)} 筆最新標的與真實收租報價至 top50.json ({updated_at_label})")


if __name__ == "__main__":
    generate_top50_json()
