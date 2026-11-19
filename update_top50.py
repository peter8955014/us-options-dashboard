def pick_best_otm_contract(df, spot_price, dte, opt_type="put", target_abs_delta=0.20, fallback_sigma=0.35):
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
        # Yahoo 收盤後常將 IV 重設為 0.00001，此時改用標的基準波動率計算 Delta，避免整張合約被略過
        sigma = raw_iv if (0.05 <= raw_iv <= 3.0) else max(0.08, fallback_sigma)

        delta = calc_bsm_delta(spot_price, strike, dte, sigma, opt_type=opt_type)
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
                "iv": round(sigma * 100.0),
                "rocPct": roc_pct,
                "annYield": ann_yield,
            }

    return best_row


def fetch_option_chain_snapshot(tk, spot_price, fallback_hv=35):
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
        base_sigma = (atm_iv if atm_iv is not None else fallback_hv) / 100.0

        best_csp = pick_best_otm_contract(chain.puts, spot_price, best_dte, opt_type="put", target_abs_delta=0.20, fallback_sigma=base_sigma)
        if best_csp:
            best_csp["exp"] = best_exp
            best_csp["dte"] = best_dte

        best_cc = pick_best_otm_contract(chain.calls, spot_price, best_dte, opt_type="call", target_abs_delta=0.20, fallback_sigma=base_sigma)
        if best_cc:
            best_cc["exp"] = best_exp
            best_cc["dte"] = best_dte

        return atm_iv, best_csp, best_cc
    except Exception:
        return None, None, None
