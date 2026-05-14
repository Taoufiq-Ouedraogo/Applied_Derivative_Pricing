# module/utils.py


from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

from scipy.optimize import minimize

from module.pricer import(
    blackScholes_closed_price,
    blackScholes_fourier_price, heston_fourier_price, merton_fourier_price,
)



SEED = 42



def get_option_data(ticker, maturity=None, option_type="call"):
    ticker = yf.Ticker(ticker)
    # Get available maturities
    expirations = ticker.options
    if maturity is None:
        Mat = expirations[-1]
    else:
        Mat = maturity if isinstance(maturity, str) else maturity.strftime("%Y-%m-%d")
    # Fetch Spot price
    S0 = ticker.history(period="1d")["Close"].iloc[-1]

    # Load option chain
    opt = ticker.option_chain(Mat)
    df = opt.calls if option_type == "call" else opt.puts

    # Clean data
    df = df.dropna(subset=["impliedVolatility", "strike", "volume", "bid", "ask"])
    # mid price
    df["midPrice"] = (df["bid"] + df["ask"]) / 2

    df = df[
        # remove broken quotes
        (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"]) &
        # liquidity filter
        (df["volume"] > 5) & (df["openInterest"] > 10) &
        # moneyness filter
        (df["strike"] >= 0.7 * S0) & (df["strike"] <= 1.3 * S0) &
        # price filter
        (df["midPrice"] > 0.1) &
        # IV filter
        (df["impliedVolatility"] > 0.05) & (df["impliedVolatility"] < 2.5)
    ]
    # Add maturity in years
    Mat_date = datetime.strptime(Mat, "%Y-%m-%d").date()
    ttm = (Mat_date - date.today()).days / 365
    df["time_to_maturity (years)"] = ttm
    df["option_type"] = option_type
    return df



def get_volatility_surface_data(ticker, option_type="call"):
    tk = yf.Ticker(ticker)
    expirations = tk.options
    S0 = tk.history(period="1d")["Close"].iloc[-1]
    all_data = []
    for Mat in expirations:
        try:
            df = get_option_data(ticker, maturity=Mat, option_type=option_type)
            #all_data.append(df[["strike", "impliedVolatility", "time_to_maturity (years)"]])
            all_data.append(df)
        except:
            continue
    surface_df = pd.concat(all_data, axis=0)
    return surface_df, S0



def implied_rate_from_parity(C, P, S0, K, T, q):
    return - (1/T) * np.log((S0 * np.exp(-q*T) - (C - P)) / K)



def implied_volatility_newton(
    market_price, S0, K, T, r, q,
    option="call", pricer=blackScholes_closed_price,
    sigma_init=0.2, tol=1e-6, max_iter=200, h=1e-4,
):
    """
    Newton-Raphson IV calibration using BlackScholes Fourier pricer
    Vega is approximated by central finite differences since
    for Fourier pricer there is no closed-form derivative
    """
    sigma = sigma_init
    for _ in range(max_iter):
        sigma = np.clip(sigma, 1e-6, 5.0)

        # price at current sigma
        price = pricer(S0, K, T, r, q, sigma, option)
        # vega via central finite difference
        price_up = pricer(S0, K, T, r, q, sigma + h, option)
        price_down = pricer(S0, K, T, r, q, sigma - h, option)
        vega = (price_up - price_down) / (2 * h)
        vega = max(abs(vega), 1e-8)
        # Newton step
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        step = np.clip(diff / vega, -0.5, 0.5)
        sigma -= step
    return sigma



def calibrate_implied_volatility(
    df, T, S0, r, q=0.0, option="call", sigma_init=0.2,
    pricer=blackScholes_fourier_price
):
    """
    Calibrates BS Fourier IV for all strikes at a given maturity
    """
    tmp = df[df["time_to_maturity (years)"] == T].copy()
    ivs_fourier, ivs_closed_form = [], []
    fourier_prices, closed_prices, market_prices = [], [], []
    for _, row in tmp.iterrows():
        K_i = row["strike"]
        market_price = row["lastPrice"]

        # calibrate IV with fourier
        iv_fourier = implied_volatility_newton(
            market_price=market_price, S0=S0, K=K_i, T=T, r=r, q=q,
            option=option, sigma_init=sigma_init, pricer=pricer
        )
        ivs_fourier.append(iv_fourier)

        # price at calibrated IV with fourier
        fourier_prices.append(
            np.nan if np.isnan(iv_fourier) else
            pricer(S0, K_i, T, r, q, iv_fourier, option)
        )
        # raw market price
        market_prices.append(market_price)
    tmp["iv_fourier"] = ivs_fourier
    tmp["fourier_price"] = fourier_prices
    tmp["market_price"] = market_prices
    return tmp



def blackScholes_fourier_mse(params, df, S0, r, q, T, option="call"):
    """
    Objective function: MSE between BS Fourier prices and market prices
    params = [sigma]
    """
    sigma = params[0]
    if sigma <= 0:
        return 1e10  # penalise invalid vol
    errors = []
    for _, row in df.iterrows():
        K_i = row["strike"]
        mkt_i = row["lastPrice"]
        try:
            model_i = blackScholes_fourier_price(S0, K_i, T, r, q, sigma, option)
        except Exception:
            model_i = 0.0
        errors.append((model_i - mkt_i) ** 2)
    return np.mean(errors)



def calibrate_blackScholes_fourier(
    df, maturity, S0, r, q=0.0,
    option="call", sigma_init=0.25
):
    """
    Calibrate Black-Scholes implied vol (sigma) by minimising MSE of
    Fourier-based model prices vs. market prices
    """
    tmp = df[df["time_to_maturity (years)"] == maturity].copy()
    tmp = tmp.dropna(subset=["lastPrice", "strike"]).reset_index(drop=True)

    res = minimize(blackScholes_fourier_mse,
        x0 = [sigma_init],
        args = (tmp, S0, r, q, maturity, option),
        method = "L-BFGS-B", bounds = [(1e-4, 5.0)],
        options= {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500}
    )

    sigma_calib = res.x[0]
    mse = res.fun
    # Reprice every strike with calibrated sigma
    model_prices, residuals = [], []
    for _, row in tmp.iterrows():
        mp = blackScholes_fourier_price(
            S0, row["strike"], maturity, r, q, sigma_calib, option
        )
        model_prices.append(mp)
        residuals.append(mp - row["lastPrice"])
    tmp["model_price_fourier"] = model_prices
    tmp["residual_fourier"]    = residuals
    return res, tmp, sigma_calib, mse



def heston_fourier_mse(params, df, S0, r, q, T, option="call"):
    """
    Objective function: MSE between Heston Fourier prices and market prices
    params = [kappa, theta, sigma, rho, V0]
    """
    kappa, theta, sigma, rho, V0 = params

    # Feller condition: 2*kappa*theta > sigma^2
    if (2 * kappa * theta <= sigma**2 or
        V0 <= 0 or theta <= 0 or sigma <= 0 or
        not (-1 < rho < 1)):
        return 1e10
    errors = []
    for _, row in df.iterrows():
        K_i   = row["strike"]
        mkt_i = row["lastPrice"]
        try:
            model_i = heston_fourier_price(
                S0, K_i, T, r, q, kappa, theta, sigma, rho, V0, option
            )
        except Exception:
            model_i = 0.0
        errors.append((model_i - mkt_i) ** 2)
    return np.mean(errors)



def calibrate_heston_fourier(
    df, maturity, S0, r, q=0.0,
    option="call", n_restarts=5,
):
    """
    Calibrate Heston parameters by minimising MSE of
    Fourier-based model prices vs. market prices
    Uses multiple random restarts to avoid local minima
    """
    tmp = df[df["time_to_maturity (years)"] == maturity].copy()
    tmp = tmp.dropna(subset=["lastPrice", "strike"]).reset_index(drop=True)

    # Parameter bounds: [kappa, theta, sigma, rho, V0]
    bounds = [
        (0.1,  15.0),
        (0.01,  1.0),
        (0.01,  2.0),
        (-0.99, 0.99),
        (0.001, 1.0),
    ]

    best_res  = None
    best_mse  = np.inf
    np.random.seed(SEED)
    for i in range(n_restarts):
        if i == 0:
            # Sensible initial guess
            x0 = [2.0, 0.05, 0.3, -0.5, 0.05]
        else:
            # Random restarts drawn uniformly within bounds
            x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]

        res = minimize(
            heston_fourier_mse, x0=x0,
            args=(tmp, S0, r, q, maturity, option),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000}
        )
        if res.fun < best_mse:
            best_mse = res.fun
            best_res = res

    kappa_c, theta_c, sigma_c, rho_c, V0_c = best_res.x
    # Reprice every strike with calibrated params
    model_prices, residuals = [], []
    for _, row in tmp.iterrows():
        mp = heston_fourier_price(S0, row["strike"], maturity, r, q,
            kappa_c, theta_c, sigma_c, rho_c, V0_c, option_type=option
        )
        model_prices.append(mp)
        residuals.append(mp - row["lastPrice"])
    tmp["model_price_heston"] = model_prices
    tmp["residual_heston"]    = residuals
    params_dict = {
        "kappa": kappa_c, "theta": theta_c,
        "sigma": sigma_c, "rho":   rho_c, "V0": V0_c
    }
    return best_res, tmp, params_dict, best_mse



def merton_mse_fourier(params, df, S0, r, q, T, option="call"):
    """
    Objective function: MSE between Merton Fourier prices and market prices
    params = [sigma, lambda_j, mu_j, delta_j]
    """
    sigma, lambda_j, mu_j, delta_j = params
    if sigma <= 0 or lambda_j < 0 or delta_j <= 0:
        return 1e1
    errors = []
    for _, row in df.iterrows():
        K_i   = row["strike"]
        mkt_i = row["lastPrice"]
        try:
            model_i = merton_fourier_price(S0, K_i, T, r, q,
                sigma, lambda_j, mu_j, delta_j, option_type=option
            )
        except Exception:
            model_i = 0.0
        errors.append((model_i - mkt_i) ** 2)
    return np.mean(errors)



def calibrate_merton_fourier(df, maturity, S0, r, q=0.0,
    option="call", n_restarts=5,
):
    """
    Calibrate Merton Jump-Diffusion parameters by minimising MSE of
    Fourier-based model prices vs. market prices.
    params = [sigma, lambda_j, mu_j, delta_j]
    """
    tmp = df[df["time_to_maturity (years)"] == maturity].copy()
    tmp = tmp.dropna(subset=["midPrice", "strike"]).reset_index(drop=True)

    # Parameter bounds: [sigma, lambda_j, mu_j, delta_j]
    bounds = [
        (1e-4,  2.0),
        (0.0,   5.0),
        (-2.0,  2.0),
        (1e-4,  2.0),
    ]

    best_res = None
    best_mse = np.inf
    np.random.seed(SEED)
    for i in range(n_restarts):
        if i == 0:
            # Warm start: sensible initial guess
            x0 = [0.2, 0.5, -0.1, 0.15]
        else:
            x0 = [np.random.uniform(lo, hi) for lo, hi in bounds]

        res = minimize(merton_mse_fourier, x0=x0,
            args=(tmp, S0, r, q, maturity, option),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000}
        )
        if res.fun < best_mse:
            best_mse = res.fun
            best_res = res
    sigma_c, lambda_c, mu_c, delta_c = best_res.x
    # Reprice every strike with calibrated params
    model_prices, residuals = [], []
    for _, row in tmp.iterrows():
        mp = merton_fourier_price(S0, row["strike"], maturity, r, q,
            sigma_c, lambda_c, mu_c, delta_c, option_type=option
        )
        model_prices.append(mp)
        residuals.append(mp - row["lastPrice"])

    tmp["model_price_merton"] = model_prices
    tmp["residual_merton"]    = residuals
    params_dict = {
        "sigma": sigma_c, "lambda_j": lambda_c,
        "mu_j":  mu_c,    "delta_j":  delta_c
    }
    return best_res, tmp, params_dict, best_mse