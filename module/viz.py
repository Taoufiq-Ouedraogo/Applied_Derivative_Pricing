# module/viz.py


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    norm, skew, kurtosis,
    jarque_bera, normaltest,
    shapiro, ttest_1samp
)

from module.pricer import (
    mc_convergence
)



def plot_model_paths(
    S_paths, V_paths=None, S0=None,
    K=None, T=None, title_prefix="Model"
):
    paths, steps_plus_1 = S_paths.shape
    steps = steps_plus_1 - 1
    time_grid = np.linspace(0, T, steps_plus_1)

    # Log returns
    R = np.log(S_paths[:, -1] / S_paths[:, 0])
    mu_R = np.mean(R)
    sigma_R = np.std(R)
    x_R = np.linspace(R.min(), R.max(), 200)

    # Price paths
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes[0, 0].plot(time_grid, S_paths.T, linewidth=0.5)
    if K is not None:
        axes[0, 0].axhline(K, color='red', linewidth=1.5, label=f"K: {K}")
        axes[0, 0].legend()
    axes[0, 0].set_title(f"{title_prefix} Price Paths")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("$S_t$")
    axes[0, 0].grid()
    # Returns distribution
    axes[0, 1].hist(R, bins=50, density=True)
    axes[0, 1].plot(x_R, norm.pdf(x_R, mu_R, sigma_R),
                    color='r', label="Normal density")
    axes[0, 1].set_title("Log-Returns Distribution")
    axes[0, 1].legend()

    # Volatility paths (if available)
    if V_paths is not None:
        axes[1, 0].plot(time_grid, V_paths.T, linewidth=0.5)
        axes[1, 0].set_title(f"{title_prefix} Volatility Paths")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("$V_t$")
        axes[1, 0].grid()
        # Volatility distribution
        V_T = V_paths[:, -1]
        mu_V = np.mean(V_T)
        sigma_V = np.std(V_T)
        x_V = np.linspace(V_T.min(), V_T.max(), 200)
        axes[1, 1].hist(V_T, bins=50, density=True)
        axes[1, 1].plot(x_V, norm.pdf(x_V, mu_V, sigma_V),
                        color='r', label="Normal density")
        axes[1, 1].set_title("Volatility Distribution")
        axes[1, 1].legend()
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
    plt.tight_layout()
    plt.show()



def plot_three_models_paths(
    S_paths_bs, S_paths_heston, S_paths_merton,
    K, T
):
    models = [
        ("Black-Scholes", S_paths_bs),
        ("Heston",        S_paths_heston),
        ("Merton",        S_paths_merton),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    steps_plus_1 = S_paths_bs.shape[1]
    time_grid = np.linspace(0, T, steps_plus_1)
    for col, (name, S_paths) in enumerate(models):
        # Price paths
        ax_path = axes[0, col]
        ax_path.plot(time_grid, S_paths.T, linewidth=0.5, alpha=0.6)
        ax_path.axhline(K, color="red", linewidth=1.5, label=f"K = {K}")
        ax_path.legend(fontsize=8)
        ax_path.set_title(f"{name}")
        ax_path.set_xlabel("Time")
        ax_path.set_ylabel("$S_t$")
        ax_path.grid(True, linewidth=0.4)
        # Log returns
        ax_dist = axes[1, col]
        R = np.log(S_paths[:, -1] / S_paths[:, 0])
        mu_R, sigma_R = np.mean(R), np.std(R)
        x_R = np.linspace(R.min(), R.max(), 200)

        ax_dist.hist(R, bins=50, density=True, alpha=0.7)
        ax_dist.plot(x_R, norm.pdf(x_R, mu_R, sigma_R), color='r',
                     linewidth=1.5, label="Normal fit")
        ax_dist.legend()
        ax_dist.set_title(f"{name} - log-returns")
        ax_dist.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()

    # Statistical tests
    results = []
    for name, S_paths in models:
        ST = S_paths[:, -1]
        R = np.log(ST / S_paths[:, 0])
        # Moments
        mean_R = np.mean(R)
        std_R = np.std(R)
        skew_R = skew(R)
        kurt_R = kurtosis(R)
        # statistical tests
        jb_stat, jb_p = jarque_bera(R)
        shapiro_stat, shapiro_p = shapiro(R)
        dagostino_stat, dagostino_p = normaltest(R)
        tt_stat, tt_p = ttest_1samp(R, popmean=0)

        results.append({
            "Model": name, "Mean_Return": mean_R, "Std_Return": std_R,
            "Skewness": skew_R, "Kurtosis": kurt_R, "JarqueBera_pvalue": jb_p,
            "Shapiro_pvalue": shapiro_p, "Dagostino_pvalue": dagostino_p,
            "ttest_pvalue": tt_p,
        })
    df_stats = pd.DataFrame(results).set_index("Model")
    return df_stats



def plot_cf_comparison(cf_partial_dict, u_min=-4, u_max=4, n=200):
    u = np.linspace(u_min, u_max, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Real part
    for name, cf in cf_partial_dict.items():
        values = cf(u)
        axes[0].plot(u, np.real(values), label=name, alpha=0.5)

    axes[0].set_title("Real part")
    axes[0].set_xlabel("u")
    axes[0].grid()
    axes[0].legend()
    # Imag part
    for name, cf in cf_partial_dict.items():
        values = cf(u)
        axes[1].plot(u, np.imag(values), label=name, alpha=0.5)
    axes[1].set_title("Imaginary part")
    axes[1].set_xlabel("u")
    axes[1].grid()
    axes[1].legend()
    plt.suptitle("Characteristic Functions Comparison")
    plt.tight_layout()
    plt.show()



def plot_characteristic_function(cf_partial, model_name=None):
    u = np.linspace(-4, 4, 200)
    cf_values = cf_partial(u)
    plt.figure(figsize=(10, 4))
    plt.plot(u, np.real(cf_values), label="Real part")
    plt.plot(u, np.imag(cf_values), label="Imaginary part", linestyle="--")
    plt.title(f"{model_name} Characteristic Function")
    plt.xlabel("u")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_mc_convergence2(S_paths, K, r, T,
                        benchmark_call, benchmark_put, model_name="Model"):
    # Compute convergence
    iters_call, prices_call, se_call = mc_convergence(S_paths, K, r, T, "call")
    iters_put,  prices_put,  se_put  = mc_convergence(S_paths, K, r, T, "put")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, iters, prices, se, benchmark, label in [
        (axes[0], iters_call, prices_call, se_call, benchmark_call, "Call"),
        (axes[1], iters_put,  prices_put,  se_put,  benchmark_put,  "Put"),
    ]:
        # MC price line
        ax.plot(iters, prices, color="steelblue", linewidth=1.5, label="MC Price")
        # Confidence band
        ax.fill_between(iters, np.array(prices) - np.array(se),
            np.array(prices) + np.array(se), alpha=0.25, color="steelblue", label="SE")
        # Benchmark line
        ax.axhline(benchmark, color="red", linestyle="--", linewidth=1.5, label="Fourier")
        ax.set_title(f"{model_name} - {label} Convergence")
        ax.set_xlabel("Number of paths")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    #plt.show()
    return axes



def plot_mc_convergence1(S_paths, K, r, T, fourier_call_price,
                        fourier_put_price, model_name="Model"
):
    # Compute convergence
    iters_call, prices_call = mc_convergence(S_paths, K, r, T, "call")
    iters_put, prices_put = mc_convergence(S_paths, K, r, T, "put")
    # Call
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(iters_call, prices_call, label="Monte Carlo")
    axes[0].axhline(fourier_call_price, color="r", linewidth=2,
                    linestyle="--", label="Fourier", alpha=0.7)
    axes[0].set_title(f"{model_name} - Call Convergence")
    axes[0].set_xlabel("Number of paths")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid()
    # Put
    axes[1].plot(iters_put, prices_put, label="Monte Carlo")
    axes[1].axhline(fourier_put_price, color="r", linewidth=2,
                    linestyle="--", label="Fourier", alpha=0.7)
    axes[1].set_title(f"{model_name} - Put Convergence")
    axes[1].set_xlabel("Number of paths")
    axes[1].set_ylabel("Price")
    axes[1].legend()
    axes[1].grid()
    plt.tight_layout()
    return axes



def plot_calibration_results(
    df_calib_call, df_calib_put, mse_call, mse_put,
    price_col="model_price_fourier",
    residual_col="residual_fourier",
    model_name="BS Fourier"
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    configs = [
        (df_calib_call, mse_call, "Call", 0),
        (df_calib_put, mse_put,  "Put", 1),
    ]
    for df, mse, label, col in configs:
        df = df.sort_values("strike")
        # Prices
        axes[0, col].plot(df["strike"], df["lastPrice"],
                          marker="o", label="Market price")
        axes[0, col].plot(df["strike"], df[price_col],
                          marker="s", linestyle="--", label=model_name)
        axes[0, col].set_title(f"{label} Prices - {model_name} Calibration")
        axes[0, col].set_xlabel("Strike")
        axes[0, col].set_ylabel("Price")
        axes[0, col].legend()
        axes[0, col].grid(True)
        # Residuals
        axes[1, col].stem(df["strike"], df[residual_col],
                          linefmt="red", markerfmt="o", basefmt="k-")
        axes[1, col].set_title(f"{label} Residuals | RMSE={np.sqrt(mse):.4f}")
        axes[1, col].set_xlabel("Strike")
        axes[1, col].set_ylabel("Residual")
        axes[1, col].grid(True)
    plt.tight_layout()
    plt.show()



def plot_option_diagnostics(df_call_, df_put_, mat):
    df_call = df_call_[df_call_["time_to_maturity (years)"]==mat]
    df_put = df_put_[df_put_["time_to_maturity (years)"]==mat]
    print(f"Call options shape: {df_call.shape}")
    print(f"Put options shape: {df_put.shape}\n")
    cols = ["volume", "impliedVolatility", "strike", "lastPrice", "openInterest"]
    n = len(cols)

    fig, axes = plt.subplots(3, n, figsize=(4*n, 9))
    for i, col in enumerate(cols):
        # Boxplot
        data_to_plot = [df_call[col].dropna(), df_put[col].dropna()]
        axes[0, i].boxplot(data_to_plot, labels=["Call", "Put"])
        axes[0, i].set_title(col)
        axes[0, i].set_xlabel("")
        # Lineplot
        axes[1, i].plot(df_call[col].values, label="Call")
        axes[1, i].plot(df_put[col].values, label="Put")
        axes[1, i].legend()
        # Histogram
        axes[2, i].hist(df_call[col].dropna(), bins=40, label="Call", density=True)
        axes[2, i].hist(df_put[col].dropna(), bins=40, label="Put", density=True)
        axes[2, i].legend()
    plt.tight_layout()
    plt.show()



def plot_volatility_smile(df_call_, df_put_, mat):
    df_call = df_call_[df_call_["time_to_maturity (years)"]==mat]
    df_put = df_put_[df_put_["time_to_maturity (years)"]==mat]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Volatility Smile
    axes[0].plot(df_call["strike"], df_call["impliedVolatility"],
                 label="Calls", marker="o", linestyle='--')
    axes[0].plot(df_put["strike"], df_put["impliedVolatility"],
                 label="Puts", marker="o", linestyle='--')
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("Implied Volatility")
    axes[0].set_title("Volatility Smile")
    axes[0].legend()
    axes[0].grid(True)
    # Price vs Strike
    axes[1].plot(df_call["strike"], df_call["lastPrice"],
                 label="Calls", marker="o", linestyle='--')
    axes[1].plot(df_put["strike"], df_put["lastPrice"],
                 label="Puts", marker="o", linestyle='--')
    axes[1].set_xlabel("Strike")
    axes[1].set_ylabel("Option Price")
    axes[1].set_title("Price vs Strike")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()



def plot_vol_surface(df):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(df["time_to_maturity (years)"], df["strike"],
                           df["impliedVolatility"], cmap=plt.cm.RdBu_r)
    ax.set_xlabel("time_to_maturity (years)")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Implied Volatility (%)")
    fig.colorbar(surf)
    plt.title("Volatility Surface")
    plt.tight_layout()
    plt.show()



def plot_vol_surface_2x2(df_call, df_put):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    dfc = df_call.copy()
    dfp = df_put.copy()
    mats_call = dfc["time_to_maturity (years)"].unique()[-5:]
    mats_put = dfp["time_to_maturity (years)"].unique()[-5:]
    def plot_block(ax_iv, ax_price, df, mats, title_prefix):

        for m in mats:
            tmp = df[df["time_to_maturity (years)"] == m].sort_values("strike")

            ax_iv.plot(tmp["strike"], tmp["impliedVolatility"],
                       marker="o", linestyle="--", label=f"T={m:.3f}")

            ax_price.plot(tmp["strike"], tmp["lastPrice"],
                          marker="o", linestyle="--", label=f"T={m:.3f}")

        ax_iv.set_title(f"{title_prefix} IV Smile")
        ax_iv.set_xlabel("Strike")
        ax_iv.set_ylabel("Implied Volatility")
        ax_iv.grid(True)
        ax_iv.legend()

        ax_price.set_title(f"{title_prefix} Price")
        ax_price.set_xlabel("Strike")
        ax_price.set_ylabel("Price")
        ax_price.grid(True)
        ax_price.legend()
    plot_block(axes[0, 0], axes[0, 1], dfc, mats_call, "Call")
    plot_block(axes[1, 0], axes[1, 1], dfp, mats_put, "Put")
    plt.tight_layout()
    plt.show()


