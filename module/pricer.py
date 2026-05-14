# module/pricer.py


import numpy as np

from functools import partial

from scipy.stats import (
    norm, skew, kurtosis,
    jarque_bera, normaltest,
    shapiro, ttest_1samp
)

from scipy.integrate import quad



SEED = 42



def compute_call_payoff(ST, K):
    return np.maximum(ST - K, 0)



def compute_put_payoff(ST, K):
    return np.maximum(K - ST, 0)



def mc_vanilla_price(S_paths, K, r, T, option="call"):
    ST = S_paths[:, -1]
    if option == "call":
        payoff = compute_call_payoff(ST, K)
    else:
        payoff = compute_put_payoff(ST, K)
    disc_payoff = np.exp(-r * T) * payoff
    price = np.mean(disc_payoff)
    std_err = np.std(disc_payoff) / np.sqrt(len(disc_payoff))
    return price, std_err



def check_put_call_parity(C, P, S0, K, r, q, T):
    lhs = C - P
    rhs = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    lhs = lhs
    rhs = rhs
    error = lhs - rhs
    print("LHS:", lhs)
    print("RHS:", rhs)
    print("Parity error:", error)
    return lhs, rhs, error



def mc_put_call_parity_check(S_paths, K, r, T):
    ST = S_paths[:, -1]
    call = np.maximum(ST - K, 0)
    put  = np.maximum(K - ST, 0)

    lhs = np.exp(-r * T) * np.mean(call - put)
    rhs = np.exp(-r * T) * np.mean(ST - K)
    error = lhs - rhs
    print("MC LHS:", lhs)
    print("MC RHS:", rhs)
    print("Parity error:", error)
    return lhs, rhs, error



def mc_convergence(S_paths, K, r, T, option="call"):
    ST = S_paths[:, -1]
    if option == "call":
        payoff = compute_call_payoff(ST, K)
    else:
        payoff = compute_put_payoff(ST, K)
    discounted = np.exp(-r * T) * payoff
    prices, std_errors, iterations = [], [], []
    for i in np.linspace(10, len(ST), 100, dtype=int):
        subset      = discounted[:i]
        prices.append(np.mean(subset))
        std_errors.append(np.std(subset) / np.sqrt(i))
        iterations.append(i)
    return iterations, prices, std_errors



def blackScholes_closed_price(S0, K, T, r, q, sigma, option="call"):
    """Black–Scholes Closed-Form Formula"""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        price = S0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return price



def blackScholes_cf(u, T, r, q, sigma, S0):
    i = 1j
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T
    return np.exp(i * u * mu - 0.5 * sigma**2 * u**2 * T)



def blackScholes_fourier_price(S0, K, T, r, q, sigma, option="call"):
    i = 1j
    logK = np.log(K)

    def phi(u):
        return blackScholes_cf(u, T, r, q, sigma, S0)
    def phi_1(u):
        return phi(u - i) / phi(-i)
    def phi_2(u):
        return phi(u)
    def P_integral(phi_func):
        def integrand(u):
            return np.real(
                np.exp(-i * u * logK) * phi_func(u) / (i * u)
            )
        integral = quad(integrand, 1e-8, 100)[0]
        return 0.5 + integral / np.pi
    P1 = P_integral(phi_1)
    P2 = P_integral(phi_2)
    call = S0 * np.exp(-q*T) * P1 - K * np.exp(-r*T) * P2
    if option == "call":
        return call
    return call - (S0 * np.exp(-q*T) - K * np.exp(-r*T))



def heston_cf(u, T, r, q, S0, V0, kappa, theta, sigma, rho):
    """
    Heston characteristic function
    """
    i = 1j
    X0 = np.log(S0)
    d = np.sqrt((rho * sigma * i * u - kappa)**2 +
                sigma**2 * (i*u + u**2))
    g = (kappa - rho * sigma * i*u - d) / \
        (kappa - rho * sigma * i*u + d)
    # to ensure numerical stability and avoid explosion
    exp_dT = np.exp(-d * T)
    C = i*u*(r - q)*T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * i*u - d)*T
        - 2*np.log((1 - g*exp_dT)/(1 - g))
    )
    D = ((kappa - rho * sigma * i*u - d) / sigma**2) * \
        ((1 - exp_dT) / (1 - g*exp_dT))
    return np.exp(C + D*V0 + i*u*X0)



def heston_fourier_price(
    S0, K, T, r, q,
    kappa, theta, sigma, rho, V0,
    option_type="call"
):
    def phi_1(u, T, r, q, S0, V0, kappa, theta, sigma, rho):
        return heston_cf(u - 1j, T, r, q, S0, V0, kappa, theta, sigma, rho) / \
           heston_cf(-1j, T, r, q, S0, V0, kappa, theta, sigma, rho)
    def phi_2(u, T, r, q, S0, V0, kappa, theta, sigma, rho):
        return heston_cf(u, T, r, q, S0, V0, kappa, theta, sigma, rho)
    def P_integral(phi_func, K, T, r, q, S0, v0, kappa, theta, sigma, rho):
        i = 1j
        logK = np.log(K)
        def integrand(u):
            phi_val = phi_func(u, T, r, q, S0, V0, kappa, theta, sigma, rho)
            return np.real(
                np.exp(-i * u * logK) * phi_val / (i * u)
            )
        integral = quad(integrand, 1e-8, 100, limit=200)[0]
        return 0.5 + integral / np.pi
    
    P1 = P_integral(phi_1, K, T, r, q, S0, V0, kappa, theta, sigma, rho)
    P2 = P_integral(phi_2, K, T, r, q, S0, V0, kappa, theta, sigma, rho)
    call = S0 * np.exp(-q*T) * P1 - K * np.exp(-r*T) * P2
    if option_type == "call":
        return call
    return call - (S0 * np.exp(-q*T) - K * np.exp(-r*T))



def merton_cf(u, T, r, q, S0, sigma, lambda_j, mu_j, delta_j):
    """
    Merton characteristic function for log-price
    """
    i = 1j
    # drift adjustment
    kappa_j = np.exp(mu_j + 0.5 * delta_j**2) - 1
    drift = np.log(S0) + (r - q - lambda_j * kappa_j - 0.5 * sigma**2) * T
    diffusion = i * u * drift - 0.5 * sigma**2 * u**2 * T
    jumps = lambda_j * T * (np.exp(1j * u * mu_j - 0.5 * sigma**2 * u**2) - 1.0)
    return np.exp(diffusion + jumps)



def merton_fourier_price(
    S0, K, T, r, q,
    sigma, lambda_j, mu_j, delta_j,
    option_type="call"
):
    i = 1j
    logK = np.log(K)
    def phi_1(u):
        return merton_cf(u - i, T, r, q, S0, sigma, lambda_j, mu_j, delta_j) / \
               merton_cf(-i, T, r, q, S0, sigma, lambda_j, mu_j, delta_j)
    def phi_2(u):
        return merton_cf(u, T, r, q, S0, sigma, lambda_j, mu_j, delta_j)
    def P(phi_func):
        def integrand(u):
            return np.real(
                np.exp(-i * u * logK) * phi_func(u) / (i * u)
            )
        integral = quad(integrand, 1e-8, 100, limit=200)[0]
        return 0.5 + integral / np.pi

    P1 = P(phi_1)
    P2 = P(phi_2)
    # Call price
    call = S0 * np.exp(-q*T) * P1 - K * np.exp(-r*T) * P2
    if option_type == "call":
        return call
    return call - (S0 * np.exp(-q*T) - K * np.exp(-r*T))



def blackScholes_greeks(S0, K, T, r, q, sigma, option="call"):
    """Closed-form BS Greeks"""
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    sign = 1 if option == "call" else -1

    delta = sign * np.exp(-q*T) * norm.cdf(sign * d1)
    gamma = np.exp(-q*T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega  = S0 * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (- S0 * np.exp(-q*T) * norm.pdf(d1) * sigma / (2*np.sqrt(T))
             - sign * r * K * np.exp(-r*T) * norm.cdf(sign * d2)
             + sign * q * S0 * np.exp(-q*T) * norm.cdf(sign * d1)) / 365
    rho   = sign * K * T * np.exp(-r*T) * norm.cdf(sign * d2) /100
    return {"delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}



# Generic Finite Difference Method bumper, works with ANY pricer
def compute_greeks_finite_difference(pricer, S0, K, T, r, q, option_type="call",
                       dS=0.01, dV=1e-4, dT=1/365, dr=1e-4, **pricer_kwargs
):
    """
    Compute Greeks via central finite differences for any Fourier pricer
    pricer_kwargs: model-specific parameters (sigma, kappa, etc.)
    dS, dV, dT, dr: bump sizes
    """
    def price(**kw):
        return pricer(**kw, option_type=option_type)
    base_kw = dict(S0=S0, K=K, T=T, r=r, q=q, **pricer_kwargs)
    P = price(**base_kw)
    # Delta
    P_Sp   = price(**{**base_kw, "S0": S0 + dS})
    P_Sm   = price(**{**base_kw, "S0": S0 - dS})
    delta  = (P_Sp - P_Sm) / (2 * dS)
    # Gamma
    gamma  = (P_Sp - 2*P + P_Sm) / (dS**2)
    # Vega
    sigma_key = "sigma"
    sig0   = pricer_kwargs[sigma_key]
    P_vp   = price(**{**base_kw, sigma_key: sig0 + dV})
    P_vm   = price(**{**base_kw, sigma_key: sig0 - dV})
    vega   = (P_vp - P_vm) / (2 * dV) / 100
    # Theta
    P_Tm   = price(**{**base_kw, "T": T - dT})
    theta  = -(P - P_Tm) / dT / 365
    # Rho
    P_rp   = price(**{**base_kw, "r": r + dr})
    P_rm   = price(**{**base_kw, "r": r - dr})
    rho    = (P_rp - P_rm) / (2 * dr) / 100
    return { "Price": P, "delta": delta, "gamma": gamma,
        "vega":  vega, "theta": theta, "rho":   rho}



# Model-specific wrappers
def bs_pricer(S0, K, T, r, q, sigma, option_type="call"):
    return blackScholes_fourier_price(S0, K, T, r, q, sigma, option_type)


def heston_pricer(S0, K, T, r, q, sigma, kappa, theta_h, rho, V0, option_type="call"):
    return heston_fourier_price(S0, K, T, r, q, kappa, theta_h, sigma, rho, V0, option_type)


def merton_pricer(S0, K, T, r, q, sigma, lambda_j, mu_j, delta_j, option_type="call"):
    return merton_fourier_price(S0, K, T, r, q, sigma, lambda_j, mu_j, delta_j, option_type)



def compute_asian_payoff(S_paths, K, option="call", average="arithmetic"):
    if average == "arithmetic":
        avg = np.mean(S_paths[:, 1:], axis=1)
    else:
        avg = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
    if option == "call":
        return np.maximum(avg - K, 0)
    return np.maximum(K - avg, 0)



def mc_asian_price(S_paths, K, r, T, option="call", average="arithmetic"):
    payoff = compute_asian_payoff(S_paths, K, option, average)
    disc = np.exp(-r * T) * payoff
    price = np.mean(disc)
    stderr = np.std(disc) / np.sqrt(len(disc))
    return price, stderr



def bs_geometric_asian_price(S0, K, T, r, q, sigma, option="call"):
    sigma_G = sigma / np.sqrt(3)
    r_G = 0.5 * (r - q - sigma**2 / 6.0)
    F_G = S0 * np.exp(r_G * T)

    d1 = (np.log(F_G / K) + 0.5 * sigma_G**2 * T) / (sigma_G * np.sqrt(T))
    d2 = d1 - sigma_G * np.sqrt(T)

    disc = np.exp(-r * T)
    if option == "call":
        return disc * (F_G * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F_G * norm.cdf(-d1))



def mc_asian_convergence(S_paths, K, r, T, option="call", average="arithmetic"):
    payoff = compute_asian_payoff(S_paths, K, option, average)
    disc = np.exp(-r * T) * payoff
    iters, prices = [], []
    for n in np.linspace(10, len(disc), 100, dtype=int):
        prices.append(np.mean(disc[:n]))
        iters.append(n)
    return iters, prices