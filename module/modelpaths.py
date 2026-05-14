# module/modelpaths.py


import numpy as np



SEED = 42



def generate_blackScholes_paths(
    S0, r, q, sigma,
    T, steps, paths
):
    np.random.seed(SEED)
    dt = T / steps
    # Brownian increments
    Z = np.random.normal(size=(paths, steps))
    dW = np.sqrt(dt) * Z
    # Cumulative Brownian motion
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((paths, 1)), W])
    t = np.linspace(0, T, steps + 1)
    # Geometric Brownian Motion
    S_paths = S0 * np.exp((r - q - 0.5 * sigma**2) * t + sigma * W)
    return S_paths



def generate_heston_paths(
    S0, V0, r, q,
    kappa, theta, sigma, rho,
    T, steps, paths
):
    np.random.seed(SEED)
    dt = T / steps
    C = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(C)
    # Underlying price paths
    S_paths = np.zeros((paths, steps + 1))
    # Stochastic volatility paths
    V_paths = np.zeros((paths, steps + 1))

    S_paths[:, 0] = S0
    V_paths[:, 0] = V0
    S = np.full(paths, S0)
    V = np.full(paths, V0)
    for t in range(steps):
        # Generate independent normals
        Z = np.random.normal(size=(2, paths))
        # Correlated Brownian motions
        dW_S, dW_V = L @ Z * np.sqrt(dt)
        # Heston Euler scheme on stochastic volatility
        V = np.maximum(
            V + kappa * (theta - V) * dt + sigma * np.sqrt(np.maximum(V, 0)) * dW_V,
            0
        )
        # Log-Euler scheme (to avoid negative prices)
        #S = S * np.exp((r - q - 0.5 * V) * dt + np.sqrt(V) * dW_S)
        S = S * (1 + (r-q)*dt + np.sqrt(V) * dW_S)
        S_paths[:, t + 1] = S
        V_paths[:, t + 1] = V
    return S_paths, V_paths



def generate_merton_paths(
    S0, r, q, sigma,
    lambda_j, mu_j, delta_j,
    T, steps, paths
):
    np.random.seed(SEED)
    dt = T / steps
    # Drift correction
    k = np.exp(mu_j + 0.5 * delta_j**2) - 1
    # Diffusion
    W = np.random.normal(size=(paths, steps))
    # Jump sizes
    J = np.random.normal(size=(paths, steps))
    # Jump occurrences
    N = np.random.poisson(lambda_j * dt, size=(paths, steps))
    diffusion = (
        (r - q - lambda_j * k - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * W
    )
    jumps = N * (mu_j + delta_j * J)
    # log prices
    log_S = np.cumsum(diffusion + jumps, axis=1)
    log_S = np.hstack([np.zeros((paths, 1)), log_S])
    S_paths = S0 * np.exp(log_S)
    return S_paths