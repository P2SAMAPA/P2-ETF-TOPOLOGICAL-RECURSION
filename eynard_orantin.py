import numpy as np
from scipy.linalg import eigh
from scipy.stats import gaussian_kde

def spectral_curve(eigvals, n_bins=50, smoothing=0.1):
    """
    Returns (x_grid, rho(x)) where rho(x) is smoothed eigenvalue density.
    """
    # Use KDE for smoothing
    kde = gaussian_kde(eigvals, bw_method=smoothing)
    x_grid = np.linspace(min(eigvals), max(eigvals), n_bins)
    rho = kde.evaluate(x_grid)
    return x_grid, rho

def topological_recursion_score(returns, n_bins=50, smoothing=0.1):
    """
    Compute per-ETF score using first symplectic invariant.
    Steps:
    1. Compute eigenvalues and eigenvectors of correlation matrix.
    2. Build spectral curve (eigenvalue density).
    3. Compute the disc amplitude W_1^1(x) as derivative of log density? Simplified:
       W_1^1(x) = (d/dx) log rho(x) * rho(x)?  Actually the first invariant is
       the "one-point function" which we approximate as the derivative of the density.
       For each ETF, score = sum_i (v_i^2) * |W_1^1(λ_i)| where v_i are eigenvector components.
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    corr = returns_clean.corr().values
    eigvals, eigvecs = eigh(corr)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Get spectral density
    x_grid, rho = spectral_curve(eigvals, n_bins, smoothing)
    # Compute derivative of log density (simplified "disc amplitude")
    # Use finite differences
    drho = np.gradient(rho, x_grid)
    # Avoid division by zero where rho is tiny
    safe_rho = np.where(rho < 1e-6, 1e-6, rho)
    w1 = drho / safe_rho   # this is d/dx log rho
    # Interpolate w1 at each eigenvalue
    w1_at_eig = np.interp(eigvals, x_grid, w1)
    # For each ETF, score = sum over modes (v_ij^2 * |w1_j|)
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            scores[i] += eigvecs[i, j]**2 * abs(w1_at_eig[j])
    tickers = returns_clean.columns
    return {ticker: scores[i] for i, ticker in enumerate(tickers)}
