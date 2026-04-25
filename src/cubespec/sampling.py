"""Samplers: independent Gaussian, Latin Hypercube, multivariate normal."""
from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.stats import norm, qmc

from .params import CSP
from .correlation import cholesky


def sample_independent(csp: CSP, n: int, seed: int = 1337) -> np.ndarray:
    """Independent Gaussian samples, shape (n, 7)."""
    rng = np.random.default_rng(seed)
    means = np.array(csp.means())
    sds = np.array(csp.sds())
    z = rng.standard_normal(size=(n, 7))
    return means + sds * z


def sample_lhs(csp: CSP, n: int, seed: int = 1337) -> np.ndarray:
    """Latin Hypercube sampling mapped through Gaussian inverse CDF."""
    sampler = qmc.LatinHypercube(d=7, seed=seed)
    u = sampler.random(n=n)
    # Avoid 0/1 endpoints
    u = np.clip(u, 1e-12, 1 - 1e-12)
    z = norm.ppf(u)
    means = np.array(csp.means())
    sds = np.array(csp.sds())
    return means + sds * z


def sample_mvn(
    csp: CSP,
    n: int,
    corr: Optional[np.ndarray] = None,
    seed: int = 1337,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Correlated multivariate normal samples, shape (n, 7)."""
    means = np.array(csp.means())
    sds = np.array(csp.sds())
    if corr is None:
        corr = np.eye(7)
    cov = (corr * np.outer(sds, sds))
    cov = cov + np.eye(7) * jitter
    L = cholesky(cov)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n, 7))
    y = z @ L.T
    return means + y
