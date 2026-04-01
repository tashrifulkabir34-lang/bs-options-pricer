"""
black_scholes.py
================
Core Black-Scholes (1973) option pricing model implementation.

Methodology
-----------
The Black-Scholes model prices European options under the following assumptions:
  1. The underlying follows geometric Brownian motion: dS = μS dt + σS dW
  2. No dividends (or continuous dividend yield q handled via cost-of-carry)
  3. Constant risk-free rate r and volatility σ over the option's life
  4. No transaction costs; continuous trading; no arbitrage
  5. Log-normal distribution of the terminal stock price

Pricing formulae (Merton 1973 extension for continuous dividends):
  d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
  d2 = d1 − σ√T
  Call = S·e^{−qT}·N(d1) − K·e^{−rT}·N(d2)
  Put  = K·e^{−rT}·N(−d2) − S·e^{−qT}·N(−d1)

Reference
---------
Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
Journal of Political Economy, 81(3), 637–654.

Author: tashrifulkabir34-lang
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Literal


OptionType = Literal["call", "put"]


@dataclass
class OptionParams:
    """Container for option pricing parameters.

    Parameters
    ----------
    S : float
        Current underlying price (spot price), > 0.
    K : float
        Strike price, > 0.
    T : float
        Time to expiration in years, > 0.
    r : float
        Continuously compounded risk-free rate (e.g. 0.05 = 5%).
    sigma : float
        Annualised implied / historical volatility (e.g. 0.20 = 20%), > 0.
    q : float, optional
        Continuous dividend yield (e.g. 0.02 = 2%). Default 0.0.
    option_type : OptionType
        ``'call'`` or ``'put'``. Default ``'call'``.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: OptionType = "call"

    def __post_init__(self) -> None:
        if self.S <= 0:
            raise ValueError(f"Spot price S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry T must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type}")


class BlackScholesModel:
    """Black-Scholes-Merton European option pricing model.

    Supports both calls and puts with continuous dividend yield.
    All methods are vectorised over numpy arrays for efficient surface
    generation.

    Examples
    --------
    >>> params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    >>> bs = BlackScholesModel(params)
    >>> round(bs.price(), 4)
    10.4506
    >>> round(bs.delta(), 4)
    0.6368
    """

    def __init__(self, params: OptionParams) -> None:
        self.params = params
        self._d1: float | None = None
        self._d2: float | None = None
        self._compute_d1_d2()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_d1_d2(self) -> None:
        """Compute and cache d1, d2 for the current parameter set."""
        p = self.params
        self._d1 = (
            np.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T
        ) / (p.sigma * np.sqrt(p.T))
        self._d2 = self._d1 - p.sigma * np.sqrt(p.T)

    def update(self, **kwargs) -> "BlackScholesModel":
        """Return a *new* model with updated parameters (immutable style).

        Parameters
        ----------
        **kwargs
            Any field of :class:`OptionParams` (S, K, T, r, sigma, q, option_type).
        """
        import dataclasses
        new_params = dataclasses.replace(self.params, **kwargs)
        return BlackScholesModel(new_params)

    # ------------------------------------------------------------------
    # Public pricing
    # ------------------------------------------------------------------

    @property
    def d1(self) -> float:
        """d1 term used in Black-Scholes formula."""
        return self._d1

    @property
    def d2(self) -> float:
        """d2 term used in Black-Scholes formula."""
        return self._d2

    def price(self) -> float:
        """Compute the theoretical option price.

        Returns
        -------
        float
            Fair value of the option.
        """
        p = self.params
        d1, d2 = self._d1, self._d2
        discount_S = p.S * np.exp(-p.q * p.T)
        discount_K = p.K * np.exp(-p.r * p.T)

        if p.option_type == "call":
            return discount_S * norm.cdf(d1) - discount_K * norm.cdf(d2)
        else:
            return discount_K * norm.cdf(-d2) - discount_S * norm.cdf(-d1)

    def intrinsic_value(self) -> float:
        """Return the intrinsic (exercise) value of the option."""
        p = self.params
        if p.option_type == "call":
            return max(p.S - p.K, 0.0)
        return max(p.K - p.S, 0.0)

    def time_value(self) -> float:
        """Return the time (extrinsic) value: price − intrinsic."""
        return self.price() - self.intrinsic_value()

    def put_call_parity_check(self) -> float:
        """Verify put-call parity: C − P = S·e^{-qT} − K·e^{-rT}.

        Returns the parity residual (should be ~0 for European options).
        """
        p = self.params
        call_price = self.update(option_type="call").price()
        put_price = self.update(option_type="put").price()
        lhs = call_price - put_price
        rhs = p.S * np.exp(-p.q * p.T) - p.K * np.exp(-p.r * p.T)
        return lhs - rhs

    # ------------------------------------------------------------------
    # Vectorised surface generation
    # ------------------------------------------------------------------

    @staticmethod
    def price_surface(
        S_range: np.ndarray,
        T_range: np.ndarray,
        K: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: OptionType = "call",
    ) -> np.ndarray:
        """Compute a price surface over a grid of (S, T) values.

        Parameters
        ----------
        S_range : np.ndarray, shape (n,)
            Spot price grid.
        T_range : np.ndarray, shape (m,)
            Time-to-expiry grid (years).
        K, r, sigma, q, option_type
            Fixed parameters.

        Returns
        -------
        np.ndarray, shape (n, m)
            Option prices.
        """
        surface = np.zeros((len(S_range), len(T_range)))
        for j, T in enumerate(T_range):
            for i, S in enumerate(S_range):
                params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma,
                                      q=q, option_type=option_type)
                surface[i, j] = BlackScholesModel(params).price()
        return surface

    @staticmethod
    def vol_surface(
        K_range: np.ndarray,
        T_range: np.ndarray,
        S: float,
        r: float,
        base_sigma: float,
        skew: float = -0.02,
        q: float = 0.0,
    ) -> np.ndarray:
        """Synthetic volatility smile/surface with linear moneyness skew.

        Used for illustration and out-of-sample scenario analysis.

        Parameters
        ----------
        K_range : np.ndarray
            Strike grid.
        T_range : np.ndarray
            Expiry grid (years).
        S : float
            Spot price.
        r : float
            Risk-free rate.
        base_sigma : float
            ATM volatility.
        skew : float
            Per-unit-moneyness slope (negative = put skew).
        q : float
            Dividend yield.

        Returns
        -------
        np.ndarray, shape (len(K_range), len(T_range))
            Implied vol surface.
        """
        surface = np.zeros((len(K_range), len(T_range)))
        for j, T in enumerate(T_range):
            for i, K in enumerate(K_range):
                moneyness = np.log(K / S) / (base_sigma * np.sqrt(T))
                surface[i, j] = max(base_sigma + skew * moneyness, 0.001)
        return surface

    def __repr__(self) -> str:
        p = self.params
        return (
            f"BlackScholesModel(S={p.S}, K={p.K}, T={p.T:.4f}, "
            f"r={p.r:.4f}, σ={p.sigma:.4f}, type={p.option_type})"
        )
