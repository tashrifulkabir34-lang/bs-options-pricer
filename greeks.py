"""
greeks.py
=========
Full first- and second-order Greeks for Black-Scholes European options.

Greeks Computed
---------------
First-order (sensitivities):
  Delta  : ∂V/∂S   — price sensitivity to spot
  Vega   : ∂V/∂σ   — price sensitivity to volatility  (per 1% move)
  Theta  : ∂V/∂t   — price decay per calendar day
  Rho    : ∂V/∂r   — price sensitivity to interest rate (per 1% move)
  Epsilon: ∂V/∂q   — price sensitivity to dividend yield

Second-order:
  Gamma  : ∂²V/∂S² — delta sensitivity to spot
  Vanna  : ∂²V/∂S∂σ— delta sensitivity to vol (or vega to spot)
  Charm  : ∂²V/∂S∂t— delta decay (delta bleed)
  Vomma  : ∂²V/∂σ² — vega sensitivity to vol (volga)
  Veta   : ∂²V/∂σ∂t— vega decay

Methodology Notes
-----------------
All formulae are analytical closed-form derivatives of the BS formula.
Theta is expressed as per-calendar-day (divided by 365).
Vega and Rho are expressed per 1% change in vol / rate (divided by 100).

Reference
---------
Hull, J.C. (2022). Options, Futures, and Other Derivatives, 11th ed.
Pearson, Chapter 19.

Author: tashrifulkabir34-lang
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from .black_scholes import BlackScholesModel, OptionParams


@dataclass
class Greeks:
    """Container holding all computed Greeks for a given option.

    Attributes
    ----------
    delta : float
        Rate of change of option price w.r.t. spot.
    gamma : float
        Rate of change of delta w.r.t. spot.
    theta : float
        Option value decay per calendar day (negative for long options).
    vega : float
        Option value change per 1% increase in volatility.
    rho : float
        Option value change per 1% increase in risk-free rate.
    vanna : float
        Cross-partial ∂²V/∂S∂σ.
    charm : float
        Delta decay per day (∂²V/∂S∂t).
    vomma : float
        Vega convexity (∂²V/∂σ²) per 1% vol change.
    veta : float
        Vega decay per day (∂²V/∂σ∂t).
    epsilon : float
        Sensitivity to continuous dividend yield.
    """

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    vanna: float = 0.0
    charm: float = 0.0
    vomma: float = 0.0
    veta: float = 0.0
    epsilon: float = 0.0

    def to_dict(self) -> dict:
        """Serialise to plain dictionary."""
        return {
            "Delta": self.delta,
            "Gamma": self.gamma,
            "Theta (per day)": self.theta,
            "Vega (per 1%)": self.vega,
            "Rho (per 1%)": self.rho,
            "Vanna": self.vanna,
            "Charm (per day)": self.charm,
            "Vomma (per 1%)": self.vomma,
            "Veta (per day)": self.veta,
            "Epsilon": self.epsilon,
        }


class GreeksCalculator:
    """Compute analytical Greeks for a Black-Scholes option.

    Parameters
    ----------
    model : BlackScholesModel
        A fully initialised pricing model.

    Examples
    --------
    >>> params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    >>> model = BlackScholesModel(params)
    >>> gc = GreeksCalculator(model)
    >>> g = gc.all_greeks()
    >>> round(g.delta, 4)
    0.6368
    """

    def __init__(self, model: BlackScholesModel) -> None:
        self.model = model
        self._p = model.params
        self._d1 = model.d1
        self._d2 = model.d2
        self._nd1 = norm.pdf(self._d1)   # φ(d1) — standard normal PDF
        self._Nd1 = norm.cdf(self._d1)   # Φ(d1)
        self._Nd2 = norm.cdf(self._d2)   # Φ(d2)

    # ------------------------------------------------------------------
    # First-order Greeks
    # ------------------------------------------------------------------

    def delta(self) -> float:
        """Compute Delta: ∂V/∂S.

        Call: e^{-qT}·Φ(d1)
        Put : e^{-qT}·(Φ(d1) − 1)
        """
        p = self._p
        factor = np.exp(-p.q * p.T)
        if p.option_type == "call":
            return factor * self._Nd1
        return factor * (self._Nd1 - 1.0)

    def gamma(self) -> float:
        """Compute Gamma: ∂²V/∂S².

        Γ = e^{-qT}·φ(d1) / (S·σ·√T)  (same for calls and puts)
        """
        p = self._p
        return (
            np.exp(-p.q * p.T) * self._nd1
            / (p.S * p.sigma * np.sqrt(p.T))
        )

    def theta(self) -> float:
        """Compute Theta: ∂V/∂t (per *calendar day*, sign convention: negative = decay).

        Call: −[S·e^{-qT}·φ(d1)·σ/(2√T)] − r·K·e^{-rT}·Φ(d2) + q·S·e^{-qT}·Φ(d1)
        Put : −[S·e^{-qT}·φ(d1)·σ/(2√T)] + r·K·e^{-rT}·Φ(−d2) − q·S·e^{-qT}·Φ(−d1)
        """
        p = self._p
        common = (
            -p.S * np.exp(-p.q * p.T) * self._nd1 * p.sigma
            / (2.0 * np.sqrt(p.T))
        )
        if p.option_type == "call":
            theta_annual = (
                common
                - p.r * p.K * np.exp(-p.r * p.T) * self._Nd2
                + p.q * p.S * np.exp(-p.q * p.T) * self._Nd1
            )
        else:
            theta_annual = (
                common
                + p.r * p.K * np.exp(-p.r * p.T) * norm.cdf(-self._d2)
                - p.q * p.S * np.exp(-p.q * p.T) * norm.cdf(-self._d1)
            )
        return theta_annual / 365.0  # per calendar day

    def vega(self) -> float:
        """Compute Vega: ∂V/∂σ per 1% change in volatility.

        V = S·e^{-qT}·φ(d1)·√T  (same for calls and puts)
        Returns value per 1% vol move (divided by 100).
        """
        p = self._p
        vega_raw = p.S * np.exp(-p.q * p.T) * self._nd1 * np.sqrt(p.T)
        return vega_raw / 100.0

    def rho(self) -> float:
        """Compute Rho: ∂V/∂r per 1% change in interest rate.

        Call: K·T·e^{-rT}·Φ(d2)
        Put : −K·T·e^{-rT}·Φ(−d2)
        Returns value per 1% rate move (divided by 100).
        """
        p = self._p
        if p.option_type == "call":
            rho_raw = p.K * p.T * np.exp(-p.r * p.T) * self._Nd2
        else:
            rho_raw = -p.K * p.T * np.exp(-p.r * p.T) * norm.cdf(-self._d2)
        return rho_raw / 100.0

    def epsilon(self) -> float:
        """Compute Epsilon (dividend rho): ∂V/∂q.

        Call: −S·T·e^{-qT}·Φ(d1)
        Put : +S·T·e^{-qT}·Φ(−d1)
        """
        p = self._p
        factor = p.S * p.T * np.exp(-p.q * p.T)
        if p.option_type == "call":
            return -factor * self._Nd1
        return factor * norm.cdf(-self._d1)

    # ------------------------------------------------------------------
    # Second-order Greeks
    # ------------------------------------------------------------------

    def vanna(self) -> float:
        """Compute Vanna: ∂²V/∂S∂σ = Vega·(d2/(S·σ)).

        Measures how delta changes with vol (and vice versa).
        """
        p = self._p
        return (
            -np.exp(-p.q * p.T)
            * self._nd1
            * self._d2
            / p.sigma
        )

    def charm(self) -> float:
        """Compute Charm (delta bleed): ∂²V/∂S∂τ, per calendar day.

        Measures how delta decays with passage of time.
        Call: e^{-qT}[q·Φ(d1) − φ(d1)·(2(r−q)T − d2·σ·√T) / (2T·σ·√T)]
        """
        p = self._p
        sq = np.sqrt(p.T)
        if p.option_type == "call":
            charm_annual = np.exp(-p.q * p.T) * (
                p.q * self._Nd1
                - self._nd1 * (
                    2 * (p.r - p.q) * p.T - self._d2 * p.sigma * sq
                ) / (2 * p.T * p.sigma * sq)
            )
        else:
            charm_annual = np.exp(-p.q * p.T) * (
                -p.q * norm.cdf(-self._d1)
                - self._nd1 * (
                    2 * (p.r - p.q) * p.T - self._d2 * p.sigma * sq
                ) / (2 * p.T * p.sigma * sq)
            )
        return charm_annual / 365.0

    def vomma(self) -> float:
        """Compute Vomma (volga): ∂²V/∂σ², per 1% vol move squared.

        Vomma = Vega · d1·d2 / σ
        """
        p = self._p
        vega_raw = p.S * np.exp(-p.q * p.T) * self._nd1 * np.sqrt(p.T)
        return vega_raw * self._d1 * self._d2 / p.sigma / 100.0

    def veta(self) -> float:
        """Compute Veta: ∂²V/∂σ∂t (vega decay per calendar day).

        Veta = −Vega·[q + d1·(r−q)/(σ·√T) − (1+d1·d2)/(2T)]
        """
        p = self._p
        vega_raw = p.S * np.exp(-p.q * p.T) * self._nd1 * np.sqrt(p.T)
        sq = np.sqrt(p.T)
        veta_annual = -vega_raw * (
            p.q
            + (p.r - p.q) * self._d1 / (p.sigma * sq)
            - (1 + self._d1 * self._d2) / (2 * p.T)
        )
        return veta_annual / 365.0 / 100.0

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def all_greeks(self) -> Greeks:
        """Compute and return all Greeks in a single call.

        Returns
        -------
        Greeks
            Dataclass containing all first- and second-order Greeks.
        """
        return Greeks(
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            vega=self.vega(),
            rho=self.rho(),
            vanna=self.vanna(),
            charm=self.charm(),
            vomma=self.vomma(),
            veta=self.veta(),
            epsilon=self.epsilon(),
        )

    def delta_surface(
        self,
        S_range: np.ndarray,
        sigma_range: np.ndarray,
    ) -> np.ndarray:
        """Compute delta surface over (S, σ) grid.

        Parameters
        ----------
        S_range : np.ndarray
            Spot price grid.
        sigma_range : np.ndarray
            Volatility grid.

        Returns
        -------
        np.ndarray, shape (len(S_range), len(sigma_range))
        """
        surface = np.zeros((len(S_range), len(sigma_range)))
        p = self._p
        for j, sigma in enumerate(sigma_range):
            for i, S in enumerate(S_range):
                params = OptionParams(
                    S=S, K=p.K, T=p.T, r=p.r,
                    sigma=sigma, q=p.q, option_type=p.option_type
                )
                m = BlackScholesModel(params)
                surface[i, j] = GreeksCalculator(m).delta()
        return surface
