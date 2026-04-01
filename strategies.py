"""
strategies.py
=============
Option strategy payoff and P&L diagram generation.

Supported Strategies
--------------------
Single-leg:
  * Long / Short Call
  * Long / Short Put

Multi-leg spreads:
  * Bull Call Spread      (long lower call, short upper call)
  * Bear Put Spread       (long upper put, short lower put)
  * Long / Short Straddle (call + put, same strike)
  * Long / Short Strangle (call + put, different strikes)
  * Long / Short Butterfly (3 strikes: long 2 wings, short 2 body)
  * Iron Condor           (put spread + call spread)
  * Covered Call
  * Protective Put

Methodology
-----------
Each strategy is modelled as a portfolio of legs. For a leg (option_type,
strike K, quantity n, option_price C):
  - At expiry payoff: n × max(S_T − K, 0) for calls, n × max(K − S_T, 0) for puts
  - P&L = payoff − net_premium (including carry at 0 for simplicity)
  - Mid-life P&L uses BS pricing at each spot scenario

Author: tashrifulkabir34-lang
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from .black_scholes import BlackScholesModel, OptionParams

LegType = Literal["call", "put", "stock"]


@dataclass
class Leg:
    """A single option (or stock) position.

    Parameters
    ----------
    option_type : str
        ``'call'``, ``'put'``, or ``'stock'``.
    K : float
        Strike price (unused for stock leg).
    quantity : float
        Number of contracts (+long, −short).
    premium : float
        Price paid (positive) or received (negative) per unit.
    S0 : float, optional
        Entry spot price (used for stock leg P&L).
    """

    option_type: LegType
    K: float
    quantity: float
    premium: float
    S0: float = 0.0

    def expiry_payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Compute payoff array at expiry for a spot price grid S_T."""
        if self.option_type == "call":
            intrinsic = np.maximum(S_T - self.K, 0.0)
        elif self.option_type == "put":
            intrinsic = np.maximum(self.K - S_T, 0.0)
        else:  # stock
            intrinsic = S_T - self.S0
        return self.quantity * intrinsic

    def bs_price(
        self,
        S_T: np.ndarray,
        T_remaining: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> np.ndarray:
        """Compute BS price across spot grid with time remaining."""
        if self.option_type == "stock":
            return self.quantity * (S_T - self.S0)

        prices = np.zeros_like(S_T, dtype=float)
        for i, S in enumerate(S_T):
            if T_remaining < 1e-6:
                prices[i] = self.expiry_payoff(np.array([S]))[0]
            else:
                params = OptionParams(
                    S=float(S), K=self.K,
                    T=T_remaining, r=r, sigma=sigma,
                    q=q, option_type=self.option_type,
                )
                prices[i] = BlackScholesModel(params).price()
        return self.quantity * prices


class OptionStrategy:
    """Build multi-leg option strategies and compute payoff / P&L.

    Parameters
    ----------
    name : str
        Human-readable strategy name.
    legs : list[Leg]
        Component legs.
    r : float
        Risk-free rate (for mid-life pricing).
    sigma : float
        Volatility assumption (for mid-life pricing).
    q : float
        Dividend yield.

    Examples
    --------
    >>> strat = OptionStrategy.long_straddle(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    >>> pnl = strat.expiry_pnl(np.linspace(60, 140, 5))
    """

    def __init__(
        self,
        name: str,
        legs: list[Leg],
        r: float = 0.05,
        sigma: float = 0.20,
        q: float = 0.0,
    ) -> None:
        self.name = name
        self.legs = legs
        self.r = r
        self.sigma = sigma
        self.q = q

    @property
    def net_premium(self) -> float:
        """Net premium paid (positive = debit, negative = credit)."""
        return sum(leg.quantity * leg.premium for leg in self.legs)

    def expiry_pnl(self, S_range: np.ndarray) -> np.ndarray:
        """Compute total P&L at expiry across spot price grid.

        P&L = sum(leg payoffs) − net_premium

        Parameters
        ----------
        S_range : np.ndarray
            Spot prices at expiry.

        Returns
        -------
        np.ndarray
            P&L per unit notional.
        """
        total_payoff = sum(leg.expiry_payoff(S_range) for leg in self.legs)
        return total_payoff - self.net_premium

    def mid_life_pnl(
        self,
        S_range: np.ndarray,
        T_remaining: float,
    ) -> np.ndarray:
        """Compute unrealised P&L at some point before expiry.

        Parameters
        ----------
        S_range : np.ndarray
            Current spot prices (scenario grid).
        T_remaining : float
            Time remaining to expiry (years).

        Returns
        -------
        np.ndarray
            Unrealised P&L.
        """
        current_values = sum(
            leg.bs_price(S_range, T_remaining, self.r, self.sigma, self.q)
            for leg in self.legs
        )
        return current_values - self.net_premium

    def breakeven_points(
        self,
        S_range: np.ndarray,
        tol: float = 0.5,
    ) -> list[float]:
        """Estimate break-even spot prices at expiry.

        Parameters
        ----------
        S_range : np.ndarray
            Fine spot grid to scan.
        tol : float
            |P&L| < tol qualifies as break-even.

        Returns
        -------
        list[float]
            List of approximate break-even spot prices.
        """
        pnl = self.expiry_pnl(S_range)
        crossings = []
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i + 1] <= 0:
                # Linear interpolation
                be = S_range[i] - pnl[i] * (S_range[i + 1] - S_range[i]) / (
                    pnl[i + 1] - pnl[i]
                )
                crossings.append(round(be, 2))
        return crossings

    def max_profit(self, S_range: np.ndarray) -> float:
        """Maximum P&L over the spot grid at expiry."""
        return float(np.max(self.expiry_pnl(S_range)))

    def max_loss(self, S_range: np.ndarray) -> float:
        """Maximum loss (minimum P&L) over the spot grid at expiry."""
        return float(np.min(self.expiry_pnl(S_range)))

    # ------------------------------------------------------------------
    # Strategy factory methods
    # ------------------------------------------------------------------

    @classmethod
    def long_call(
        cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> "OptionStrategy":
        params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call")
        premium = BlackScholesModel(params).price()
        legs = [Leg("call", K, 1.0, premium)]
        return cls("Long Call", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def long_put(
        cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> "OptionStrategy":
        params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put")
        premium = BlackScholesModel(params).price()
        legs = [Leg("put", K, 1.0, premium)]
        return cls("Long Put", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def long_straddle(
        cls, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> "OptionStrategy":
        call_price = BlackScholesModel(
            OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call")
        ).price()
        put_price = BlackScholesModel(
            OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put")
        ).price()
        legs = [
            Leg("call", K, 1.0, call_price),
            Leg("put", K, 1.0, put_price),
        ]
        return cls("Long Straddle", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def long_strangle(
        cls,
        S: float,
        K_put: float,
        K_call: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> "OptionStrategy":
        call_price = BlackScholesModel(
            OptionParams(S=S, K=K_call, T=T, r=r, sigma=sigma, q=q, option_type="call")
        ).price()
        put_price = BlackScholesModel(
            OptionParams(S=S, K=K_put, T=T, r=r, sigma=sigma, q=q, option_type="put")
        ).price()
        legs = [
            Leg("call", K_call, 1.0, call_price),
            Leg("put", K_put, 1.0, put_price),
        ]
        return cls("Long Strangle", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def bull_call_spread(
        cls,
        S: float,
        K_lo: float,
        K_hi: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> "OptionStrategy":
        price_lo = BlackScholesModel(
            OptionParams(S=S, K=K_lo, T=T, r=r, sigma=sigma, q=q, option_type="call")
        ).price()
        price_hi = BlackScholesModel(
            OptionParams(S=S, K=K_hi, T=T, r=r, sigma=sigma, q=q, option_type="call")
        ).price()
        legs = [
            Leg("call", K_lo, 1.0, price_lo),
            Leg("call", K_hi, -1.0, price_hi),
        ]
        return cls("Bull Call Spread", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def bear_put_spread(
        cls,
        S: float,
        K_lo: float,
        K_hi: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> "OptionStrategy":
        price_hi = BlackScholesModel(
            OptionParams(S=S, K=K_hi, T=T, r=r, sigma=sigma, q=q, option_type="put")
        ).price()
        price_lo = BlackScholesModel(
            OptionParams(S=S, K=K_lo, T=T, r=r, sigma=sigma, q=q, option_type="put")
        ).price()
        legs = [
            Leg("put", K_hi, 1.0, price_hi),
            Leg("put", K_lo, -1.0, price_lo),
        ]
        return cls("Bear Put Spread", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def iron_condor(
        cls,
        S: float,
        K1: float,
        K2: float,
        K3: float,
        K4: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> "OptionStrategy":
        """Iron Condor: K1 < K2 < K3 < K4.

        Short put spread (K1/K2) + Short call spread (K3/K4).
        """
        def price(K, otype):
            return BlackScholesModel(
                OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=otype)
            ).price()

        legs = [
            Leg("put", K1, 1.0, price(K1, "put")),   # long put K1
            Leg("put", K2, -1.0, price(K2, "put")),  # short put K2
            Leg("call", K3, -1.0, price(K3, "call")), # short call K3
            Leg("call", K4, 1.0, price(K4, "call")),  # long call K4
        ]
        return cls("Iron Condor", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def long_butterfly(
        cls,
        S: float,
        K1: float,
        K2: float,
        K3: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> "OptionStrategy":
        """Long Call Butterfly: long K1 call, short 2×K2 call, long K3 call."""
        def price(K):
            return BlackScholesModel(
                OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call")
            ).price()

        legs = [
            Leg("call", K1, 1.0, price(K1)),
            Leg("call", K2, -2.0, price(K2)),
            Leg("call", K3, 1.0, price(K3)),
        ]
        return cls("Long Butterfly", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def covered_call(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> "OptionStrategy":
        """Covered Call: long stock + short call."""
        call_price = BlackScholesModel(
            OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call")
        ).price()
        legs = [
            Leg("stock", K, 1.0, S, S0=S),
            Leg("call", K, -1.0, call_price),
        ]
        return cls("Covered Call", legs, r=r, sigma=sigma, q=q)

    @classmethod
    def protective_put(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> "OptionStrategy":
        """Protective Put: long stock + long put."""
        put_price = BlackScholesModel(
            OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put")
        ).price()
        legs = [
            Leg("stock", K, 1.0, S, S0=S),
            Leg("put", K, 1.0, put_price),
        ]
        return cls("Protective Put", legs, r=r, sigma=sigma, q=q)

    def __repr__(self) -> str:
        return f"OptionStrategy(name='{self.name}', legs={len(self.legs)}, net_premium={self.net_premium:.4f})"
