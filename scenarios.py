"""
scenarios.py
============
P&L scenario analysis and sensitivity (what-if) grids for option positions.

Features
--------
  1. Spot × Volatility P&L heatmap
  2. Spot × Time-decay P&L heatmap
  3. Single-parameter sensitivity curves (spot, vol, time, rate)
  4. Out-of-sample scenario testing: stress scenarios (crash, vol spike, etc.)
  5. Greeks sensitivity profile across spot range

Methodology Notes
-----------------
All mid-life P&L uses BS re-pricing, not delta-approximation, to capture
full non-linear (gamma) effects. Stress tests use percentage shocks
consistent with historical market extremes (e.g. 1987 crash: −22% spot,
+100pp vol; 2008: −40% spot; COVID: −35% spot, +200% vol).

Limitations
-----------
  * Assumes European options (no early exercise).
  * Vol in scenarios is a flat scalar; no smile/skew simulation.
  * Carry costs and bid-ask spreads are not modelled.

Author: tashrifulkabir34-lang
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .black_scholes import BlackScholesModel, OptionParams
from .greeks import GreeksCalculator


@dataclass
class StressScenario:
    """A named stress-test scenario.

    Parameters
    ----------
    name : str
        Descriptive scenario name.
    dS_pct : float
        Percentage change in spot (e.g. -0.20 = down 20%).
    d_sigma_abs : float
        Absolute change in volatility (e.g. 0.15 = +15pp).
    dT : float
        Time elapsed in years (e.g. 0.0 = instantaneous).
    dr : float
        Change in risk-free rate (absolute).
    """

    name: str
    dS_pct: float
    d_sigma_abs: float
    dT: float = 0.0
    dr: float = 0.0


# Historical stress scenarios (out-of-sample reference events)
HISTORICAL_STRESS_SCENARIOS: list[StressScenario] = [
    StressScenario("Black Monday 1987",     dS_pct=-0.22, d_sigma_abs=0.50, dT=1/365),
    StressScenario("Dot-com Crash 2000",    dS_pct=-0.40, d_sigma_abs=0.20, dT=30/365),
    StressScenario("GFC 2008",              dS_pct=-0.40, d_sigma_abs=0.40, dT=60/365),
    StressScenario("Flash Crash 2010",      dS_pct=-0.09, d_sigma_abs=0.30, dT=1/365),
    StressScenario("COVID Crash Mar 2020",  dS_pct=-0.35, d_sigma_abs=1.00, dT=30/365),
    StressScenario("Vol Crush (post-event)",dS_pct= 0.02, d_sigma_abs=-0.15, dT=1/365),
    StressScenario("Soft Landing Rally",    dS_pct= 0.15, d_sigma_abs=-0.05, dT=30/365),
]


class ScenarioAnalyzer:
    """Compute P&L and Greek profiles under various market scenarios.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry (years).
    r : float
        Risk-free rate.
    sigma : float
        Current implied volatility.
    q : float
        Dividend yield.
    option_type : str
        ``'call'`` or ``'put'``.

    Examples
    --------
    >>> sa = ScenarioAnalyzer(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    >>> results = sa.stress_test()
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type
        self._base_model = BlackScholesModel(
            OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
        )
        self._base_price = self._base_model.price()

    def _reprice(self, S=None, sigma=None, T=None, r=None) -> float:
        """Reprice with overridden parameters."""
        params = OptionParams(
            S=S if S is not None else self.S,
            K=self.K,
            T=max(T if T is not None else self.T, 1e-6),
            r=r if r is not None else self.r,
            sigma=max(sigma if sigma is not None else self.sigma, 0.001),
            q=self.q,
            option_type=self.option_type,
        )
        return BlackScholesModel(params).price()

    # ------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------

    def spot_vol_heatmap(
        self,
        S_range: np.ndarray,
        sigma_range: np.ndarray,
        T_remaining: float | None = None,
    ) -> np.ndarray:
        """P&L heatmap: rows=spot, cols=volatility.

        Parameters
        ----------
        S_range : np.ndarray, shape (n,)
        sigma_range : np.ndarray, shape (m,)
        T_remaining : float, optional
            Time to expiry for mid-life pricing; defaults to self.T.

        Returns
        -------
        np.ndarray, shape (n, m)
            P&L relative to initial position cost.
        """
        T = T_remaining if T_remaining is not None else self.T
        grid = np.zeros((len(S_range), len(sigma_range)))
        for i, S in enumerate(S_range):
            for j, sigma in enumerate(sigma_range):
                grid[i, j] = self._reprice(S=S, sigma=sigma, T=T) - self._base_price
        return grid

    def spot_time_heatmap(
        self,
        S_range: np.ndarray,
        days_range: np.ndarray,
    ) -> np.ndarray:
        """P&L heatmap: rows=spot, cols=days elapsed.

        Parameters
        ----------
        S_range : np.ndarray
        days_range : np.ndarray
            Days elapsed from today.

        Returns
        -------
        np.ndarray, shape (n, m)
        """
        grid = np.zeros((len(S_range), len(days_range)))
        for i, S in enumerate(S_range):
            for j, days in enumerate(days_range):
                T_rem = max(self.T - days / 365.0, 1e-6)
                grid[i, j] = self._reprice(S=S, T=T_rem) - self._base_price
        return grid

    # ------------------------------------------------------------------
    # Sensitivity curves
    # ------------------------------------------------------------------

    def spot_sensitivity(self, S_range: np.ndarray) -> dict[str, np.ndarray]:
        """Compute price, P&L, and Greeks across spot range.

        Returns
        -------
        dict with keys: 'price', 'pnl', 'delta', 'gamma', 'theta', 'vega'
        """
        prices, pnl, deltas, gammas, thetas, vegas = [], [], [], [], [], []
        for S in S_range:
            params = OptionParams(
                S=float(S), K=self.K, T=self.T, r=self.r,
                sigma=self.sigma, q=self.q, option_type=self.option_type
            )
            m = BlackScholesModel(params)
            gc = GreeksCalculator(m)
            g = gc.all_greeks()
            p = m.price()
            prices.append(p)
            pnl.append(p - self._base_price)
            deltas.append(g.delta)
            gammas.append(g.gamma)
            thetas.append(g.theta)
            vegas.append(g.vega)
        return {
            "price": np.array(prices),
            "pnl": np.array(pnl),
            "delta": np.array(deltas),
            "gamma": np.array(gammas),
            "theta": np.array(thetas),
            "vega": np.array(vegas),
        }

    def vol_sensitivity(self, sigma_range: np.ndarray) -> np.ndarray:
        """Option price vs. implied volatility."""
        return np.array([self._reprice(sigma=s) for s in sigma_range])

    def time_decay_curve(self, days_range: np.ndarray) -> np.ndarray:
        """Option price vs. days elapsed (theta decay curve)."""
        prices = []
        for days in days_range:
            T_rem = max(self.T - days / 365.0, 1e-6)
            prices.append(self._reprice(T=T_rem))
        return np.array(prices)

    def rate_sensitivity(self, r_range: np.ndarray) -> np.ndarray:
        """Option price vs. risk-free rate."""
        return np.array([self._reprice(r=r) for r in r_range])

    # ------------------------------------------------------------------
    # Stress testing (out-of-sample scenarios)
    # ------------------------------------------------------------------

    def stress_test(
        self,
        scenarios: list[StressScenario] | None = None,
    ) -> list[dict]:
        """Apply named stress scenarios and report P&L impact.

        Parameters
        ----------
        scenarios : list[StressScenario], optional
            List of scenarios to test. Defaults to HISTORICAL_STRESS_SCENARIOS.

        Returns
        -------
        list[dict]
            Each dict: {name, new_S, new_sigma, new_T, pnl, pnl_pct, new_price}
        """
        if scenarios is None:
            scenarios = HISTORICAL_STRESS_SCENARIOS

        results = []
        for sc in scenarios:
            new_S = self.S * (1 + sc.dS_pct)
            new_sigma = max(self.sigma + sc.d_sigma_abs, 0.001)
            new_T = max(self.T - sc.dT, 1e-6)
            new_r = self.r + sc.dr
            new_price = self._reprice(S=new_S, sigma=new_sigma, T=new_T, r=new_r)
            pnl = new_price - self._base_price
            pnl_pct = pnl / self._base_price * 100 if self._base_price > 0 else float("nan")
            results.append({
                "Scenario": sc.name,
                "New Spot": round(new_S, 2),
                "New σ": round(new_sigma, 4),
                "New Price": round(new_price, 4),
                "P&L ($)": round(pnl, 4),
                "P&L (%)": round(pnl_pct, 2),
            })
        return results

    def delta_hedge_analysis(
        self,
        S_range: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute delta-hedged portfolio P&L (option + short delta shares).

        For each spot price scenario, the hedged P&L is:
          Π = (option_price − base_price) − delta_base × (S − S0)

        This shows the gamma P&L (residual after delta hedge).

        Returns
        -------
        dict with 'unhedged_pnl', 'hedged_pnl', 'delta_base'
        """
        base_delta = GreeksCalculator(self._base_model).delta()
        sens = self.spot_sensitivity(S_range)
        unhedged = sens["pnl"]
        hedge_pnl = base_delta * (S_range - self.S)
        hedged = unhedged - hedge_pnl
        return {
            "unhedged_pnl": unhedged,
            "hedged_pnl": hedged,
            "delta_base": base_delta,
        }
