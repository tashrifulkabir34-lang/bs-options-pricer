"""
implied_vol.py
==============
Implied Volatility (IV) solver using Newton-Raphson with bisection fallback.

Methodology
-----------
Given a market-observed option price V_mkt, implied volatility σ* satisfies:

    BS(S, K, T, r, σ*, q, type) = V_mkt

Since BS is monotone and smooth in σ, we can solve with Newton-Raphson:

    σ_{n+1} = σ_n − [BS(σ_n) − V_mkt] / Vega(σ_n)

Convergence is fast (typically 3–5 iterations) near the solution.

Fallback strategy:
  - If Newton-Raphson diverges or Vega is near-zero (deep OTM / ITM),
    we switch to Brent's method (scipy.optimize.brentq) over [σ_lo, σ_hi].
  - If no solution exists (price below intrinsic / above theoretical max),
    we return NaN with a descriptive warning.

Assumptions / Limitations
--------------------------
  * European options only (no early exercise premium).
  * The model assumes flat vol; term structure and smile are ignored.
  * Deep ITM/OTM options may have near-zero Vega, causing slow convergence.
  * Extreme parameter combinations (T → 0, σ → ∞) may fail gracefully.

Reference
---------
Jaeckel, P. (2015). Let's be rational. Wilmott Magazine, 2015(75), 40–53.
Brenner, M. & Subrahmanyam, M. (1988). A simple formula to compute the
implied standard deviation. Financial Analysts Journal, 44(5), 80–83.

Author: tashrifulkabir34-lang
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import brentq
from .black_scholes import BlackScholesModel, OptionParams
from .greeks import GreeksCalculator

# Solver configuration constants
_MAX_ITER: int = 100
_TOL: float = 1e-8
_SIGMA_LO: float = 1e-6
_SIGMA_HI: float = 10.0  # 1000% vol upper bound
_VEGA_FLOOR: float = 1e-10  # below which NR is abandoned


class IVSolverError(ValueError):
    """Raised when no valid implied volatility can be found."""


class ImpliedVolatilitySolver:
    """Solve for the Black-Scholes implied volatility given market price.

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
    q : float, optional
        Continuous dividend yield. Default 0.0.
    option_type : str
        ``'call'`` or ``'put'``.

    Examples
    --------
    >>> solver = ImpliedVolatilitySolver(S=100, K=100, T=1.0, r=0.05)
    >>> round(solver.solve(market_price=10.4506), 4)
    0.2
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.option_type = option_type

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_model(self, sigma: float) -> BlackScholesModel:
        """Construct a BlackScholesModel for a trial sigma."""
        params = OptionParams(
            S=self.S, K=self.K, T=self.T, r=self.r,
            sigma=sigma, q=self.q, option_type=self.option_type,
        )
        return BlackScholesModel(params)

    def _price_error(self, sigma: float, market_price: float) -> float:
        """Return BS(σ) − V_mkt."""
        return self._make_model(sigma).price() - market_price

    def _validate_bounds(self, market_price: float) -> None:
        """Check price is within arbitrage-free bounds."""
        params_lo = OptionParams(
            S=self.S, K=self.K, T=self.T, r=self.r,
            sigma=_SIGMA_LO, q=self.q, option_type=self.option_type,
        )
        params_hi = OptionParams(
            S=self.S, K=self.K, T=self.T, r=self.r,
            sigma=_SIGMA_HI, q=self.q, option_type=self.option_type,
        )
        price_lo = BlackScholesModel(params_lo).price()
        price_hi = BlackScholesModel(params_hi).price()

        if market_price < price_lo - 1e-6:
            raise IVSolverError(
                f"Market price {market_price:.4f} is below the minimum "
                f"BS price {price_lo:.4f} at σ={_SIGMA_LO}. "
                "Check for arbitrage or input errors."
            )
        if market_price > price_hi + 1e-6:
            raise IVSolverError(
                f"Market price {market_price:.4f} exceeds maximum "
                f"BS price {price_hi:.4f} at σ={_SIGMA_HI}. "
                "Implied vol would be unrealistically high."
            )

    # ------------------------------------------------------------------
    # Brenner-Subrahmanyam initial guess
    # ------------------------------------------------------------------

    def _initial_sigma(self, market_price: float) -> float:
        """Brenner & Subrahmanyam (1988) ATM approximation as seed.

        σ₀ ≈ V_mkt·√(2π/T) / S  (ATM approximation)
        Falls back to 0.20 if result is out of range.
        """
        sigma0 = market_price * np.sqrt(2 * np.pi / self.T) / self.S
        if _SIGMA_LO < sigma0 < _SIGMA_HI:
            return sigma0
        return 0.20

    # ------------------------------------------------------------------
    # Newton-Raphson solver
    # ------------------------------------------------------------------

    def _newton_raphson(
        self, market_price: float, sigma0: float
    ) -> tuple[float, int, str]:
        """Run Newton-Raphson iterations.

        Returns
        -------
        sigma : float
            Converged implied vol (or last iterate).
        n_iter : int
            Number of iterations used.
        status : str
            ``'converged'`` or ``'diverged'``.
        """
        sigma = sigma0
        for i in range(1, _MAX_ITER + 1):
            model = self._make_model(sigma)
            error = model.price() - market_price
            vega_per_unit = GreeksCalculator(model).vega() * 100.0  # raw vega

            if abs(error) < _TOL:
                return sigma, i, "converged"

            if abs(vega_per_unit) < _VEGA_FLOOR:
                return sigma, i, "low_vega"

            sigma_new = sigma - error / vega_per_unit
            # Keep sigma in valid range
            sigma_new = np.clip(sigma_new, _SIGMA_LO, _SIGMA_HI)
            sigma = sigma_new

        return sigma, _MAX_ITER, "max_iter"

    # ------------------------------------------------------------------
    # Brent fallback
    # ------------------------------------------------------------------

    def _brent(self, market_price: float) -> float:
        """Brent's method (bracketed root-finding) as a robust fallback."""
        try:
            sigma = brentq(
                self._price_error,
                _SIGMA_LO,
                _SIGMA_HI,
                args=(market_price,),
                xtol=_TOL,
                maxiter=500,
            )
            return sigma
        except ValueError as exc:
            raise IVSolverError(
                f"Brent solver failed to bracket root for price={market_price:.4f}. "
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        market_price: float,
        method: str = "newton",
        return_diagnostics: bool = False,
    ) -> float | dict:
        """Compute implied volatility for a given market price.

        Parameters
        ----------
        market_price : float
            Observed market price of the option.
        method : str
            Primary method: ``'newton'`` (default) or ``'brent'``.
        return_diagnostics : bool
            If True, return a dict with sigma, iterations, and method used.

        Returns
        -------
        float or dict
            Implied volatility (annualised), or diagnostics dict.

        Raises
        ------
        IVSolverError
            If no valid implied vol exists for the given market price.
        """
        self._validate_bounds(market_price)
        sigma0 = self._initial_sigma(market_price)

        if method == "brent":
            sigma = self._brent(market_price)
            diag = {"sigma": sigma, "iterations": None, "method": "brent", "status": "converged"}
        else:
            sigma, n_iter, status = self._newton_raphson(market_price, sigma0)

            if status == "converged":
                diag = {"sigma": sigma, "iterations": n_iter, "method": "newton", "status": "converged"}
            else:
                # Fallback to Brent
                warnings.warn(
                    f"Newton-Raphson {status} after {n_iter} iterations. "
                    "Falling back to Brent's method.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                sigma = self._brent(market_price)
                diag = {
                    "sigma": sigma,
                    "iterations": n_iter,
                    "method": "brent_fallback",
                    "status": "converged",
                }

        if return_diagnostics:
            return diag
        return diag["sigma"]

    def solve_smile(
        self,
        market_prices: dict[float, float],
    ) -> dict[float, float]:
        """Solve IV for multiple strikes (building a vol smile).

        Parameters
        ----------
        market_prices : dict
            Mapping of {strike: market_price}.

        Returns
        -------
        dict
            Mapping of {strike: implied_vol}. Returns NaN for failed strikes.
        """
        result = {}
        original_K = self.K
        for K, price in market_prices.items():
            self.K = K
            try:
                result[K] = self.solve(price)
            except IVSolverError as exc:
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
                result[K] = float("nan")
        self.K = original_K
        return result
