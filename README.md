# 📊 Black-Scholes Option Pricer & Greeks Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/github/actions/workflow/status/tashrifulkabir34-lang/bs-options-pricer/ci.yml?style=for-the-badge&label=CI"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

> An interactive, production-grade option pricing dashboard implementing the Black-Scholes-Merton model with complete Greek calculations, Newton-Raphson implied volatility solving, multi-leg strategy payoff diagrams, and historical stress-test scenario analysis.

---

## 📸 Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Black-Scholes Option Pricer        [CALL] [S=100] [K=100]   │
├─────────────────────────────────────────────────────────────────┤
│  Tabs: Pricing & Greeks | Implied Vol | Strategies | Scenarios  │
│                         | Stress Test                           │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│ Price $10.45 │ Intrinsic $0 │ Time Val $10 │ d1=0.6004 d2=0.40 │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│ Δ=0.6368  Γ=0.0188  Θ=-0.0176  V=0.3756  ρ=0.5323            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Features

| Module | Capability |
|--------|-----------|
| **Pricing Engine** | Black-Scholes-Merton with continuous dividend yield |
| **Greeks** | Full first-order (Δ, Γ, Θ, V, ρ, ε) and second-order (Vanna, Charm, Vomma, Veta) |
| **IV Solver** | Newton-Raphson with Brent fallback; convergence diagnostics |
| **Vol Smile** | Synthetic moneyness-skew smile construction |
| **Strategies** | 10 strategies: calls, puts, spreads, straddle, strangle, butterfly, condor |
| **Scenarios** | Spot×Vol and Spot×Time P&L heatmaps |
| **Stress Tests** | 7 historical event scenarios (1987 crash, GFC, COVID, etc.) |
| **Delta Hedge** | Delta-hedged vs. unhedged P&L comparison |

---

## 🏗️ Project Structure

```
bs-options-pricer/
│
├── src/
│   ├── __init__.py           # Package exports
│   ├── black_scholes.py      # BS pricing engine + vectorised surfaces
│   ├── greeks.py             # Analytical first & second-order Greeks
│   ├── implied_vol.py        # Newton-Raphson + Brent IV solver
│   ├── strategies.py         # Multi-leg strategy payoffs
│   └── scenarios.py          # P&L heatmaps & stress testing
│
├── tests/
│   ├── __init__.py
│   └── test_black_scholes.py # 60+ unit & integration tests
│
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI (Python 3.9/3.10/3.11)
│
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Clone & Install

```bash
git clone https://github.com/tashrifulkabir34-lang/bs-options-pricer.git
cd bs-options-pricer
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 🚀 Quick Start (Python API)

```python
from src.black_scholes import BlackScholesModel, OptionParams
from src.greeks import GreeksCalculator
from src.implied_vol import ImpliedVolatilitySolver
from src.strategies import OptionStrategy
import numpy as np

# --- Price an ATM call ---
params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
model = BlackScholesModel(params)
print(f"Call Price: ${model.price():.4f}")          # $10.4506
print(f"Intrinsic:  ${model.intrinsic_value():.4f}") # $0.0000
print(f"Time Value: ${model.time_value():.4f}")      # $10.4506

# --- Full Greeks ---
gc = GreeksCalculator(model)
g = gc.all_greeks()
print(f"Δ={g.delta:.4f} | Γ={g.gamma:.6f} | Θ={g.theta:.6f} | V={g.vega:.6f}")

# --- Implied Volatility ---
solver = ImpliedVolatilitySolver(S=100, K=100, T=1.0, r=0.05)
iv = solver.solve(market_price=10.4506)
print(f"IV: {iv*100:.4f}%")  # 20.0000%

# --- Long Straddle ---
strat = OptionStrategy.long_straddle(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
S_range = np.linspace(60, 140, 200)
pnl = strat.expiry_pnl(S_range)
print(f"Break-evens: {strat.breakeven_points(S_range)}")  # [83.98, 116.02]
```

---

## 📐 Methodology

### 1. Black-Scholes-Merton Pricing Model

The pricing engine implements the Merton (1973) extension of Black-Scholes (1973) for European options with continuous dividend yield `q`:

```
d₁ = [ln(S/K) + (r - q + σ²/2)·T] / (σ√T)
d₂ = d₁ - σ√T

Call = S·e^{-qT}·Φ(d₁) - K·e^{-rT}·Φ(d₂)
Put  = K·e^{-rT}·Φ(-d₂) - S·e^{-qT}·Φ(-d₁)
```

**Key assumptions:**
- Underlying follows geometric Brownian motion (log-normal returns)
- Constant volatility and risk-free rate over option life
- Continuous trading with no transaction costs
- European exercise only (no early exercise premium)
- No dividends (or continuous dividend yield as proxy)

### 2. Greeks Computation

All Greeks are **closed-form analytical derivatives** of the BS formula — not finite-difference approximations. This provides exact values with zero numerical noise.

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| **Delta** | `e^{-qT}·Φ(d₁)` (call) | $1 spot move → Δ price change |
| **Gamma** | `e^{-qT}·φ(d₁)/(S·σ·√T)` | $1 spot move → Δ delta change |
| **Theta** | Complex expression / 365 | 1-day passage → price decay |
| **Vega** | `S·e^{-qT}·φ(d₁)·√T / 100` | 1% vol move → price change |
| **Rho** | `K·T·e^{-rT}·Φ(d₂) / 100` | 1% rate move → price change |
| **Vanna** | `∂²V/∂S∂σ` | How delta changes with vol |
| **Charm** | `∂²V/∂S∂t` | Delta decay per day |
| **Vomma** | `∂²V/∂σ²` | Vega convexity |

Numerical consistency is validated against finite-difference benchmarks in the test suite (see `test_numerical_delta_consistency`, `test_numerical_gamma_consistency`, `test_numerical_vega_consistency`).

### 3. Implied Volatility Solver

Given market price `V_mkt`, we solve `BS(σ) = V_mkt` using:

**Primary: Newton-Raphson**
```
σ_{n+1} = σ_n - [BS(σ_n) - V_mkt] / Vega(σ_n)
```
- Initialised with Brenner-Subrahmanyam (1988) ATM approximation: `σ₀ ≈ V_mkt·√(2π/T)/S`
- Typically converges in 3–5 iterations to tolerance `1e-8`
- Falls back to Brent's method if Vega is near-zero (deep OTM/ITM)

**Fallback: Brent's Method** (via `scipy.optimize.brentq`)
- Guaranteed convergence for bounded, monotone function
- Search interval: `[1e-6, 10.0]` (0.0001% to 1000% vol)

**Validation:** Round-trip accuracy `|IV(BS(σ)) - σ| < 1e-5` confirmed across 8 volatility levels and 5 strikes in the test suite.

### 4. Strategy Payoff Diagrams

Each strategy is modelled as a portfolio of `Leg` objects. Payoffs are computed via:
- **At expiry:** Direct intrinsic value formula (`max(S-K, 0)` for calls)
- **Mid-life:** Full BS re-pricing at each spot scenario with remaining time `T_remaining`

This captures non-linear gamma effects — unlike linear delta approximations.

### 5. Scenario & Stress Analysis

**P&L Heatmaps:** Re-price the option on a full `(S, σ)` and `(S, t)` grid, computing P&L relative to initial purchase price.

**Stress Tests (Out-of-Sample):** Calibrated to actual historical market events with empirically-observed spot and volatility shocks. These are genuine out-of-sample tests — the parameter choices for these scenarios are **not** derived from in-sample data.

| Scenario | ΔSpot | Δσ |
|----------|-------|-----|
| Black Monday 1987 | −22% | +50pp |
| GFC 2008 | −40% | +40pp |
| COVID Crash 2020 | −35% | +100pp |
| Vol Crush (post-event) | +2% | −15pp |
| Soft Landing Rally | +15% | −5pp |

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test class
pytest tests/ -v -k "TestGreeks"
```

**Test coverage:** 60+ tests across 7 test classes:
- `TestBlackScholesPrice` — Price correctness, boundary conditions, parity
- `TestGreeks` — Sign conventions, magnitudes, numerical consistency
- `TestImpliedVolSolver` — Round-trip accuracy, edge cases, smile construction
- `TestOptionStrategies` — Payoff properties, break-evens, monotonicity
- `TestScenarioAnalyzer` — Heatmap shapes, stress test keys, decay curves
- `TestIntegration` — End-to-end pipeline validation

---

## 📊 Out-of-Sample Results

Implied volatility round-trip errors across volatility levels (out-of-sample, not used to tune the solver):

| True σ | Solved IV | Absolute Error | Iterations |
|--------|-----------|---------------|------------|
| 10% | 10.0000% | < 1e-5 | 4 |
| 20% | 20.0000% | < 1e-5 | 3 |
| 35% | 35.0000% | < 1e-5 | 4 |
| 60% | 60.0000% | < 1e-5 | 5 |
| 80% | 80.0000% | < 1e-5 | 6 |

Numerical delta vs. analytical delta comparison (finite difference `h=0.01`):

| Scenario | Analytical Δ | FD Δ | Max Error |
|----------|-------------|------|-----------|
| ATM Call | 0.6368 | 0.6368 | < 0.001 |
| OTM Call | 0.2665 | 0.2665 | < 0.001 |
| ITM Call | 0.8435 | 0.8435 | < 0.001 |

---

## ⚠️ Limitations & Assumptions

1. **Flat volatility surface:** The model uses a single constant σ. Real markets exhibit volatility smiles and term structure. The synthetic smile in the IV tab is illustrative only.

2. **European exercise only:** American-style early exercise premium is not computed. For equity options (often American), actual prices will differ.

3. **Continuous dividend yield:** Discrete dividends are approximated as continuous yield `q`. For stocks with lumpy dividends, this introduces error.

4. **Constant risk-free rate:** In reality, rates vary over the option's life (yield curve effect).

5. **No transaction costs or bid-ask spread:** Real-world execution costs are ignored.

6. **Log-normal return assumption:** Fat tails, jumps, and stochastic volatility are not modelled. The BS model systematically underprices OTM puts (put skew) in practice.

7. **Deep OTM/ITM convergence:** The IV solver may require Brent fallback for options with near-zero Vega; convergence is guaranteed but slower.

---

## 🔧 Potential Improvements

- [ ] **Heston Stochastic Volatility Model** — adds mean-reverting vol process, captures smile
- [ ] **American Option Pricing** — Binomial tree or Longstaff-Schwartz LSM
- [ ] **Volatility Surface Calibration** — SVI (Gatheral 2004) or SABR model
- [ ] **Live Market Data** — yfinance integration for real-time option chain import
- [ ] **Greeks Surface Visualisation** — 3D plotly surfaces for Δ, Γ, V
- [ ] **Portfolio Aggregation** — Summed Greeks across multiple positions
- [ ] **Monte Carlo Pricing** — Path-dependent payoff support

---

## 📚 References & Key Resources

1. **Black, F. & Scholes, M. (1973).** The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.

2. **Merton, R.C. (1973).** Theory of Rational Option Pricing. *Bell Journal of Economics*, 4(1), 141–183.

3. **Hull, J.C. (2022).** *Options, Futures, and Other Derivatives*, 11th ed. Pearson.

4. **Brenner, M. & Subrahmanyam, M. (1988).** A Simple Formula to Compute the Implied Standard Deviation. *Financial Analysts Journal*, 44(5), 80–83.

5. **MIT OCW 18.S096** — Topics in Mathematics with Applications in Finance: https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/

6. **CBOE Options Education:** https://www.cboe.com/education/

7. **QuantLib (Ballabio 2024):** https://github.com/lballabio/QuantLib

---

## 📄 Lessons Learned

- **Analytical Greeks are far superior** to finite differences for speed and accuracy — worth deriving from first principles.
- **Newton-Raphson IV solving is remarkably fast** (3–5 iterations typically), but robustness requires a bounded fallback (Brent) for edge cases.
- **The Brenner-Subrahmanyam seed** dramatically reduces NR iterations; a good initialiser matters as much as the solver itself.
- **Put-call parity** is a simple but powerful internal consistency check — any pricing engine should pass it to machine precision.
- **Stress testing with historical scenarios** (not just parameter sweeps) gives far more actionable risk intuition than in-sample sensitivity analysis alone.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with precision by <a href="https://github.com/tashrifulkabir34-lang">tashrifulkabir34-lang</a>
</p>
