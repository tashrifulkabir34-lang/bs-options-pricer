"""
app.py
======
Black-Scholes Option Pricer & Greeks Dashboard — Streamlit Application.

Launch with:
    streamlit run app.py

Author: tashrifulkabir34-lang
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import streamlit as st
import warnings

from src.black_scholes import BlackScholesModel, OptionParams
from src.greeks import GreeksCalculator
from src.implied_vol import ImpliedVolatilitySolver, IVSolverError
from src.strategies import OptionStrategy
from src.scenarios import ScenarioAnalyzer, HISTORICAL_STRESS_SCENARIOS

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BS Option Pricer & Greeks Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark professional theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d27, #252836);
        border: 1px solid #2d3142;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-label { color: #8892b0; font-size: 0.78rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ccd6f6; font-size: 1.55rem; font-weight: 700; margin: 4px 0; }
    .metric-positive { color: #64ffda; }
    .metric-negative { color: #ff6b6b; }
    .section-header {
        color: #ccd6f6; font-size: 1.1rem; font-weight: 700;
        border-left: 3px solid #64ffda; padding-left: 10px; margin: 20px 0 12px;
    }
    .badge {
        display: inline-block; background: #172a45;
        border: 1px solid #64ffda; border-radius: 20px;
        color: #64ffda; font-size: 0.72rem; padding: 2px 10px;
        font-weight: 600; letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib style
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#2d3142",
    "axes.labelcolor": "#8892b0",
    "xtick.color": "#8892b0",
    "ytick.color": "#8892b0",
    "grid.color": "#2d3142",
    "grid.alpha": 0.8,
    "text.color": "#ccd6f6",
    "lines.linewidth": 2.0,
    "font.family": "monospace",
})

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — global parameters
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Option Parameters")
    st.markdown("---")

    S = st.number_input("Spot Price (S)", min_value=1.0, max_value=10000.0,
                         value=100.0, step=1.0, help="Current underlying price")
    K = st.number_input("Strike Price (K)", min_value=1.0, max_value=10000.0,
                         value=100.0, step=1.0, help="Option strike price")
    T_days = st.slider("Days to Expiry", min_value=1, max_value=730,
                        value=90, step=1, help="Calendar days until expiration")
    T = T_days / 365.0

    r = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=20.0,
                   value=5.0, step=0.25) / 100.0
    sigma = st.slider("Implied Volatility (%)", min_value=1.0, max_value=200.0,
                       value=20.0, step=0.5) / 100.0
    q = st.slider("Dividend Yield (%)", min_value=0.0, max_value=15.0,
                   value=0.0, step=0.25) / 100.0
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)

    st.markdown("---")
    st.markdown("### 📐 Display Range")
    spot_lo_pct = st.slider("Spot Range Lower (%ATM)", 50, 99, 70)
    spot_hi_pct = st.slider("Spot Range Upper (%ATM)", 101, 200, 130)
    n_points = st.select_slider("Grid Points", options=[50, 100, 200, 300], value=100)

    st.markdown("---")
    st.caption("© 2024 tashrifulkabir34-lang | Black-Scholes Dashboard")

# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────
params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
model = BlackScholesModel(params)
gc = GreeksCalculator(model)
greeks = gc.all_greeks()
price = model.price()
S_range = np.linspace(S * spot_lo_pct / 100, S * spot_hi_pct / 100, n_points)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
with col_h1:
    st.markdown("# 📊 Black-Scholes Option Pricer")
    st.markdown(
        f'<span class="badge">{"CALL" if option_type == "call" else "PUT"}</span> '
        f'<span class="badge">S={S:.2f}</span> '
        f'<span class="badge">K={K:.2f}</span> '
        f'<span class="badge">T={T_days}d</span> '
        f'<span class="badge">σ={sigma*100:.1f}%</span>',
        unsafe_allow_html=True
    )
with col_h2:
    moneyness = S / K - 1
    m_label = "ATM" if abs(moneyness) < 0.01 else ("ITM" if (
        (option_type == "call" and moneyness > 0) or
        (option_type == "put" and moneyness < 0)
    ) else "OTM")
    st.metric("Moneyness", m_label, f"{moneyness*100:+.2f}%")
with col_h3:
    parity_err = model.put_call_parity_check()
    st.metric("Parity Check", f"{abs(parity_err):.2e}", help="Put-Call parity residual (should be ~0)")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Pricing & Greeks",
    "🔧 Implied Volatility",
    "📉 Strategy Payoffs",
    "🌡️ Scenario Analysis",
    "🔥 Stress Testing",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pricing & Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Option Price</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Theoretical Price</div>
            <div class="metric-value metric-positive">${price:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        iv = model.intrinsic_value()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Intrinsic Value</div>
            <div class="metric-value">${iv:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        tv = model.time_value()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Time Value</div>
            <div class="metric-value">${tv:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        d1_val = model.d1
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">d1 / d2</div>
            <div class="metric-value">{d1_val:.4f} / {model.d2:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">First-Order Greeks</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    greek_data = [
        ("Delta", greeks.delta, "Price change per $1 spot move", greeks.delta >= 0),
        ("Gamma", greeks.gamma, "Delta change per $1 spot move", True),
        ("Theta/day", greeks.theta, "Price decay per calendar day", greeks.theta >= 0),
        ("Vega/1%", greeks.vega, "Price change per 1% vol increase", True),
        ("Rho/1%", greeks.rho, "Price change per 1% rate increase", greeks.rho >= 0),
    ]
    for col, (label, val, tooltip, positive) in zip([c1, c2, c3, c4, c5], greek_data):
        color_class = "metric-positive" if positive else "metric-negative"
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color_class}">{val:.5f}</div>
            </div>""", unsafe_allow_html=True)
            st.caption(tooltip)

    st.markdown('<div class="section-header">Second-Order Greeks</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    second_data = [
        ("Vanna", greeks.vanna, "∂Delta/∂σ  or  ∂Vega/∂S"),
        ("Charm/day", greeks.charm, "Delta decay per calendar day"),
        ("Vomma/1%", greeks.vomma, "Vega change per 1% vol"),
        ("Veta/day", greeks.veta, "Vega decay per calendar day"),
    ]
    for col, (label, val, tooltip) in zip([c1, c2, c3, c4], second_data):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val:.6f}</div>
            </div>""", unsafe_allow_html=True)
            st.caption(tooltip)

    # Greeks profile charts
    st.markdown('<div class="section-header">Greeks vs. Spot Price</div>', unsafe_allow_html=True)
    sa = ScenarioAnalyzer(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
    sens = sa.spot_sensitivity(S_range)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor("#0e1117")
    plot_pairs = [
        (axes[0, 0], "Option Price", sens["price"], "#64ffda"),
        (axes[0, 1], "Delta", sens["delta"], "#7ec8e3"),
        (axes[0, 2], "Gamma", sens["gamma"], "#f4c87e"),
        (axes[1, 0], "Theta (per day)", sens["theta"], "#ff6b6b"),
        (axes[1, 1], "Vega (per 1%)", sens["vega"], "#c3a6ff"),
        (axes[1, 2], "P&L vs Base", sens["pnl"], "#64ffda"),
    ]
    for ax, title, data, color in plot_pairs:
        ax.plot(S_range, data, color=color, linewidth=2)
        ax.axvline(S, color="#8892b0", linestyle="--", alpha=0.6, linewidth=1)
        ax.axvline(K, color="#f4c87e", linestyle=":", alpha=0.6, linewidth=1)
        if title == "P&L vs Base":
            ax.axhline(0, color="#8892b0", linestyle="-", alpha=0.4, linewidth=1)
            ax.fill_between(S_range, data, 0,
                            where=(data >= 0), alpha=0.15, color="#64ffda")
            ax.fill_between(S_range, data, 0,
                            where=(data < 0), alpha=0.15, color="#ff6b6b")
        ax.set_title(title, color="#ccd6f6", fontsize=10, pad=8)
        ax.set_xlabel("Spot Price", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Greeks table
    with st.expander("📋 Full Greeks Table"):
        greeks_df = pd.DataFrame(
            list(greeks.to_dict().items()),
            columns=["Greek", "Value"]
        )
        greeks_df["Value"] = greeks_df["Value"].apply(lambda x: f"{x:.8f}")
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Implied Volatility Solver
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Implied Volatility Solver</div>', unsafe_allow_html=True)
    st.markdown("Enter a market-observed option price to back out the implied volatility.")

    col_iv1, col_iv2 = st.columns([2, 3])
    with col_iv1:
        market_price = st.number_input(
            "Market Price",
            min_value=0.001,
            max_value=float(S),
            value=round(float(price), 4),
            step=0.01,
            format="%.4f",
        )
        iv_method = st.radio("Solver Method", ["newton", "brent"], horizontal=True)
        solve_btn = st.button("🔍 Solve IV", type="primary")

        if solve_btn or True:
            try:
                solver = ImpliedVolatilitySolver(
                    S=S, K=K, T=T, r=r, q=q, option_type=option_type
                )
                diag = solver.solve(market_price, method=iv_method, return_diagnostics=True)
                iv_result = diag["sigma"]

                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Implied Volatility</div>
                    <div class="metric-value metric-positive">{iv_result*100:.4f}%</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Method Used</div>
                    <div class="metric-value">{diag['method']}</div>
                </div>""", unsafe_allow_html=True)
                if diag["iterations"]:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Iterations</div>
                        <div class="metric-value">{diag['iterations']}</div>
                    </div>""", unsafe_allow_html=True)
            except IVSolverError as e:
                st.error(f"❌ IV Solver Error: {e}")

    with col_iv2:
        # IV vs market price curve
        st.markdown("**Implied Volatility vs. Market Price**")
        price_grid = np.linspace(
            max(price * 0.3, 0.01),
            price * 2.5,
            80
        )
        iv_grid = []
        solver_curve = ImpliedVolatilitySolver(S=S, K=K, T=T, r=r, q=q, option_type=option_type)
        for mp in price_grid:
            try:
                iv_grid.append(solver_curve.solve(mp) * 100)
            except IVSolverError:
                iv_grid.append(float("nan"))

        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.plot(price_grid, iv_grid, color="#64ffda", linewidth=2)
        ax.axvline(market_price, color="#f4c87e", linestyle="--", alpha=0.8, linewidth=1.5,
                   label=f"Market Price ${market_price:.2f}")
        ax.set_xlabel("Market Price ($)")
        ax.set_ylabel("Implied Vol (%)")
        ax.set_title("IV vs. Market Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Vol smile construction
    st.markdown('<div class="section-header">Volatility Smile (Synthetic)</div>', unsafe_allow_html=True)
    st.markdown(
        "Constructs a synthetic vol smile using a linear moneyness skew. "
        "This illustrates the implied vol surface, not flat-vol assumption."
    )
    skew_val = st.slider("Skew Slope", min_value=-0.10, max_value=0.10, value=-0.03, step=0.005)
    K_smile = np.linspace(S * 0.70, S * 1.30, 25)
    sigma_smile = []
    for Ki in K_smile:
        moneyness = np.log(Ki / S) / (sigma * np.sqrt(T))
        sigma_smile.append(max(sigma + skew_val * moneyness, 0.001))

    fig3, ax = plt.subplots(figsize=(10, 4))
    ax.plot(K_smile, np.array(sigma_smile) * 100, color="#c3a6ff", linewidth=2.5, label="Implied Vol Smile")
    ax.axvline(K, color="#f4c87e", linestyle="--", alpha=0.7, linewidth=1.5, label=f"Strike K={K:.0f}")
    ax.axvline(S, color="#64ffda", linestyle=":", alpha=0.7, linewidth=1.5, label=f"Spot S={S:.0f}")
    ax.axhline(sigma * 100, color="#8892b0", linestyle="--", alpha=0.5, linewidth=1, label="Flat Vol")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title("Synthetic Volatility Smile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Strategy Payoffs
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Strategy Payoff Diagrams</div>', unsafe_allow_html=True)

    strategy_name = st.selectbox(
        "Select Strategy",
        [
            "Long Call", "Long Put",
            "Long Straddle", "Long Strangle",
            "Bull Call Spread", "Bear Put Spread",
            "Iron Condor", "Long Butterfly",
            "Covered Call", "Protective Put",
        ]
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        K2 = st.number_input("Strike K2 (spread strategies)", value=float(K * 1.05), step=1.0)
    with col_s2:
        K_put_str = st.number_input("Put Strike (strangle)", value=float(K * 0.95), step=1.0)

    T_mid_days = st.slider("Mid-Life T Remaining (days)", 1, T_days - 1, T_days // 2)
    T_mid = T_mid_days / 365.0

    # Build strategy
    strategy_map = {
        "Long Call": lambda: OptionStrategy.long_call(S, K, T, r, sigma, q),
        "Long Put": lambda: OptionStrategy.long_put(S, K, T, r, sigma, q),
        "Long Straddle": lambda: OptionStrategy.long_straddle(S, K, T, r, sigma, q),
        "Long Strangle": lambda: OptionStrategy.long_strangle(S, K_put_str, K2, T, r, sigma, q),
        "Bull Call Spread": lambda: OptionStrategy.bull_call_spread(S, K, K2, T, r, sigma, q),
        "Bear Put Spread": lambda: OptionStrategy.bear_put_spread(S, K_put_str, K, T, r, sigma, q),
        "Iron Condor": lambda: OptionStrategy.iron_condor(S, K * 0.85, K * 0.95, K * 1.05, K * 1.15, T, r, sigma, q),
        "Long Butterfly": lambda: OptionStrategy.long_butterfly(S, K * 0.90, K, K * 1.10, T, r, sigma, q),
        "Covered Call": lambda: OptionStrategy.covered_call(S, K, T, r, sigma, q),
        "Protective Put": lambda: OptionStrategy.protective_put(S, K, T, r, sigma, q),
    }

    try:
        strat = strategy_map[strategy_name]()
        expiry_pnl = strat.expiry_pnl(S_range)
        mid_pnl = strat.mid_life_pnl(S_range, T_mid)
        breakevens = strat.breakeven_points(S_range)
        max_p = strat.max_profit(S_range)
        max_l = strat.max_loss(S_range)

        # Summary
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Net Premium", f"${strat.net_premium:.4f}")
        with mc2:
            st.metric("Max Profit", f"${max_p:.2f}" if max_p < 9999 else "Unlimited")
        with mc3:
            st.metric("Max Loss", f"${max_l:.2f}")
        with mc4:
            be_str = ", ".join([f"${b:.1f}" for b in breakevens]) if breakevens else "N/A"
            st.metric("Break-Even(s)", be_str)

        # Plot
        fig4, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, data, label, color, title in [
            (axes[0], expiry_pnl, "At Expiry", "#64ffda", "Expiry P&L"),
            (axes[1], mid_pnl, f"T−{T_mid_days}d remaining", "#c3a6ff", "Mid-Life P&L"),
        ]:
            ax.plot(S_range, data, color=color, linewidth=2.5, label=label)
            ax.axhline(0, color="#8892b0", linewidth=1, alpha=0.5)
            ax.fill_between(S_range, data, 0, where=(data >= 0), alpha=0.12, color="#64ffda")
            ax.fill_between(S_range, data, 0, where=(data < 0), alpha=0.12, color="#ff6b6b")
            ax.axvline(S, color="#f4c87e", linestyle="--", alpha=0.7, linewidth=1.5, label=f"Spot {S:.0f}")
            for be in breakevens:
                ax.axvline(be, color="#ff6b6b", linestyle=":", alpha=0.6, linewidth=1)
            ax.set_xlabel("Spot Price at Expiry")
            ax.set_ylabel("P&L ($)")
            ax.set_title(f"{strategy_name} — {title}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    except Exception as e:
        st.error(f"Strategy construction error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Scenario Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">P&L Heatmaps</div>', unsafe_allow_html=True)
    sa4 = ScenarioAnalyzer(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("**Spot × Volatility P&L Heatmap**")
        sigma_range = np.linspace(max(sigma * 0.3, 0.01), sigma * 2.5, 30)
        S_heat = np.linspace(S * 0.7, S * 1.3, 30)
        heatmap_sv = sa4.spot_vol_heatmap(S_heat, sigma_range)

        fig5, ax = plt.subplots(figsize=(7, 5))
        vmax = np.abs(heatmap_sv).max() or 1
        im = ax.imshow(
            heatmap_sv,
            aspect="auto",
            origin="lower",
            extent=[sigma_range[0] * 100, sigma_range[-1] * 100, S_heat[0], S_heat[-1]],
            cmap="RdYlGn",
            vmin=-vmax, vmax=vmax,
        )
        ax.axhline(S, color="white", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(sigma * 100, color="white", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Implied Vol (%)")
        ax.set_ylabel("Spot Price")
        ax.set_title("P&L: Spot × Volatility")
        plt.colorbar(im, ax=ax, label="P&L ($)")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with col_h2:
        st.markdown("**Spot × Time P&L Heatmap**")
        days_range = np.linspace(0, T_days * 0.9, 30)
        heatmap_st = sa4.spot_time_heatmap(S_heat, days_range)

        fig6, ax = plt.subplots(figsize=(7, 5))
        vmax2 = np.abs(heatmap_st).max() or 1
        im2 = ax.imshow(
            heatmap_st,
            aspect="auto",
            origin="lower",
            extent=[days_range[0], days_range[-1], S_heat[0], S_heat[-1]],
            cmap="RdYlGn",
            vmin=-vmax2, vmax=vmax2,
        )
        ax.axhline(S, color="white", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Days Elapsed")
        ax.set_ylabel("Spot Price")
        ax.set_title("P&L: Spot × Time")
        plt.colorbar(im2, ax=ax, label="P&L ($)")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    # Theta decay and delta hedge
    st.markdown('<div class="section-header">Sensitivity Curves</div>', unsafe_allow_html=True)
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        days_arr = np.linspace(0, T_days * 0.95, 100)
        decay = sa4.time_decay_curve(days_arr)
        fig7, ax = plt.subplots(figsize=(7, 4))
        ax.plot(days_arr, decay, color="#ff6b6b", linewidth=2)
        ax.axhline(price, color="#8892b0", linestyle="--", alpha=0.5)
        ax.set_xlabel("Days Elapsed")
        ax.set_ylabel("Option Price ($)")
        ax.set_title("Theta Decay Curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

    with col_sc2:
        dh = sa4.delta_hedge_analysis(S_range)
        fig8, ax = plt.subplots(figsize=(7, 4))
        ax.plot(S_range, dh["unhedged_pnl"], color="#64ffda", linewidth=2, label="Unhedged P&L")
        ax.plot(S_range, dh["hedged_pnl"], color="#f4c87e", linewidth=2, linestyle="--",
                label=f"Delta-Hedged P&L (Δ={dh['delta_base']:.3f})")
        ax.axhline(0, color="#8892b0", linewidth=1, alpha=0.4)
        ax.axvline(S, color="#8892b0", linestyle=":", alpha=0.5)
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("P&L ($)")
        ax.set_title("Delta-Hedged vs Unhedged P&L")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close(fig8)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Stress Testing (Out-of-Sample)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Historical Stress Test Scenarios</div>', unsafe_allow_html=True)
    st.markdown(
        "These are **out-of-sample** shock scenarios calibrated to actual historical market events. "
        "They test option P&L under conditions unseen in the in-sample parameter space."
    )

    sa5 = ScenarioAnalyzer(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
    stress_results = sa5.stress_test()
    df_stress = pd.DataFrame(stress_results)
    df_stress["Signal"] = df_stress["P&L ($)"].apply(
        lambda x: "✅ Profit" if x > 0 else "❌ Loss"
    )

    # Colour positive/negative
    def highlight_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #64ffda; font-weight: bold;"
            elif val < 0:
                return "color: #ff6b6b; font-weight: bold;"
        return ""

    st.dataframe(
        df_stress.style.applymap(highlight_pnl, subset=["P&L ($)", "P&L (%)"]),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart
    fig9, ax = plt.subplots(figsize=(12, 5))
    colors = ["#64ffda" if x >= 0 else "#ff6b6b" for x in df_stress["P&L ($)"]]
    bars = ax.barh(df_stress["Scenario"], df_stress["P&L ($)"], color=colors, alpha=0.85)
    ax.axvline(0, color="#ccd6f6", linewidth=1, alpha=0.6)
    for bar, val in zip(bars, df_stress["P&L ($)"]):
        ax.text(
            bar.get_width() + (0.01 * abs(df_stress["P&L ($)"].min())),
            bar.get_y() + bar.get_height() / 2,
            f"${val:.2f}",
            va="center", ha="left" if val >= 0 else "right",
            color="#ccd6f6", fontsize=8
        )
    ax.set_xlabel("P&L ($)")
    ax.set_title("Stress Test P&L by Historical Scenario")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9)

    # Worst-case summary
    worst = df_stress.loc[df_stress["P&L ($)"].idxmin()]
    best = df_stress.loc[df_stress["P&L ($)"].idxmax()]
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.error(f"**Worst Scenario:** {worst['Scenario']} → P&L = ${worst['P&L ($)']:.4f} ({worst['P&L (%)']:.1f}%)")
    with col_w2:
        st.success(f"**Best Scenario:** {best['Scenario']} → P&L = ${best['P&L ($)']:.4f} ({best['P&L (%)']:.1f}%)")

    st.markdown("---")
    st.caption(
        "⚠️ **Limitations**: These scenarios assume flat volatility (no smile/skew), "
        "European exercise only, and no transaction costs. Real-world P&L will differ."
    )
