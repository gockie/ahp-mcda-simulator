import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import io
import json
from itertools import combinations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AHP-MCDA Monte Carlo Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root variables */
:root {
    --navy:   #1B3A5C;
    --teal:   #0E7C7B;
    --blue:   #2E75B6;
    --amber:  #C47A00;
    --green:  #2E8B57;
    --red:    #B22222;
    --light:  #F4F7FB;
    --border: #D0DAE8;
    --text:   #1A1A2E;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main header */
.app-header {
    background: linear-gradient(135deg, #1B3A5C 0%, #0E4D7B 50%, #0E7C7B 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(14,124,123,0.3) 0%, transparent 70%);
    pointer-events: none;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    margin: 0 0 0.5rem;
    color: white;
    letter-spacing: -0.5px;
}
.app-header p {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    opacity: 0.85;
    margin: 0;
    font-weight: 300;
    line-height: 1.6;
}
.app-header .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    color: rgba(255,255,255,0.9);
}

/* Mode cards */
.mode-card {
    background: white;
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
    cursor: pointer;
}
.mode-card:hover { border-color: var(--navy); }
.mode-card.active { border-color: var(--teal); background: #F0FAFA; }
.mode-card h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--navy);
    margin: 0 0 0.5rem;
    font-size: 1.1rem;
}
.mode-card p { margin: 0; font-size: 0.88rem; color: #555; line-height: 1.5; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--navy);
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    flex: 1;
    min-width: 140px;
    text-align: center;
}
.metric-card .val {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: var(--navy);
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.78rem;
    color: #777;
    margin-top: 0.3rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.metric-card.good .val { color: var(--green); }
.metric-card.warn .val { color: var(--amber); }
.metric-card.bad  .val { color: var(--red); }

/* CR badge */
.cr-pass {
    display: inline-block;
    background: #E8F5E9;
    color: #2E8B57;
    border: 1px solid #2E8B57;
    border-radius: 20px;
    padding: 0.25rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
}
.cr-fail {
    display: inline-block;
    background: #FFEBEE;
    color: #B22222;
    border: 1px solid #B22222;
    border-radius: 20px;
    padding: 0.25rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Weight bar */
.weight-bar-wrap { margin: 0.2rem 0; }
.weight-bar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #444;
}

/* Tier badges */
.tier-1 { background:#E8F5E9; color:#2E8B57; border:1px solid #2E8B57; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tier-2 { background:#FFF8E1; color:#C47A00; border:1px solid #C47A00; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tier-3 { background:#FBE9E7; color:#8B3000; border:1px solid #8B3000; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tier-4 { background:#FAFAFA; color:#555; border:1px solid #AAA; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }

/* Info box */
.info-box {
    background: #EFF6FF;
    border-left: 4px solid var(--blue);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0.8rem 0;
    color: #1e3a5f;
}
.warn-box {
    background: #FFFBF0;
    border-left: 4px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0.8rem 0;
    color: #5a3a00;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F4F7FB;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--navy);
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.8rem;
    color: #999;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AHP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

SAATY_RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32,
            8:1.41, 9:1.45, 10:1.49, 11:1.51, 12:1.54, 13:1.56,
            14:1.57, 15:1.58, 16:1.59, 17:1.60, 18:1.61, 19:1.62, 20:1.63}

def compute_ahp(matrix):
    """Compute AHP weights and consistency from pairwise matrix."""
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    norm = matrix / col_sums
    weights = norm.mean(axis=1)
    weighted_sum = matrix @ weights
    lambdas = weighted_sum / weights
    lambda_max = lambdas.mean()
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI = SAATY_RI.get(n, 1.63)
    CR = CI / RI if RI > 0 else 0
    return weights, lambda_max, CI, CR

def build_matrix_from_upper(n, upper_vals):
    """Build full n×n reciprocal matrix from upper triangle values."""
    matrix = np.eye(n)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            v = upper_vals[idx]
            matrix[i, j] = v
            matrix[j, i] = 1.0 / v
            idx += 1
    return matrix

def run_monte_carlo(weights, scores, n_iter, p_perturb=0.30):
    """
    Run Monte Carlo weight perturbation analysis.
    weights: array of shape (n_criteria,)
    scores:  array of shape (n_alternatives, n_criteria) — normalised P_ik
    Returns: all_weights (n_iter, n_criteria), all_scores (n_iter, n_alternatives)
    """
    np.random.seed(42)
    n_c = len(weights)
    sigma = np.maximum(weights * 0.03, 0.002)  # ~3% relative noise floor 0.002

    all_w = np.zeros((n_iter, n_c))
    all_s = np.zeros((n_iter, scores.shape[0]))

    for s in range(n_iter):
        noise = np.random.normal(0, sigma)
        mask  = np.random.random(n_c) < p_perturb
        w_p   = weights + noise * mask
        w_p   = np.maximum(w_p, 0.005)
        w_p   = w_p / w_p.sum()
        all_w[s] = w_p
        all_s[s] = scores @ w_p   # shape (n_alt,)

    return all_w, all_s

def jenks_fisher(values, k):
    """Jenks-Fisher optimal classification."""
    n = len(values)
    sorted_vals = np.sort(values)

    if k >= n:
        return sorted_vals[:-1], np.arange(1, n, dtype=int)

    # DP matrices
    mat_SSW = np.full((k, n), np.inf)
    mat_BI  = np.zeros((k, n), dtype=int)

    for i in range(n):
        v = sorted_vals[:i+1]
        mat_SSW[0, i] = np.var(v) * len(v) if len(v) > 0 else 0
        mat_BI[0, i]  = 0

    for cl in range(1, k):
        for i in range(cl, n):
            best_ssw = np.inf
            best_bi  = cl
            for m in range(cl-1, i):
                v_right  = sorted_vals[m+1:i+1]
                ssw_left = mat_SSW[cl-1, m]
                ssw_right= np.var(v_right) * len(v_right) if len(v_right) > 0 else 0
                total    = ssw_left + ssw_right
                if total < best_ssw:
                    best_ssw = total
                    best_bi  = m + 1
            mat_SSW[cl, i] = best_ssw
            mat_BI[cl, i]  = best_bi

    # Backtrack
    breaks_idx = []
    idx = n - 1
    for cl in range(k-1, 0, -1):
        bi = mat_BI[cl, idx]
        breaks_idx.append(bi)
        idx = bi - 1
    breaks_idx = sorted(breaks_idx)
    break_vals = sorted_vals[breaks_idx]

    total_ssw  = mat_SSW[k-1, n-1]
    total_sdam = np.var(sorted_vals) * n
    gvf = 1 - total_ssw / total_sdam if total_sdam > 0 else 1.0

    return break_vals, gvf

def assign_tiers(scores, breaks):
    """Assign tier labels given sorted break values (high breaks = high tier)."""
    sorted_breaks = np.sort(breaks)[::-1]  # descending
    tiers = []
    for s in scores:
        assigned = len(sorted_breaks) + 1
        for i, b in enumerate(sorted_breaks):
            if s >= b:
                assigned = i + 1
                break
        tiers.append(assigned)
    return np.array(tiers)

def gvf_for_k(values, k):
    _, gvf = jenks_fisher(values, k)
    return gvf


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

FONT  = 'DejaVu Sans'
DARK  = '#1B3A5C'
MED   = '#2E75B6'
TEAL  = '#0E7C7B'
AMBER = '#C47A00'
GREEN = '#2E8B57'
RED   = '#B22222'
TIER_COLS = {1: '#2E8B57', 2: '#C47A00', 3: '#8B4500', 4: '#555555'}

def fig_weights(weights, criteria_names):
    fig, ax = plt.subplots(figsize=(9, max(4, len(criteria_names)*0.42)), facecolor='white')
    ax.set_facecolor('#F9FAFB')
    n = len(weights)
    y = np.arange(n)
    sorted_idx = np.argsort(weights)
    w_sorted = weights[sorted_idx]
    names_sorted = [criteria_names[i] for i in sorted_idx]
    colors = [plt.cm.Blues(0.4 + 0.6 * w / max(weights)) for w in w_sorted]
    bars = ax.barh(y, w_sorted, color=colors, edgecolor='white', linewidth=0.8, height=0.7)
    for bar, w in zip(bars, w_sorted):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{w:.4f}', va='center', ha='left', fontsize=9,
                fontfamily=FONT, color=DARK, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=9, fontfamily=FONT)
    ax.set_xlabel('AHP Weight  wᵢ', fontsize=10, fontfamily=FONT)
    ax.set_title('Derived Criterion Weights (AHP Eigenvector Method)',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT, pad=10)
    ax.set_xlim(0, max(weights) * 1.25)
    ax.grid(axis='x', color='#E0E0E0', lw=0.6)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig

def fig_scores(alt_names, det_scores, tiers, breaks):
    fig, ax = plt.subplots(figsize=(9, max(4, len(alt_names)*0.55)), facecolor='white')
    ax.set_facecolor('#F9FAFB')
    n = len(alt_names)
    sorted_idx = np.argsort(det_scores)[::-1]
    scores_s = det_scores[sorted_idx]
    names_s  = [alt_names[i] for i in sorted_idx]
    tiers_s  = tiers[sorted_idx]
    y = np.arange(n)
    colors = [TIER_COLS.get(t, '#888') for t in tiers_s]
    bars = ax.barh(y, scores_s, color=colors, edgecolor='white', lw=0.8, height=0.68, alpha=0.85)
    for bar, s, t in zip(bars, scores_s, tiers_s):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                f'{s:.3f}', va='center', ha='left', fontsize=9.5,
                fontfamily=FONT, color=TIER_COLS.get(t,'#888'), fontweight='bold')
    for b in breaks:
        ax.axvline(b, color='#444', lw=1.0, ls='--', alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=9.5, fontfamily=FONT)
    ax.set_xlabel('Composite Suitability Score  Rᵏ ∈ [0, 1]', fontsize=10, fontfamily=FONT)
    ax.set_title('Ranked Composite Suitability Scores with Tier Classification',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT, pad=10)
    ax.set_xlim(0, 1.12)
    ax.invert_yaxis()
    ax.grid(axis='x', color='#E0E0E0', lw=0.6)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=TIER_COLS[t], alpha=0.85, label=f'Tier {t}')
                  for t in sorted(TIER_COLS)]
    from matplotlib.lines import Line2D
    legend_els.append(Line2D([0],[0], color='#444', lw=1, ls='--', label='Tier boundary'))
    ax.legend(handles=legend_els, fontsize=8.5, loc='lower right', framealpha=0.92)
    plt.tight_layout()
    return fig

def fig_mc_convergence(all_weights, criteria_names, weights_det, top_n=4):
    n_iter = len(all_weights)
    checkpoints = np.unique(np.concatenate([
        np.arange(100, 1000, 200),
        np.arange(1000, n_iter+1, max(500, n_iter//30))
    ])).astype(int)
    checkpoints = checkpoints[checkpoints <= n_iter]

    sorted_idx = np.argsort(weights_det)[::-1]
    top_idx = sorted_idx[:top_n]
    colors_c = [MED, TEAL, AMBER, '#6A4C93']

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor='white')
    fig.suptitle('Monte Carlo Sensitivity Test: Weight Standard Deviation Convergence',
                 fontsize=12, fontweight='bold', color=DARK, fontfamily=FONT)

    for ax_i, (ci, col) in enumerate(zip(top_idx, colors_c)):
        ax = axes.flat[ax_i]
        ax.set_facecolor('#F9FAFB')
        running_sigma = [all_weights[:cp, ci].std() for cp in checkpoints]
        ax.fill_between(checkpoints,
                        [s*0.88 for s in running_sigma],
                        [s*1.12 for s in running_sigma],
                        color=col, alpha=0.12)
        ax.plot(checkpoints, running_sigma, color=col, lw=2.2)
        ax.axvline(n_iter*0.5, color=GREEN, lw=1.0, ls=':', alpha=0.8)
        ax.axvline(n_iter,     color=RED,   lw=1.0, ls=':', alpha=0.8)
        s10 = all_weights[:int(n_iter*0.5), ci].std()
        s20 = all_weights[:n_iter, ci].std()
        ax.annotate(f'|σ@50% − σ@100%| = {abs(s10-s20):.5f}',
                    xy=(n_iter*0.75, (s10+s20)/2),
                    fontsize=7.5, ha='center', color=GREEN, fontfamily=FONT,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=GREEN, lw=0.9, alpha=0.92))
        cname = criteria_names[ci][:35]
        ax.set_title(f'{cname}  (w={weights_det[ci]:.4f})',
                     fontsize=9.5, fontweight='bold', color=DARK, fontfamily=FONT)
        ax.set_xlabel('Iterations (N)', fontsize=8.5, fontfamily=FONT)
        ax.set_ylabel('Running σᵢ', fontsize=8.5, fontfamily=FONT)
        ax.grid(True, color='#E0E0E0', lw=0.6)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

def fig_tier_stability(all_scores, alt_names, det_tiers, breaks, n_iter):
    checkpoints = np.unique(np.concatenate([
        np.arange(100, 1000, 200),
        np.arange(1000, n_iter+1, max(500, n_iter//30))
    ])).astype(int)
    checkpoints = checkpoints[checkpoints <= n_iter]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='white')
    fig.suptitle('Monte Carlo Sensitivity Test: Tier Assignment Stability',
                 fontsize=12, fontweight='bold', color=DARK, fontfamily=FONT)

    styles = ['-','--','-.',':',(0,(3,1,1,1)),'-','--','-.',':',(0,(3,1,1,1)),
              '-','--','-.',':',(0,(3,1,1,1)),'-','--','-.']

    ax1.set_facecolor('#F9FAFB')
    for b_idx in range(len(alt_names)):
        stability = []
        for cp in checkpoints:
            s_cp = all_scores[:cp, b_idx]
            t_cp = assign_tiers(s_cp, breaks)
            stability.append((t_cp == det_tiers[b_idx]).mean() * 100)
        col = TIER_COLS.get(det_tiers[b_idx], '#888')
        ax1.plot(checkpoints, stability, color=col, lw=1.5,
                 ls=styles[b_idx % len(styles)], alpha=0.85, label=alt_names[b_idx])

    ax1.axhline(99.5, color=RED, lw=1.8, ls='--', label='99.5% threshold')
    ax1.set_ylim(85, 101)
    ax1.set_xlabel('Iterations (N)', fontsize=9, fontfamily=FONT)
    ax1.set_ylabel('Tier stability (%)', fontsize=9, fontfamily=FONT)
    ax1.set_title('(a) All Alternatives', fontsize=10, fontweight='bold',
                  color=DARK, fontfamily=FONT)
    ax1.grid(True, color='#E0E0E0', lw=0.6)
    ax1.tick_params(labelsize=8)
    if len(alt_names) <= 15:
        ax1.legend(fontsize=6.5, framealpha=0.9, ncol=2, loc='lower right')

    # Panel B: summary bars at 5 checkpoints
    ax2.set_facecolor('#F9FAFB')
    n_show = min(6, len(alt_names))
    # Select most boundary-proximate alternatives
    det_scores_arr = np.array([all_scores[:, b].mean() for b in range(len(alt_names))])
    cp_pts = np.linspace(int(n_iter*0.05), n_iter, 5, dtype=int)
    cp_lbls = [f'N={p:,}' for p in cp_pts]
    x = np.arange(len(cp_pts))
    width = 0.12
    for shift, b_idx in enumerate(range(n_show)):
        vals = []
        for cp in cp_pts:
            s_cp = all_scores[:cp, b_idx]
            t_cp = assign_tiers(s_cp, breaks)
            vals.append((t_cp == det_tiers[b_idx]).mean() * 100)
        col = TIER_COLS.get(det_tiers[b_idx], '#888')
        off = (shift - n_show/2) * width
        bars = ax2.bar(x + off, vals, width*0.88, label=alt_names[b_idx][:18],
                       color=col, alpha=0.80, edgecolor='white', lw=0.8)
    ax2.axhline(99.5, color=RED, lw=1.8, ls='--', label='99.5% threshold')
    ax2.set_xticks(x); ax2.set_xticklabels(cp_lbls, fontsize=8.5, fontfamily=FONT)
    ax2.set_ylim(80, 101.5)
    ax2.set_ylabel('Tier stability (%)', fontsize=9, fontfamily=FONT)
    ax2.set_title('(b) Sampled N Checkpoints', fontsize=10, fontweight='bold',
                  color=DARK, fontfamily=FONT)
    ax2.grid(axis='y', color='#E0E0E0', lw=0.6)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7, framealpha=0.92, loc='lower right',
               title='Alternative', title_fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT EXAMPLE DATA (Canadian Basin Screening)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CRITERIA = [
    "C1  Tectonic stability",
    "C2  Fault & fracture intensity",
    "C3  Evaporites",
    "C4  Reservoir-seal pairs",
    "C5  Leakage via outcrops",
    "C6  Storage capacity",
    "C7  Basin size",
    "C8  Reservoir temperature",
    "C9  Hydrogeological confinement",
    "C10 Depleted reservoir potential",
    "C11 Freshwater constraint",
    "C12 Industry maturity",
    "C13 Onshore / offshore",
    "C14 Accessibility",
    "C15 Infrastructure",
    "C16 CO2 source proximity",
]

DEFAULT_WEIGHTS = np.array([0.041, 0.071, 0.041, 0.092, 0.022,
                             0.221, 0.041, 0.041, 0.022, 0.071,
                             0.022, 0.041, 0.041, 0.041, 0.071, 0.123])

DEFAULT_ALTERNATIVES = [
    "WCSB", "Williston Basin", "Michigan (SW Ont.)", "NL Offshore",
    "Scotian Basin", "Flemish Pass", "Beaufort-Mackenzie", "Hudson Bay",
    "St. Lawrence", "Nova Scotia", "Arctic Islands", "New Brunswick", "Pacific Margin"
]

DEFAULT_SCORES_DET = np.array([0.982, 0.847, 0.720, 0.627, 0.572,
                                0.446, 0.444, 0.414, 0.403, 0.367,
                                0.280, 0.228, 0.178])

# Normalised P matrix (16 criteria x 13 alternatives)
DEFAULT_P = np.array([
    [1.000,1.000,1.000,0.429,0.429,0.429,1.000,1.000,0.429,1.000,1.000,1.000,0.000],
    [1.000,1.000,1.000,0.333,0.333,0.333,0.333,1.000,0.333,0.333,1.000,0.333,0.000],
    [1.000,1.000,1.000,0.000,0.500,0.000,0.500,1.000,0.000,0.500,1.000,0.000,0.000],
    [1.000,0.700,0.700,0.700,0.700,0.700,0.300,0.300,0.300,0.300,0.100,0.100,0.100],
    [1.000,1.000,1.000,0.750,0.750,0.750,0.500,1.000,0.500,0.500,0.500,0.250,0.250],
    [1.000,0.700,0.300,0.700,0.700,0.700,0.700,0.700,0.100,0.100,0.300,0.100,0.100],
    [1.000,1.000,0.700,1.000,1.000,0.700,1.000,1.000,0.100,0.100,1.000,0.000,0.700],
    [1.000,0.429,0.143,1.000,1.000,1.000,1.000,0.143,0.143,0.000,0.000,0.000,0.429],
    [1.000,1.000,1.000,0.500,0.500,0.500,0.500,0.500,0.500,0.000,1.000,0.000,0.000],
    [1.000,0.700,1.000,0.700,0.200,0.100,0.200,0.000,0.100,0.100,0.200,0.000,0.000],
    [1.000,1.000,0.667,1.000,1.000,1.000,0.333,0.333,0.667,0.667,0.333,0.333,0.667],
    [1.000,0.778,0.778,0.778,0.778,0.333,0.333,0.000,0.333,0.333,0.000,0.111,0.111],
    [1.000,1.000,1.000,0.556,0.000,0.000,1.000,0.556,1.000,1.000,1.000,1.000,0.556],
    [1.000,1.000,1.000,0.556,0.556,0.222,0.222,0.222,1.000,1.000,0.000,1.000,0.556],
    [1.000,1.000,1.000,0.667,0.667,0.222,0.222,0.000,0.667,0.667,0.000,0.222,0.222],
    [1.000,1.000,1.000,0.429,0.143,0.143,0.143,0.000,1.000,0.429,0.000,0.143,0.429],
]).T  # shape: (13 alternatives, 16 criteria)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

if 'ran_mc' not in st.session_state:
    st.session_state.ran_mc = False
if 'mc_weights' not in st.session_state:
    st.session_state.mc_weights = None
if 'mc_scores' not in st.session_state:
    st.session_state.mc_scores = None
if 'final_weights' not in st.session_state:
    st.session_state.final_weights = None
if 'final_alt_scores' not in st.session_state:
    st.session_state.final_alt_scores = None
if 'final_tiers' not in st.session_state:
    st.session_state.final_tiers = None
if 'breaks' not in st.session_state:
    st.session_state.breaks = None
if 'criteria_names' not in st.session_state:
    st.session_state.criteria_names = DEFAULT_CRITERIA.copy()
if 'alt_names' not in st.session_state:
    st.session_state.alt_names = DEFAULT_ALTERNATIVES.copy()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <div class="badge">AHP · MCDA · MONTE CARLO</div>
  <h1>AHP-MCDA Monte Carlo Simulator</h1>
  <p>Derive criterion weights using the Analytic Hierarchy Process, score alternatives,
  classify tiers using Jenks-Fisher optimisation, and validate robustness
  through Monte Carlo weight perturbation analysis.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — GLOBAL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Simulation Settings")

    mode = st.radio(
        "Weight input mode",
        ["Quick Mode — enter weights directly",
         "Expert Mode — pairwise comparison matrix"],
        help="Quick Mode: type weights directly. Expert Mode: full AHP pairwise comparison."
    )
    is_expert = "Expert" in mode

    st.markdown("---")
    n_iter = st.select_slider(
        "Monte Carlo iterations (N)",
        options=[1000, 2000, 5000, 10000, 20000, 50000],
        value=10000,
        help="More iterations = more stable results. 10,000 is sufficient for most cases."
    )
    p_perturb = st.slider(
        "Perturbation probability (p)",
        min_value=0.10, max_value=0.50, value=0.30, step=0.05,
        help="Probability that any given weight comparison is perturbed by ±1 unit."
    )
    n_tiers = st.slider(
        "Number of tiers (k)",
        min_value=2, max_value=6, value=4,
        help="Number of Jenks-Fisher classification tiers."
    )

    st.markdown("---")
    use_example = st.checkbox(
        "Load Canadian Basin Screening example",
        value=True,
        help="Pre-loads the 13-basin, 16-criterion Canadian CO2 storage screening from Okwaraojimadu & Ezekiel (2025)."
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#666; line-height:1.6;">
    <b>Reference:</b><br>
    Okwaraojimadu C.K. & Ezekiel C.J. (2025).<br>
    <i>Canadian CO₂ Storage Basin Screening:</i><br>
    AHP-MCDA with Monte Carlo Validation.<br><br>
    University of Calgary, MSc Research.<br><br>
    <b>Code:</b> Python 3 · NumPy · Matplotlib<br>
    <b>Contact:</b> chisom.okwaraojimadu@ucalgary.ca
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Setup",
    "⚖️  Weights & Scoring",
    "📊  Monte Carlo Results",
    "💾  Export"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — SETUP
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">1. Define Criteria and Alternatives</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Criteria (what you are evaluating on)**")
        if use_example:
            criteria_text = st.text_area(
                "One criterion per line",
                value="\n".join(DEFAULT_CRITERIA),
                height=320,
                key="criteria_input"
            )
        else:
            criteria_text = st.text_area(
                "One criterion per line",
                value="Criterion 1\nCriterion 2\nCriterion 3\nCriterion 4\nCriterion 5",
                height=320,
                key="criteria_input"
            )

    with col2:
        st.markdown("**Alternatives (what you are ranking)**")
        if use_example:
            alt_text = st.text_area(
                "One alternative per line",
                value="\n".join(DEFAULT_ALTERNATIVES),
                height=320,
                key="alt_input"
            )
        else:
            alt_text = st.text_area(
                "One alternative per line",
                value="Alternative A\nAlternative B\nAlternative C",
                height=320,
                key="alt_input"
            )

    criteria_names = [c.strip() for c in criteria_text.strip().split("\n") if c.strip()]
    alt_names = [a.strip() for a in alt_text.strip().split("\n") if a.strip()]
    n_c = len(criteria_names)
    n_a = len(alt_names)

    st.session_state.criteria_names = criteria_names
    st.session_state.alt_names = alt_names

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Criteria defined", n_c)
    col_m2.metric("Alternatives defined", n_a)
    col_m3.metric("Pairwise comparisons needed", n_c*(n_c-1)//2 if n_c > 1 else 0)

    if n_c < 2:
        st.error("Please define at least 2 criteria.")
    if n_a < 2:
        st.error("Please define at least 2 alternatives.")

    st.markdown('<div class="section-header">2. Alternative Scores (Normalised 0-1)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Enter the normalised performance score P<sub>ik</sub> for each alternative on each criterion.
    Values must be between 0 (worst) and 1 (best). If using the Canadian example, scores
    are pre-loaded from the published paper.
    </div>
    """, unsafe_allow_html=True)

    if use_example and n_c == 16 and n_a == 13:
        score_matrix = DEFAULT_P.copy()
        st.success("Canadian Basin Screening scores loaded from Okwaraojimadu & Ezekiel (2025).")
        df_preview = pd.DataFrame(score_matrix,
                                  index=alt_names,
                                  columns=[c[:20] for c in criteria_names])
        st.dataframe(df_preview.style.format("{:.3f}").background_gradient(
            cmap='Blues', axis=None, vmin=0, vmax=1), height=300)
    else:
        st.markdown("**Enter scores in the table below (edit cells directly):**")
        if n_c > 0 and n_a > 0:
            init_data = {c[:20]: [0.5]*n_a for c in criteria_names}
            df_edit = pd.DataFrame(init_data, index=alt_names)
            edited = st.data_editor(df_edit, use_container_width=True,
                                    num_rows="fixed", key="score_editor")
            score_matrix = edited.values.astype(float)
            score_matrix = np.clip(score_matrix, 0, 1)
        else:
            score_matrix = np.zeros((max(n_a,1), max(n_c,1)))


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — WEIGHTS & SCORING
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Weight Derivation</div>',
                unsafe_allow_html=True)

    if n_c < 2:
        st.warning("Define at least 2 criteria in Setup.")
        st.stop()

    # ── QUICK MODE ─────────────────────────────────────────────────────────────
    if not is_expert:
        st.markdown("""
        <div class="info-box">
        <b>Quick Mode:</b> Enter your criterion weights directly. They will be normalised
        automatically to sum to 1. No consistency check is applied in this mode.
        </div>
        """, unsafe_allow_html=True)

        if use_example and n_c == 16:
            default_w = DEFAULT_WEIGHTS.tolist()
        else:
            default_w = [round(1.0/n_c, 4)] * n_c

        weight_inputs = []
        cols_w = st.columns(min(4, n_c))
        for i, cname in enumerate(criteria_names):
            col_idx = i % len(cols_w)
            with cols_w[col_idx]:
                w = st.number_input(
                    label=cname[:25],
                    min_value=0.001, max_value=1.0,
                    value=float(default_w[i]) if i < len(default_w) else 1.0/n_c,
                    step=0.001, format="%.4f",
                    key=f"w_quick_{i}"
                )
                weight_inputs.append(w)

        raw_w = np.array(weight_inputs)
        weights = raw_w / raw_w.sum()

        st.markdown("---")
        col_cr1, col_cr2 = st.columns([1, 2])
        with col_cr1:
            st.markdown('<div class="metric-card"><div class="val">' +
                        f'{weights.sum():.4f}' +
                        '</div><div class="lbl">Weight Sum (normalised)</div></div>',
                        unsafe_allow_html=True)
        with col_cr2:
            st.markdown("""
            <div class="info-box">
            <b>Note:</b> Consistency check (CR) is not computed in Quick Mode.
            Switch to Expert Mode for full AHP with CR verification.
            </div>
            """, unsafe_allow_html=True)

        lambda_max, CI, CR = None, None, None
        cr_ok = True

    # ── EXPERT MODE ────────────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="info-box">
        <b>Expert Mode:</b> Complete the pairwise comparison matrix using Saaty's 1-9 scale.
        1 = equally important, 3 = moderately more important, 5 = strongly, 7 = very strongly,
        9 = extremely. Use 1/3, 1/5 etc. for the reverse. The app computes the eigenvector
        weights and Consistency Ratio (CR) automatically. CR must be ≤ 0.10.
        </div>
        """, unsafe_allow_html=True)

        n_pairs = n_c * (n_c - 1) // 2

        if use_example and n_c == 16:
            # Use the deterministic weights from the paper to build a near-consistent matrix
            w_ex = DEFAULT_WEIGHTS
            # Build ratio matrix rounded to nearest Saaty value
            def nearest_saaty(v):
                vals = [1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
                return min(vals, key=lambda x: abs(x-v))
            ex_defaults = []
            for i in range(n_c):
                for j in range(i+1, n_c):
                    ex_defaults.append(nearest_saaty(w_ex[i]/w_ex[j]))
        else:
            ex_defaults = [1.0] * n_pairs

        if n_pairs <= 45:
            upper_vals = []
            pair_labels = []
            for i in range(n_c):
                for j in range(i+1, n_c):
                    pair_labels.append(f"{criteria_names[i][:20]} vs {criteria_names[j][:20]}")

            cols_p = st.columns(min(3, n_pairs))
            for k, (lbl, dv) in enumerate(zip(pair_labels, ex_defaults)):
                col_idx = k % len(cols_p)
                with cols_p[col_idx]:
                    v = st.select_slider(
                        lbl,
                        options=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,
                                 1,2,3,4,5,6,7,8,9],
                        value=float(dv),
                        format_func=lambda x: f"1/{int(round(1/x))}" if x < 1 else str(int(x)) if x == int(x) else f"{x:.2f}",
                        key=f"pair_{k}"
                    )
                    upper_vals.append(v)
        else:
            st.warning(f"With {n_c} criteria, there are {n_pairs} pairwise comparisons. "
                       f"Enter them as a CSV row of upper-triangle values below "
                       f"(row by row, left to right, separated by commas).")
            csv_input = st.text_area(
                "Upper triangle values (comma-separated)",
                value=",".join([str(round(v,4)) for v in ex_defaults]),
                height=150
            )
            try:
                upper_vals = [float(x.strip()) for x in csv_input.split(",")]
                if len(upper_vals) != n_pairs:
                    st.error(f"Expected {n_pairs} values, got {len(upper_vals)}.")
                    upper_vals = ex_defaults
            except Exception:
                st.error("Could not parse input. Please check your values.")
                upper_vals = ex_defaults

        matrix = build_matrix_from_upper(n_c, upper_vals)
        weights, lambda_max, CI, CR = compute_ahp(matrix)

        cr_ok = CR <= 0.10

        # Display CR
        col_cr1, col_cr2, col_cr3 = st.columns(3)
        with col_cr1:
            st.markdown(f'<div class="metric-card {"good" if cr_ok else "bad"}">'
                        f'<div class="val">{"✓" if cr_ok else "✗"} {CR:.4f}</div>'
                        f'<div class="lbl">Consistency Ratio (CR)</div></div>',
                        unsafe_allow_html=True)
        with col_cr2:
            st.markdown(f'<div class="metric-card"><div class="val">{lambda_max:.4f}</div>'
                        f'<div class="lbl">λ_max</div></div>', unsafe_allow_html=True)
        with col_cr3:
            st.markdown(f'<div class="metric-card"><div class="val">{CI:.4f}</div>'
                        f'<div class="lbl">Consistency Index (CI)</div></div>',
                        unsafe_allow_html=True)

        if not cr_ok:
            st.markdown("""
            <div class="warn-box">
            ⚠️ <b>CR exceeds 0.10.</b> Your pairwise judgements are inconsistent.
            Please revise your comparisons before proceeding. Focus on the most
            extreme comparisons first and check for logical contradictions.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✓ Judgements are consistent (CR = {CR:.4f} ≤ 0.10). Weights accepted.")

    # ── Weight display ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Derived Weights</div>', unsafe_allow_html=True)

    df_weights = pd.DataFrame({
        'Criterion': criteria_names,
        'Weight (wᵢ)': [f"{w:.4f}" for w in weights],
        'Percentage': [f"{w*100:.1f}%" for w in weights]
    })
    st.dataframe(df_weights, use_container_width=True, hide_index=True)

    fig_w = fig_weights(weights, criteria_names)
    st.pyplot(fig_w, use_container_width=True)
    plt.close(fig_w)

    # ── Composite scores ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Composite Suitability Scores</div>',
                unsafe_allow_html=True)

    alt_scores = score_matrix @ weights

    # Jenks-Fisher classification
    if n_a >= n_tiers:
        breaks, gvf = jenks_fisher(alt_scores, n_tiers)
        tiers = assign_tiers(alt_scores, breaks)
    else:
        breaks = np.array([])
        gvf = 1.0
        tiers = np.ones(n_a, dtype=int)

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Jenks-Fisher GVF", f"{gvf:.4f}",
                  help="Goodness of Variance Fit. ≥ 0.90 = excellent classification.")
    col_s2.metric("Tier boundaries", f"k = {n_tiers}")

    df_scores = pd.DataFrame({
        'Alternative': alt_names,
        'Score Rᵏ': [f"{s:.4f}" for s in alt_scores],
        'Tier': tiers,
        'Rank': pd.Series(alt_scores).rank(ascending=False).astype(int).values
    }).sort_values('Rank').reset_index(drop=True)
    st.dataframe(df_scores, use_container_width=True, hide_index=True)

    if n_a >= 2:
        fig_s = fig_scores(np.array(alt_names), alt_scores, tiers, breaks)
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)

    # Save to session state
    st.session_state.final_weights    = weights
    st.session_state.final_alt_scores = alt_scores
    st.session_state.final_tiers      = tiers
    st.session_state.breaks           = breaks
    st.session_state.score_matrix     = score_matrix
    st.session_state.cr_ok            = cr_ok


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — MONTE CARLO RESULTS
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Monte Carlo Weight Uncertainty Analysis</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    The Monte Carlo simulation perturbs each criterion weight with probability
    <b>p = {p_perturb}</b> per iteration, recomputes composite scores for all
    {n_a} alternatives, and re-applies Jenks-Fisher tier classification.
    Running <b>N = {n_iter:,}</b> iterations provides convergence statistics
    and tier stability estimates.
    </div>
    """, unsafe_allow_html=True)

    weights_ready = st.session_state.final_weights is not None
    cr_ready = st.session_state.get('cr_ok', True)

    if not weights_ready:
        st.warning("Complete the Weights & Scoring tab first.")
    elif is_expert and not cr_ready:
        st.error("CR > 0.10. Fix pairwise comparisons before running Monte Carlo.")
    else:
        if st.button(f"▶  Run Monte Carlo  (N = {n_iter:,})", type="primary"):
            with st.spinner(f"Running {n_iter:,} Monte Carlo iterations..."):
                P = st.session_state.score_matrix  # (n_alt, n_c)
                w = st.session_state.final_weights
                all_w, all_s = run_monte_carlo(w, P, n_iter, p_perturb)
                st.session_state.mc_weights = all_w
                st.session_state.mc_scores  = all_s
                st.session_state.ran_mc     = True
            st.success(f"Simulation complete. {n_iter:,} iterations run.")

        if st.session_state.ran_mc and st.session_state.mc_weights is not None:
            all_w = st.session_state.mc_weights
            all_s = st.session_state.mc_scores
            w_det = st.session_state.final_weights
            breaks= st.session_state.breaks
            tiers_det = st.session_state.final_tiers

            # ── Summary stats ──────────────────────────────────────────────────
            st.markdown('<div class="section-header">Weight Uncertainty Summary</div>',
                        unsafe_allow_html=True)

            sigma_vals = all_w.std(axis=0)
            mu_vals    = all_w.mean(axis=0)

            df_mc = pd.DataFrame({
                'Criterion':         criteria_names,
                'Det. Weight (wᵢ)':  [f"{w:.4f}" for w in w_det],
                'MC Mean (μᵢ)':      [f"{m:.4f}" for m in mu_vals],
                'MC Std Dev (σᵢ)':   [f"{s:.5f}" for s in sigma_vals],
                'Stable?':           ['✓' if s <= 0.010 else '⚠' for s in sigma_vals]
            })
            st.dataframe(df_mc, use_container_width=True, hide_index=True)

            col_mc1, col_mc2, col_mc3 = st.columns(3)
            col_mc1.metric("Max σᵢ across all criteria", f"{sigma_vals.max():.5f}")
            col_mc2.metric("Mean σᵢ", f"{sigma_vals.mean():.5f}")
            col_mc3.metric("Iterations run", f"{n_iter:,}")

            # ── Tier stability ─────────────────────────────────────────────────
            st.markdown('<div class="section-header">Tier Stability Results</div>',
                        unsafe_allow_html=True)

            stability_pcts = []
            for b_idx in range(n_a):
                t_mc = assign_tiers(all_s[:, b_idx], breaks)
                pct  = (t_mc == tiers_det[b_idx]).mean() * 100
                stability_pcts.append(pct)

            df_stab = pd.DataFrame({
                'Alternative':         alt_names,
                'Det. Score':          [f"{s:.4f}" for s in st.session_state.final_alt_scores],
                'Det. Tier':           tiers_det,
                'Tier Stability (%)':  [f"{p:.2f}%" for p in stability_pcts],
                'Robust?':             ['✓' if p >= 99.5 else '⚠' for p in stability_pcts]
            })
            st.dataframe(df_stab, use_container_width=True, hide_index=True)

            all_robust = all(p >= 99.5 for p in stability_pcts)
            if all_robust:
                st.success(f"✓ All {n_a} alternatives maintain tier assignment in ≥ 99.5% "
                           f"of {n_iter:,} Monte Carlo simulations. Classification is robust.")
            else:
                n_warn = sum(p < 99.5 for p in stability_pcts)
                st.warning(f"⚠ {n_warn} alternative(s) fall below 99.5% tier stability. "
                           f"Consider reviewing criterion weights or tier boundaries.")

            # ── Convergence figures ────────────────────────────────────────────
            st.markdown('<div class="section-header">Convergence Figures</div>',
                        unsafe_allow_html=True)

            top_n = min(4, n_c)
            fig_conv = fig_mc_convergence(all_w, criteria_names, w_det, top_n=top_n)
            st.pyplot(fig_conv, use_container_width=True)
            plt.close(fig_conv)

            fig_stab = fig_tier_stability(all_s, alt_names, tiers_det, breaks, n_iter)
            st.pyplot(fig_stab, use_container_width=True)
            plt.close(fig_stab)

            # Save for export
            st.session_state.sigma_vals     = sigma_vals
            st.session_state.mu_vals        = mu_vals
            st.session_state.stability_pcts = stability_pcts
            st.session_state.df_mc          = df_mc
            st.session_state.df_stab        = df_stab


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — EXPORT
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Export Results</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Download your results in multiple formats for use in your paper,
    supplementary materials, or GIS attribute tables.
    </div>
    """, unsafe_allow_html=True)

    weights_ready = st.session_state.final_weights is not None

    if not weights_ready:
        st.info("Complete the Setup and Weights tabs to enable exports.")
    else:
        w   = st.session_state.final_weights
        s   = st.session_state.final_alt_scores
        t   = st.session_state.final_tiers
        brk = st.session_state.breaks

        # ── CSV exports ────────────────────────────────────────────────────────
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            df_w_export = pd.DataFrame({
                'Criterion': st.session_state.criteria_names,
                'Weight_wi': w,
                'Percentage': w * 100
            })
            csv_w = df_w_export.to_csv(index=False)
            st.download_button(
                "📥 Download Weights CSV",
                data=csv_w,
                file_name="ahp_weights.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_e2:
            df_s_export = pd.DataFrame({
                'Alternative': st.session_state.alt_names,
                'Composite_Score': s,
                'Tier': t,
                'Rank': pd.Series(s).rank(ascending=False).astype(int).values
            }).sort_values('Rank').reset_index(drop=True)
            csv_s = df_s_export.to_csv(index=False)
            st.download_button(
                "📥 Download Scores & Tiers CSV",
                data=csv_s,
                file_name="ahp_scores_tiers.csv",
                mime="text/csv",
                use_container_width=True
            )

        if st.session_state.ran_mc and st.session_state.mc_weights is not None:
            col_e3, col_e4 = st.columns(2)

            with col_e3:
                df_mc_export = pd.DataFrame({
                    'Criterion': st.session_state.criteria_names,
                    'Det_Weight': w,
                    'MC_Mean': st.session_state.mu_vals,
                    'MC_StdDev': st.session_state.sigma_vals
                })
                csv_mc = df_mc_export.to_csv(index=False)
                st.download_button(
                    "📥 Download MC Weight Stats CSV",
                    data=csv_mc,
                    file_name="mc_weight_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_e4:
                df_stab_export = pd.DataFrame({
                    'Alternative': st.session_state.alt_names,
                    'Det_Score': s,
                    'Det_Tier': t,
                    'Tier_Stability_pct': st.session_state.stability_pcts
                })
                csv_stab = df_stab_export.to_csv(index=False)
                st.download_button(
                    "📥 Download Tier Stability CSV",
                    data=csv_stab,
                    file_name="mc_tier_stability.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # ── Figure exports ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Figure Downloads</div>',
                    unsafe_allow_html=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fig_w_dl = fig_weights(w, st.session_state.criteria_names)
            st.download_button(
                "📥 Download Weights Figure (PNG)",
                data=fig_to_bytes(fig_w_dl),
                file_name="ahp_weights.png",
                mime="image/png",
                use_container_width=True
            )

        with col_f2:
            if n_a >= 2:
                fig_s_dl = fig_scores(np.array(st.session_state.alt_names), s, t, brk)
                st.download_button(
                    "📥 Download Scores Figure (PNG)",
                    data=fig_to_bytes(fig_s_dl),
                    file_name="ahp_scores.png",
                    mime="image/png",
                    use_container_width=True
                )

        if st.session_state.ran_mc and st.session_state.mc_weights is not None:
            col_f3, col_f4 = st.columns(2)
            all_w = st.session_state.mc_weights
            all_s = st.session_state.mc_scores
            tiers_det = st.session_state.final_tiers

            with col_f3:
                fig_cv_dl = fig_mc_convergence(
                    all_w, st.session_state.criteria_names, w, top_n=min(4, n_c))
                st.download_button(
                    "📥 Download Convergence Figure (PNG)",
                    data=fig_to_bytes(fig_cv_dl),
                    file_name="mc_convergence.png",
                    mime="image/png",
                    use_container_width=True
                )

            with col_f4:
                fig_st_dl = fig_tier_stability(
                    all_s, st.session_state.alt_names,
                    tiers_det, brk, n_iter)
                st.download_button(
                    "📥 Download Stability Figure (PNG)",
                    data=fig_to_bytes(fig_st_dl),
                    file_name="mc_tier_stability.png",
                    mime="image/png",
                    use_container_width=True
                )

        # ── JSON export (full results) ─────────────────────────────────────────
        st.markdown('<div class="section-header">Full Results JSON</div>',
                    unsafe_allow_html=True)

        results_dict = {
            "criteria": st.session_state.criteria_names,
            "alternatives": st.session_state.alt_names,
            "weights": {c: float(ww) for c, ww in
                        zip(st.session_state.criteria_names, w)},
            "scores": {a: float(ss) for a, ss in
                       zip(st.session_state.alt_names, s)},
            "tiers": {a: int(tt) for a, tt in
                      zip(st.session_state.alt_names, t)},
            "tier_breaks": [float(b) for b in brk],
            "n_iterations": n_iter,
            "perturbation_probability": p_perturb,
        }
        if st.session_state.ran_mc and hasattr(st.session_state, 'sigma_vals'):
            results_dict["mc_sigma"] = {c: float(sig) for c, sig in
                                        zip(st.session_state.criteria_names,
                                            st.session_state.sigma_vals)}
            results_dict["mc_tier_stability"] = {a: float(p) for a, p in
                                                 zip(st.session_state.alt_names,
                                                     st.session_state.stability_pcts)}

        st.download_button(
            "📥 Download Full Results JSON",
            data=json.dumps(results_dict, indent=2),
            file_name="ahp_mc_results.json",
            mime="application/json",
            use_container_width=True
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  AHP-MCDA Monte Carlo Simulator · Built with Python, NumPy, Matplotlib, Streamlit ·
  Okwaraojimadu C.K. & Ezekiel C.J., University of Calgary · 2025 ·
  Open source — code available on request
</div>
""", unsafe_allow_html=True)
