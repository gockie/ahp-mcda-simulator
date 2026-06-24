import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import json

st.set_page_config(
    page_title="AHP-MCDA Monte Carlo Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --navy:#1B3A5C; --teal:#0E7C7B; --blue:#2E75B6;
    --amber:#C47A00; --green:#2E8B57; --red:#B22222;
    --border:#D0DAE8; --text:#1A1A2E; --bg:#F8FAFC;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--text);}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
[data-testid="collapsedControl"]{display:none!important;}

.hero{
    background:linear-gradient(135deg,#1B3A5C 0%,#0E4D7B 55%,#0E7C7B 100%);
    padding:2.4rem 2.2rem 2rem;border-radius:18px;margin-bottom:1.6rem;
    color:white;position:relative;overflow:hidden;
}
.hero::before{
    content:'';position:absolute;top:-60%;right:-8%;
    width:420px;height:420px;
    background:radial-gradient(circle,rgba(14,124,123,.28) 0%,transparent 70%);
    pointer-events:none;
}
.hero .pill{
    display:inline-block;background:rgba(255,255,255,.15);
    border:1px solid rgba(255,255,255,.3);border-radius:20px;
    padding:.22rem .85rem;font-size:.72rem;letter-spacing:1.2px;
    margin-bottom:.9rem;color:rgba(255,255,255,.92);font-weight:500;
}
.hero h1{font-size:2.05rem;font-weight:700;margin:0 0 .5rem;color:white;line-height:1.2;}
.hero .sub{font-size:.97rem;opacity:.83;margin:0;font-weight:300;line-height:1.65;}

.section-hd{
    font-size:1.15rem;font-weight:700;color:var(--navy);
    border-bottom:2.5px solid var(--border);padding-bottom:.38rem;
    margin:1.5rem 0 .9rem;
}
.callout{
    border-left:4px solid var(--blue);background:#EFF6FF;
    border-radius:0 10px 10px 0;padding:.8rem 1.05rem;
    font-size:.875rem;line-height:1.68;margin:.7rem 0;color:#1e3a5f;
}
.callout.warn{border-color:var(--amber);background:#FFFBF0;color:#5a3a00;}
.callout.ok{border-color:var(--green);background:#F0FFF4;color:#1a4a2a;}
.callout.tip{border-color:var(--teal);background:#F0FFFE;color:#0a3a3a;}

.kpi{
    background:white;border:1.5px solid var(--border);border-radius:12px;
    padding:.9rem 1rem;text-align:center;
}
.kpi .val{font-family:'JetBrains Mono',monospace;font-size:1.55rem;
    font-weight:600;color:var(--navy);line-height:1;}
.kpi .lbl{font-size:.72rem;color:#888;margin-top:.28rem;
    font-weight:600;letter-spacing:.6px;text-transform:uppercase;}
.kpi.good .val{color:var(--green);}
.kpi.bad  .val{color:var(--red);}

.step-row{
    display:flex;gap:1rem;align-items:flex-start;
    background:white;border:1.5px solid var(--border);
    border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:.85rem;
}
.step-num{
    background:linear-gradient(135deg,#1B3A5C,#0E7C7B);
    color:white;font-weight:700;font-size:1rem;
    min-width:40px;height:40px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.step-body h4{font-weight:700;color:var(--navy);margin:0 0 .3rem;font-size:.97rem;}
.step-body p{margin:0;font-size:.85rem;color:#444;line-height:1.65;}

.gloss-card{background:white;border:1px solid var(--border);
    border-radius:10px;padding:.85rem 1rem;margin-bottom:.6rem;}
.gloss-term{font-family:'JetBrains Mono',monospace;font-size:.87rem;
    font-weight:600;color:var(--navy);margin-bottom:.22rem;}
.gloss-def{font-size:.83rem;color:#444;line-height:1.65;margin:0;}
.gloss-ex{font-size:.78rem;color:var(--teal);font-style:italic;margin-top:.22rem;}

.footer{text-align:center;padding:1.8rem 0 .8rem;font-size:.75rem;color:#aaa;
    border-top:1px solid var(--border);margin-top:2.5rem;
    font-family:'JetBrains Mono',monospace;}

.tier-badge{display:inline-block;border-radius:6px;padding:.18rem .55rem;
    font-size:.75rem;font-weight:700;color:white;}
.t1{background:#2E8B57;}.t2{background:#C47A00;}
.t3{background:#8B4500;}.t4{background:#555;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CORE MATH
# ══════════════════════════════════════════════════════════════════════════════

SAATY_RI = {1:0,2:0,3:.58,4:.90,5:1.12,6:1.24,7:1.32,8:1.41,
            9:1.45,10:1.49,11:1.51,12:1.54,13:1.56,14:1.57,
            15:1.58,16:1.59,17:1.60,18:1.61,19:1.62,20:1.63}

def compute_ahp(M):
    n = M.shape[0]
    cs = M.sum(axis=0)
    w  = (M / cs).mean(axis=1)
    lam = (M @ w) / w
    lmax = lam.mean()
    CI = (lmax - n) / (n - 1) if n > 1 else 0
    RI = SAATY_RI.get(n, 1.63)
    CR = CI / RI if RI > 0 else 0
    return w, lmax, CI, CR

def build_matrix(n, upper):
    M = np.eye(n); idx = 0
    for i in range(n):
        for j in range(i+1, n):
            M[i,j] = upper[idx]; M[j,i] = 1.0 / upper[idx]; idx += 1
    return M

def find_inconsistent_pairs(M, names, top_n=3):
    """Return the top_n most inconsistent pairwise comparisons."""
    n = M.shape[0]
    w, *_ = compute_ahp(M)
    issues = []
    for i in range(n):
        for j in range(i+1, n):
            ideal = w[i] / w[j] if w[j] > 0 else 1.0
            actual = M[i, j]
            ratio = max(actual/ideal, ideal/actual)
            issues.append((ratio, names[i], names[j], actual, ideal))
    issues.sort(reverse=True)
    return issues[:top_n]

def jenks(values, k):
    """Jenks-Fisher natural breaks."""
    n = len(values)
    sv = np.sort(values)
    if k <= 1:
        return np.array([]), float(np.var(sv) == 0)
    if k >= n:
        return sv[1:], 1.0
    SSW = np.full((k, n), np.inf)
    BI  = np.zeros((k, n), dtype=int)
    for i in range(n):
        v = sv[:i+1]
        SSW[0, i] = np.var(v) * len(v)
    for cl in range(1, k):
        for i in range(cl, n):
            best, bb = np.inf, cl
            for m in range(cl-1, i):
                vr  = sv[m+1:i+1]
                tot = SSW[cl-1, m] + (np.var(vr) * len(vr) if len(vr) > 0 else 0)
                if tot < best:
                    best = tot; bb = m + 1
            SSW[cl, i] = best; BI[cl, i] = bb
    idx = n - 1; bks = []
    for cl in range(k-1, 0, -1):
        bi = BI[cl, idx]; bks.append(bi); idx = bi - 1
    bv   = sv[sorted(bks)]
    sdam = np.var(sv) * n
    gvf  = 1 - SSW[k-1, n-1] / sdam if sdam > 0 else 1.0
    return bv, gvf

def assign_tiers(scores, breaks):
    sb = np.sort(breaks)[::-1]
    out = []
    for s in scores:
        t = len(sb) + 1
        for i, b in enumerate(sb):
            if s >= b:
                t = i + 1; break
        out.append(t)
    return np.array(out)

def run_mc_matrix(weights_det, P, pairwise_M, n_iter, p_perturb):
    """
    Scientifically correct MC: perturb the pairwise matrix elements by
    ±1 Saaty unit with probability p, re-derive weights via eigenvector,
    recompute scores.  Falls back to weight-space perturbation when no
    pairwise matrix is available (Quick Mode).
    """
    np.random.seed(42)
    n_c = len(weights_det)
    n_a = P.shape[0]
    aw  = np.zeros((n_iter, n_c))
    as_ = np.zeros((n_iter, n_a))

    if pairwise_M is not None:
        # --- Expert mode: perturb matrix elements ---
        n = pairwise_M.shape[0]
        saaty_scale = np.array([1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,
                                  1,2,3,4,5,6,7,8,9])
        for s in range(n_iter):
            Mp = pairwise_M.copy()
            for i in range(n):
                for j in range(i+1, n):
                    if np.random.random() < p_perturb:
                        cur = Mp[i, j]
                        cur_idx = np.argmin(np.abs(saaty_scale - cur))
                        delta = np.random.choice([-1, 0, 1])
                        new_idx = np.clip(cur_idx + delta, 0, len(saaty_scale)-1)
                        new_val = saaty_scale[new_idx]
                        Mp[i, j] = new_val
                        Mp[j, i] = 1.0 / new_val
            wp, *_ = compute_ahp(Mp)
            wp = np.maximum(wp, 1e-6); wp /= wp.sum()
            aw[s]  = wp
            as_[s] = P @ wp
    else:
        # --- Quick mode: Gaussian weight perturbation ---
        sig = np.maximum(weights_det * 0.03, 0.002)
        for s in range(n_iter):
            noise = np.random.normal(0, sig)
            mask  = np.random.random(n_c) < p_perturb
            wp    = np.maximum(weights_det + noise * mask, 0.005)
            wp   /= wp.sum()
            aw[s]  = wp
            as_[s] = P @ wp

    return aw, as_


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

FONT='DejaVu Sans'
DARK='#1B3A5C'; MED='#2E75B6'; TEAL='#0E7C7B'
AMBER='#C47A00'; GREEN='#2E8B57'; RED='#B22222'; PURPLE='#6A4C93'
TC={1:'#2E8B57',2:'#C47A00',3:'#8B4500',4:'#555555'}

def fig2bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0); plt.close(fig)
    return buf.getvalue()

def plot_pareto(w, names):
    """Pareto chart of criterion weights showing 80/20 rule."""
    fig, ax1 = plt.subplots(figsize=(10, max(4, len(names)*.45)), facecolor='white')
    ax1.set_facecolor('#F9FAFB')
    idx = np.argsort(w)[::-1]
    ws  = w[idx]; ns = [names[i] for i in idx]
    cumw = np.cumsum(ws / ws.sum() * 100)
    y = np.arange(len(ws))
    cols = [plt.cm.Blues(.4 + .6 * ww / max(w)) for ww in ws]
    bars = ax1.barh(y, ws, color=cols, edgecolor='white', lw=.8, height=.7)
    for bar, ww in zip(bars, ws):
        ax1.text(bar.get_width() + .002, bar.get_y() + bar.get_height()/2,
                 f'{ww:.4f}', va='center', ha='left', fontsize=8.5,
                 fontfamily=FONT, color=DARK, fontweight='bold')
    ax1.set_yticks(y); ax1.set_yticklabels(ns, fontsize=8.5, fontfamily=FONT)
    ax1.set_xlabel('AHP Weight  wᵢ', fontsize=10, fontfamily=FONT)
    ax1.set_title('Criterion Weights — Pareto Chart (highest to lowest)',
                  fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT, pad=8)
    ax2 = ax1.twiny()
    ax2.plot(cumw, y, color=RED, lw=2, marker='o', ms=4, label='Cumulative %')
    ax2.axvline(80, color=AMBER, lw=1.5, ls='--', alpha=.8)
    ax2.text(81, len(ws)*.05, '80%', color=AMBER, fontsize=8.5, fontfamily=FONT)
    ax2.set_xlabel('Cumulative Weight (%)', fontsize=9, color=RED, fontfamily=FONT)
    ax2.tick_params(axis='x', colors=RED, labelsize=8)
    ax1.set_xlim(0, max(w)*1.28)
    ax1.grid(axis='x', color='#E0E0E0', lw=.6)
    ax1.invert_yaxis()
    plt.tight_layout(); return fig

def plot_scores(names, scores, tiers, breaks):
    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*.55)), facecolor='white')
    ax.set_facecolor('#F9FAFB')
    idx = np.argsort(scores)[::-1]
    ss = scores[idx]; ns = [names[i] for i in idx]; ts = tiers[idx]
    y  = np.arange(len(names))
    cols = [TC.get(t, '#888') for t in ts]
    ax.barh(y, ss, color=cols, edgecolor='white', lw=.8, height=.68, alpha=.85)
    for i, (s, t) in enumerate(zip(ss, ts)):
        ax.text(s + .008, i, f'{s:.3f}', va='center', ha='left', fontsize=9.5,
                fontfamily=FONT, color=TC.get(t, '#888'), fontweight='bold')
    for b in breaks:
        ax.axvline(b, color='#333', lw=1.2, ls='--', alpha=.7)
    ax.set_yticks(y); ax.set_yticklabels(ns, fontsize=9.5, fontfamily=FONT)
    ax.set_xlabel('Composite Score  Rᵏ ∈ [0, 1]', fontsize=10, fontfamily=FONT)
    ax.set_title('Ranked Composite Scores with Tier Classification',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT, pad=8)
    ax.set_xlim(0, 1.14); ax.invert_yaxis()
    ax.grid(axis='x', color='#E0E0E0', lw=.6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [Patch(facecolor=TC[t], alpha=.85, label=f'Tier {t} — {"Priority" if t==1 else "Secondary" if t==2 else "Low priority" if t==3 else "Marginal"}')
           for t in sorted(TC) if t <= max(tiers)]
    leg.append(Line2D([0],[0], color='#333', lw=1.2, ls='--', label='Tier boundary'))
    ax.legend(handles=leg, fontsize=8, loc='lower right', framealpha=.93)
    plt.tight_layout(); return fig

def plot_conv(aw, names, w_det, top_n=4):
    n_iter = len(aw)
    cps = np.unique(np.concatenate([
        np.arange(100, min(1000, n_iter), 200),
        np.arange(1000, n_iter+1, max(500, n_iter//30))
    ])).astype(int)
    cps = cps[cps <= n_iter]
    idx  = np.argsort(w_det)[::-1][:top_n]
    cols = [MED, TEAL, AMBER, PURPLE]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor='white')
    fig.suptitle('Monte Carlo Convergence: Do the weight uncertainties stabilise?',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT)
    for ax_i, (ci, col) in enumerate(zip(idx, cols)):
        ax = axes.flat[ax_i]; ax.set_facecolor('#F9FAFB')
        rs = [aw[:cp, ci].std() for cp in cps]
        ax.fill_between(cps, [s*.88 for s in rs], [s*1.12 for s in rs],
                        color=col, alpha=.12)
        ax.plot(cps, rs, color=col, lw=2.2)
        ax.axvline(n_iter*.5, color=GREEN, lw=1., ls=':', alpha=.8)
        ax.axvline(n_iter,    color=RED,   lw=1., ls=':', alpha=.8)
        s50  = aw[:int(n_iter*.5), ci].std()
        s100 = aw[:n_iter, ci].std()
        ax.annotate(f'Change 50%→100%: {abs(s50-s100):.5f}',
                    xy=(n_iter*.72, (s50+s100)/2), fontsize=7.5, ha='center',
                    color=GREEN, fontfamily=FONT,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=GREEN, lw=.9, alpha=.92))
        ax.set_title(f'{names[ci][:30]}  (w={w_det[ci]:.4f})',
                     fontsize=9, fontweight='bold', color=DARK, fontfamily=FONT)
        ax.set_xlabel('Number of simulations run', fontsize=8.5, fontfamily=FONT)
        ax.set_ylabel('Weight variability (σᵢ)', fontsize=8.5, fontfamily=FONT)
        ax.grid(True, color='#E0E0E0', lw=.6); ax.tick_params(labelsize=8)
    plt.tight_layout(rect=[0,0,1,.93]); return fig

def plot_stability(as_, names, det_tiers, breaks, n_iter):
    cps = np.unique(np.concatenate([
        np.arange(100, min(1000, n_iter), 200),
        np.arange(1000, n_iter+1, max(500, n_iter//30))
    ])).astype(int)
    cps = cps[cps <= n_iter]
    styles = ['-','--','-.',':', (0,(3,1,1,1)),'-','--','-.',':',(0,(3,1,1,1)),
              '-','--','-.',':', (0,(3,1,1,1)),'-','--','-.']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='white')
    fig.suptitle('Monte Carlo Stability: Does each option stay in its tier?',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT)
    ax1.set_facecolor('#F9FAFB')
    for b in range(len(names)):
        stab = [(assign_tiers(as_[:cp, b], breaks) == det_tiers[b]).mean()*100
                for cp in cps]
        ax1.plot(cps, stab, color=TC.get(det_tiers[b], '#888'),
                 lw=1.5, ls=styles[b % len(styles)], alpha=.85, label=names[b])
    ax1.axhline(99.5, color=RED, lw=1.8, ls='--', label='99.5% robust threshold')
    ax1.set_ylim(85, 101)
    ax1.set_xlabel('Number of simulations run', fontsize=9, fontfamily=FONT)
    ax1.set_ylabel('% of simulations where tier stays the same', fontsize=9, fontfamily=FONT)
    ax1.set_title('(a) All alternatives', fontsize=10, fontweight='bold',
                  color=DARK, fontfamily=FONT)
    ax1.grid(True, color='#E0E0E0', lw=.6); ax1.tick_params(labelsize=8)
    if len(names) <= 15:
        ax1.legend(fontsize=6.5, framealpha=.9, ncol=2, loc='lower right')
    ax2.set_facecolor('#F9FAFB')
    n_show = min(6, len(names))
    cp_pts = np.linspace(int(n_iter*.05), n_iter, 5, dtype=int)
    cp_lbls = [f'N={p:,}' for p in cp_pts]
    x = np.arange(len(cp_pts)); width = .12
    for shift, b in enumerate(range(n_show)):
        vals = [(assign_tiers(as_[:cp, b], breaks) == det_tiers[b]).mean()*100
                for cp in cp_pts]
        ax2.bar(x + (shift - n_show/2)*width, vals, width*.88,
                label=names[b][:18], color=TC.get(det_tiers[b], '#888'),
                alpha=.80, edgecolor='white', lw=.8)
    ax2.axhline(99.5, color=RED, lw=1.8, ls='--', label='99.5% threshold')
    ax2.set_xticks(x); ax2.set_xticklabels(cp_lbls, fontsize=8.5, fontfamily=FONT)
    ax2.set_ylim(80, 101.5)
    ax2.set_ylabel('% tier stability', fontsize=9, fontfamily=FONT)
    ax2.set_title('(b) Spot-checks at key simulation counts',
                  fontsize=10, fontweight='bold', color=DARK, fontfamily=FONT)
    ax2.grid(axis='y', color='#E0E0E0', lw=.6); ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7, framealpha=.92, loc='lower right',
               title='Alternative', title_fontsize=7)
    plt.tight_layout(rect=[0,0,1,.93]); return fig

def plot_mc_boxplot(aw, names, w_det):
    """Box plot showing weight distribution across all MC iterations."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(names)*.45)), facecolor='white')
    ax.set_facecolor('#F9FAFB')
    idx = np.argsort(w_det)[::-1]
    data = [aw[:, i] for i in idx]
    ns   = [names[i][:28] for i in idx]
    bp = ax.boxplot(data, vert=False, patch_artist=True,
                    medianprops=dict(color=RED, lw=2),
                    whiskerprops=dict(color=DARK, lw=1),
                    capprops=dict(color=DARK, lw=1),
                    flierprops=dict(marker='.', ms=3, alpha=.3, color=MED))
    for patch, i in zip(bp['boxes'], idx):
        patch.set_facecolor(plt.cm.Blues(.3 + .7 * w_det[i] / max(w_det)))
        patch.set_alpha(.75)
    ax.set_yticks(range(1, len(ns)+1))
    ax.set_yticklabels(ns, fontsize=8.5, fontfamily=FONT)
    for i, orig_i in enumerate(idx):
        ax.axvline(w_det[orig_i], ymin=(i)/len(idx),
                   ymax=(i+1)/len(idx), color=GREEN, lw=1.5, ls='--', alpha=.7)
    ax.set_xlabel('Weight value across all simulations', fontsize=10, fontfamily=FONT)
    ax.set_title('Weight Distribution from Monte Carlo Simulations\n'
                 '(green dashes = deterministic weight, boxes = simulated spread)',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT, pad=8)
    ax.grid(axis='x', color='#E0E0E0', lw=.6)
    plt.tight_layout(); return fig


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT DATA  — actual normalised Pᵢₖ scores from the published study
# ══════════════════════════════════════════════════════════════════════════════

DEF_CRITERIA = [
    "C1  Tectonic stability",     "C2  Fault & fracture intensity",
    "C3  Evaporites",             "C4  Reservoir-seal pairs",
    "C5  Leakage via outcrops",   "C6  Storage capacity",
    "C7  Basin size",             "C8  Reservoir temperature",
    "C9  Hydrogeological confinement", "C10 Depleted reservoir potential",
    "C11 Freshwater constraint",  "C12 Industry maturity",
    "C13 Onshore / offshore",     "C14 Accessibility",
    "C15 Infrastructure",         "C16 CO₂ source proximity",
]

DEF_W = np.array([.041,.071,.041,.092,.022,.221,.041,.041,
                   .022,.071,.022,.041,.041,.041,.071,.123])

DEF_ALTS = [
    "WCSB", "Williston Basin", "Michigan (SW Ont.)", "NL Offshore",
    "Scotian Basin", "Flemish Pass", "Beaufort-Mackenzie", "Hudson Bay",
    "St. Lawrence", "Nova Scotia", "Arctic Islands", "New Brunswick", "Pacific Margin"
]

# Normalised Pᵢₖ scores (13 alts × 16 criteria) — from published study
DEF_P = np.array([
 #  C1    C2    C3    C4    C5    C6    C7    C8    C9   C10   C11   C12   C13   C14   C15   C16
 [1.00, 1.00, 1.00, 1.00, 0.00, 1.00, 1.00, 0.50, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # WCSB
 [1.00, 1.00, 1.00, 0.75, 0.50, 0.75, 0.75, 0.75, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # Williston
 [1.00, 1.00, 0.50, 0.75, 0.50, 0.50, 0.50, 0.75, 0.50, 1.00, 1.00, 0.75, 1.00, 1.00, 1.00, 0.75],  # Michigan
 [1.00, 0.75, 0.00, 0.75, 0.50, 0.50, 0.75, 1.00, 0.50, 0.75, 0.50, 0.50, 0.00, 0.50, 0.50, 0.50],  # NL Offshore
 [1.00, 0.75, 0.00, 0.75, 0.50, 0.50, 0.75, 1.00, 0.50, 0.50, 0.50, 0.25, 0.00, 0.50, 0.25, 0.25],  # Scotian
 [1.00, 0.75, 0.00, 0.50, 0.50, 0.25, 0.50, 1.00, 0.25, 0.25, 0.50, 0.25, 0.00, 0.25, 0.25, 0.25],  # Flemish Pass
 [0.75, 0.75, 0.00, 0.75, 0.50, 0.50, 0.75, 0.50, 0.50, 0.25, 0.75, 0.25, 0.50, 0.25, 0.25, 0.25],  # Beaufort-Mac
 [0.75, 0.75, 0.00, 0.50, 0.50, 0.25, 0.75, 0.25, 0.25, 0.00, 0.75, 0.00, 0.50, 0.25, 0.00, 0.00],  # Hudson Bay
 [1.00, 1.00, 0.50, 0.50, 0.50, 0.25, 0.25, 0.75, 0.50, 0.50, 1.00, 0.50, 1.00, 1.00, 0.75, 0.75],  # St. Lawrence
 [0.75, 0.75, 0.00, 0.50, 0.50, 0.25, 0.25, 0.75, 0.25, 0.25, 0.50, 0.25, 0.50, 0.75, 0.50, 0.50],  # Nova Scotia
 [0.75, 0.75, 0.00, 0.25, 0.50, 0.25, 0.50, 0.25, 0.25, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00],  # Arctic Islands
 [0.75, 0.75, 0.25, 0.25, 0.50, 0.25, 0.25, 0.75, 0.25, 0.00, 0.50, 0.25, 1.00, 0.75, 0.25, 0.25],  # New Brunswick
 [0.25, 0.25, 0.00, 0.25, 1.00, 0.00, 0.25, 0.75, 0.00, 0.00, 0.75, 0.00, 0.00, 0.25, 0.00, 0.00],  # Pacific Margin
])


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

defaults = dict(ran_mc=False, mc_w=None, mc_s=None, weights=None,
                scores=None, tiers=None, breaks=None, P=None,
                cr_ok=True, sigma=None, mu=None, stab=None,
                pairwise_M=None, cnames=DEF_CRITERIA, anames=DEF_ALTS)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="pill">AHP &middot; MCDA &middot; MONTE CARLO</div>
  <h1>🌍 Rank Anything — Confidently</h1>
  <p class="sub">
    Compare options across multiple factors &middot; Derive scientifically grounded weights &middot;
    Classify priority tiers automatically &middot; Prove your rankings hold up under uncertainty.
    <br><br>
    <b>Built for CO₂ storage basin screening</b> — but works for any ranking or site-selection problem.
    No coding required.
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS BAR
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("⚙️  Settings — click to expand", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([2,1,1,1,1])
    with c1:
        mode = st.radio("How do you want to enter criterion importance?",
            ["✏️  Quick Mode — I'll type importance scores directly",
             "🔬  Expert Mode — I'll compare criteria one pair at a time (full AHP)"],
            help="Quick Mode is faster. Expert Mode is more rigorous and checks for internal consistency.")
    with c2:
        n_iter = st.select_slider("Simulations to run (N)",
            options=[1000,2000,5000,10000,20000,50000], value=10000,
            help="More simulations = more reliable confidence statistics. 10,000 is a solid default.")
    with c3:
        p_perturb = st.slider("Perturbation chance per weight",
            min_value=0.10, max_value=0.50, value=0.30, step=0.05,
            help="Chance that each weight is randomly nudged in each simulation run. 0.30 = 30%.")
    with c4:
        n_tiers = st.slider("Number of priority tiers (k)",
            min_value=2, max_value=6, value=4,
            help="How many groups to classify your options into. 4 = Priority / Secondary / Low / Marginal.")
    with c5:
        use_example = st.checkbox("Load CO₂ basin example", value=True,
            help="Pre-loads 13 Canadian basins and 16 criteria from Okwaraojimadu & Ezekiel (2025).")

is_expert = "Expert" in mode


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_setup, tab_w, tab_mc, tab_exp, tab_how, tab_gl = st.tabs([
    "📋  Step 1 — Setup",
    "⚖️  Step 2 — Weights & Scores",
    "📊  Step 3 — Simulation Results",
    "💾  Step 4 — Export",
    "❓  How to Use",
    "📖  Glossary",
])


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — SETUP
# ──────────────────────────────────────────────────────────────────────────────
with tab_setup:
    st.markdown('<div class="section-hd">What are you comparing and on what factors?</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="callout tip">
    <b>🗺️ What to do here:</b> List the <b>things you want to rank</b> (your alternatives)
    and the <b>factors that matter</b> (your criteria). Then fill in how well each thing
    performs on each factor, on a scale of 0 (worst) to 1 (best).
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📌 Criteria** — the factors you are judging by")
        st.caption("One per line. Example: Cost, Environmental impact, Accessibility")
        crit_txt = st.text_area("Criteria (one per line)",
            value="\n".join(DEF_CRITERIA) if use_example else
                  "Criterion 1\nCriterion 2\nCriterion 3\nCriterion 4\nCriterion 5",
            height=280, key="crit_in", label_visibility="collapsed")
    with col2:
        st.markdown("**🏛️ Alternatives** — the options you are ranking")
        st.caption("One per line. Example: Site A, Site B, Site C")
        alt_txt = st.text_area("Alternatives (one per line)",
            value="\n".join(DEF_ALTS) if use_example else
                  "Option A\nOption B\nOption C",
            height=280, key="alt_in", label_visibility="collapsed")

    cnames = [c.strip() for c in crit_txt.strip().split("\n") if c.strip()]
    anames = [a.strip() for a in alt_txt.strip().split("\n") if a.strip()]
    n_c, n_a = len(cnames), len(anames)
    st.session_state['cnames'] = cnames
    st.session_state['anames'] = anames

    m1, m2, m3 = st.columns(3)
    m1.metric("Criteria defined", n_c)
    m2.metric("Alternatives defined", n_a)
    m3.metric("Pairwise comparisons (Expert Mode)", n_c*(n_c-1)//2 if n_c > 1 else 0)

    if n_c < 2: st.error("⚠️ Please define at least 2 criteria.")
    if n_a < 2: st.error("⚠️ Please define at least 2 alternatives.")

    st.markdown('<div class="section-hd">How does each option perform on each factor?</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="callout">
    <b>Score each option on each factor</b> using a number from <b>0</b> (worst / not suitable)
    to <b>1</b> (best / most suitable). A score of <b>0.5</b> = average performance.
    The example data below is pre-filled from the published CO₂ storage study.
    </div>""", unsafe_allow_html=True)

    if use_example and n_c == 16 and n_a == 13:
        P = DEF_P.copy()
        st.success("✅ Example performance scores loaded from Okwaraojimadu & Ezekiel (2025).")
        df_p = pd.DataFrame(P, index=anames, columns=[c[:16] for c in cnames])
        st.dataframe(df_p.style.format("{:.2f}").background_gradient(
            cmap='Blues', axis=None, vmin=0, vmax=1), height=300)
        st.caption("Colour intensity = score (darker blue = better). "
                   "Each row is one alternative; each column is one criterion.")
    else:
        if n_c > 0 and n_a > 0:
            st.caption("Edit the table below — click any cell to change a value.")
            init = {c[:16]: [0.5]*n_a for c in cnames}
            ed   = st.data_editor(pd.DataFrame(init, index=anames),
                                  use_container_width=True, num_rows="fixed",
                                  key="score_ed")
            P = np.clip(ed.values.astype(float), 0, 1)
        else:
            P = np.zeros((max(n_a,1), max(n_c,1)))

    st.session_state['P'] = P
    st.markdown("""<div class="callout tip">
    ✅ <b>Done with Step 1?</b> Click the <b>Step 2 — Weights & Scores</b> tab above to continue.
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — WEIGHTS & SCORING
# ──────────────────────────────────────────────────────────────────────────────
with tab_w:
    st.markdown('<div class="section-hd">How important is each factor?</div>',
                unsafe_allow_html=True)

    cnames = st.session_state.get('cnames', DEF_CRITERIA)
    anames = st.session_state.get('anames', DEF_ALTS)
    P      = st.session_state.get('P', DEF_P)
    n_c, n_a = len(cnames), len(anames)

    if n_c < 2:
        st.warning("⚠️ Go to Step 1 first and define at least 2 criteria.")
        st.stop()

    pairwise_M = None

    if not is_expert:
        # ── QUICK MODE ──────────────────────────────────────────────────────
        st.markdown("""<div class="callout">
        <b>✏️ Quick Mode:</b> Type a number for each factor showing how important it is.
        Larger numbers = more important. They do not need to add up to any particular total —
        the app rescales them automatically.
        </div>""", unsafe_allow_html=True)

        def_w = DEF_W.tolist() if (use_example and n_c == 16) else [1./n_c]*n_c
        wi = []
        cols_w = st.columns(min(4, n_c))
        for i, cn in enumerate(cnames):
            with cols_w[i % len(cols_w)]:
                wi.append(st.number_input(cn[:28], min_value=0.001, max_value=1.,
                    value=float(def_w[i]) if i < len(def_w) else 1./n_c,
                    step=0.001, format="%.4f", key=f"wq{i}"))
        raw = np.array(wi); weights = raw / raw.sum()
        lambda_max = CI = CR = None; cr_ok = True

    else:
        # ── EXPERT MODE ─────────────────────────────────────────────────────
        st.markdown("""<div class="callout">
        <b>🔬 Expert Mode:</b> Compare every pair of criteria using Saaty's 1–9 scale.
        <br>
        <b>1</b> = equally important &nbsp;|&nbsp;
        <b>3</b> = one is moderately more important &nbsp;|&nbsp;
        <b>5</b> = strongly more important &nbsp;|&nbsp;
        <b>7</b> = very strongly &nbsp;|&nbsp;
        <b>9</b> = extremely more important.
        <br>Values like <b>1/3</b> mean the <em>second</em> criterion is more important.
        The app will check that your answers are internally consistent (CR must be below 0.10).
        </div>""", unsafe_allow_html=True)

        n_pairs = n_c*(n_c-1)//2
        if use_example and n_c == 16:
            def ns(v):
                opts = [1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
                return min(opts, key=lambda x: abs(x-v))
            def_up = [ns(DEF_W[i]/DEF_W[j])
                      for i in range(n_c) for j in range(i+1, n_c)]
        else:
            def_up = [1.]*n_pairs

        upper = []
        if n_pairs <= 45:
            pairs = [(cnames[i][:22], cnames[j][:22])
                     for i in range(n_c) for j in range(i+1, n_c)]
            cp = st.columns(min(3, n_pairs))
            for k, ((a, b), dv) in enumerate(zip(pairs, def_up)):
                with cp[k % len(cp)]:
                    v = st.select_slider(f"{a}  vs  {b}",
                        options=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9],
                        value=float(dv),
                        format_func=lambda x: (f"1/{int(round(1/x))}" if x < 1
                                               else str(int(x)) if x == int(x)
                                               else f"{x:.2f}"),
                        key=f"pair{k}")
                    upper.append(v)
        else:
            st.info(f"You have {n_pairs} pairs — too many for sliders. "
                    "Paste comma-separated upper-triangle values below.")
            csv_in = st.text_area("Upper triangle values (comma-separated)",
                value=",".join([str(round(v,4)) for v in def_up]), height=120)
            try:
                upper = [float(x.strip()) for x in csv_in.split(",")]
                if len(upper) != n_pairs:
                    st.error(f"Expected {n_pairs} values, got {len(upper)}.")
                    upper = def_up
            except:
                st.error("Could not parse values."); upper = def_up

        M = build_matrix(n_c, upper)
        pairwise_M = M
        weights, lambda_max, CI, CR = compute_ahp(M)
        cr_ok = CR <= 0.10

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f'<div class="kpi {"good" if cr_ok else "bad"}">'
                f'<div class="val">{"✓" if cr_ok else "✗"} {CR:.4f}</div>'
                f'<div class="lbl">Consistency Ratio (CR) — must be &lt; 0.10</div></div>',
                unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="kpi"><div class="val">{lambda_max:.4f}</div>'
                        f'<div class="lbl">Lambda Max (λmax)</div></div>',
                        unsafe_allow_html=True)
        with col_c:
            st.markdown(f'<div class="kpi"><div class="val">{CI:.4f}</div>'
                        f'<div class="lbl">Consistency Index (CI)</div></div>',
                        unsafe_allow_html=True)

        if not cr_ok:
            st.markdown("""<div class="callout warn">
            <b>⚠️ CR &gt; 0.10 — your comparisons are inconsistent.</b>
            This means some of your answers contradict each other.
            Below are the three most inconsistent pairs — revise those first.
            </div>""", unsafe_allow_html=True)
            issues = find_inconsistent_pairs(M, cnames, top_n=3)
            for rank_i, (ratio, na, nb, actual, ideal) in enumerate(issues, 1):
                st.markdown(
                    f"**{rank_i}.** &nbsp; *{na}* vs *{nb}* — "
                    f"you said **{actual:.2f}** but the weights suggest **{ideal:.2f}**. "
                    f"Inconsistency factor: **{ratio:.2f}×**")
        else:
            st.markdown(f"""<div class="callout ok">
            ✅ <b>Consistent — CR = {CR:.4f}.</b>
            Your pairwise comparisons are internally coherent. Weights accepted.
            </div>""", unsafe_allow_html=True)

        st.session_state['pairwise_M'] = M

    # ── WEIGHT TABLE + PARETO CHART ─────────────────────────────────────────
    st.markdown('<div class="section-hd">Derived criterion weights</div>',
                unsafe_allow_html=True)
    st.caption("These weights tell you how much each factor contributes to the final score. "
               "They all add up to 1.0 (100%).")

    df_w = pd.DataFrame({
        'Criterion': cnames,
        'Weight (wᵢ)': [f"{w:.4f}" for w in weights],
        'Share of total': [f"{w*100:.1f}%" for w in weights],
        'Importance bar': weights
    })
    st.dataframe(df_w.drop(columns=['Importance bar']),
                 use_container_width=True, hide_index=True)

    fp = plot_pareto(weights, cnames)
    st.pyplot(fp, use_container_width=True); plt.close(fp)
    st.caption("The red line shows cumulative weight. The dotted line at 80% marks the "
               "Pareto point — the fewest criteria that together explain 80% of the ranking.")

    # ── COMPOSITE SCORES ────────────────────────────────────────────────────
    st.markdown('<div class="section-hd">Final rankings and tier classification</div>',
                unsafe_allow_html=True)

    alt_scores = P @ weights

    if n_a >= n_tiers:
        breaks, gvf = jenks(alt_scores, n_tiers)
        tiers = assign_tiers(alt_scores, breaks)
    else:
        breaks = np.array([]); gvf = 1.; tiers = np.ones(n_a, dtype=int)

    ca, cb, cc = st.columns(3)
    ca.metric("Classification quality (GVF)", f"{gvf:.4f}",
              help="Goodness of Variance Fit. Above 0.90 = excellent tier separation.")
    cb.metric("Number of tiers", n_tiers)
    cc.metric("Alternatives ranked", n_a)

    tier_labels = {1:"🟢 Tier 1 — Priority", 2:"🟡 Tier 2 — Secondary",
                   3:"🟠 Tier 3 — Low priority", 4:"⚫ Tier 4 — Marginal"}

    df_s = pd.DataFrame({
        'Alternative': anames,
        'Score (0–1)': [f"{s:.4f}" for s in alt_scores],
        'National Rank': pd.Series(alt_scores).rank(ascending=False).astype(int).values,
        'Priority Tier': [tier_labels.get(t, f"Tier {t}") for t in tiers]
    }).sort_values('National Rank').reset_index(drop=True)
    st.dataframe(df_s, use_container_width=True, hide_index=True)

    if n_a >= 2:
        fs = plot_scores(np.array(anames), alt_scores, tiers, breaks)
        st.pyplot(fs, use_container_width=True); plt.close(fs)
        st.caption("Bar length = composite score. Colour = tier. "
                   "Dashed vertical lines = Jenks-Fisher tier boundaries (data-driven, not arbitrary).")

    st.session_state.update({
        'weights': weights, 'scores': alt_scores, 'tiers': tiers,
        'breaks': breaks, 'cr_ok': cr_ok, 'P': P,
        'pairwise_M': pairwise_M
    })

    st.markdown("""<div class="callout tip">
    ✅ <b>Done with Step 2?</b> Click <b>Step 3 — Simulation Results</b> to test
    how robust these rankings are.
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — MONTE CARLO
# ──────────────────────────────────────────────────────────────────────────────
with tab_mc:
    st.markdown('<div class="section-hd">Are these rankings trustworthy?</div>',
                unsafe_allow_html=True)

    cnames   = st.session_state.get('cnames', DEF_CRITERIA)
    anames   = st.session_state.get('anames', DEF_ALTS)
    w        = st.session_state['weights']
    s_det    = st.session_state['scores']
    t_det    = st.session_state['tiers']
    brk      = st.session_state['breaks']
    P        = st.session_state['P']
    pm       = st.session_state.get('pairwise_M', None)
    n_a      = len(anames) if anames else 0

    st.markdown(f"""<div class="callout">
    <b>What this does:</b> The app reruns your entire analysis <b>{n_iter:,} times</b>,
    each time slightly changing the criterion weights at random — simulating the natural
    uncertainty in any expert judgement. If the rankings and tier assignments stay the same
    across almost all of those runs, your results are <b>robust and trustworthy</b>.
    <br><br>
    An alternative that stays in its tier in <b>≥ 99.5%</b> of simulations is considered
    <b>robustly classified</b>. Anything below that deserves a closer look.
    <br><br>
    <b>Method:</b> {"Pairwise matrix element perturbation (±1 Saaty unit) — scientifically correct for Expert Mode." if pm is not None else "Weight-space Gaussian perturbation (Quick Mode)."}
    </div>""", unsafe_allow_html=True)

    if w is None:
        st.warning("⚠️ Complete Step 2 first.")
    elif is_expert and not st.session_state.get('cr_ok', True):
        st.error("⚠️ CR > 0.10 — fix your pairwise comparisons in Step 2 before running.")
    else:
        if st.button(f"▶  Run {n_iter:,} simulations now", type="primary",
                     help="This may take a few seconds for large N."):
            with st.spinner(f"Running {n_iter:,} simulations — please wait..."):
                aw, as_ = run_mc_matrix(w, P, pm, n_iter, p_perturb)
                st.session_state.update({'mc_w': aw, 'mc_s': as_, 'ran_mc': True})
            st.success(f"✅ Done — {n_iter:,} simulations complete.")

        if st.session_state['ran_mc'] and st.session_state['mc_w'] is not None:
            aw  = st.session_state['mc_w']
            as_ = st.session_state['mc_s']
            sig = aw.std(axis=0)
            mu  = aw.mean(axis=0)

            # ── WEIGHT UNCERTAINTY TABLE ────────────────────────────────────
            st.markdown('<div class="section-hd">How much did the weights vary?</div>',
                        unsafe_allow_html=True)
            st.caption("σᵢ = standard deviation of each weight across all simulations. "
                       "Smaller = more stable. Values ≤ 0.006 are considered very stable.")

            df_mc = pd.DataFrame({
                'Criterion': cnames,
                'Your weight': [f"{ww:.4f}" for ww in w],
                'Average across simulations': [f"{m:.4f}" for m in mu],
                'Variability (σᵢ)': [f"{s:.5f}" for s in sig],
                'Stable?': ['✅ Yes' if s <= 0.010 else '⚠️ Check' for s in sig]
            })
            st.dataframe(df_mc, use_container_width=True, hide_index=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("Highest weight variability (max σᵢ)", f"{sig.max():.5f}",
                      help="Lower is better. < 0.006 is excellent.")
            r2.metric("Average weight variability (mean σᵢ)", f"{sig.mean():.5f}")
            r3.metric("Simulations run", f"{n_iter:,}")

            # ── TIER STABILITY TABLE ────────────────────────────────────────
            st.markdown('<div class="section-hd">Did the tier assignments hold up?</div>',
                        unsafe_allow_html=True)
            st.caption("Each row shows what percentage of simulations kept that option in its original tier. "
                       "Above 99.5% = robust classification.")

            stab = []
            for b in range(n_a):
                stab.append(
                    (assign_tiers(as_[:, b], brk) == t_det[b]).mean() * 100)

            tier_labels = {1:"🟢 Tier 1 — Priority",2:"🟡 Tier 2 — Secondary",
                           3:"🟠 Tier 3 — Low priority",4:"⚫ Tier 4 — Marginal"}
            df_st = pd.DataFrame({
                'Alternative': anames,
                'Score': [f"{ss:.4f}" for ss in s_det],
                'Tier': [tier_labels.get(t, f"Tier {t}") for t in t_det],
                'Tier stability': [f"{p:.2f}%" for p in stab],
                'Verdict': ['✅ Robust' if p >= 99.5 else '⚠️ Review' for p in stab]
            })
            st.dataframe(df_st, use_container_width=True, hide_index=True)

            if all(p >= 99.5 for p in stab):
                st.markdown(f"""<div class="callout ok">
                ✅ <b>All {n_a} alternatives are robustly classified.</b>
                Every option stayed in its tier in ≥ 99.5% of {n_iter:,} simulations.
                You can be highly confident in these results.
                </div>""", unsafe_allow_html=True)
            else:
                n_w = sum(p < 99.5 for p in stab)
                st.markdown(f"""<div class="callout warn">
                ⚠️ <b>{n_w} alternative(s) are below the 99.5% stability threshold.</b>
                Their tier assignment is sensitive to how you weight the criteria.
                Consider reviewing those criteria weights or collecting more data for those options.
                </div>""", unsafe_allow_html=True)

            # ── CONVERGENCE PLOTS ───────────────────────────────────────────
            st.markdown('<div class="section-hd">Do the results converge? (Quality check)</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="callout">
            These charts show the weight variability stabilising as more simulations are run.
            Flat lines near the right = the simulation has converged and you have enough iterations.
            </div>""", unsafe_allow_html=True)

            fig_cv = plot_conv(aw, cnames, w, top_n=min(4, len(cnames)))
            st.pyplot(fig_cv, use_container_width=True); plt.close(fig_cv)

            fig_bp = plot_mc_boxplot(aw, cnames, w)
            st.pyplot(fig_bp, use_container_width=True); plt.close(fig_bp)
            st.caption("Each box shows the spread of simulated weight values. "
                       "Narrow boxes = stable weights. Green dashes = your original weight.")

            fig_st = plot_stability(as_, anames, t_det, brk, n_iter)
            st.pyplot(fig_st, use_container_width=True); plt.close(fig_st)

            st.session_state.update({'sigma': sig, 'mu': mu, 'stab': stab})

            st.markdown("""<div class="callout tip">
            ✅ <b>Done with Step 3?</b> Click <b>Step 4 — Export</b> to download all results.
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — EXPORT
# ──────────────────────────────────────────────────────────────────────────────
with tab_exp:
    st.markdown('<div class="section-hd">Download your results</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="callout">
    <b>CSV files</b> open in Excel or Google Sheets. &nbsp;
    <b>PNG files</b> are high-resolution images ready for papers and presentations. &nbsp;
    <b>JSON</b> contains everything in one file for GIS or programming workflows.
    </div>""", unsafe_allow_html=True)

    w   = st.session_state['weights']
    s   = st.session_state['scores']
    t   = st.session_state['tiers']
    brk = st.session_state['breaks']
    cn  = st.session_state.get('cnames', DEF_CRITERIA)
    an  = st.session_state.get('anames', DEF_ALTS)

    if w is None:
        st.info("Complete Steps 1 and 2 to enable downloads.")
    else:
        st.markdown("**📄 Data tables (CSV)**")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥  Criterion weights",
                pd.DataFrame({'Criterion': cn, 'Weight': w, 'Pct (%)': w*100}
                             ).to_csv(index=False),
                "ahp_weights.csv", "text/csv", use_container_width=True)
        with c2:
            df_se = pd.DataFrame({
                'Alternative': an, 'Score': s, 'Tier': t,
                'Rank': pd.Series(s).rank(ascending=False).astype(int).values
            }).sort_values('Rank').reset_index(drop=True)
            st.download_button("📥  Scores and tier rankings",
                df_se.to_csv(index=False), "ahp_scores.csv",
                "text/csv", use_container_width=True)

        if st.session_state['ran_mc'] and st.session_state['sigma'] is not None:
            c3, c4 = st.columns(2)
            with c3:
                st.download_button("📥  Monte Carlo weight statistics",
                    pd.DataFrame({'Criterion': cn, 'Det_Weight': w,
                        'MC_Mean': st.session_state['mu'],
                        'MC_StdDev': st.session_state['sigma']}
                    ).to_csv(index=False),
                    "mc_weights.csv", "text/csv", use_container_width=True)
            with c4:
                st.download_button("📥  Tier stability results",
                    pd.DataFrame({'Alternative': an, 'Score': s, 'Tier': t,
                        'Stability_pct': st.session_state['stab']}
                    ).to_csv(index=False),
                    "tier_stability.csv", "text/csv", use_container_width=True)

        st.markdown("**🖼️ Figures (PNG)**")
        cf1, cf2 = st.columns(2)
        with cf1:
            fw2 = plot_pareto(w, cn)
            st.download_button("📥  Criterion weights chart",
                fig2bytes(fw2), "weights_pareto.png",
                "image/png", use_container_width=True)
        with cf2:
            if s is not None and len(s) >= 2:
                fs2 = plot_scores(np.array(an), s, t, brk)
                st.download_button("📥  Ranked scores chart",
                    fig2bytes(fs2), "scores.png",
                    "image/png", use_container_width=True)

        if st.session_state['ran_mc']:
            aw  = st.session_state['mc_w']
            as_ = st.session_state['mc_s']
            cf3, cf4 = st.columns(2)
            with cf3:
                fc = plot_conv(aw, cn, w, top_n=min(4, len(cn)))
                st.download_button("📥  Convergence chart",
                    fig2bytes(fc), "convergence.png",
                    "image/png", use_container_width=True)
            with cf4:
                fst = plot_stability(as_, an, t, brk, n_iter)
                st.download_button("📥  Tier stability chart",
                    fig2bytes(fst), "stability.png",
                    "image/png", use_container_width=True)
            cf5, _ = st.columns(2)
            with cf5:
                fbp = plot_mc_boxplot(aw, cn, w)
                st.download_button("📥  Weight distribution boxplot",
                    fig2bytes(fbp), "weight_boxplot.png",
                    "image/png", use_container_width=True)

        st.markdown("**📦 Full results (JSON)**")
        rd = {
            'criteria': cn, 'alternatives': an,
            'weights': {c: float(ww) for c, ww in zip(cn, w)},
            'scores': {a: float(ss) for a, ss in zip(an, s)},
            'tiers': {a: int(tt) for a, tt in zip(an, t)},
            'tier_breaks': [float(b) for b in brk],
            'n_iterations': n_iter,
            'perturbation_probability': p_perturb
        }
        if st.session_state['sigma'] is not None:
            rd['mc_sigma']     = {c: float(sig) for c, sig in zip(cn, st.session_state['sigma'])}
            rd['mc_stability'] = {a: float(p)   for a, p   in zip(an, st.session_state['stab'])}
        st.download_button("📥  Full results (JSON)",
            json.dumps(rd, indent=2), "ahp_mc_results.json",
            "application/json", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# HOW TO USE
# ──────────────────────────────────────────────────────────────────────────────
with tab_how:
    st.markdown('<div class="section-hd">What does this app do?</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="callout tip">
    This app helps you <b>rank a list of options</b> across <b>multiple factors</b>
    and then tells you <b>how confident you can be in those rankings</b> — even when
    you are not 100% sure how important each factor is.
    <br><br>
    It was built to rank Canadian sedimentary basins for CO₂ geological storage,
    but it works for <em>any</em> ranking or site-selection problem.
    <b>No programming knowledge needed.</b>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hd">Step-by-step guide</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Open Settings at the top",
         "Click the <b>Settings — click to expand</b> bar. Choose whether you want Quick Mode "
         "(type importance scores directly) or Expert Mode (full AHP pairwise comparison). "
         "Set how many simulations to run (10,000 is a solid default) and tick or untick the "
         "example data checkbox."),
        ("2", "Define your criteria and alternatives (Step 1 tab)",
         "In the left box, list the <b>factors</b> you want to evaluate on — one per line. "
         "In the right box, list the <b>options</b> you want to rank — one per line. "
         "Then fill in the score table: how well does each option perform on each factor, "
         "on a scale of 0 (worst) to 1 (best)?"),
        ("3", "Enter criterion importance (Step 2 tab)",
         "<b>Quick Mode:</b> Type a number for each factor. Higher = more important. They "
         "do not need to add to 1.<br><br>"
         "<b>Expert Mode:</b> Compare every factor against every other, one pair at a time, "
         "using Saaty's 1-9 scale. The app derives weights mathematically and checks that "
         "your answers are internally consistent (CR must be below 0.10)."),
        ("4", "See your rankings",
         "Scroll down in Step 2 to see a ranked list of your alternatives with their composite "
         "scores and colour-coded priority tiers. Tiers are set by the Jenks-Fisher algorithm — "
         "a statistical method that finds <em>natural</em> groupings in your scores, not arbitrary cutoffs."),
        ("5", "Test robustness (Step 3 tab)",
         "Click <b>Run simulations</b>. The app reruns your analysis thousands of times with slightly "
         "different weights, checking whether rankings and tier assignments hold. "
         "Options stable in ≥ 99.5% of runs are robustly classified. "
         "Two convergence charts confirm the simulation has enough iterations."),
        ("6", "Download everything (Step 4 tab)",
         "Download weights, scores, tier assignments, Monte Carlo statistics, and all figures "
         "as CSV, PNG, or JSON. All files are ready for papers, GIS, or reports."),
        ("7", "Look up any term you do not recognise (Glossary tab)",
         "Every technical term used in the app — CR, GVF, eigenvector, Jenks-Fisher, σᵢ — "
         "is explained in plain English with a real example."),
    ]
    for num, title, body in steps:
        st.markdown(f"""
        <div class="step-row">
          <div class="step-num">{num}</div>
          <div class="step-body"><h4>{title}</h4><p>{body}</p></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hd">Frequently asked questions</div>',
                unsafe_allow_html=True)
    faqs = [
        ("Do I need to know AHP or statistics?",
         "No. Quick Mode lets anyone type importance scores and run the simulation immediately. "
         "Expert Mode adds full AHP rigour if you want it."),
        ("What is the difference between Quick Mode and Expert Mode?",
         "Quick Mode is faster — you type how important each factor is directly. "
         "Expert Mode is more rigorous — you compare factors in pairs and the app derives "
         "weights mathematically, then verifies that your judgements are internally consistent."),
        ("What is a good number of simulations?",
         "10,000 is a solid default. The convergence charts will show that results stabilise "
         "well before the end — typically by 3,000 iterations."),
        ("My CR is above 0.10 — what do I do?",
         "Some of your pairwise comparisons are contradicting each other. The app now shows you "
         "the three most inconsistent pairs — revise those first. Usually two or three adjustments "
         "are enough to bring CR below 0.10."),
        ("Why are some options shown as '⚠️ Review'?",
         "Those options are close to a tier boundary, so their tier assignment changes in more than "
         "0.5% of simulations. This does not mean your result is wrong — it means you should look "
         "at those cases more carefully before making high-stakes decisions."),
        ("Can I use this for problems other than CO₂ storage?",
         "Yes — completely. Replace the example with your own criteria and alternatives for any "
         "ranking problem: supplier selection, policy prioritisation, site evaluation, project ranking."),
        ("How do I cite this tool?",
         "Okwaraojimadu, C.K. & Ezekiel, C.J. (2025). AHP-MCDA Monte Carlo Simulator "
         "[Web application]. University of Calgary. chisom.okwaraojimadu@ucalgary.ca"),
    ]
    for q, a in faqs:
        with st.expander(f"▸  {q}"):
            st.markdown(f"<p style='font-size:.9rem;line-height:1.7;color:#333;'>{a}</p>",
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# GLOSSARY
# ──────────────────────────────────────────────────────────────────────────────
with tab_gl:
    st.markdown('<div class="section-hd">Plain-English glossary</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="callout tip">
    Every technical term used in the app, explained in plain English with real examples.
    </div>""", unsafe_allow_html=True)

    sections = {
        "The basics": [
            ("Criterion (plural: criteria)",
             "A factor you use to evaluate your options — like cost, capacity, or accessibility.",
             "In a CO₂ storage study, criteria include storage capacity and tectonic stability."),
            ("Alternative",
             "One of the options you are comparing. Also called a candidate, site, or option.",
             "WCSB, Williston Basin, and Michigan Basin are the alternatives in the example."),
            ("Performance score (Pᵢₖ)",
             "A number between 0 and 1 saying how well an alternative performs on a criterion. 1 = best, 0 = worst.",
             "WCSB scores 1.0 on storage capacity because it has the largest capacity class."),
            ("Composite score (Rᵏ)",
             "The overall score for an alternative — the weighted sum of all its performance scores. Higher = better.",
             "WCSB scores 0.982 out of 1.0."),
            ("Weight (wᵢ)",
             "How important a criterion is, expressed as a fraction of the total. All weights add up to 1.0.",
             "Storage capacity has weight 0.221, meaning it accounts for 22.1% of the final score."),
            ("Tier",
             "A priority group. Tier 1 = highest priority. Boundaries are set by data, not arbitrary cutoffs.",
             "WCSB and Williston are Tier 1. Michigan is Tier 2."),
        ],
        "AHP — Analytic Hierarchy Process": [
            ("AHP (Analytic Hierarchy Process)",
             "A method that derives weights by asking you to compare criteria in pairs, rather than guessing. Developed by Thomas Saaty (1977).",
             ""),
            ("Pairwise comparison",
             "Asking 'How much more important is A than B?' for every pair of criteria. With 16 criteria, there are 120 pairs.",
             ""),
            ("Saaty scale (1 to 9)",
             "The scale used for pairwise answers. 1 = equally important. 3 = moderately more. 5 = strongly more. 7 = very strongly. 9 = extremely more. Values below 1 (e.g. 1/3) mean the other criterion is more important.",
             ""),
            ("Eigenvector method",
             "The mathematical technique AHP uses to turn the comparison matrix into weights. The app handles this automatically — you never need to do the maths yourself.",
             ""),
            ("Lambda max (λmax)",
             "A number derived from your comparisons. For a perfectly consistent matrix it exactly equals n (the number of criteria). Values close to n = good consistency.",
             "With 16 criteria, perfect λmax = 16. This study achieved λmax = 16.154."),
            ("Consistency Index (CI)",
             "Measures how inconsistent your comparisons are. CI = (λmax − n) / (n − 1). Smaller is better.",
             ""),
            ("Random Index (RI)",
             "The consistency index you would expect from a completely random matrix of the same size. Used to put CI in perspective.",
             "For 16 criteria, RI = 1.59."),
            ("Consistency Ratio (CR)",
             "CR = CI ÷ RI. Must be below 0.10 for the weights to be considered valid. Think of it as a quality control check on your judgements.",
             "This study achieved CR = 0.0064 — 15 times better than the 0.10 threshold."),
        ],
        "Monte Carlo simulation": [
            ("Monte Carlo simulation",
             "Running the entire analysis thousands of times with slightly different weights each time, to see how much the results change under uncertainty.",
             ""),
            ("Iteration",
             "One single run of the simulation with one set of randomly adjusted weights. N = 10,000 means 10,000 runs.",
             ""),
            ("Perturbation",
             "A small random change applied to a weight or matrix element in each iteration, simulating the natural uncertainty in expert judgement.",
             ""),
            ("Perturbation probability (p)",
             "The chance that any given weight is changed in a particular iteration. p = 0.30 means a 30% chance per weight per run.",
             ""),
            ("Weight standard deviation (σᵢ)",
             "How much a criterion's weight varied across all simulations. Very small σᵢ = weight is stable = result is trustworthy.",
             "Storage capacity weight has σ = 0.006 — barely changes across 10,000 simulations."),
            ("Convergence",
             "When the running statistics stop changing as more iterations are added. If σᵢ looks the same at N = 10,000 and N = 20,000, the simulation has converged — you have enough iterations.",
             ""),
            ("Tier stability",
             "The percentage of simulations where an alternative stays in its original tier. Above 99.5% is considered robustly classified.",
             "WCSB has 100% tier stability — it is Tier 1 in every single simulation."),
        ],
        "Tier classification": [
            ("Jenks-Fisher (Natural Breaks)",
             "A statistical algorithm that finds the best boundary positions between tiers based on where the natural gaps are in your scores — not arbitrary cutoffs chosen by the analyst.",
             "The gap between 0.847 and 0.720 is the largest natural break, so the Tier 1 / Tier 2 boundary falls there."),
            ("Goodness of Variance Fit (GVF)",
             "A number from 0 to 1 rating how well the tiers capture the structure in your data. Above 0.90 = excellent.",
             "This study achieved GVF = 0.957 with 4 tiers — excellent tier separation."),
            ("SDCM (Sum of Squared Deviations from Class Means)",
             "The total within-tier variance. Jenks-Fisher minimises this to find the best boundaries — tighter groups within each tier, bigger gaps between tiers.",
             ""),
            ("SDAM (Sum of Squared Deviations from the Array Mean)",
             "The total variance across all scores. Used together with SDCM to calculate GVF.",
             ""),
        ],
    }

    for sec, terms in sections.items():
        st.markdown(f'<div class="section-hd">{sec}</div>', unsafe_allow_html=True)
        for term, definition, example in terms:
            ex_html = f'<p class="gloss-ex">&#9656; Example: {example}</p>' if example else ''
            st.markdown(f"""
            <div class="gloss-card">
              <div class="gloss-term">{term}</div>
              <p class="gloss-def">{definition}</p>
              {ex_html}
            </div>""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  AHP-MCDA Monte Carlo Simulator &nbsp;·&nbsp; Python 3 &nbsp;·&nbsp;
  NumPy · Matplotlib · Streamlit &nbsp;·&nbsp;
  Okwaraojimadu C.K. &amp; Ezekiel C.J., University of Calgary, 2025 &nbsp;·&nbsp;
  chisom.okwaraojimadu@ucalgary.ca
</div>
""", unsafe_allow_html=True)
