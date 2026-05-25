import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AHP-MCDA Monte Carlo Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state: sidebar toggle ─────────────────────────────────────────────
if 'sidebar_open' not in st.session_state:
    st.session_state.sidebar_open = True

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:   #1B3A5C;
    --teal:   #0E7C7B;
    --blue:   #2E75B6;
    --amber:  #C47A00;
    --green:  #2E8B57;
    --red:    #B22222;
    --border: #D0DAE8;
    --text:   #1A1A2E;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {visibility: hidden;}

/* ── Sidebar toggle button fixed top-left ── */
.sidebar-toggle-btn {
    position: fixed;
    top: 0.75rem;
    left: 0.75rem;
    z-index: 999999;
    background: #1B3A5C;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
    font-size: 1.2rem;
    cursor: pointer;
    line-height: 1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    transition: background 0.2s;
}
.sidebar-toggle-btn:hover { background: #0E7C7B; }

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, #1B3A5C 0%, #0E4D7B 50%, #0E7C7B 100%);
    padding: 2.2rem 2rem 1.8rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(14,124,123,0.3) 0%, transparent 70%);
    pointer-events: none;
}
.app-header .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
    margin-bottom: 0.9rem;
    color: rgba(255,255,255,0.9);
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    font-weight: 400;
    margin: 0 0 0.45rem;
    color: white;
}
.app-header p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
    font-weight: 300;
    line-height: 1.6;
}

/* ── Section headers ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: var(--navy);
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.45rem;
    margin: 1.5rem 0 1rem;
}

/* ── Info / warn boxes ── */
.info-box {
    background: #EFF6FF;
    border-left: 4px solid var(--blue);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    line-height: 1.65;
    margin: 0.8rem 0;
    color: #1e3a5f;
}
.warn-box {
    background: #FFFBF0;
    border-left: 4px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    line-height: 1.65;
    margin: 0.8rem 0;
    color: #5a3a00;
}
.success-box {
    background: #F0FFF4;
    border-left: 4px solid var(--green);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    line-height: 1.65;
    margin: 0.8rem 0;
    color: #1a4a2a;
}

/* ── How-to step cards ── */
.step-card {
    background: white;
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.1rem;
    display: flex;
    gap: 1.2rem;
    align-items: flex-start;
    transition: box-shadow 0.2s;
}
.step-card:hover { box-shadow: 0 4px 18px rgba(27,58,92,0.10); }
.step-num {
    background: linear-gradient(135deg, #1B3A5C, #0E7C7B);
    color: white;
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    width: 48px; height: 48px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(27,58,92,0.25);
}
.step-content h4 {
    font-family: 'DM Serif Display', serif;
    color: var(--navy);
    margin: 0 0 0.4rem;
    font-size: 1.05rem;
}
.step-content p {
    margin: 0;
    font-size: 0.88rem;
    color: #444;
    line-height: 1.65;
}

/* ── Glossary cards ── */
.gloss-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.gloss-term {
    font-family: 'DM Mono', monospace;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--navy);
    margin-bottom: 0.3rem;
}
.gloss-def {
    font-size: 0.86rem;
    color: #444;
    line-height: 1.65;
    margin: 0;
}
.gloss-example {
    font-size: 0.80rem;
    color: var(--teal);
    font-style: italic;
    margin-top: 0.3rem;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'DM Mono', monospace;
    font-size: 1.7rem;
    font-weight: 500;
    color: var(--navy);
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.76rem;
    color: #777;
    margin-top: 0.3rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.metric-card.good .val { color: var(--green); }
.metric-card.bad  .val { color: var(--red);   }

/* ── Tier badges ── */
.tier-1 { background:#E8F5E9; color:#2E8B57; border:1px solid #2E8B57; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tier-2 { background:#FFF8E1; color:#C47A00; border:1px solid #C47A00; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tier-3 { background:#FBE9E7; color:#8B3000; border:1px solid #8B3000; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }
.tier-4 { background:#FAFAFA; color:#555;    border:1px solid #AAA;    border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; }

/* ── App footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.78rem;
    color: #999;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    font-family: 'DM Mono', monospace;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #F4F7FB;
    border-right: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar toggle button (always visible, top-left) ──────────────────────────
# Streamlit doesn't expose the native toggle after collapse, so we render
# a persistent fixed button that re-opens the sidebar via JS interaction note.
st.markdown("""
<div style="position:fixed;top:0.65rem;left:0.65rem;z-index:999999;">
  <button onclick="
    var btn = window.parent.document.querySelector('[data-testid=collapsedControl]');
    if(btn){ btn.click(); }
    return false;
  "
  style="display:inline-flex;align-items:center;justify-content:center;
         width:38px;height:38px;background:#1B3A5C;color:white;
         border-radius:8px;font-size:1.25rem;border:none;cursor:pointer;
         box-shadow:0 2px 8px rgba(0,0,0,0.28);transition:background 0.2s;"
  title="Open / close settings panel"
  onmouseover="this.style.background='#0E7C7B'"
  onmouseout="this.style.background='#1B3A5C'">
    &#9776;
  </button>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AHP CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

SAATY_RI = {1:0,2:0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,
            9:1.45,10:1.49,11:1.51,12:1.54,13:1.56,14:1.57,
            15:1.58,16:1.59,17:1.60,18:1.61,19:1.62,20:1.63}

def compute_ahp(matrix):
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    norm = matrix / col_sums
    weights = norm.mean(axis=1)
    lam = (matrix @ weights) / weights
    lambda_max = lam.mean()
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI = SAATY_RI.get(n, 1.63)
    CR = CI / RI if RI > 0 else 0
    return weights, lambda_max, CI, CR

def build_matrix(n, upper_vals):
    M = np.eye(n)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            v = upper_vals[idx]
            M[i,j] = v;  M[j,i] = 1.0/v
            idx += 1
    return M

def jenks_fisher(values, k):
    n = len(values)
    sv = np.sort(values)
    if k >= n:
        return sv[:-1], 1.0
    SSW = np.full((k, n), np.inf)
    BI  = np.zeros((k, n), dtype=int)
    for i in range(n):
        v = sv[:i+1]
        SSW[0,i] = np.var(v)*len(v) if len(v)>0 else 0
    for cl in range(1, k):
        for i in range(cl, n):
            best, bst_bi = np.inf, cl
            for m in range(cl-1, i):
                vr = sv[m+1:i+1]
                tot = SSW[cl-1,m] + (np.var(vr)*len(vr) if len(vr)>0 else 0)
                if tot < best: best=tot; bst_bi=m+1
            SSW[cl,i]=best; BI[cl,i]=bst_bi
    idx=n-1; bks=[]
    for cl in range(k-1,0,-1):
        bi=BI[cl,idx]; bks.append(bi); idx=bi-1
    bks=sorted(bks)
    bv = sv[bks]
    sdam = np.var(sv)*n
    gvf  = 1 - SSW[k-1,n-1]/sdam if sdam>0 else 1.0
    return bv, gvf

def assign_tiers(scores, breaks):
    sb = np.sort(breaks)[::-1]
    out = []
    for s in scores:
        t = len(sb)+1
        for i,b in enumerate(sb):
            if s >= b: t=i+1; break
        out.append(t)
    return np.array(out)

def run_mc(weights, score_matrix, n_iter, p_perturb):
    np.random.seed(42)
    n_c = len(weights)
    sigma = np.maximum(weights*0.03, 0.002)
    all_w = np.zeros((n_iter, n_c))
    all_s = np.zeros((n_iter, score_matrix.shape[0]))
    for s in range(n_iter):
        noise = np.random.normal(0, sigma)
        mask  = np.random.random(n_c) < p_perturb
        wp    = np.maximum(weights + noise*mask, 0.005)
        wp    = wp / wp.sum()
        all_w[s] = wp
        all_s[s] = score_matrix @ wp
    return all_w, all_s


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

FONT='DejaVu Sans'; DARK='#1B3A5C'; MED='#2E75B6'; TEAL='#0E7C7B'
AMBER='#C47A00'; GREEN='#2E8B57'; RED='#B22222'; PURPLE='#6A4C93'
TIER_COLS={1:'#2E8B57',2:'#C47A00',3:'#8B4500',4:'#555555'}

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0); plt.close(fig)
    return buf.getvalue()

def plot_weights(weights, names):
    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*0.42)), facecolor='white')
    ax.set_facecolor('#F9FAFB')
    n=len(weights); y=np.arange(n)
    idx=np.argsort(weights); ws=weights[idx]; ns=[names[i] for i in idx]
    cols=[plt.cm.Blues(0.4+0.6*w/max(weights)) for w in ws]
    bars=ax.barh(y, ws, color=cols, edgecolor='white', linewidth=0.8, height=0.7)
    for bar,w in zip(bars,ws):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                f'{w:.4f}', va='center', ha='left', fontsize=9,
                fontfamily=FONT, color=DARK, fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels(ns, fontsize=9, fontfamily=FONT)
    ax.set_xlabel('AHP Weight  wᵢ', fontsize=10, fontfamily=FONT)
    ax.set_title('Derived Criterion Weights', fontsize=11, fontweight='bold',
                 color=DARK, fontfamily=FONT, pad=8)
    ax.set_xlim(0, max(weights)*1.25)
    ax.grid(axis='x', color='#E0E0E0', lw=0.6)
    plt.tight_layout(); return fig

def plot_scores(names, scores, tiers, breaks):
    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*0.55)), facecolor='white')
    ax.set_facecolor('#F9FAFB')
    idx=np.argsort(scores)[::-1]; ss=scores[idx]
    ns=[names[i] for i in idx]; ts=tiers[idx]
    y=np.arange(len(names)); cols=[TIER_COLS.get(t,'#888') for t in ts]
    ax.barh(y, ss, color=cols, edgecolor='white', lw=0.8, height=0.68, alpha=0.85)
    for i,(s,t) in enumerate(zip(ss,ts)):
        ax.text(s+0.008, i, f'{s:.3f}', va='center', ha='left',
                fontsize=9.5, fontfamily=FONT, color=TIER_COLS.get(t,'#888'), fontweight='bold')
    for b in breaks: ax.axvline(b, color='#444', lw=1.0, ls='--', alpha=0.7)
    ax.set_yticks(y); ax.set_yticklabels(ns, fontsize=9.5, fontfamily=FONT)
    ax.set_xlabel('Composite Score  Rᵏ ∈ [0,1]', fontsize=10, fontfamily=FONT)
    ax.set_title('Ranked Composite Scores with Tier Classification',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT, pad=8)
    ax.set_xlim(0, 1.12); ax.invert_yaxis()
    ax.grid(axis='x', color='#E0E0E0', lw=0.6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg=[Patch(facecolor=TIER_COLS[t], alpha=0.85, label=f'Tier {t}') for t in sorted(TIER_COLS)]
    leg.append(Line2D([0],[0], color='#444', lw=1, ls='--', label='Tier boundary'))
    ax.legend(handles=leg, fontsize=8.5, loc='lower right', framealpha=0.92)
    plt.tight_layout(); return fig

def plot_conv(all_w, names, w_det, top_n=4):
    n_iter=len(all_w)
    cps=np.unique(np.concatenate([
        np.arange(100,1000,200),
        np.arange(1000,n_iter+1,max(500,n_iter//30))
    ])).astype(int)
    cps=cps[cps<=n_iter]
    idx=np.argsort(w_det)[::-1][:top_n]
    cols=[MED,TEAL,AMBER,PURPLE]
    fig,axes=plt.subplots(2,2,figsize=(11,7),facecolor='white')
    fig.suptitle('Monte Carlo Sensitivity: Weight Standard Deviation Convergence',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT)
    for ax_i,(ci,col) in enumerate(zip(idx,cols)):
        ax=axes.flat[ax_i]; ax.set_facecolor('#F9FAFB')
        rs=[all_w[:cp,ci].std() for cp in cps]
        ax.fill_between(cps,[s*0.88 for s in rs],[s*1.12 for s in rs],color=col,alpha=0.12)
        ax.plot(cps,rs,color=col,lw=2.2)
        ax.axvline(n_iter*0.5,color=GREEN,lw=1.0,ls=':',alpha=0.8)
        ax.axvline(n_iter,    color=RED,  lw=1.0,ls=':',alpha=0.8)
        s50=all_w[:int(n_iter*0.5),ci].std(); s100=all_w[:n_iter,ci].std()
        ax.annotate(f'|σ@50%−σ@100%| = {abs(s50-s100):.5f}',
                    xy=(n_iter*0.72,(s50+s100)/2), fontsize=7.5, ha='center',
                    color=GREEN, fontfamily=FONT,
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',
                              edgecolor=GREEN,lw=0.9,alpha=0.92))
        ax.set_title(f'{names[ci][:32]}  (w={w_det[ci]:.4f})',
                     fontsize=9.5,fontweight='bold',color=DARK,fontfamily=FONT)
        ax.set_xlabel('Iterations (N)',fontsize=8.5,fontfamily=FONT)
        ax.set_ylabel('Running σᵢ',fontsize=8.5,fontfamily=FONT)
        ax.grid(True,color='#E0E0E0',lw=0.6); ax.tick_params(labelsize=8)
    plt.tight_layout(rect=[0,0,1,0.94]); return fig

def plot_stability(all_s, names, det_tiers, breaks, n_iter):
    cps=np.unique(np.concatenate([
        np.arange(100,1000,200),
        np.arange(1000,n_iter+1,max(500,n_iter//30))
    ])).astype(int)
    cps=cps[cps<=n_iter]
    styles=['-','--','-.',':',(0,(3,1,1,1)),'-','--','-.',':',(0,(3,1,1,1)),
            '-','--','-.',':',(0,(3,1,1,1)),'-','--','-.']
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.5),facecolor='white')
    fig.suptitle('Monte Carlo Sensitivity: Tier Assignment Stability',
                 fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT)
    ax1.set_facecolor('#F9FAFB')
    for b in range(len(names)):
        stab=[( assign_tiers(all_s[:cp,b],breaks)==det_tiers[b] ).mean()*100 for cp in cps]
        ax1.plot(cps,stab,color=TIER_COLS.get(det_tiers[b],'#888'),
                 lw=1.5,ls=styles[b%len(styles)],alpha=0.85,label=names[b])
    ax1.axhline(99.5,color=RED,lw=1.8,ls='--',label='99.5% threshold')
    ax1.set_ylim(85,101); ax1.set_xlabel('Iterations (N)',fontsize=9,fontfamily=FONT)
    ax1.set_ylabel('Tier stability (%)',fontsize=9,fontfamily=FONT)
    ax1.set_title('(a) All Alternatives',fontsize=10,fontweight='bold',color=DARK,fontfamily=FONT)
    ax1.grid(True,color='#E0E0E0',lw=0.6); ax1.tick_params(labelsize=8)
    if len(names)<=15:
        ax1.legend(fontsize=6.5,framealpha=0.9,ncol=2,loc='lower right')
    ax2.set_facecolor('#F9FAFB')
    n_show=min(6,len(names)); cp_pts=np.linspace(int(n_iter*0.05),n_iter,5,dtype=int)
    cp_lbls=[f'N={p:,}' for p in cp_pts]; x=np.arange(len(cp_pts)); width=0.12
    for shift,b in enumerate(range(n_show)):
        vals=[(assign_tiers(all_s[:cp,b],breaks)==det_tiers[b]).mean()*100 for cp in cp_pts]
        off=(shift-n_show/2)*width
        ax2.bar(x+off,vals,width*0.88,label=names[b][:18],
                color=TIER_COLS.get(det_tiers[b],'#888'),alpha=0.80,edgecolor='white',lw=0.8)
    ax2.axhline(99.5,color=RED,lw=1.8,ls='--',label='99.5% threshold')
    ax2.set_xticks(x); ax2.set_xticklabels(cp_lbls,fontsize=8.5,fontfamily=FONT)
    ax2.set_ylim(80,101.5); ax2.set_ylabel('Tier stability (%)',fontsize=9,fontfamily=FONT)
    ax2.set_title('(b) Sampled N Checkpoints',fontsize=10,fontweight='bold',color=DARK,fontfamily=FONT)
    ax2.grid(axis='y',color='#E0E0E0',lw=0.6); ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7,framealpha=0.92,loc='lower right',title='Alternative',title_fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.93]); return fig


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT DATA — Canadian Basin Screening
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CRITERIA = [
    "C1  Tectonic stability","C2  Fault & fracture intensity","C3  Evaporites",
    "C4  Reservoir-seal pairs","C5  Leakage via outcrops","C6  Storage capacity",
    "C7  Basin size","C8  Reservoir temperature","C9  Hydrogeological confinement",
    "C10 Depleted reservoir potential","C11 Freshwater constraint",
    "C12 Industry maturity","C13 Onshore / offshore","C14 Accessibility",
    "C15 Infrastructure","C16 CO2 source proximity",
]
DEFAULT_WEIGHTS = np.array([0.041,0.071,0.041,0.092,0.022,0.221,0.041,0.041,
                             0.022,0.071,0.022,0.041,0.041,0.041,0.071,0.123])
DEFAULT_ALTS = ["WCSB","Williston Basin","Michigan (SW Ont.)","NL Offshore",
                "Scotian Basin","Flemish Pass","Beaufort-Mackenzie","Hudson Bay",
                "St. Lawrence","Nova Scotia","Arctic Islands","New Brunswick","Pacific Margin"]
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
]).T  # shape (13 alts, 16 criteria)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

for key, val in [('ran_mc',False),('mc_weights',None),('mc_scores',None),
                 ('final_weights',None),('final_scores',None),('final_tiers',None),
                 ('breaks',None),('score_matrix',None),('cr_ok',True),
                 ('sigma_vals',None),('mu_vals',None),('stability_pcts',None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.radio(
        "Weight input mode",
        ["Quick Mode — enter weights directly",
         "Expert Mode — pairwise comparison matrix"],
        help="Quick Mode: type weights directly and run. Expert Mode: full AHP with automatic consistency check."
    )
    is_expert = "Expert" in mode
    st.markdown("---")
    n_iter = st.select_slider(
        "Monte Carlo iterations (N)",
        options=[1000,2000,5000,10000,20000,50000], value=10000,
        help="How many times to randomly perturb the weights. 10,000 is a good default."
    )
    p_perturb = st.slider(
        "Perturbation probability (p)",
        min_value=0.10, max_value=0.50, value=0.30, step=0.05,
        help="How likely each weight is to be nudged in any given iteration. 0.30 = 30%."
    )
    n_tiers = st.slider(
        "Number of tiers (k)", min_value=2, max_value=6, value=4,
        help="How many priority groups to split your alternatives into."
    )
    st.markdown("---")
    use_example = st.checkbox(
        "Load Canadian Basin Screening example", value=True,
        help="Pre-loads the 13-basin, 16-criterion study by Okwaraojimadu & Ezekiel (2025)."
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.77rem;color:#666;line-height:1.7;">
    <b>Reference</b><br>
    Okwaraojimadu C.K. &amp; Ezekiel C.J. (2025)<br>
    <i>Canadian CO&#x2082; Storage Basin Screening</i><br>
    University of Calgary, MSc Research<br><br>
    <b>Contact</b><br>chisom.okwaraojimadu@ucalgary.ca
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <div class="badge">AHP &middot; MCDA &middot; MONTE CARLO</div>
  <h1>AHP-MCDA Monte Carlo Simulator</h1>
  <p>Derive criterion weights with the Analytic Hierarchy Process &middot;
     Score and rank alternatives &middot;
     Classify priority tiers with Jenks-Fisher optimisation &middot;
     Validate robustness with Monte Carlo simulation.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_how, tab_setup, tab_weights, tab_mc, tab_export, tab_gloss = st.tabs([
    "❓  How to Use",
    "📋  Setup",
    "⚖️  Weights & Scoring",
    "📊  Monte Carlo Results",
    "💾  Export",
    "📖  Glossary",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB: HOW TO USE
# ══════════════════════════════════════════════════════════════════════════════

with tab_how:
    st.markdown('<div class="section-header">Welcome — What Does This App Do?</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    This app helps you <b>rank and compare a list of options</b> (called <i>alternatives</i>) across
    <b>multiple factors</b> (called <i>criteria</i>) — and then tells you <b>how confident you can be</b>
    in those rankings. It was originally built to rank Canadian sedimentary basins for CO&#x2082;
    geological storage, but it works for <i>any</i> ranking or site-selection problem.<br><br>
    You do not need any programming knowledge. Just follow the steps below.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Step-by-Step Guide</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Open the Settings panel",
         "Click the <b>&#9776; menu icon</b> at the very top-left of the screen to open the "
         "Settings panel. If the sidebar is already open, you will see options for mode, "
         "number of iterations, and more. If you accidentally closed it, click the "
         "&#9776; icon again to bring it back."),

        ("2", "Choose your mode",
         "In the Settings panel, pick one of two modes:<br><br>"
         "<b>Quick Mode</b> — You already know how important each factor is (or you want to "
         "type in rough percentage weights). This is the fastest way to get started.<br><br>"
         "<b>Expert Mode</b> — You compare every factor against every other factor, one pair "
         "at a time (e.g. 'Is storage capacity more important than infrastructure?'). The app "
         "then works out the weights mathematically and checks that your answers are "
         "internally consistent. This is the full academic AHP method."),

        ("3", "Try the built-in example",
         "The <b>Load Canadian Basin Screening example</b> checkbox in Settings is ticked by "
         "default. This loads a real published study with 13 Canadian sedimentary basins and "
         "16 evaluation criteria. It is a great way to explore all the features before entering "
         "your own data. Untick the box when you are ready to use your own problem."),

        ("4", "Set up your criteria and alternatives  (Setup tab)",
         "Go to the <b>Setup</b> tab. In the left box, type your criteria — one per line "
         "(e.g. Cost, Environmental Impact, Accessibility). In the right box, type your "
         "alternatives — one per line (e.g. Site A, Site B, Site C).<br><br>"
         "Then fill in the score table at the bottom. Each score must be a number between "
         "<b>0</b> (worst possible performance) and <b>1</b> (best possible performance). "
         "For example, if Site A is perfect on Cost, give it 1.0. If it is average, give it 0.5."),

        ("5", "Enter your weights  (Weights & Scoring tab)",
         "Go to the <b>Weights & Scoring</b> tab.<br><br>"
         "<b>Quick Mode:</b> Type a number for each criterion representing how important it is. "
         "It does not matter if they do not add up to 1 — the app normalises them automatically.<br><br>"
         "<b>Expert Mode:</b> For each pair of criteria, move the slider to say how much more "
         "important one is than the other. The app checks your answers for consistency and "
         "warns you if something does not add up. Once consistent, the weights are calculated "
         "and shown in a chart."),

        ("6", "See your rankings",
         "Still in the Weights & Scoring tab, scroll down to see your alternatives ranked "
         "from highest to lowest composite score, colour-coded by priority tier. The tiers "
         "are determined automatically using a statistical method (Jenks-Fisher) that finds "
         "the natural groupings in your scores — not arbitrary cutoffs."),

        ("7", "Run the Monte Carlo simulation  (Monte Carlo Results tab)",
         "Go to the <b>Monte Carlo Results</b> tab and click <b>Run Monte Carlo</b>. The app "
         "will run your analysis thousands of times, each time slightly varying the weights, "
         "and check whether your rankings stay the same. If they do — your results are robust. "
         "If some alternatives change tier often, that tells you those rankings are sensitive "
         "to the weights you chose.<br><br>"
         "You will see two charts: one showing that the statistics settle down (converge) "
         "well before the end of the simulation, and one showing the percentage of simulations "
         "where each alternative kept its tier. Anything above 99.5% is considered robust."),

        ("8", "Download your results  (Export tab)",
         "Go to the <b>Export</b> tab to download your weights, scores, tier assignments, "
         "Monte Carlo statistics, and figures as CSV files, PNG images, or a JSON file. "
         "These are ready to paste into a paper, a GIS attribute table, or a report."),

        ("9", "Look up unfamiliar terms  (Glossary tab)",
         "If you come across a term you do not recognise — like CR, GVF, eigenvector, or "
         "Jenks-Fisher — click the <b>Glossary</b> tab for plain-English explanations "
         "with examples."),
    ]

    for num, title, body in steps:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">{num}</div>
          <div class="step-content">
            <h4>{title}</h4>
            <p>{body}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Frequently Asked Questions</div>',
                unsafe_allow_html=True)

    faqs = [
        ("Do I need to know anything about AHP or statistics?",
         "No. Quick Mode is designed so that anyone can use it — just type in how important "
         "each factor is and the app does the rest. Expert Mode is there if you want the "
         "full academic method, but it is optional."),
        ("What is a good number of Monte Carlo iterations?",
         "10,000 is a safe default for most problems. The app shows you a convergence chart "
         "that confirms when the results have settled — typically by 3,000 to 5,000 iterations. "
         "Choosing 20,000 or 50,000 gives you extra confidence but takes a bit longer."),
        ("My Consistency Ratio (CR) is above 0.10 — what do I do?",
         "This means some of your pairwise comparisons contradict each other. For example, "
         "if you said A is more important than B, B is more important than C, but C is more "
         "important than A — that is a contradiction. Go back to your comparisons and look "
         "for the ones that feel least certain, and adjust them."),
        ("Can I use this for something other than CO2 storage?",
         "Yes, completely. The app is general purpose. It works for any ranking problem: "
         "site selection, supplier evaluation, policy prioritisation, investment screening, "
         "and more. Just replace the default example with your own criteria and alternatives."),
        ("How do I cite this tool in my paper?",
         "Okwaraojimadu, C.K. & Ezekiel, C.J. (2025). AHP-MCDA Monte Carlo Simulator "
         "[Web application]. University of Calgary. Available at: [your app URL]. "
         "Contact chisom.okwaraojimadu@ucalgary.ca for the source code."),
    ]

    for q, a in faqs:
        with st.expander(f"▸  {q}"):
            st.markdown(f"<p style='font-size:0.9rem;line-height:1.7;color:#333;'>{a}</p>",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: SETUP
# ══════════════════════════════════════════════════════════════════════════════

with tab_setup:
    st.markdown('<div class="section-header">1. Define Your Criteria and Alternatives</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Criteria** — the factors you are evaluating on")
        crit_text = st.text_area(
            "One criterion per line",
            value="\n".join(DEFAULT_CRITERIA) if use_example else
                  "Criterion 1\nCriterion 2\nCriterion 3\nCriterion 4\nCriterion 5",
            height=320, key="crit_input"
        )
    with col2:
        st.markdown("**Alternatives** — the options you are ranking")
        alt_text = st.text_area(
            "One alternative per line",
            value="\n".join(DEFAULT_ALTS) if use_example else
                  "Option A\nOption B\nOption C",
            height=320, key="alt_input"
        )

    criteria_names = [c.strip() for c in crit_text.strip().split("\n") if c.strip()]
    alt_names      = [a.strip() for a in alt_text.strip().split("\n") if a.strip()]
    n_c, n_a = len(criteria_names), len(alt_names)
    st.session_state.criteria_names = criteria_names
    st.session_state.alt_names      = alt_names

    c1,c2,c3 = st.columns(3)
    c1.metric("Criteria defined", n_c)
    c2.metric("Alternatives defined", n_a)
    c3.metric("Pairwise comparisons needed", n_c*(n_c-1)//2 if n_c>1 else 0)

    if n_c < 2: st.error("Please define at least 2 criteria.")
    if n_a < 2: st.error("Please define at least 2 alternatives.")

    st.markdown('<div class="section-header">2. Performance Scores  (0 = worst · 1 = best)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    For each combination of alternative and criterion, enter a number between <b>0</b> (performs
    the worst possible on that criterion) and <b>1</b> (performs the best possible).
    These are your <i>normalised performance scores</i>. If you loaded the example,
    they are pre-filled from the published paper.
    </div>
    """, unsafe_allow_html=True)

    if use_example and n_c==16 and n_a==13:
        score_matrix = DEFAULT_P.copy()
        st.success("Example scores loaded from Okwaraojimadu & Ezekiel (2025).")
        df_prev = pd.DataFrame(score_matrix, index=alt_names,
                               columns=[c[:18] for c in criteria_names])
        st.dataframe(df_prev.style.format("{:.3f}").background_gradient(
            cmap='Blues', axis=None, vmin=0, vmax=1), height=300)
    else:
        if n_c>0 and n_a>0:
            init = {c[:18]:[0.5]*n_a for c in criteria_names}
            df_edit = pd.DataFrame(init, index=alt_names)
            edited  = st.data_editor(df_edit, use_container_width=True,
                                     num_rows="fixed", key="score_editor")
            score_matrix = np.clip(edited.values.astype(float), 0, 1)
        else:
            score_matrix = np.zeros((max(n_a,1), max(n_c,1)))

    st.session_state.score_matrix = score_matrix


# ══════════════════════════════════════════════════════════════════════════════
# TAB: WEIGHTS & SCORING
# ══════════════════════════════════════════════════════════════════════════════

with tab_weights:
    st.markdown('<div class="section-header">Weight Derivation</div>', unsafe_allow_html=True)

    criteria_names = st.session_state.get('criteria_names', DEFAULT_CRITERIA)
    alt_names      = st.session_state.get('alt_names',      DEFAULT_ALTS)
    score_matrix   = st.session_state.get('score_matrix',   DEFAULT_P)
    n_c = len(criteria_names); n_a = len(alt_names)

    if n_c < 2:
        st.warning("Go to the Setup tab first and define at least 2 criteria.")
        st.stop()

    # ── Quick Mode ─────────────────────────────────────────────────────────────
    if not is_expert:
        st.markdown("""
        <div class="info-box">
        <b>Quick Mode:</b> Type a number next to each criterion representing its relative
        importance. The numbers do not need to add to 1 — the app rescales them automatically.
        A higher number means more important. For example: 5 = very important, 1 = not very important.
        </div>
        """, unsafe_allow_html=True)

        def_w = DEFAULT_WEIGHTS.tolist() if (use_example and n_c==16) else [1.0/n_c]*n_c
        w_inputs=[]
        cols_w = st.columns(min(4,n_c))
        for i,cn in enumerate(criteria_names):
            with cols_w[i % len(cols_w)]:
                w=st.number_input(cn[:25], min_value=0.001, max_value=1.0,
                                  value=float(def_w[i]) if i<len(def_w) else 1.0/n_c,
                                  step=0.001, format="%.4f", key=f"wq_{i}")
                w_inputs.append(w)
        raw_w = np.array(w_inputs)
        weights = raw_w / raw_w.sum()
        lambda_max=CI=CR=None; cr_ok=True

    # ── Expert Mode ────────────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div class="info-box">
        <b>Expert Mode — Pairwise Comparison:</b> For each pair of criteria below, move the
        slider to express how much more important the <i>left</i> criterion is compared to the
        <i>right</i> one. Use the Saaty scale: <b>1</b> = equally important,
        <b>3</b> = moderately more important, <b>5</b> = strongly more important,
        <b>7</b> = very strongly, <b>9</b> = extremely. Values below 1 (like 1/3 or 1/5)
        mean the right criterion is more important. The app will check your answers and
        warn you if they are inconsistent.
        </div>
        """, unsafe_allow_html=True)

        n_pairs = n_c*(n_c-1)//2
        if use_example and n_c==16:
            def ns(v):
                opts=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
                return min(opts, key=lambda x: abs(x-v))
            def_upper=[ns(DEFAULT_WEIGHTS[i]/DEFAULT_WEIGHTS[j])
                       for i in range(n_c) for j in range(i+1,n_c)]
        else:
            def_upper=[1.0]*n_pairs

        upper_vals=[]
        if n_pairs<=45:
            pair_lbls=[]
            for i in range(n_c):
                for j in range(i+1,n_c):
                    pair_lbls.append(f"{criteria_names[i][:22]}  vs  {criteria_names[j][:22]}")
            cols_p=st.columns(min(3,n_pairs))
            for k,(lbl,dv) in enumerate(zip(pair_lbls,def_upper)):
                with cols_p[k%len(cols_p)]:
                    v=st.select_slider(lbl,
                        options=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9],
                        value=float(dv),
                        format_func=lambda x: f"1/{int(round(1/x))}" if x<1
                                              else str(int(x)) if x==int(x) else f"{x:.2f}",
                        key=f"pair_{k}")
                    upper_vals.append(v)
        else:
            st.warning(f"With {n_c} criteria there are {n_pairs} pairs. "
                       "Paste comma-separated upper-triangle values below.")
            csv_in=st.text_area("Upper triangle values",
                                value=",".join([str(round(v,4)) for v in def_upper]),height=120)
            try:
                upper_vals=[float(x.strip()) for x in csv_in.split(",")]
                if len(upper_vals)!=n_pairs:
                    st.error(f"Expected {n_pairs} values, got {len(upper_vals)}.")
                    upper_vals=def_upper
            except Exception:
                st.error("Could not parse input."); upper_vals=def_upper

        matrix  = build_matrix(n_c, upper_vals)
        weights, lambda_max, CI, CR = compute_ahp(matrix)
        cr_ok   = CR<=0.10

        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card {"good" if cr_ok else "bad"}">'
                        f'<div class="val">{"✓" if cr_ok else "✗"} {CR:.4f}</div>'
                        f'<div class="lbl">Consistency Ratio (CR)</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="val">{lambda_max:.4f}</div>'
                        f'<div class="lbl">Lambda Max</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="val">{CI:.4f}</div>'
                        f'<div class="lbl">Consistency Index</div></div>', unsafe_allow_html=True)

        if not cr_ok:
            st.markdown("""
            <div class="warn-box">
            <b>Your CR is above 0.10.</b> Some comparisons are contradicting each other.
            Look for pairs where your judgement might be inconsistent and revise them.
            A CR below 0.10 means your comparisons are acceptably consistent.
            </div>""", unsafe_allow_html=True)
        else:
            st.success(f"Consistent (CR = {CR:.4f}). Weights accepted.")

    # ── Weight table + chart ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Your Derived Weights</div>', unsafe_allow_html=True)
    df_w=pd.DataFrame({'Criterion':criteria_names,
                       'Weight (wᵢ)':[f"{w:.4f}" for w in weights],
                       'Share':[f"{w*100:.1f}%" for w in weights]})
    st.dataframe(df_w, use_container_width=True, hide_index=True)
    fig_w=plot_weights(weights, criteria_names)
    st.pyplot(fig_w, use_container_width=True); plt.close(fig_w)

    # ── Composite scores ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Composite Scores and Tier Rankings</div>',
                unsafe_allow_html=True)
    alt_scores = score_matrix @ weights
    if n_a>=n_tiers:
        breaks, gvf = jenks_fisher(alt_scores, n_tiers)
        tiers = assign_tiers(alt_scores, breaks)
    else:
        breaks=np.array([]); gvf=1.0; tiers=np.ones(n_a,dtype=int)

    ca,cb=st.columns(2)
    ca.metric("Jenks-Fisher GVF", f"{gvf:.4f}",
              help="Goodness of Variance Fit. Above 0.90 = excellent grouping.")
    cb.metric("Number of tiers", n_tiers)

    df_s=pd.DataFrame({'Alternative':alt_names,
                       'Score Rᵏ':[f"{s:.4f}" for s in alt_scores],
                       'Tier':tiers,
                       'Rank':pd.Series(alt_scores).rank(ascending=False).astype(int).values
                      }).sort_values('Rank').reset_index(drop=True)
    st.dataframe(df_s, use_container_width=True, hide_index=True)
    if n_a>=2:
        fig_s=plot_scores(np.array(alt_names), alt_scores, tiers, breaks)
        st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)

    # Save to session state
    st.session_state.final_weights=weights; st.session_state.final_scores=alt_scores
    st.session_state.final_tiers=tiers;    st.session_state.breaks=breaks
    st.session_state.score_matrix=score_matrix; st.session_state.cr_ok=cr_ok


# ══════════════════════════════════════════════════════════════════════════════
# TAB: MONTE CARLO RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_mc:
    st.markdown('<div class="section-header">Monte Carlo Weight Uncertainty Analysis</div>',
                unsafe_allow_html=True)

    criteria_names = st.session_state.get('criteria_names', DEFAULT_CRITERIA)
    alt_names      = st.session_state.get('alt_names',      DEFAULT_ALTS)
    weights   = st.session_state.final_weights
    scores    = st.session_state.final_scores
    tiers_det = st.session_state.final_tiers
    breaks    = st.session_state.breaks
    sm        = st.session_state.score_matrix

    st.markdown(f"""
    <div class="info-box">
    The simulation will run <b>{n_iter:,} iterations</b>. In each iteration it slightly
    adjusts your criterion weights at random (with probability {p_perturb}),
    recomputes all the scores, and checks whether the tier assignments change.
    If your rankings stay the same in nearly every iteration, your results are robust.
    If some alternatives switch tiers often, those rankings are sensitive to the weights.
    </div>
    """, unsafe_allow_html=True)

    if weights is None:
        st.warning("Complete the Weights & Scoring tab first.")
    elif is_expert and not st.session_state.cr_ok:
        st.error("CR > 0.10 — fix pairwise comparisons before running the simulation.")
    else:
        if st.button(f"▶  Run Monte Carlo  (N = {n_iter:,})", type="primary"):
            with st.spinner(f"Running {n_iter:,} simulations — please wait..."):
                aw, as_ = run_mc(weights, sm, n_iter, p_perturb)
                st.session_state.mc_weights=aw; st.session_state.mc_scores=as_
                st.session_state.ran_mc=True
            st.success(f"Done! {n_iter:,} iterations completed.")

        if st.session_state.ran_mc and st.session_state.mc_weights is not None:
            aw=st.session_state.mc_weights; as_=st.session_state.mc_scores
            sigma_v=aw.std(axis=0); mu_v=aw.mean(axis=0)

            st.markdown('<div class="section-header">Weight Uncertainty Summary</div>',
                        unsafe_allow_html=True)
            df_mc=pd.DataFrame({'Criterion':criteria_names,
                                'Deterministic Weight':[f"{w:.4f}" for w in weights],
                                'MC Mean':[f"{m:.4f}" for m in mu_v],
                                'MC Std Dev (σᵢ)':[f"{s:.5f}" for s in sigma_v],
                                'Stable?':['Yes' if s<=0.010 else 'Check' for s in sigma_v]})
            st.dataframe(df_mc, use_container_width=True, hide_index=True)

            m1,m2,m3=st.columns(3)
            m1.metric("Maximum σᵢ",  f"{sigma_v.max():.5f}")
            m2.metric("Average σᵢ",  f"{sigma_v.mean():.5f}")
            m3.metric("Iterations",  f"{n_iter:,}")

            st.markdown('<div class="section-header">Tier Stability</div>', unsafe_allow_html=True)
            stab_pcts=[]
            for b in range(n_a):
                t_mc=assign_tiers(as_[:,b], breaks)
                stab_pcts.append((t_mc==tiers_det[b]).mean()*100)

            df_stab=pd.DataFrame({'Alternative':alt_names,
                                  'Score':[f"{s:.4f}" for s in scores],
                                  'Tier':tiers_det,
                                  'Stability (%)':[f"{p:.2f}%" for p in stab_pcts],
                                  'Robust?':['Yes' if p>=99.5 else 'Review' for p in stab_pcts]})
            st.dataframe(df_stab, use_container_width=True, hide_index=True)

            if all(p>=99.5 for p in stab_pcts):
                st.markdown(f"""
                <div class="success-box">
                All {n_a} alternatives retain their tier in at least 99.5% of {n_iter:,}
                simulations. Your tier classification is robust.
                </div>""", unsafe_allow_html=True)
            else:
                n_w=sum(p<99.5 for p in stab_pcts)
                st.markdown(f"""
                <div class="warn-box">
                {n_w} alternative(s) fall below 99.5% stability. Their tier assignment is
                sensitive to weight variation. Consider reviewing those criteria weights or
                whether the tier boundary is in the right place.
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Convergence Figures</div>',
                        unsafe_allow_html=True)
            fig_cv=plot_conv(aw, criteria_names, weights, top_n=min(4,n_c))
            st.pyplot(fig_cv, use_container_width=True); plt.close(fig_cv)
            fig_st=plot_stability(as_, alt_names, tiers_det, breaks, n_iter)
            st.pyplot(fig_st, use_container_width=True); plt.close(fig_st)

            st.session_state.sigma_vals=sigma_v; st.session_state.mu_vals=mu_v
            st.session_state.stability_pcts=stab_pcts


# ══════════════════════════════════════════════════════════════════════════════
# TAB: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.markdown('<div class="section-header">Download Your Results</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    All downloads are available below. CSV files open in Excel or Google Sheets.
    PNG files are high-resolution images ready for papers or presentations.
    JSON contains everything in one file for use in GIS or programming workflows.
    </div>
    """, unsafe_allow_html=True)

    w=st.session_state.final_weights; s=st.session_state.final_scores
    t=st.session_state.final_tiers;   brk=st.session_state.breaks
    cn=st.session_state.get('criteria_names',DEFAULT_CRITERIA)
    an=st.session_state.get('alt_names',     DEFAULT_ALTS)

    if w is None:
        st.info("Complete the Setup and Weights tabs to enable downloads.")
    else:
        c1,c2=st.columns(2)
        with c1:
            df_we=pd.DataFrame({'Criterion':cn,'Weight':w,'Percentage':w*100})
            st.download_button("📥  Weights CSV", df_we.to_csv(index=False),
                               "ahp_weights.csv","text/csv",use_container_width=True)
        with c2:
            df_se=pd.DataFrame({'Alternative':an,'Score':s,'Tier':t,
                                'Rank':pd.Series(s).rank(ascending=False).astype(int).values
                               }).sort_values('Rank').reset_index(drop=True)
            st.download_button("📥  Scores & Tiers CSV", df_se.to_csv(index=False),
                               "ahp_scores.csv","text/csv",use_container_width=True)

        if st.session_state.ran_mc and st.session_state.sigma_vals is not None:
            c3,c4=st.columns(2)
            with c3:
                df_mce=pd.DataFrame({'Criterion':cn,'Det_Weight':w,
                                     'MC_Mean':st.session_state.mu_vals,
                                     'MC_StdDev':st.session_state.sigma_vals})
                st.download_button("📥  MC Weight Stats CSV",df_mce.to_csv(index=False),
                                   "mc_weights.csv","text/csv",use_container_width=True)
            with c4:
                df_ste=pd.DataFrame({'Alternative':an,'Score':s,'Tier':t,
                                     'Stability_pct':st.session_state.stability_pcts})
                st.download_button("📥  Tier Stability CSV",df_ste.to_csv(index=False),
                                   "tier_stability.csv","text/csv",use_container_width=True)

        st.markdown("**Figures**")
        cf1,cf2=st.columns(2)
        with cf1:
            fw=plot_weights(w,cn)
            st.download_button("📥  Weights Figure PNG",fig_to_bytes(fw),
                               "weights.png","image/png",use_container_width=True)
        with cf2:
            if s is not None and len(s)>=2:
                fs=plot_scores(np.array(an),s,t,brk)
                st.download_button("📥  Scores Figure PNG",fig_to_bytes(fs),
                                   "scores.png","image/png",use_container_width=True)

        if st.session_state.ran_mc:
            aw=st.session_state.mc_weights; as_=st.session_state.mc_scores
            cf3,cf4=st.columns(2)
            with cf3:
                fc=plot_conv(aw,cn,w,top_n=min(4,len(cn)))
                st.download_button("📥  Convergence Figure PNG",fig_to_bytes(fc),
                                   "convergence.png","image/png",use_container_width=True)
            with cf4:
                fst=plot_stability(as_,an,t,brk,n_iter)
                st.download_button("📥  Stability Figure PNG",fig_to_bytes(fst),
                                   "stability.png","image/png",use_container_width=True)

        st.markdown("**Full Results JSON**")
        rd={'criteria':cn,'alternatives':an,
            'weights':{c:float(ww) for c,ww in zip(cn,w)},
            'scores':{a:float(ss) for a,ss in zip(an,s)},
            'tiers':{a:int(tt) for a,tt in zip(an,t)},
            'tier_breaks':[float(b) for b in brk],
            'n_iterations':n_iter,'perturbation_probability':p_perturb}
        if st.session_state.sigma_vals is not None:
            rd['mc_sigma']={c:float(sig) for c,sig in zip(cn,st.session_state.sigma_vals)}
            rd['mc_stability']={a:float(p) for a,p in zip(an,st.session_state.stability_pcts)}
        st.download_button("📥  Full Results JSON",json.dumps(rd,indent=2),
                           "ahp_mc_results.json","application/json",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: GLOSSARY
# ══════════════════════════════════════════════════════════════════════════════

with tab_gloss:
    st.markdown('<div class="section-header">Glossary of Terms</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Plain-English explanations of every technical term used in this app.
    Terms are grouped by topic.
    </div>
    """, unsafe_allow_html=True)

    sections = {
        "The Basics": [
            ("Criterion (plural: criteria)",
             "A factor or characteristic you use to evaluate your options. Think of it as a "
             "question you are asking about each option, like 'How costly is it?' or "
             "'How close is it to an emission source?'",
             "Example: In a CO2 storage study, criteria include storage capacity, "
             "tectonic stability, and proximity to CO2 sources."),

            ("Alternative",
             "One of the options you are comparing and ranking. Also called a candidate, "
             "site, or option depending on the field.",
             "Example: In a basin screening study, the alternatives are the sedimentary "
             "basins being evaluated (e.g. WCSB, Williston Basin, Michigan Basin)."),

            ("Performance score  (Pᵢₖ)",
             "A number between 0 and 1 that says how well a particular alternative performs "
             "on a particular criterion. 1 means it performs as well as possible; 0 means it "
             "performs as badly as possible.",
             "Example: If WCSB has very large storage capacity (the best possible class), "
             "its performance score on the storage capacity criterion is 1.0."),

            ("Composite score  (Rᵏ)",
             "The final overall score for each alternative, calculated by multiplying each "
             "performance score by its criterion weight and adding everything up. "
             "Higher is better. The formula is: Rᵏ = sum of (weight × score) for all criteria.",
             "Example: WCSB scores 0.982 out of 1.0, meaning it performs near-optimally "
             "across all 16 criteria when weighted by their importance."),

            ("Weight  (wᵢ)",
             "A number that represents how important a criterion is relative to the others. "
             "All weights add up to 1 (or 100%). A weight of 0.221 means that criterion "
             "accounts for 22.1% of the overall score.",
             "Example: Storage capacity has the highest weight (0.221) because it is the "
             "most important factor for CO2 storage site selection."),

            ("Tier",
             "A priority group. Alternatives are sorted into tiers based on their composite "
             "scores: Tier 1 = highest priority, Tier 2 = secondary, and so on. The number "
             "of tiers (and where the boundaries fall) is determined by the Jenks-Fisher method.",
             "Example: WCSB and Williston Basin are in Tier 1 (Priority). Michigan Basin "
             "is in Tier 2 (Secondary)."),
        ],

        "AHP — Analytic Hierarchy Process": [
            ("AHP  (Analytic Hierarchy Process)",
             "A structured method for working out how important each criterion is, developed "
             "by mathematician Thomas Saaty in the 1970s. Instead of guessing weights, you "
             "compare every criterion against every other one — one pair at a time — and the "
             "app calculates the weights mathematically from your answers.",
             "Example: You say storage capacity is 'strongly more important' than accessibility. "
             "AHP turns this judgement into precise numerical weights."),

            ("Pairwise comparison",
             "Asking 'How much more important is criterion A compared to criterion B?' for "
             "every possible pair of criteria. If you have 16 criteria, you need to answer "
             "120 such questions. The answers form the pairwise comparison matrix.",
             "Example: Comparing C6 (storage capacity) vs C15 (infrastructure): "
             "you might say capacity is 5x more important (strongly more important)."),

            ("Saaty scale  (1 to 9)",
             "The scale used to answer pairwise comparison questions. "
             "1 = equally important, 3 = moderately more important, "
             "5 = strongly more important, 7 = very strongly, 9 = extremely. "
             "Values like 1/3 or 1/5 mean the other criterion is more important.",
             ""),

            ("Eigenvector method",
             "The mathematical technique AHP uses to turn the pairwise comparison matrix "
             "into weights. The app does this automatically. You do not need to understand "
             "the mathematics — just know that it produces the most consistent weights "
             "possible given your comparisons.",
             ""),

            ("Lambda max  (λmax)",
             "A number the app calculates to check your comparisons. For a perfectly "
             "consistent set of answers, λmax equals the number of criteria (n). "
             "The further λmax is from n, the more inconsistent your comparisons are.",
             "Example: With 16 criteria, a perfect λmax = 16.000. "
             "This study achieved λmax = 16.154, which is very close."),

            ("Consistency Index  (CI)",
             "A measure derived from λmax that quantifies how inconsistent your pairwise "
             "comparisons are. Smaller is better. CI = (λmax - n) / (n - 1).",
             ""),

            ("Random Index  (RI)",
             "A reference value that depends only on the number of criteria. It represents "
             "the average CI you would get from completely random comparisons. The app looks "
             "this up automatically from a published table (Saaty, 1980).",
             "Example: For 16 criteria, RI = 1.59."),

            ("Consistency Ratio  (CR)",
             "The key consistency check. CR = CI / RI. If CR is below 0.10 (10%), your "
             "comparisons are considered acceptably consistent and the weights are valid. "
             "If CR is above 0.10, some comparisons are contradicting each other and you "
             "need to revise them.",
             "Example: This study achieved CR = 0.0064, which is far below 0.10 and "
             "indicates excellent consistency."),
        ],

        "Monte Carlo Simulation": [
            ("Monte Carlo simulation",
             "A technique that runs your analysis thousands of times, each time with slightly "
             "different (randomly varied) inputs, to see how much the results change. "
             "Named after the Monte Carlo casino because it involves randomness. "
             "In this app, we vary the criterion weights slightly in each run.",
             "Example: If your weights change by a tiny random amount in each of 10,000 "
             "runs, and the WCSB is still in Tier 1 in 100% of those runs, you can be "
             "very confident that it truly belongs in Tier 1."),

            ("Iteration",
             "One run of the simulation with one set of randomly perturbed weights. "
             "The more iterations you run, the more reliable your statistics.",
             "Example: N = 10,000 means the simulation runs 10,000 times."),

            ("Perturbation",
             "A small random change applied to a weight in each iteration. "
             "The perturbation is designed to mimic the uncertainty in your judgements — "
             "similar to what would happen if you varied your pairwise comparisons by "
             "one step on the Saaty scale.",
             ""),

            ("Perturbation probability  (p)",
             "The chance that any given weight is perturbed in a particular iteration. "
             "p = 0.30 means each weight has a 30% chance of being nudged. "
             "The other 70% of the time it stays at its original value.",
             ""),

            ("Weight standard deviation  (σᵢ)",
             "After running all iterations, this measures how much a particular weight "
             "varied across the simulations. A very small σᵢ (like 0.005) means that "
             "even with perturbation, the weight barely changes — so it is stable.",
             "Example: The storage capacity weight (C6) has σ = 0.006, meaning it "
             "fluctuated by about 0.006 on average across 10,000 simulations."),

            ("Convergence",
             "When the running statistics (like σᵢ) stop changing as you add more "
             "iterations. If the statistics are the same at N = 10,000 as at N = 20,000, "
             "the simulation has converged — more iterations would not tell you anything new.",
             ""),

            ("Tier stability",
             "The percentage of Monte Carlo iterations in which an alternative stays in "
             "its deterministic (original) tier. 100% stability means it never changed tiers "
             "across all simulations. Anything above 99.5% is considered robust.",
             "Example: WCSB has 100% tier stability — it is in Tier 1 in every one of "
             "the 10,000 simulations."),
        ],

        "Tier Classification": [
            ("Jenks-Fisher method  (also called Natural Breaks)",
             "A statistical algorithm that finds the best places to draw the boundaries "
             "between tiers, based on the natural gaps in your score data. It maximises the "
             "difference between tiers while minimising the difference within each tier. "
             "This means the boundaries are not arbitrary — they reflect where the real "
             "gaps are in your scores.",
             "Example: The scores 0.982, 0.847 (Tier 1) and 0.720, 0.627, 0.572 (Tier 2) "
             "have a natural gap between them, which Jenks-Fisher correctly identifies."),

            ("Goodness of Variance Fit  (GVF)",
             "A number between 0 and 1 that tells you how well the tier boundaries fit the "
             "data. GVF = 1 would mean perfect classification. GVF above 0.90 is considered "
             "excellent. It is calculated as: GVF = 1 minus (within-class variance / total variance).",
             "Example: This study achieved GVF = 0.957 with 4 tiers, which is excellent — "
             "meaning the 4-tier classification captures 95.7% of the variance in scores."),

            ("SDCM  (Sum of Squared Deviations from Class Means)",
             "The within-class variance — how spread out the scores are within each tier. "
             "The Jenks-Fisher algorithm minimises this number to find the best tier boundaries.",
             ""),

            ("SDAM  (Sum of Squared Deviations from the Array Mean)",
             "The total variance in all scores, regardless of tier. Used to compute GVF.",
             ""),
        ],
    }

    for section_title, terms in sections.items():
        st.markdown(f'<div class="section-header">{section_title}</div>', unsafe_allow_html=True)
        for term, definition, example in terms:
            ex_html = (f'<p class="gloss-example">&#9656; {example}</p>' if example else '')
            st.markdown(f"""
            <div class="gloss-card">
              <div class="gloss-term">{term}</div>
              <p class="gloss-def">{definition}</p>
              {ex_html}
            </div>
            """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  AHP-MCDA Monte Carlo Simulator &middot; Python 3 &middot; NumPy &middot;
  Matplotlib &middot; Streamlit &middot;
  Okwaraojimadu C.K. &amp; Ezekiel C.J., University of Calgary, 2025 &middot;
  chisom.okwaraojimadu@ucalgary.ca
</div>
""", unsafe_allow_html=True)
