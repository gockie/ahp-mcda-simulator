import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import json
import re

st.set_page_config(
    page_title="AHP-MCDA Monte Carlo Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:  #1B3A5C; --teal: #0E7C7B; --blue: #2E75B6;
    --amber: #C47A00; --green:#2E8B57; --red:  #B22222;
    --border:#D0DAE8; --text: #1A1A2E;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text); }
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}

/* hide the native sidebar toggle arrow */
[data-testid="collapsedControl"]{ display:none !important; }

.app-header{
    background:linear-gradient(135deg,#1B3A5C 0%,#0E4D7B 50%,#0E7C7B 100%);
    padding:2.2rem 2rem 1.8rem; border-radius:16px; margin-bottom:1.4rem;
    color:white; position:relative; overflow:hidden;
}
.app-header::before{
    content:''; position:absolute; top:-50%; right:-10%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(14,124,123,.3) 0%,transparent 70%);
    pointer-events:none;
}
.app-header .badge{
    display:inline-block; background:rgba(255,255,255,.15);
    border:1px solid rgba(255,255,255,.3); border-radius:20px;
    padding:.2rem .8rem; font-size:.72rem; font-family:'DM Mono',monospace;
    letter-spacing:1px; margin-bottom:.9rem; color:rgba(255,255,255,.9);
}
.app-header h1{
    font-family:'DM Serif Display',serif; font-size:2.1rem;
    font-weight:400; margin:0 0 .45rem; color:white;
}
.app-header p{ font-size:.95rem; opacity:.85; margin:0; font-weight:300; line-height:1.6; }

.section-header{
    font-family:'DM Serif Display',serif; font-size:1.3rem; color:var(--navy);
    border-bottom:2px solid var(--border); padding-bottom:.4rem; margin:1.4rem 0 .9rem;
}
.info-box{
    background:#EFF6FF; border-left:4px solid var(--blue);
    border-radius:0 8px 8px 0; padding:.75rem 1rem;
    font-size:.87rem; line-height:1.65; margin:.7rem 0; color:#1e3a5f;
}
.warn-box{
    background:#FFFBF0; border-left:4px solid var(--amber);
    border-radius:0 8px 8px 0; padding:.75rem 1rem;
    font-size:.87rem; line-height:1.65; margin:.7rem 0; color:#5a3a00;
}
.success-box{
    background:#F0FFF4; border-left:4px solid var(--green);
    border-radius:0 8px 8px 0; padding:.75rem 1rem;
    font-size:.87rem; line-height:1.65; margin:.7rem 0; color:#1a4a2a;
}
.settings-bar{
    background:#F4F7FB; border:1px solid var(--border); border-radius:12px;
    padding:1rem 1.2rem; margin-bottom:1.2rem;
}
.step-card{
    background:white; border:1.5px solid var(--border); border-radius:14px;
    padding:1.2rem 1.4rem; margin-bottom:1rem;
    display:flex; gap:1.1rem; align-items:flex-start;
}
.step-num{
    background:linear-gradient(135deg,#1B3A5C,#0E7C7B); color:white;
    font-family:'DM Serif Display',serif; font-size:1.2rem;
    width:44px; height:44px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; box-shadow:0 2px 8px rgba(27,58,92,.25);
}
.step-content h4{
    font-family:'DM Serif Display',serif; color:var(--navy);
    margin:0 0 .35rem; font-size:1rem;
}
.step-content p{ margin:0; font-size:.86rem; color:#444; line-height:1.65; }
.gloss-card{
    background:white; border:1px solid var(--border);
    border-radius:10px; padding:.9rem 1.1rem; margin-bottom:.65rem;
}
.gloss-term{ font-family:'DM Mono',monospace; font-size:.9rem; font-weight:600; color:var(--navy); margin-bottom:.25rem; }
.gloss-def{ font-size:.84rem; color:#444; line-height:1.65; margin:0; }
.gloss-ex{ font-size:.79rem; color:var(--teal); font-style:italic; margin-top:.25rem; }
.metric-card{
    background:white; border:1px solid var(--border); border-radius:10px;
    padding:.9rem 1.1rem; text-align:center;
}
.metric-card .val{ font-family:'DM Mono',monospace; font-size:1.6rem; font-weight:500; color:var(--navy); line-height:1; }
.metric-card .lbl{ font-size:.74rem; color:#777; margin-top:.25rem; font-weight:500; letter-spacing:.5px; text-transform:uppercase; }
.metric-card.good .val{ color:var(--green); }
.metric-card.bad  .val{ color:var(--red); }
.app-footer{
    text-align:center; padding:1.8rem 0 1rem; font-size:.77rem; color:#999;
    border-top:1px solid var(--border); margin-top:2.5rem;
    font-family:'DM Mono',monospace;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

SAATY_RI = {1:0,2:0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,
            9:1.45,10:1.49,11:1.51,12:1.54,13:1.56,14:1.57,
            15:1.58,16:1.59,17:1.60,18:1.61,19:1.62,20:1.63}

def compute_ahp(M):
    n=M.shape[0]; cs=M.sum(axis=0); w=(M/cs).mean(axis=1)
    lam=(M@w)/w; lmax=lam.mean()
    CI=(lmax-n)/(n-1) if n>1 else 0
    RI=SAATY_RI.get(n,1.63)
    return w, lmax, CI, CI/RI if RI>0 else 0

def build_matrix(n, upper):
    M=np.eye(n); idx=0
    for i in range(n):
        for j in range(i+1,n):
            M[i,j]=upper[idx]; M[j,i]=1/upper[idx]; idx+=1
    return M

def jenks(values, k):
    n=len(values); sv=np.sort(values)
    if k>=n: return sv[:-1],1.0
    SSW=np.full((k,n),np.inf); BI=np.zeros((k,n),dtype=int)
    for i in range(n):
        v=sv[:i+1]; SSW[0,i]=np.var(v)*len(v)
    for cl in range(1,k):
        for i in range(cl,n):
            best,bb=np.inf,cl
            for m in range(cl-1,i):
                vr=sv[m+1:i+1]
                tot=SSW[cl-1,m]+(np.var(vr)*len(vr) if len(vr)>0 else 0)
                if tot<best: best=tot; bb=m+1
            SSW[cl,i]=best; BI[cl,i]=bb
    idx=n-1; bks=[]
    for cl in range(k-1,0,-1):
        bi=BI[cl,idx]; bks.append(bi); idx=bi-1
    bv=sv[sorted(bks)]; sdam=np.var(sv)*n
    return bv, 1-SSW[k-1,n-1]/sdam if sdam>0 else 1.0

def assign_tiers(scores, breaks):
    sb=np.sort(breaks)[::-1]
    out=[]
    for s in scores:
        t=len(sb)+1
        for i,b in enumerate(sb):
            if s>=b: t=i+1; break
        out.append(t)
    return np.array(out)

def run_mc(weights, P, n_iter, p):
    np.random.seed(42); n_c=len(weights)
    sig=np.maximum(weights*0.03,0.002)
    aw=np.zeros((n_iter,n_c)); as_=np.zeros((n_iter,P.shape[0]))
    for s in range(n_iter):
        noise=np.random.normal(0,sig); mask=np.random.random(n_c)<p
        wp=np.maximum(weights+noise*mask,0.005); wp/=wp.sum()
        aw[s]=wp; as_[s]=P@wp
    return aw, as_


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

FONT='DejaVu Sans'; DARK='#1B3A5C'; MED='#2E75B6'; TEAL='#0E7C7B'
AMBER='#C47A00'; GREEN='#2E8B57'; RED='#B22222'; PURPLE='#6A4C93'
TC={1:'#2E8B57',2:'#C47A00',3:'#8B4500',4:'#555555'}

def fig2bytes(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=150,bbox_inches='tight',facecolor='white')
    buf.seek(0); plt.close(fig); return buf.getvalue()

def plot_weights(w, names):
    fig,ax=plt.subplots(figsize=(9,max(4,len(names)*.42)),facecolor='white')
    ax.set_facecolor('#F9FAFB'); n=len(w); y=np.arange(n)
    idx=np.argsort(w); ws=w[idx]; ns=[names[i] for i in idx]
    cols=[plt.cm.Blues(.4+.6*ww/max(w)) for ww in ws]
    bars=ax.barh(y,ws,color=cols,edgecolor='white',lw=.8,height=.7)
    for bar,ww in zip(bars,ws):
        ax.text(bar.get_width()+.002,bar.get_y()+bar.get_height()/2,
                f'{ww:.4f}',va='center',ha='left',fontsize=9,
                fontfamily=FONT,color=DARK,fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels(ns,fontsize=9,fontfamily=FONT)
    ax.set_xlabel('AHP Weight  wᵢ',fontsize=10,fontfamily=FONT)
    ax.set_title('Derived Criterion Weights',fontsize=11,fontweight='bold',
                 color=DARK,fontfamily=FONT,pad=8)
    ax.set_xlim(0,max(w)*1.25); ax.grid(axis='x',color='#E0E0E0',lw=.6)
    plt.tight_layout(); return fig

def plot_scores(names, scores, tiers, breaks):
    fig,ax=plt.subplots(figsize=(9,max(4,len(names)*.55)),facecolor='white')
    ax.set_facecolor('#F9FAFB')
    idx=np.argsort(scores)[::-1]; ss=scores[idx]
    ns=[names[i] for i in idx]; ts=tiers[idx]
    y=np.arange(len(names)); cols=[TC.get(t,'#888') for t in ts]
    ax.barh(y,ss,color=cols,edgecolor='white',lw=.8,height=.68,alpha=.85)
    for i,(s,t) in enumerate(zip(ss,ts)):
        ax.text(s+.008,i,f'{s:.3f}',va='center',ha='left',fontsize=9.5,
                fontfamily=FONT,color=TC.get(t,'#888'),fontweight='bold')
    for b in breaks: ax.axvline(b,color='#444',lw=1.,ls='--',alpha=.7)
    ax.set_yticks(y); ax.set_yticklabels(ns,fontsize=9.5,fontfamily=FONT)
    ax.set_xlabel('Composite Score  Rᵏ ∈ [0,1]',fontsize=10,fontfamily=FONT)
    ax.set_title('Ranked Composite Scores with Tier Classification',
                 fontsize=11,fontweight='bold',color=DARK,fontfamily=FONT,pad=8)
    ax.set_xlim(0,1.12); ax.invert_yaxis(); ax.grid(axis='x',color='#E0E0E0',lw=.6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg=[Patch(facecolor=TC[t],alpha=.85,label=f'Tier {t}') for t in sorted(TC)]
    leg.append(Line2D([0],[0],color='#444',lw=1,ls='--',label='Tier boundary'))
    ax.legend(handles=leg,fontsize=8.5,loc='lower right',framealpha=.92)
    plt.tight_layout(); return fig

def plot_conv(aw, names, w_det, top_n=4):
    n_iter=len(aw)
    cps=np.unique(np.concatenate([np.arange(100,1000,200),
        np.arange(1000,n_iter+1,max(500,n_iter//30))])).astype(int)
    cps=cps[cps<=n_iter]
    idx=np.argsort(w_det)[::-1][:top_n]
    cols=[MED,TEAL,AMBER,PURPLE]
    fig,axes=plt.subplots(2,2,figsize=(11,7),facecolor='white')
    fig.suptitle('Monte Carlo Sensitivity: Weight Std Dev Convergence',
                 fontsize=11,fontweight='bold',color=DARK,fontfamily=FONT)
    for ax_i,(ci,col) in enumerate(zip(idx,cols)):
        ax=axes.flat[ax_i]; ax.set_facecolor('#F9FAFB')
        rs=[aw[:cp,ci].std() for cp in cps]
        ax.fill_between(cps,[s*.88 for s in rs],[s*1.12 for s in rs],color=col,alpha=.12)
        ax.plot(cps,rs,color=col,lw=2.2)
        ax.axvline(n_iter*.5,color=GREEN,lw=1.,ls=':',alpha=.8)
        ax.axvline(n_iter,   color=RED,  lw=1.,ls=':',alpha=.8)
        s50=aw[:int(n_iter*.5),ci].std(); s100=aw[:n_iter,ci].std()
        ax.annotate(f'|σ@50%−σ@100%| = {abs(s50-s100):.5f}',
                    xy=(n_iter*.72,(s50+s100)/2),fontsize=7.5,ha='center',
                    color=GREEN,fontfamily=FONT,
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',
                              edgecolor=GREEN,lw=.9,alpha=.92))
        ax.set_title(f'{names[ci][:32]}  (w={w_det[ci]:.4f})',
                     fontsize=9.5,fontweight='bold',color=DARK,fontfamily=FONT)
        ax.set_xlabel('Iterations (N)',fontsize=8.5,fontfamily=FONT)
        ax.set_ylabel('Running σᵢ',fontsize=8.5,fontfamily=FONT)
        ax.grid(True,color='#E0E0E0',lw=.6); ax.tick_params(labelsize=8)
    plt.tight_layout(rect=[0,0,1,.94]); return fig

def plot_stability(as_, names, det_tiers, breaks, n_iter):
    cps=np.unique(np.concatenate([np.arange(100,1000,200),
        np.arange(1000,n_iter+1,max(500,n_iter//30))])).astype(int)
    cps=cps[cps<=n_iter]
    styles=['-','--','-.',':', (0,(3,1,1,1)),'-','--','-.',':', (0,(3,1,1,1)),
            '-','--','-.',':', (0,(3,1,1,1)),'-','--','-.']
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.5),facecolor='white')
    fig.suptitle('Monte Carlo Sensitivity: Tier Assignment Stability',
                 fontsize=11,fontweight='bold',color=DARK,fontfamily=FONT)
    ax1.set_facecolor('#F9FAFB')
    for b in range(len(names)):
        stab=[(assign_tiers(as_[:cp,b],breaks)==det_tiers[b]).mean()*100 for cp in cps]
        ax1.plot(cps,stab,color=TC.get(det_tiers[b],'#888'),
                 lw=1.5,ls=styles[b%len(styles)],alpha=.85,label=names[b])
    ax1.axhline(99.5,color=RED,lw=1.8,ls='--',label='99.5% threshold')
    ax1.set_ylim(85,101); ax1.set_xlabel('Iterations (N)',fontsize=9,fontfamily=FONT)
    ax1.set_ylabel('Tier stability (%)',fontsize=9,fontfamily=FONT)
    ax1.set_title('(a) All Alternatives',fontsize=10,fontweight='bold',color=DARK,fontfamily=FONT)
    ax1.grid(True,color='#E0E0E0',lw=.6); ax1.tick_params(labelsize=8)
    if len(names)<=15:
        ax1.legend(fontsize=6.5,framealpha=.9,ncol=2,loc='lower right')
    ax2.set_facecolor('#F9FAFB')
    n_show=min(6,len(names))
    cp_pts=np.linspace(int(n_iter*.05),n_iter,5,dtype=int)
    cp_lbls=[f'N={p:,}' for p in cp_pts]; x=np.arange(len(cp_pts)); width=.12
    for shift,b in enumerate(range(n_show)):
        vals=[(assign_tiers(as_[:cp,b],breaks)==det_tiers[b]).mean()*100 for cp in cp_pts]
        ax2.bar(x+(shift-n_show/2)*width,vals,width*.88,label=names[b][:18],
                color=TC.get(det_tiers[b],'#888'),alpha=.80,edgecolor='white',lw=.8)
    ax2.axhline(99.5,color=RED,lw=1.8,ls='--',label='99.5% threshold')
    ax2.set_xticks(x); ax2.set_xticklabels(cp_lbls,fontsize=8.5,fontfamily=FONT)
    ax2.set_ylim(80,101.5); ax2.set_ylabel('Tier stability (%)',fontsize=9,fontfamily=FONT)
    ax2.set_title('(b) Sampled N Checkpoints',fontsize=10,fontweight='bold',color=DARK,fontfamily=FONT)
    ax2.grid(axis='y',color='#E0E0E0',lw=.6); ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7,framealpha=.92,loc='lower right',
               title='Alternative',title_fontsize=7)
    plt.tight_layout(rect=[0,0,1,.93]); return fig


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT DATA
# ══════════════════════════════════════════════════════════════════════════════

DEF_CRITERIA = [
    "C1  Tectonic stability","C2  Fault & fracture intensity","C3  Evaporites",
    "C4  Reservoir-seal pairs","C5  Leakage via outcrops","C6  Storage capacity",
    "C7  Basin size","C8  Reservoir temperature","C9  Hydrogeological confinement",
    "C10 Depleted reservoir potential","C11 Freshwater constraint",
    "C12 Industry maturity","C13 Onshore / offshore","C14 Accessibility",
    "C15 Infrastructure","C16 CO2 source proximity",
]
DEF_W = np.array([.041,.071,.041,.092,.022,.221,.041,.041,
                   .022,.071,.022,.041,.041,.041,.071,.123])
DEF_ALTS = ["WCSB","Williston Basin","Michigan (SW Ont.)","NL Offshore",
            "Scotian Basin","Flemish Pass","Beaufort-Mackenzie","Hudson Bay",
            "St. Lawrence","Nova Scotia","Arctic Islands","New Brunswick","Pacific Margin"]
DEF_P = np.array([
    [1.,.071/DEF_W[0] if DEF_W[0]>0 else 1,1.,0.429,0.429,0.429,1.,1.,0.429,1.,1.,1.,0.],
    [1.,1.,1.,0.333,0.333,0.333,0.333,1.,0.333,0.333,1.,0.333,0.],
    [1.,1.,1.,0.,0.5,0.,0.5,1.,0.,0.5,1.,0.,0.],
    [1.,0.7,0.7,0.7,0.7,0.7,0.3,0.3,0.3,0.3,0.1,0.1,0.1],
    [1.,1.,1.,0.75,0.75,0.75,0.5,1.,0.5,0.5,0.5,0.25,0.25],
    [1.,0.7,0.3,0.7,0.7,0.7,0.7,0.7,0.1,0.1,0.3,0.1,0.1],
    [1.,1.,0.7,1.,1.,0.7,1.,1.,0.1,0.1,1.,0.,0.7],
    [1.,0.429,0.143,1.,1.,1.,1.,0.143,0.143,0.,0.,0.,0.429],
    [1.,1.,1.,0.5,0.5,0.5,0.5,0.5,0.5,0.,1.,0.,0.],
    [1.,0.7,1.,0.7,0.2,0.1,0.2,0.,0.1,0.1,0.2,0.,0.],
    [1.,1.,0.667,1.,1.,1.,0.333,0.333,0.667,0.667,0.333,0.333,0.667],
    [1.,0.778,0.778,0.778,0.778,0.333,0.333,0.,0.333,0.333,0.,0.111,0.111],
    [1.,1.,1.,0.556,0.,0.,1.,0.556,1.,1.,1.,1.,0.556],
    [1.,1.,1.,0.556,0.556,0.222,0.222,0.222,1.,1.,0.,1.,0.556],
    [1.,1.,1.,0.667,0.667,0.222,0.222,0.,0.667,0.667,0.,0.222,0.222],
    [1.,1.,1.,0.429,0.143,0.143,0.143,0.,1.,0.429,0.,0.143,0.429],
]).T  # (13 alts, 16 criteria)
# Fix row 0 (tectonic) - simple correction
DEF_P[:,0] = [1.,1.,1.,0.429,0.429,0.429,1.,1.,0.429,1.,1.,1.,0.]


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for k,v in [('ran_mc',False),('mc_w',None),('mc_s',None),
            ('weights',None),('scores',None),('tiers',None),
            ('breaks',None),('P',None),('cr_ok',True),
            ('sigma',None),('mu',None),('stab',None)]:
    if k not in st.session_state: st.session_state[k]=v


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <div class="badge">AHP &middot; MCDA &middot; MONTE CARLO</div>
  <h1>AHP-MCDA Monte Carlo Simulator</h1>
  <p>Derive criterion weights &middot; Score and rank alternatives &middot;
  Classify priority tiers with Jenks-Fisher &middot;
  Validate robustness with Monte Carlo simulation.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS BAR  (replaces sidebar — always visible at top)
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("⚙️  Settings — click to open / close", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([2,1,1,1,1])
    with c1:
        mode = st.radio("Weight input mode",
            ["Quick Mode — enter weights directly",
             "Expert Mode — pairwise comparison matrix"],
            horizontal=False,
            help="Quick: type weights directly. Expert: full AHP pairwise comparison with consistency check.")
    with c2:
        n_iter = st.select_slider("Monte Carlo iterations (N)",
            options=[1000,2000,5000,10000,20000,50000], value=10000,
            help="More iterations = more reliable uncertainty statistics.")
    with c3:
        p_perturb = st.slider("Perturbation probability",
            min_value=0.10, max_value=0.50, value=0.30, step=0.05,
            help="Chance that each weight is nudged per iteration.")
    with c4:
        n_tiers = st.slider("Number of tiers (k)",
            min_value=2, max_value=6, value=4,
            help="How many priority groups to classify alternatives into.")
    with c5:
        use_example = st.checkbox("Load Canadian example", value=True,
            help="Pre-loads 13 basins and 16 criteria from Okwaraojimadu & Ezekiel (2025).")

is_expert = "Expert" in mode


# ══════════════════════════════════════════════════════════════════════════════
# TABS  — Setup first, How to Use and Glossary at the end
# ══════════════════════════════════════════════════════════════════════════════

tab_setup, tab_w, tab_mc, tab_exp, tab_how, tab_gl = st.tabs([
    "📋  Setup",
    "⚖️  Weights & Scoring",
    "📊  Monte Carlo Results",
    "💾  Export",
    "❓  How to Use",
    "📖  Glossary",
])


# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
with tab_setup:
    st.markdown('<div class="section-header">1. Define Criteria and Alternatives</div>',
                unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Criteria** — the factors you are evaluating on")
        crit_txt = st.text_area("One per line",
            value="\n".join(DEF_CRITERIA) if use_example else
                  "Criterion 1\nCriterion 2\nCriterion 3\nCriterion 4\nCriterion 5",
            height=300, key="crit_in")
    with col2:
        st.markdown("**Alternatives** — the options you are ranking")
        alt_txt = st.text_area("One per line",
            value="\n".join(DEF_ALTS) if use_example else
                  "Option A\nOption B\nOption C",
            height=300, key="alt_in")

    cnames = [c.strip() for c in crit_txt.strip().split("\n") if c.strip()]
    anames = [a.strip() for a in alt_txt.strip().split("\n") if a.strip()]
    n_c, n_a = len(cnames), len(anames)
    st.session_state['cnames'] = cnames
    st.session_state['anames'] = anames

    m1,m2,m3 = st.columns(3)
    m1.metric("Criteria", n_c)
    m2.metric("Alternatives", n_a)
    m3.metric("Pairwise comparisons needed", n_c*(n_c-1)//2 if n_c>1 else 0)

    if n_c<2: st.error("Define at least 2 criteria.")
    if n_a<2: st.error("Define at least 2 alternatives.")

    st.markdown('<div class="section-header">2. Performance Scores (0 = worst · 1 = best)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    For each alternative on each criterion, enter a score between <b>0</b> (worst) and
    <b>1</b> (best). The example data is pre-filled from the published study.
    </div>""", unsafe_allow_html=True)

    if use_example and n_c==16 and n_a==13:
        P = DEF_P.copy()
        st.success("Example scores loaded from Okwaraojimadu & Ezekiel (2025).")
        df_p = pd.DataFrame(P, index=anames, columns=[c[:16] for c in cnames])
        st.dataframe(df_p.style.format("{:.3f}").background_gradient(
            cmap='Blues', axis=None, vmin=0, vmax=1), height=280)
    else:
        if n_c>0 and n_a>0:
            init = {c[:16]:[0.5]*n_a for c in cnames}
            ed   = st.data_editor(pd.DataFrame(init,index=anames),
                                  use_container_width=True, num_rows="fixed",
                                  key="score_ed")
            P = np.clip(ed.values.astype(float), 0, 1)
        else:
            P = np.zeros((max(n_a,1), max(n_c,1)))

    st.session_state['P'] = P


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTS & SCORING
# ──────────────────────────────────────────────────────────────────────────────
with tab_w:
    st.markdown('<div class="section-header">Weight Derivation</div>', unsafe_allow_html=True)
    cnames = st.session_state.get('cnames', DEF_CRITERIA)
    anames = st.session_state.get('anames', DEF_ALTS)
    P      = st.session_state.get('P',      DEF_P)
    n_c, n_a = len(cnames), len(anames)

    if n_c < 2:
        st.warning("Go to Setup first and define at least 2 criteria."); st.stop()

    # Quick mode
    if not is_expert:
        st.markdown("""<div class="info-box">
        <b>Quick Mode:</b> Type how important each criterion is.
        Numbers do not need to add to 1 — the app rescales them automatically.
        Higher number = more important.
        </div>""", unsafe_allow_html=True)
        def_w = DEF_W.tolist() if (use_example and n_c==16) else [1./n_c]*n_c
        wi=[]
        cols_w=st.columns(min(4,n_c))
        for i,cn in enumerate(cnames):
            with cols_w[i%len(cols_w)]:
                wi.append(st.number_input(cn[:24], min_value=0.001, max_value=1.,
                    value=float(def_w[i]) if i<len(def_w) else 1./n_c,
                    step=0.001, format="%.4f", key=f"wq{i}"))
        raw=np.array(wi); weights=raw/raw.sum()
        lambda_max=CI=CR=None; cr_ok=True

    # Expert mode
    else:
        st.markdown("""<div class="info-box">
        <b>Expert Mode:</b> Compare every pair of criteria using Saaty's 1-9 scale.
        1 = equally important, 3 = moderately more, 5 = strongly more,
        7 = very strongly, 9 = extremely. The app checks consistency automatically.
        </div>""", unsafe_allow_html=True)
        n_pairs=n_c*(n_c-1)//2
        if use_example and n_c==16:
            def ns(v):
                opts=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
                return min(opts,key=lambda x:abs(x-v))
            def_up=[ns(DEF_W[i]/DEF_W[j]) for i in range(n_c) for j in range(i+1,n_c)]
        else:
            def_up=[1.]*n_pairs
        upper=[]
        if n_pairs<=45:
            pairs=[(cnames[i][:20],cnames[j][:20])
                   for i in range(n_c) for j in range(i+1,n_c)]
            cp=st.columns(min(3,n_pairs))
            for k,((a,b),dv) in enumerate(zip(pairs,def_up)):
                with cp[k%len(cp)]:
                    v=st.select_slider(f"{a}  vs  {b}",
                        options=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9],
                        value=float(dv),
                        format_func=lambda x:f"1/{int(round(1/x))}" if x<1
                                             else str(int(x)) if x==int(x) else f"{x:.2f}",
                        key=f"pair{k}")
                    upper.append(v)
        else:
            csv_in=st.text_area("Upper triangle values (comma-separated)",
                value=",".join([str(round(v,4)) for v in def_up]),height=120)
            try:
                upper=[float(x.strip()) for x in csv_in.split(",")]
                if len(upper)!=n_pairs:
                    st.error(f"Expected {n_pairs}, got {len(upper)}."); upper=def_up
            except: st.error("Could not parse."); upper=def_up

        M=build_matrix(n_c,upper)
        weights,lambda_max,CI,CR=compute_ahp(M); cr_ok=CR<=0.10

        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card {"good" if cr_ok else "bad"}">'
                        f'<div class="val">{"✓" if cr_ok else "✗"} {CR:.4f}</div>'
                        f'<div class="lbl">Consistency Ratio (CR)</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="val">{lambda_max:.4f}</div>'
                        f'<div class="lbl">Lambda Max</div></div>',unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="val">{CI:.4f}</div>'
                        f'<div class="lbl">Consistency Index</div></div>',unsafe_allow_html=True)
        if not cr_ok:
            st.markdown("""<div class="warn-box">
            <b>CR &gt; 0.10.</b> Some comparisons contradict each other.
            Revise the most uncertain ones until CR falls below 0.10.
            </div>""", unsafe_allow_html=True)
        else:
            st.success(f"Consistent — CR = {CR:.4f}. Weights accepted.")

    # Weight table + chart
    st.markdown('<div class="section-header">Derived Weights</div>', unsafe_allow_html=True)
    df_w=pd.DataFrame({'Criterion':cnames,
                       'Weight (wᵢ)':[f"{w:.4f}" for w in weights],
                       'Share':[f"{w*100:.1f}%" for w in weights]})
    st.dataframe(df_w, use_container_width=True, hide_index=True)
    fw=plot_weights(weights,cnames)
    st.pyplot(fw,use_container_width=True); plt.close(fw)

    # Composite scores
    st.markdown('<div class="section-header">Composite Scores and Tier Rankings</div>',
                unsafe_allow_html=True)
    alt_scores = P @ weights
    if n_a>=n_tiers:
        breaks,gvf=jenks(alt_scores,n_tiers)
        tiers=assign_tiers(alt_scores,breaks)
    else:
        breaks=np.array([]); gvf=1.; tiers=np.ones(n_a,dtype=int)

    ca,cb=st.columns(2)
    ca.metric("Jenks-Fisher GVF",f"{gvf:.4f}",
              help="Goodness of Variance Fit. Above 0.90 = excellent grouping.")
    cb.metric("Tiers", n_tiers)

    df_s=pd.DataFrame({'Alternative':anames,
                       'Score Rᵏ':[f"{s:.4f}" for s in alt_scores],
                       'Tier':tiers,
                       'Rank':pd.Series(alt_scores).rank(ascending=False).astype(int).values
                      }).sort_values('Rank').reset_index(drop=True)
    st.dataframe(df_s,use_container_width=True,hide_index=True)
    if n_a>=2:
        fs=plot_scores(np.array(anames),alt_scores,tiers,breaks)
        st.pyplot(fs,use_container_width=True); plt.close(fs)

    st.session_state.update({'weights':weights,'scores':alt_scores,
                              'tiers':tiers,'breaks':breaks,'cr_ok':cr_ok,'P':P})


# ──────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ──────────────────────────────────────────────────────────────────────────────
with tab_mc:
    st.markdown('<div class="section-header">Monte Carlo Weight Uncertainty Analysis</div>',
                unsafe_allow_html=True)
    cnames = st.session_state.get('cnames', DEF_CRITERIA)
    anames = st.session_state.get('anames', DEF_ALTS)
    w      = st.session_state['weights']
    s_det  = st.session_state['scores']
    t_det  = st.session_state['tiers']
    brk    = st.session_state['breaks']
    P      = st.session_state['P']
    n_a    = len(anames)

    st.markdown(f"""<div class="info-box">
    The simulation runs <b>{n_iter:,} iterations</b>, each time slightly varying the
    criterion weights at random (probability {p_perturb} per weight per iteration),
    recomputing all scores, and checking whether tier assignments change.
    Alternatives stable in &ge;99.5% of runs are considered robustly classified.
    </div>""", unsafe_allow_html=True)

    if w is None:
        st.warning("Complete the Weights & Scoring tab first.")
    elif is_expert and not st.session_state['cr_ok']:
        st.error("CR > 0.10 — fix pairwise comparisons before running.")
    else:
        if st.button(f"▶  Run Monte Carlo  (N = {n_iter:,})", type="primary"):
            with st.spinner(f"Running {n_iter:,} simulations..."):
                aw,as_=run_mc(w,P,n_iter,p_perturb)
                st.session_state.update({'mc_w':aw,'mc_s':as_,'ran_mc':True})
            st.success(f"Done — {n_iter:,} iterations complete.")

        if st.session_state['ran_mc'] and st.session_state['mc_w'] is not None:
            aw=st.session_state['mc_w']; as_=st.session_state['mc_s']
            sig=aw.std(axis=0); mu=aw.mean(axis=0)

            st.markdown('<div class="section-header">Weight Uncertainty Summary</div>',
                        unsafe_allow_html=True)
            df_mc=pd.DataFrame({'Criterion':cnames,
                                 'Det. Weight':[f"{ww:.4f}" for ww in w],
                                 'MC Mean':[f"{m:.4f}" for m in mu],
                                 'MC Std Dev σᵢ':[f"{s:.5f}" for s in sig],
                                 'Stable?':['Yes' if s<=0.01 else 'Check' for s in sig]})
            st.dataframe(df_mc,use_container_width=True,hide_index=True)
            r1,r2,r3=st.columns(3)
            r1.metric("Max σᵢ",f"{sig.max():.5f}")
            r2.metric("Mean σᵢ",f"{sig.mean():.5f}")
            r3.metric("Iterations",f"{n_iter:,}")

            st.markdown('<div class="section-header">Tier Stability</div>',
                        unsafe_allow_html=True)
            stab=[]
            for b in range(n_a):
                stab.append((assign_tiers(as_[:,b],brk)==t_det[b]).mean()*100)

            df_st=pd.DataFrame({'Alternative':anames,
                                 'Score':[f"{ss:.4f}" for ss in s_det],
                                 'Tier':t_det,
                                 'Stability':[f"{p:.2f}%" for p in stab],
                                 'Robust?':['Yes' if p>=99.5 else 'Review' for p in stab]})
            st.dataframe(df_st,use_container_width=True,hide_index=True)

            if all(p>=99.5 for p in stab):
                st.markdown(f"""<div class="success-box">
                All {n_a} alternatives retain their tier in &ge;99.5% of {n_iter:,}
                simulations. Classification is robust.
                </div>""", unsafe_allow_html=True)
            else:
                n_w=sum(p<99.5 for p in stab)
                st.markdown(f"""<div class="warn-box">
                {n_w} alternative(s) below 99.5% stability. Their tier assignment
                is sensitive to weight variation — review those criteria weights.
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Convergence Figures</div>',
                        unsafe_allow_html=True)
            fig_cv=plot_conv(aw,cnames,w,top_n=min(4,len(cnames)))
            st.pyplot(fig_cv,use_container_width=True); plt.close(fig_cv)
            fig_st=plot_stability(as_,anames,t_det,brk,n_iter)
            st.pyplot(fig_st,use_container_width=True); plt.close(fig_st)

            st.session_state.update({'sigma':sig,'mu':mu,'stab':stab})


# ──────────────────────────────────────────────────────────────────────────────
# EXPORT
# ──────────────────────────────────────────────────────────────────────────────
with tab_exp:
    st.markdown('<div class="section-header">Download Your Results</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    CSV files open in Excel or Google Sheets.
    PNG files are high-resolution images ready for papers and presentations.
    JSON contains everything in one file for GIS or programming workflows.
    </div>""", unsafe_allow_html=True)

    w=st.session_state['weights']; s=st.session_state['scores']
    t=st.session_state['tiers'];   brk=st.session_state['breaks']
    cn=st.session_state.get('cnames',DEF_CRITERIA)
    an=st.session_state.get('anames',DEF_ALTS)

    if w is None:
        st.info("Complete the Setup and Weights tabs to enable downloads.")
    else:
        c1,c2=st.columns(2)
        with c1:
            st.download_button("📥  Weights CSV",
                pd.DataFrame({'Criterion':cn,'Weight':w,'Pct':w*100}).to_csv(index=False),
                "ahp_weights.csv","text/csv",use_container_width=True)
        with c2:
            df_se=pd.DataFrame({'Alternative':an,'Score':s,'Tier':t,
                'Rank':pd.Series(s).rank(ascending=False).astype(int).values
                }).sort_values('Rank').reset_index(drop=True)
            st.download_button("📥  Scores & Tiers CSV",df_se.to_csv(index=False),
                "ahp_scores.csv","text/csv",use_container_width=True)

        if st.session_state['ran_mc'] and st.session_state['sigma'] is not None:
            c3,c4=st.columns(2)
            with c3:
                st.download_button("📥  MC Weight Stats CSV",
                    pd.DataFrame({'Criterion':cn,'Det_Weight':w,
                        'MC_Mean':st.session_state['mu'],
                        'MC_StdDev':st.session_state['sigma']}).to_csv(index=False),
                    "mc_weights.csv","text/csv",use_container_width=True)
            with c4:
                st.download_button("📥  Tier Stability CSV",
                    pd.DataFrame({'Alternative':an,'Score':s,'Tier':t,
                        'Stability_pct':st.session_state['stab']}).to_csv(index=False),
                    "tier_stability.csv","text/csv",use_container_width=True)

        st.markdown("**Figures**")
        cf1,cf2=st.columns(2)
        with cf1:
            fw2=plot_weights(w,cn)
            st.download_button("📥  Weights PNG",fig2bytes(fw2),
                "weights.png","image/png",use_container_width=True)
        with cf2:
            if s is not None and len(s)>=2:
                fs2=plot_scores(np.array(an),s,t,brk)
                st.download_button("📥  Scores PNG",fig2bytes(fs2),
                    "scores.png","image/png",use_container_width=True)

        if st.session_state['ran_mc']:
            aw=st.session_state['mc_w']; as_=st.session_state['mc_s']
            cf3,cf4=st.columns(2)
            with cf3:
                fc=plot_conv(aw,cn,w,top_n=min(4,len(cn)))
                st.download_button("📥  Convergence PNG",fig2bytes(fc),
                    "convergence.png","image/png",use_container_width=True)
            with cf4:
                fst=plot_stability(as_,an,t,brk,n_iter)
                st.download_button("📥  Stability PNG",fig2bytes(fst),
                    "stability.png","image/png",use_container_width=True)

        st.markdown("**Full Results JSON**")
        rd={'criteria':cn,'alternatives':an,
            'weights':{c:float(ww) for c,ww in zip(cn,w)},
            'scores':{a:float(ss) for a,ss in zip(an,s)},
            'tiers':{a:int(tt) for a,tt in zip(an,t)},
            'tier_breaks':[float(b) for b in brk],
            'n_iterations':n_iter,'perturbation_probability':p_perturb}
        if st.session_state['sigma'] is not None:
            rd['mc_sigma']={c:float(sig) for c,sig in zip(cn,st.session_state['sigma'])}
            rd['mc_stability']={a:float(p) for a,p in zip(an,st.session_state['stab'])}
        st.download_button("📥  Full Results JSON",json.dumps(rd,indent=2),
            "ahp_mc_results.json","application/json",use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# HOW TO USE
# ──────────────────────────────────────────────────────────────────────────────
with tab_how:
    st.markdown('<div class="section-header">What Does This App Do?</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    This app helps you <b>rank and compare a list of options</b> across <b>multiple factors</b>
    and then tells you <b>how confident you can be in those rankings</b>.
    It was built to rank Canadian sedimentary basins for CO&#x2082; geological storage,
    but works for <i>any</i> ranking or site-selection problem.
    No programming knowledge needed.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Step-by-Step Guide</div>',
                unsafe_allow_html=True)

    steps = [
        ("1","Open Settings",
         "Click the <b>Settings — click to open/close</b> bar at the top of the page. "
         "This is where you choose your mode, set the number of simulations, and decide "
         "whether to load the built-in example. It is always visible — no sidebar needed."),
        ("2","Choose your mode",
         "<b>Quick Mode</b> — You already know the relative importance of each factor "
         "and want to type weights in directly. Fast and simple.<br><br>"
         "<b>Expert Mode</b> — You compare every factor against every other factor one "
         "pair at a time. The app calculates the weights mathematically and checks that "
         "your answers are internally consistent (Consistency Ratio must be below 0.10)."),
        ("3","Try the built-in example",
         "The <b>Load Canadian example</b> checkbox is ticked by default. This loads a "
         "published study with 13 Canadian sedimentary basins and 16 criteria — a great "
         "way to explore all features before entering your own data. Untick it to start fresh."),
        ("4","Define criteria and alternatives (Setup tab)",
         "Go to <b>Setup</b>. In the left box type your criteria — one per line "
         "(e.g. Cost, Environmental Impact, Accessibility). In the right box type your "
         "alternatives — one per line (e.g. Site A, Site B, Site C).<br><br>"
         "Then fill in the score table. Each score must be between <b>0</b> (worst) "
         "and <b>1</b> (best). A score of 0.5 means average performance."),
        ("5","Enter weights (Weights & Scoring tab)",
         "<b>Quick Mode:</b> Type a number for each criterion. Larger = more important. "
         "They do not need to add to 1.<br><br>"
         "<b>Expert Mode:</b> Use the sliders to compare each pair of criteria. "
         "The app tells you immediately if your comparisons are consistent."),
        ("6","See your rankings",
         "Scroll down in Weights & Scoring to see your alternatives ranked by their "
         "composite scores, colour-coded by priority tier. Tiers are set automatically "
         "using Jenks-Fisher — a statistical method that finds natural groupings in your "
         "scores rather than using arbitrary cutoffs."),
        ("7","Run the simulation (Monte Carlo Results tab)",
         "Click <b>Run Monte Carlo</b>. The app reruns your analysis thousands of times "
         "with slightly different weights each time, checking whether your rankings hold. "
         "If they do — your results are robust. You will see two charts confirming that "
         "the statistics settle down well before the end of the simulation."),
        ("8","Download results (Export tab)",
         "Download weights, scores, tier assignments, Monte Carlo statistics, and "
         "figures as CSV files, PNG images, or a JSON file. All are ready for papers, "
         "GIS attribute tables, or reports."),
        ("9","Look up terms (Glossary tab)",
         "If you see an unfamiliar term — CR, GVF, eigenvector, Jenks-Fisher — "
         "the Glossary tab explains every technical term in plain English with examples."),
    ]
    for num,title,body in steps:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">{num}</div>
          <div class="step-content"><h4>{title}</h4><p>{body}</p></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Frequently Asked Questions</div>',
                unsafe_allow_html=True)
    faqs = [
        ("Do I need to know AHP or statistics?",
         "No. Quick Mode lets anyone type weights and run the simulation immediately. "
         "Expert Mode adds the full AHP pairwise method if you want it."),
        ("What is a good number of iterations?",
         "10,000 is a solid default. The convergence charts will show you that results "
         "stabilise well before the end — typically by 3,000 to 5,000 iterations."),
        ("My CR is above 0.10 — what do I do?",
         "Some of your pairwise comparisons contradict each other. Look for the pairs "
         "where your judgement is least certain and revise them until CR falls below 0.10."),
        ("Can I use this for problems other than CO2 storage?",
         "Yes — completely. Replace the example with your own criteria and alternatives "
         "for any ranking problem: site selection, supplier evaluation, policy prioritisation."),
        ("How do I cite this tool?",
         "Okwaraojimadu, C.K. & Ezekiel, C.J. (2025). AHP-MCDA Monte Carlo Simulator "
         "[Web application]. University of Calgary. "
         "Contact: chisom.okwaraojimadu@ucalgary.ca"),
    ]
    for q,a in faqs:
        with st.expander(f"▸  {q}"):
            st.markdown(f"<p style='font-size:.9rem;line-height:1.7;color:#333;'>{a}</p>",
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# GLOSSARY
# ──────────────────────────────────────────────────────────────────────────────
with tab_gl:
    st.markdown('<div class="section-header">Glossary of Terms</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Plain-English explanations of every technical term used in the app,
    grouped by topic.
    </div>""", unsafe_allow_html=True)

    sections = {
        "The Basics": [
            ("Criterion (plural: criteria)",
             "A factor you use to evaluate your options — like cost, capacity, or accessibility.",
             "Example: In a CO2 storage study, criteria include storage capacity and tectonic stability."),
            ("Alternative",
             "One of the options you are comparing. Also called a candidate, site, or option.",
             "Example: WCSB, Williston Basin, and Michigan Basin are the alternatives."),
            ("Performance score (Pᵢₖ)",
             "A number between 0 and 1 saying how well an alternative performs on a criterion. 1 = best, 0 = worst.",
             "Example: WCSB scores 1.0 on storage capacity because it has the largest capacity class."),
            ("Composite score (Rᵏ)",
             "The overall score for an alternative — the weighted sum of all its performance scores. Higher is better.",
             "Example: WCSB scores 0.982 out of 1.0."),
            ("Weight (wᵢ)",
             "How important a criterion is as a fraction of the total. All weights sum to 1.",
             "Example: Storage capacity has weight 0.221, meaning it accounts for 22.1% of the score."),
            ("Tier",
             "A priority group. Tier 1 = highest priority. Boundaries are set by Jenks-Fisher, not arbitrary cutoffs.",
             "Example: WCSB and Williston are Tier 1. Michigan is Tier 2."),
        ],
        "AHP — Analytic Hierarchy Process": [
            ("AHP (Analytic Hierarchy Process)",
             "A method that derives weights by comparing criteria in pairs rather than guessing. Developed by Thomas Saaty (1977).",
             ""),
            ("Pairwise comparison",
             "Asking 'How much more important is A than B?' for every pair of criteria. With 16 criteria, there are 120 pairs.",
             ""),
            ("Saaty scale (1 to 9)",
             "The scale for answering pairwise questions. 1 = equal, 3 = moderately more, 5 = strongly, 7 = very strongly, 9 = extremely. Values below 1 (e.g. 1/3) mean the other criterion is more important.",
             ""),
            ("Eigenvector method",
             "The mathematical technique AHP uses to turn the comparison matrix into weights. The app handles this automatically.",
             ""),
            ("Lambda max (λmax)",
             "A number derived from your comparisons. For a perfectly consistent matrix it equals n (the number of criteria).",
             "Example: With 16 criteria, perfect λmax = 16. This study achieved λmax = 16.154."),
            ("Consistency Index (CI)",
             "Measures how inconsistent your comparisons are. CI = (λmax - n) / (n - 1). Smaller is better.",
             ""),
            ("Random Index (RI)",
             "The expected CI for a completely random matrix of size n. Used to compute CR. Looked up from Saaty's table.",
             "Example: For 16 criteria, RI = 1.59."),
            ("Consistency Ratio (CR)",
             "CR = CI / RI. Must be below 0.10 for the weights to be considered valid.",
             "Example: This study achieved CR = 0.0064 — well below 0.10."),
        ],
        "Monte Carlo Simulation": [
            ("Monte Carlo simulation",
             "Running the analysis thousands of times with slightly different (randomly varied) weights each time, to see how much results change.",
             ""),
            ("Iteration",
             "One run of the simulation with one set of perturbed weights. N = 10,000 means 10,000 runs.",
             ""),
            ("Perturbation",
             "A small random change applied to a weight in each iteration, simulating uncertainty in your judgements.",
             ""),
            ("Perturbation probability (p)",
             "The chance that any given weight is perturbed in a particular iteration. p = 0.30 means a 30% chance per weight.",
             ""),
            ("Weight standard deviation (σᵢ)",
             "How much a criterion's weight varied across all simulations. Very small σᵢ means the weight is stable.",
             "Example: Storage capacity weight has σ = 0.006 — barely changes across 10,000 simulations."),
            ("Convergence",
             "When the running statistics stop changing as more iterations are added. If σᵢ is the same at N = 10,000 and N = 20,000, the simulation has converged.",
             ""),
            ("Tier stability",
             "The percentage of simulations where an alternative stays in its original tier. Above 99.5% is considered robust.",
             "Example: WCSB has 100% tier stability — it is Tier 1 in every simulation."),
        ],
        "Tier Classification": [
            ("Jenks-Fisher (Natural Breaks)",
             "A statistical algorithm that finds the best boundary positions between tiers based on where the natural gaps are in your scores — not arbitrary cutoffs.",
             "Example: The gap between 0.847 and 0.720 is the largest break, so the Tier 1/Tier 2 boundary falls there."),
            ("Goodness of Variance Fit (GVF)",
             "A number from 0 to 1 rating how well the tiers fit the data. Above 0.90 = excellent.",
             "Example: This study achieved GVF = 0.957 with 4 tiers — excellent."),
            ("SDCM (Sum of Squared Deviations from Class Means)",
             "The within-tier variance. Jenks-Fisher minimises this to find the best boundaries.",
             ""),
            ("SDAM (Sum of Squared Deviations from the Array Mean)",
             "The total variance across all scores. Used to calculate GVF.",
             ""),
        ],
    }

    for sec, terms in sections.items():
        st.markdown(f'<div class="section-header">{sec}</div>', unsafe_allow_html=True)
        for term, definition, example in terms:
            ex_html = f'<p class="gloss-ex">&#9656; {example}</p>' if example else ''
            st.markdown(f"""
            <div class="gloss-card">
              <div class="gloss-term">{term}</div>
              <p class="gloss-def">{definition}</p>
              {ex_html}
            </div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  AHP-MCDA Monte Carlo Simulator &middot; Python 3 &middot; NumPy &middot;
  Matplotlib &middot; Streamlit &middot;
  Okwaraojimadu C.K. &amp; Ezekiel C.J., University of Calgary, 2025 &middot;
  chisom.okwaraojimadu@ucalgary.ca
</div>
""", unsafe_allow_html=True)
