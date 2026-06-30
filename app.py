import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import json

st.set_page_config(
    page_title="AHP-MCDA Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--navy:#1B3A5C;--teal:#0E7C7B;--blue:#2E75B6;--amber:#C47A00;
      --green:#2E8B57;--red:#B22222;--border:#D0DAE8;--text:#1A1A2E;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="collapsedControl"]{display:none!important;}

.hero{background:linear-gradient(135deg,#1B3A5C 0%,#0E4D7B 55%,#0E7C7B 100%);
  padding:2rem 2.2rem 1.8rem;border-radius:16px;margin-bottom:1.4rem;color:white;}
.hero h1{font-size:1.9rem;font-weight:700;margin:0 0 .4rem;}
.hero p{font-size:.93rem;opacity:.85;margin:0;line-height:1.6;}

/* Wizard progress bar */
.wizard-bar{display:flex;gap:0;margin-bottom:1.8rem;border-radius:12px;
  overflow:hidden;border:1.5px solid var(--border);}
.wstep{flex:1;padding:.65rem .5rem;text-align:center;font-size:.78rem;
  font-weight:600;background:#F4F7FB;color:#999;border-right:1px solid var(--border);
  transition:all .2s;}
.wstep:last-child{border-right:none;}
.wstep.done{background:#E8F5E9;color:#2E8B57;}
.wstep.active{background:var(--navy);color:white;}
.wstep .num{display:block;font-size:1.1rem;margin-bottom:.1rem;}

.card{background:white;border:1.5px solid var(--border);border-radius:14px;
  padding:1.3rem 1.5rem;margin-bottom:1rem;}
.card-title{font-size:1.05rem;font-weight:700;color:var(--navy);margin-bottom:.6rem;}

.callout{border-left:4px solid var(--blue);background:#EFF6FF;border-radius:0 10px 10px 0;
  padding:.75rem 1rem;font-size:.875rem;line-height:1.68;margin:.6rem 0;color:#1e3a5f;}
.callout.ok{border-color:var(--green);background:#F0FFF4;color:#1a4a2a;}
.callout.warn{border-color:var(--amber);background:#FFFBF0;color:#5a3a00;}
.callout.tip{border-color:var(--teal);background:#F0FFFE;color:#0a3a3a;}

.kpi{background:white;border:1.5px solid var(--border);border-radius:12px;
  padding:.85rem 1rem;text-align:center;}
.kpi .val{font-family:'JetBrains Mono',monospace;font-size:1.5rem;
  font-weight:600;color:var(--navy);line-height:1;}
.kpi .lbl{font-size:.7rem;color:#888;margin-top:.25rem;font-weight:600;
  letter-spacing:.5px;text-transform:uppercase;}
.kpi.good .val{color:var(--green);}
.kpi.bad .val{color:var(--red);}

.nav-row{display:flex;gap:.8rem;margin-top:1.4rem;}
.footer{text-align:center;padding:1.5rem 0 .6rem;font-size:.74rem;color:#aaa;
  border-top:1px solid var(--border);margin-top:2rem;
  font-family:'JetBrains Mono',monospace;}

.result-tier-1{background:#E8F5E9;border-left:5px solid #2E8B57;
  border-radius:0 10px 10px 0;padding:.6rem 1rem;margin:.3rem 0;}
.result-tier-2{background:#FFF8E1;border-left:5px solid #C47A00;
  border-radius:0 10px 10px 0;padding:.6rem 1rem;margin:.3rem 0;}
.result-tier-3{background:#FBE9E7;border-left:5px solid #8B4500;
  border-radius:0 10px 10px 0;padding:.6rem 1rem;margin:.3rem 0;}
.result-tier-4{background:#F5F5F5;border-left:5px solid #555;
  border-radius:0 10px 10px 0;padding:.6rem 1rem;margin:.3rem 0;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CORE MATH
# ══════════════════════════════════════════════════════════════════════════════

SAATY_RI = {1:0,2:0,3:.58,4:.90,5:1.12,6:1.24,7:1.32,8:1.41,
            9:1.45,10:1.49,11:1.51,12:1.54,13:1.56,14:1.57,
            15:1.58,16:1.59,17:1.60,18:1.61,19:1.62,20:1.63}

def compute_ahp(M):
    n=M.shape[0]; cs=M.sum(axis=0); w=(M/cs).mean(axis=1)
    lam=(M@w)/w; lmax=lam.mean()
    CI=(lmax-n)/(n-1) if n>1 else 0
    RI=SAATY_RI.get(n,1.63)
    return w,lmax,CI,CI/RI if RI>0 else 0

def build_matrix(n,upper):
    M=np.eye(n); idx=0
    for i in range(n):
        for j in range(i+1,n):
            M[i,j]=upper[idx]; M[j,i]=1/upper[idx]; idx+=1
    return M

def find_inconsistent_pairs(M,names,top_n=3):
    n=M.shape[0]; w,*_=compute_ahp(M); issues=[]
    for i in range(n):
        for j in range(i+1,n):
            ideal=w[i]/w[j] if w[j]>0 else 1.
            actual=M[i,j]
            ratio=max(actual/ideal,ideal/actual)
            issues.append((ratio,names[i],names[j],actual,ideal))
    issues.sort(reverse=True)
    return issues[:top_n]

def jenks(values,k):
    n=len(values); sv=np.sort(values)
    if k<=1: return np.array([]),float(np.var(sv)==0)
    if k>=n: return sv[1:],1.0
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
    return bv,1-SSW[k-1,n-1]/sdam if sdam>0 else 1.0

def assign_tiers(scores,breaks):
    sb=np.sort(breaks)[::-1]; out=[]
    for s in scores:
        t=len(sb)+1
        for i,b in enumerate(sb):
            if s>=b: t=i+1; break
        out.append(t)
    return np.array(out)

def normalise_scores(raw_scores, all_possible_raw):
    """Min-max normalise using the full range of possible raw scores."""
    mn=min(all_possible_raw); mx=max(all_possible_raw)
    if mx==mn: return np.ones(len(raw_scores))
    return np.array([(r-mn)/(mx-mn) for r in raw_scores])

def run_mc(weights, P, pairwise_M, n_iter, p_perturb):
    np.random.seed(42)
    n_c=len(weights); n_a=P.shape[0]
    aw=np.zeros((n_iter,n_c)); as_=np.zeros((n_iter,n_a))
    if pairwise_M is not None:
        saaty_scale=np.array([1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,
                               1,2,3,4,5,6,7,8,9])
        n=pairwise_M.shape[0]
        for s in range(n_iter):
            Mp=pairwise_M.copy()
            for i in range(n):
                for j in range(i+1,n):
                    if np.random.random()<p_perturb:
                        cur=Mp[i,j]
                        ci=np.argmin(np.abs(saaty_scale-cur))
                        delta=np.random.choice([-1,0,1])
                        ni=np.clip(ci+delta,0,len(saaty_scale)-1)
                        Mp[i,j]=saaty_scale[ni]; Mp[j,i]=1/saaty_scale[ni]
            wp,*_=compute_ahp(Mp)
            wp=np.maximum(wp,1e-6); wp/=wp.sum()
            aw[s]=wp; as_[s]=P@wp
    else:
        sig=np.maximum(weights*0.03,0.002)
        for s in range(n_iter):
            noise=np.random.normal(0,sig)
            mask=np.random.random(n_c)<p_perturb
            wp=np.maximum(weights+noise*mask,0.005); wp/=wp.sum()
            aw[s]=wp; as_[s]=P@wp
    return aw,as_


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

FONT='DejaVu Sans'; DARK='#1B3A5C'; MED='#2E75B6'; TEAL='#0E7C7B'
AMBER='#C47A00'; GREEN='#2E8B57'; RED='#B22222'; PURPLE='#6A4C93'
TC={1:'#2E8B57',2:'#C47A00',3:'#8B4500',4:'#555555'}

def fig2bytes(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=150,bbox_inches='tight',facecolor='white')
    buf.seek(0); plt.close(fig); return buf.getvalue()

def plot_pareto(w, names):
    """Vertical Pareto chart — bars rise upward, cumulative line overlaid."""
    idx = np.argsort(w)[::-1]
    ws  = w[idx]
    ns  = [names[i] for i in idx]
    cumw = np.cumsum(ws / ws.sum() * 100)
    x    = np.arange(len(ws))

    fig, ax1 = plt.subplots(figsize=(max(8, len(names) * 0.65), 5.5), facecolor='white')
    ax1.set_facecolor('#F9FAFB')

    cols = [plt.cm.Blues(0.35 + 0.65 * ww / max(w)) for ww in ws]
    bars = ax1.bar(x, ws, color=cols, edgecolor='white', lw=0.8, width=0.7)

    for bar, ww in zip(bars, ws):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(w) * 0.015,
                 f'{ww:.4f}', ha='center', va='bottom',
                 fontsize=8, fontfamily=FONT, color=DARK, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(ns, rotation=40, ha='right', fontsize=8.5, fontfamily=FONT)
    ax1.set_ylabel('Weight (wᵢ)', fontsize=10, fontfamily=FONT)
    ax1.set_ylim(0, max(w) * 1.25)
    ax1.grid(axis='y', color='#E0E0E0', lw=0.6)
    ax1.set_title('Criterion Weights — Pareto Chart\n'
                  'Bars = individual weights · Line = cumulative %',
                  fontsize=11, fontweight='bold', color=DARK, fontfamily=FONT, pad=8)

    ax2 = ax1.twinx()
    ax2.plot(x, cumw, color=RED, lw=2.2, marker='o', ms=5, zorder=5)
    ax2.axhline(80, color=AMBER, lw=1.5, ls='--', alpha=0.85)
    ax2.text(len(ws) - 0.5, 81.5, '80% threshold',
             color=AMBER, fontsize=8, fontfamily=FONT, ha='right')
    ax2.set_ylabel('Cumulative Weight (%)', fontsize=9, color=RED, fontfamily=FONT)
    ax2.tick_params(axis='y', colors=RED, labelsize=8)
    ax2.set_ylim(0, 110)

    plt.tight_layout()
    return fig

def plot_scores(names,scores,tiers,breaks,n_tiers):
    fig,ax=plt.subplots(figsize=(9,max(4,len(names)*.58)),facecolor='white')
    ax.set_facecolor('#F9FAFB')
    idx=np.argsort(scores)[::-1]; ss=scores[idx]
    ns=[names[i] for i in idx]; ts=tiers[idx]
    y=np.arange(len(names))
    cols=[TC.get(t,'#888') for t in ts]
    ax.barh(y,ss,color=cols,edgecolor='white',lw=.8,height=.68,alpha=.85)
    for i,(s,t) in enumerate(zip(ss,ts)):
        ax.text(s+.008,i,f'{s:.3f}',va='center',ha='left',fontsize=9.5,
                fontfamily=FONT,color=TC.get(t,'#888'),fontweight='bold')
    for b in breaks: ax.axvline(b,color='#333',lw=1.2,ls='--',alpha=.7)
    ax.set_yticks(y); ax.set_yticklabels(ns,fontsize=9.5,fontfamily=FONT)
    ax.set_xlabel('Composite Score Rᵏ ∈ [0,1]',fontsize=10,fontfamily=FONT)
    ax.set_title('Ranked Composite Scores with Tier Classification',
                 fontsize=11,fontweight='bold',color=DARK,fontfamily=FONT,pad=8)
    ax.set_xlim(0,1.14); ax.invert_yaxis()
    ax.grid(axis='x',color='#E0E0E0',lw=.6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    tlabels={1:'Priority',2:'Secondary',3:'Low priority',4:'Marginal'}
    leg=[Patch(facecolor=TC[t],alpha=.85,label=f'Tier {t} — {tlabels.get(t,"")}')
         for t in sorted(TC) if t<=max(tiers)]
    leg.append(Line2D([0],[0],color='#333',lw=1.2,ls='--',label='Tier boundary'))
    ax.legend(handles=leg,fontsize=8,loc='lower right',framealpha=.93)
    plt.tight_layout(); return fig

def plot_conv(aw,names,w_det,top_n=4):
    n_iter=len(aw)
    cps=np.unique(np.concatenate([
        np.arange(100,min(1000,n_iter),200),
        np.arange(1000,n_iter+1,max(500,n_iter//30))])).astype(int)
    cps=cps[cps<=n_iter]
    idx=np.argsort(w_det)[::-1][:top_n]
    cols=[MED,TEAL,AMBER,PURPLE]
    fig,axes=plt.subplots(2,2,figsize=(11,7),facecolor='white')
    fig.suptitle('Monte Carlo Convergence — Do weight uncertainties stabilise?',
                 fontsize=11,fontweight='bold',color=DARK,fontfamily=FONT)
    for ax_i,(ci,col) in enumerate(zip(idx,cols)):
        ax=axes.flat[ax_i]; ax.set_facecolor('#F9FAFB')
        rs=[aw[:cp,ci].std() for cp in cps]
        ax.fill_between(cps,[s*.88 for s in rs],[s*1.12 for s in rs],color=col,alpha=.12)
        ax.plot(cps,rs,color=col,lw=2.2)
        ax.axvline(n_iter*.5,color=GREEN,lw=1.,ls=':',alpha=.8)
        ax.axvline(n_iter,color=RED,lw=1.,ls=':',alpha=.8)
        s50=aw[:int(n_iter*.5),ci].std(); s100=aw[:n_iter,ci].std()
        ax.annotate(f'Change 50%→100%: {abs(s50-s100):.5f}',
                    xy=(n_iter*.72,(s50+s100)/2),fontsize=7.5,ha='center',
                    color=GREEN,fontfamily=FONT,
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',
                              edgecolor=GREEN,lw=.9,alpha=.92))
        ax.set_title(f'{names[ci][:30]}  (w={w_det[ci]:.4f})',
                     fontsize=9,fontweight='bold',color=DARK,fontfamily=FONT)
        ax.set_xlabel('Simulations run',fontsize=8.5,fontfamily=FONT)
        ax.set_ylabel('Weight variability (σᵢ)',fontsize=8.5,fontfamily=FONT)
        ax.grid(True,color='#E0E0E0',lw=.6); ax.tick_params(labelsize=8)
    plt.tight_layout(rect=[0,0,1,.93]); return fig

def plot_stability(as_,names,det_tiers,breaks,n_iter):
    cps=np.unique(np.concatenate([
        np.arange(100,min(1000,n_iter),200),
        np.arange(1000,n_iter+1,max(500,n_iter//30))])).astype(int)
    cps=cps[cps<=n_iter]
    styles=['-','--','-.',':', (0,(3,1,1,1)),'-','--','-.',':',(0,(3,1,1,1)),
            '-','--','-.',':', (0,(3,1,1,1)),'-','--','-.']
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.5),facecolor='white')
    fig.suptitle('Monte Carlo Stability — Does each option stay in its tier?',
                 fontsize=11,fontweight='bold',color=DARK,fontfamily=FONT)
    ax1.set_facecolor('#F9FAFB')
    for b in range(len(names)):
        stab=[(assign_tiers(as_[:cp,b],breaks)==det_tiers[b]).mean()*100 for cp in cps]
        ax1.plot(cps,stab,color=TC.get(det_tiers[b],'#888'),
                 lw=1.5,ls=styles[b%len(styles)],alpha=.85,label=names[b])
    ax1.axhline(99.5,color=RED,lw=1.8,ls='--',label='99.5% threshold')
    ax1.set_ylim(85,101)
    ax1.set_xlabel('Simulations run',fontsize=9,fontfamily=FONT)
    ax1.set_ylabel('% simulations where tier stays the same',fontsize=9,fontfamily=FONT)
    ax1.set_title('(a) All alternatives',fontsize=10,fontweight='bold',
                  color=DARK,fontfamily=FONT)
    ax1.grid(True,color='#E0E0E0',lw=.6); ax1.tick_params(labelsize=8)
    if len(names)<=15: ax1.legend(fontsize=6.5,framealpha=.9,ncol=2,loc='lower right')
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
    ax2.set_title('(b) Spot-checks at key simulation counts',fontsize=10,
                  fontweight='bold',color=DARK,fontfamily=FONT)
    ax2.grid(axis='y',color='#E0E0E0',lw=.6); ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7,framealpha=.92,loc='lower right',
               title='Alternative',title_fontsize=7)
    plt.tight_layout(rect=[0,0,1,.93]); return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = dict(
        step=1,
        # Step 1 — criteria names
        criteria=[],
        is_demo=False,
        # Step 2 — weights per criterion (raw, will be normalised)
        raw_weights={},
        weight_mode='quick',   # 'quick' or 'expert'
        pairwise_M=None,
        cr_ok=True,
        # Step 3 — alternatives
        alternatives=[],
        # Step 4 — class definitions per criterion
        # {crit_name: {'n_classes': int, 'labels': [...], 'scores': [...]}}
        class_defs={},
        # Step 5 — assignments: {alt_name: {crit_name: class_label}}
        assignments={},
        # Results
        weights=None, scores=None, tiers=None, breaks=None, P=None,
        ran_mc=False, mc_w=None, mc_s=None, sigma=None, mu=None, stab=None,
        n_tiers=4, n_iter=10000, p_perturb=0.30,
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v

init_state()
S = st.session_state


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div style="display:flex;align-items:center;gap:1.4rem;margin-bottom:.9rem;">
    <svg width="72" height="72" viewBox="0 0 180 140" style="flex-shrink:0;border-radius:14px;background:rgba(255,255,255,0.13);padding:8px;">
      <rect x="66" y="4"   width="48"  height="26" rx="4" fill="#2E8B57"/>
      <rect x="49" y="36"  width="82"  height="26" rx="4" fill="#C47A00"/>
      <rect x="32" y="68"  width="116" height="26" rx="4" fill="#8B4500"/>
      <rect x="15" y="100" width="150" height="26" rx="4" fill="#94a3b8"/>
      <line x1="78"  y1="17"  x2="102" y2="17"  stroke="white" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
      <line x1="61"  y1="49"  x2="119" y2="49"  stroke="white" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
      <line x1="44"  y1="81"  x2="136" y2="81"  stroke="white" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
      <circle cx="6"   cy="52"  r="5" fill="#2E75B6" opacity="0.75"/>
      <circle cx="174" cy="42"  r="4" fill="#2E75B6" opacity="0.65"/>
      <circle cx="4"   cy="80"  r="4" fill="#2E75B6" opacity="0.55"/>
      <circle cx="176" cy="75"  r="5" fill="#2E75B6" opacity="0.75"/>
      <circle cx="7"   cy="28"  r="3" fill="#2E75B6" opacity="0.50"/>
      <circle cx="173" cy="108" r="4" fill="#2E75B6" opacity="0.65"/>
    </svg>
    <h1 style="margin:0;">AHP-MCDA Monte Carlo Simulator</h1>
  </div>
  <p>Rank any set of options across multiple criteria — with scientifically grounded weights,
  automatic tier classification, and Monte Carlo robustness validation.
  Follow the steps below. Each step unlocks after the previous one is complete.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WIZARD PROGRESS BAR
# ══════════════════════════════════════════════════════════════════════════════

step_labels = [
    ("1","Criteria"),
    ("2","Weights"),
    ("3","Alternatives"),
    ("4","Class Definitions"),
    ("5","Assignments"),
    ("6","Results"),
]

bar_html = '<div class="wizard-bar">'
for i,(num,label) in enumerate(step_labels):
    s = i+1
    cls = "active" if S.step==s else ("done" if S.step>s else "")
    icon = "✓" if S.step>s else num
    bar_html += f'<div class="wstep {cls}"><span class="num">{icon}</span>{label}</div>'
bar_html += '</div>'
st.markdown(bar_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DEFINE CRITERIA
# ══════════════════════════════════════════════════════════════════════════════

if S.step == 1:
    st.markdown("## Step 1 — What factors will you evaluate on?")

    with st.expander("📖  How to use this app — read before you start", expanded=False):
        st.markdown("""
**What does this app do?**

This app helps you rank a list of options across multiple factors and tells you how confident
you can be in those rankings even when you are not 100% certain how important each factor is.
It implements AHP for weight derivation, Jenks-Fisher for tier classification, and Monte Carlo
simulation for robustness validation. No programming knowledge needed.

---

**Step 1 — Criteria:** List the factors you will evaluate on. These are the properties that matter
when comparing your options. Example: Storage capacity, Tectonic stability, Infrastructure.
Minimum 2, recommended 3 to 16.

**Step 2 — Weights:** Tell the app how important each criterion is.
*Quick Mode* — type a number for each criterion (higher = more important).
*Expert Mode* — compare every pair using Saaty's 1–9 scale. The app checks consistency
(Consistency Ratio must be below 0.10).

**Step 3 — Alternatives:** List the options you want to rank. These do not have to be basins —
they can be any set of options. Minimum 2.

**Step 4 — Class definitions:** For each criterion, define 3 to 5 classes from least to most
favourable. Each class needs a plain English label (e.g. "Very large") and a raw score
(higher = better, non-linear scores recommended e.g. 1, 3, 7, 15, 21).
The app normalises all scores to [0, 1] automatically. This is where you encode the
domain-specific thresholds for each factor.

**Step 5 — Assignments:** For every combination of alternative and criterion, select the class
that best describes that alternative's performance. This is where your domain expertise comes in.
Example: WCSB on Storage capacity → select "Very large". The app looks up the raw score,
normalises it, and builds the scoring matrix automatically.

**Step 6 — Results:** The app computes composite scores, classifies options into priority tiers
using Jenks-Fisher natural breaks, and shows you a ranked chart. Run the Monte Carlo simulation
to test whether your rankings hold under weight uncertainty, then download everything as CSV,
PNG, or JSON.

---

**Tips:**
- Use the ← Back button to return to any previous step at any time.
- Load the Canadian CO₂ basin example first to understand the full flow before entering your own data.
- For research use, Expert Mode is recommended as it produces auditable, CR-verified weights.
- Non-linear class scores (1, 3, 7, 15, 21) amplify differences between classes and are
  recommended for geoscience applications.
        """)

    st.markdown("""<div class="callout tip">
    List the <b>criteria</b> — the factors that matter when comparing your options.
    These could be geological properties, economic factors, environmental indicators, or anything else.
    <br><br>
    <b>Example:</b> Storage capacity, Tectonic stability, CO₂ source proximity, Infrastructure
    </div>""", unsafe_allow_html=True)

    with st.form("form_criteria"):
        st.markdown("**Enter your criteria — one per line (minimum 2, recommended 3–16):**")
        default_crit = "\n".join(S.criteria) if S.criteria else (
            "Tectonic stability\nFault and fracture intensity\nStorage capacity\n"
            "Reservoir quality\nCO₂ source proximity\nInfrastructure")
        crit_txt = st.text_area("Criteria", value=default_crit,
                                height=220, label_visibility="collapsed")
        load_example = st.checkbox("Load Canadian CO₂ basin example (16 criteria, 13 basins)",
                                   value=False)
        submitted = st.form_submit_button("Save criteria and continue →", type="primary")

        if submitted:
            if load_example:
                S.criteria = [
                    "Tectonic stability","Fault and fracture intensity","Evaporites",
                    "Reservoir–seal pairs","Leakage via outcrops","Storage capacity",
                    "Basin size","Reservoir temperature","Hydrogeological confinement",
                    "Depleted reservoir potential","Freshwater constraint",
                    "Industry maturity","Onshore / offshore","Accessibility",
                    "Infrastructure","CO₂ source proximity"
                ]
                S.is_demo = True
            else:
                cnames = [c.strip() for c in crit_txt.strip().split("\n") if c.strip()]
                if len(cnames) < 2:
                    st.error("Please enter at least 2 criteria.")
                    st.stop()
                S.criteria = cnames
                S.is_demo = False

            # Reset downstream state when criteria change
            S.raw_weights = {}; S.class_defs = {}
            S.assignments = {}; S.weights = None
            S.step = 2
            st.rerun()

    if S.criteria:
        st.markdown(f"**Currently defined:** {len(S.criteria)} criteria")
        for c in S.criteria:
            st.markdown(f"- {c}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 2:
    st.markdown("## Step 2 — How important is each criterion?")

    cnames = S.criteria
    n_c = len(cnames)

    mode = st.radio("Choose your weighting method:",
        ["✏️  Quick Mode — I'll type importance scores directly",
         "🔬  Expert Mode — Full AHP pairwise comparison (recommended for research)"],
        key="weight_mode_radio")
    S.weight_mode = "expert" if "Expert" in mode else "quick"

    if S.weight_mode == "quick":
        st.markdown("""<div class="callout">
        Type a number for each criterion. <b>Higher number = more important.</b>
        They do not need to add up to any particular total — the app rescales them to sum to 1.
        </div>""", unsafe_allow_html=True)

        with st.form("form_weights_quick"):
            # Pre-fill with example weights if example was loaded
            example_w = [.041,.071,.041,.092,.022,.221,.041,.041,
                         .022,.071,.022,.041,.041,.041,.071,.123]
            wi = []
            cols = st.columns(min(4, n_c))
            for i, cn in enumerate(cnames):
                with cols[i % len(cols)]:
                    default_val = float(example_w[i]) if (
                        len(cnames)==16 and i<len(example_w)) else round(1/n_c, 4)
                    wi.append(st.number_input(cn[:30], min_value=0.001,
                        max_value=100., value=default_val,
                        step=0.001, format="%.4f", key=f"wq_{i}"))

            col_a, col_b = st.columns(2)
            with col_a:
                back = st.form_submit_button("← Back")
            with col_b:
                fwd = st.form_submit_button("Save weights and continue →", type="primary")

            if back:
                S.step = 1; st.rerun()
            if fwd:
                raw = np.array(wi)
                S.raw_weights = {cn: float(raw[i]) for i,cn in enumerate(cnames)}
                S.weights = raw / raw.sum()
                S.pairwise_M = None; S.cr_ok = True
                S.step = 3; st.rerun()

    else:  # Expert mode
        st.markdown("""<div class="callout">
        Compare every pair of criteria using <b>Saaty's 1–9 scale</b>.<br>
        <b>1</b> = equally important &nbsp;|&nbsp; <b>3</b> = moderately more &nbsp;|&nbsp;
        <b>5</b> = strongly more &nbsp;|&nbsp; <b>7</b> = very strongly &nbsp;|&nbsp;
        <b>9</b> = extremely more important.<br>
        Values like <b>1/3</b> mean the <em>second</em> criterion is more important.
        CR must be below 0.10 to proceed.
        </div>""", unsafe_allow_html=True)

        n_pairs = n_c*(n_c-1)//2
        example_w = np.array([.041,.071,.041,.092,.022,.221,.041,.041,
                               .022,.071,.022,.041,.041,.041,.071,.123])

        def ns(v):
            opts=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
            return min(opts,key=lambda x:abs(x-v))

        if n_pairs <= 55:
            pairs=[(cnames[i],cnames[j]) for i in range(n_c) for j in range(i+1,n_c)]
            with st.form("form_weights_expert"):
                upper=[]
                cp=st.columns(min(3,n_pairs))
                for k_i,((a,b)) in enumerate(pairs):
                    if len(cnames)==16:
                        i_=cnames.index(a); j_=cnames.index(b)
                        dv=float(ns(example_w[i_]/example_w[j_])) if example_w[j_]>0 else 1.
                    else:
                        dv=1.
                    with cp[k_i%len(cp)]:
                        v=st.select_slider(f"{a[:20]} vs {b[:20]}",
                            options=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9],
                            value=dv,
                            format_func=lambda x:(f"1/{int(round(1/x))}" if x<1
                                                  else str(int(x)) if x==int(x)
                                                  else f"{x:.2f}"),
                            key=f"pair_{k_i}")
                        upper.append(v)

                M=build_matrix(n_c,upper)
                weights_exp,lmax,CI,CR=compute_ahp(M)
                cr_ok=CR<=0.10

                c1,c2,c3=st.columns(3)
                c1.metric("Consistency Ratio (CR)","✓ "+f"{CR:.4f}" if cr_ok else "✗ "+f"{CR:.4f}")
                c2.metric("Lambda Max",f"{lmax:.4f}")
                c3.metric("Consistency Index",f"{CI:.4f}")

                if not cr_ok:
                    st.markdown("""<div class="callout warn">
                    ⚠️ <b>CR > 0.10.</b> Fix these inconsistent pairs first:
                    </div>""", unsafe_allow_html=True)
                    issues=find_inconsistent_pairs(M,cnames,top_n=3)
                    for ri,(ratio,na,nb,actual,ideal) in enumerate(issues,1):
                        st.markdown(f"**{ri}.** *{na}* vs *{nb}* — "
                                    f"you said {actual:.2f}, weights suggest {ideal:.2f} "
                                    f"(inconsistency: {ratio:.2f}×)")

                col_a,col_b=st.columns(2)
                with col_a: back=st.form_submit_button("← Back")
                with col_b: fwd=st.form_submit_button("Save and continue →",type="primary",
                                                       disabled=not cr_ok)
                if back: S.step=1; st.rerun()
                if fwd and cr_ok:
                    S.weights=weights_exp; S.pairwise_M=M
                    S.cr_ok=True; S.step=3; st.rerun()
        else:
            st.info(f"You have {n_pairs} pairs — too many for sliders. "
                    "Switch to Quick Mode or reduce the number of criteria.")
            if st.button("← Back"): S.step=1; st.rerun()

    # Weight preview outside form
    if S.weights is not None and len(S.weights)==n_c:
        st.markdown("---")
        st.markdown("**Current weight distribution:**")
        df_w=pd.DataFrame({'Criterion':cnames,
                           'Weight':[f"{w:.4f}" for w in S.weights],
                           'Share':[f"{w*100:.1f}%" for w in S.weights]})
        st.dataframe(df_w,use_container_width=True,hide_index=True)

    # Literature reference box for demo data
    if S.is_demo:
        with st.expander("📚  Literature sources for the example weights", expanded=False):
            st.markdown("""
The default example weights follow the criterion importance hierarchy established in the
CO₂ storage screening literature, specifically storage capacity as the dominant criterion,
CO₂ source proximity as the primary economic feasibility driver, and reservoir-seal quality
as the principal containment criterion:

- Bachu, S. (2003). Screening and ranking of sedimentary basins for sequestration of CO₂
  in geological media in response to climate change. *Environmental Geology*, 44(3), 277–289.
  https://doi.org/10.1007/s00254-003-0762-9
- Metz, B., Davidson, O., de Coninck, H., Loos, M., & Meyer, L. (Eds.) (2005).
  *IPCC Special Report on Carbon Dioxide Capture and Storage*. Cambridge University Press.
- Celia, M.A., Bachu, S., Nordbotten, J.M., & Bandilla, K.W. (2015). Status of CO₂ storage
  in deep saline aquifers with emphasis on modeling approaches and practical simulations.
  *Water Resources Research*, 51(9), 6846–6892. https://doi.org/10.1002/2015WR017609
- Middleton, R.S., & Bielicki, J.M. (2009). A scalable infrastructure model for carbon
  capture and storage: SimCCS. *Energy Policy*, 37(3), 1052–1060.

These weights are a starting reference only — adjust them to reflect your own judgement
or research context.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ALTERNATIVES
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 3:
    st.markdown("## Step 3 — What are you ranking?")
    st.markdown("""<div class="callout tip">
    List the <b>alternatives</b> — the options you want to rank and compare.
    These can be basins, sites, countries, suppliers, projects, or anything else.
    <br><br>
    <b>Example:</b> WCSB, Williston Basin, Michigan Basin, Scotian Basin
    </div>""", unsafe_allow_html=True)

    with st.form("form_alts"):
        st.markdown("**Enter your alternatives — one per line (minimum 2):**")
        default_alts = "\n".join(S.alternatives) if S.alternatives else (
            "Option A\nOption B\nOption C\nOption D")
        if len(S.criteria)==16 and not S.alternatives:
            default_alts=("WCSB\nWilliston Basin\nMichigan Basin\nNL Offshore\n"
                          "Scotian Basin\nFlemish Pass\nBeaufort-Mackenzie\nHudson Bay\n"
                          "St. Lawrence\nNova Scotia\nArctic Islands\nNew Brunswick\nPacific Margin")

        alt_txt=st.text_area("Alternatives",value=default_alts,
                             height=220,label_visibility="collapsed")
        col_a,col_b=st.columns(2)
        with col_a: back=st.form_submit_button("← Back")
        with col_b: fwd=st.form_submit_button("Save alternatives and continue →",type="primary")

        if back: S.step=2; st.rerun()
        if fwd:
            anames=[a.strip() for a in alt_txt.strip().split("\n") if a.strip()]
            if len(anames)<2:
                st.error("Please enter at least 2 alternatives.")
                st.stop()
            S.alternatives=anames
            S.assignments={}  # reset assignments if alternatives changed
            S.step=4; st.rerun()

    if S.alternatives:
        st.markdown(f"**Currently defined:** {len(S.alternatives)} alternatives")
        cols=st.columns(min(4,len(S.alternatives)))
        for i,a in enumerate(S.alternatives):
            cols[i%len(cols)].markdown(f"• {a}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CLASS DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 4:
    st.markdown("## Step 4 — Define the classes for each criterion")
    st.markdown("""<div class="callout tip">
    For each criterion, define <b>3 to 5 classes</b> — from least favourable to most favourable.
    Each class needs a <b>label</b> (plain English description) and a <b>raw score</b>
    (a number reflecting its relative importance — higher = better).
    <br><br>
    Use <b>non-linear scores</b> to amplify differences between classes
    (e.g. 1, 3, 7, 15, 21 rather than 1, 2, 3, 4, 5).
    The app normalises all scores to [0, 1] automatically.
    <br><br>
    <b>Example — Storage capacity:</b><br>
    Class 1 (Negligible) → score 1 &nbsp;|&nbsp;
    Class 2 (Small) → score 3 &nbsp;|&nbsp;
    Class 3 (Medium) → score 7 &nbsp;|&nbsp;
    Class 4 (Large) → score 15 &nbsp;|&nbsp;
    Class 5 (Very large) → score 21
    </div>""", unsafe_allow_html=True)

    # Pre-fill example data
    EXAMPLE_CLASSES = {
        "Tectonic stability":       {"n":4,"labels":[">200 cm/s²","100–200 cm/s²","50–100 cm/s²","<50 cm/s²"],"scores":[1,3,7,15]},
        "Fault and fracture intensity":{"n":3,"labels":["High","Moderate","Low"],"scores":[1,3,7]},
        "Evaporites":               {"n":3,"labels":["None","Domes","Bedded"],"scores":[1,2,3]},
        "Reservoir–seal pairs":     {"n":5,"labels":["None","Poor","Intermediate","Good","Excellent"],"scores":[1,3,7,15,21]},
        "Leakage via outcrops":     {"n":5,"labels":["Very high","High","Intermediate","Low","Very low"],"scores":[1,3,5,7,9]},
        "Storage capacity":         {"n":5,"labels":["Negligible","Small","Medium","Large","Very large"],"scores":[1,3,7,15,21]},
        "Basin size":               {"n":5,"labels":["<600 km²","<1,000 km²","<2,000 km²","<200,000 km²",">200,000 km²"],"scores":[1,3,7,15,21]},
        "Reservoir temperature":    {"n":4,"labels":["<40°C","40–70°C","70–100°C",">100°C"],"scores":[1,3,7,15]},
        "Hydrogeological confinement":{"n":3,"labels":["Local","Intermediate","Regional"],"scores":[1,3,5]},
        "Depleted reservoir potential":{"n":5,"labels":["None","Minor (<50 Mboe)","Moderate (50–500 Mboe)","Large (500 Mboe–1 Gboe)","Very large (>1 Gboe)"],"scores":[1,3,5,15,21]},
        "Freshwater constraint":    {"n":4,"labels":["All fresh","Abundant","Some","Limited"],"scores":[1,3,5,7]},
        "Industry maturity":        {"n":5,"labels":["Unexplored","Exploring","Developing","Mature","Very mature"],"scores":[1,2,4,8,10]},
        "Onshore / offshore":       {"n":3,"labels":["Deep offshore","Shallow offshore","Onshore"],"scores":[1,6,10]},
        "Accessibility":            {"n":4,"labels":["Inaccessible","Difficult","Acceptable","Easy"],"scores":[1,3,6,10]},
        "Infrastructure":           {"n":4,"labels":["None","Minor","Moderate","Extensive"],"scores":[1,3,7,10]},
        "CO₂ source proximity":     {"n":5,"labels":["None (>200 km)","Minor","Moderate","<200 km","<100 km"],"scores":[1,3,7,15,15]},
    }

    cnames = S.criteria
    n_c = len(cnames)

    st.markdown(f"**Defining classes for {n_c} criteria.** "
                "Expand each criterion below to set its classes.")
    st.markdown("---")

    new_class_defs = {}
    all_valid = True

    for ci, cname in enumerate(cnames):
        # Pre-fill from example or previous session
        ex = EXAMPLE_CLASSES.get(cname, None)
        prev = S.class_defs.get(cname, None)
        init = prev or ({"n": ex["n"], "labels": ex["labels"], "scores": ex["scores"]}
                        if ex else {"n": 3, "labels": ["Low","Medium","High"], "scores": [1,3,7]})

        with st.expander(f"📌 {cname}", expanded=(ci==0 and not prev)):
            n_classes = st.selectbox(
                f"Number of classes for {cname}",
                options=[3,4,5],
                index=[3,4,5].index(min(max(init["n"],3),5)),
                key=f"nclass_{ci}",
                help="Choose between 3 and 5 classes. Class 1 = least favourable, Class N = most favourable.")

            labels = []; scores = []
            col_l, col_s = st.columns([3,1])
            with col_l: st.markdown("**Class label** (plain English description)")
            with col_s: st.markdown("**Raw score** (higher = better)")

            valid_criterion = True
            for j in range(n_classes):
                prev_label = init["labels"][j] if j < len(init["labels"]) else f"Class {j+1}"
                prev_score = init["scores"][j] if j < len(init["scores"]) else (j+1)*2
                cl, cs_ = st.columns([3,1])
                with cl:
                    lbl = st.text_input(
                        f"Class {j+1} label",
                        value=prev_label, key=f"lbl_{ci}_{j}",
                        placeholder=f"e.g. {'Low' if j==0 else 'Medium' if j==1 else 'High'}",
                        label_visibility="collapsed")
                    labels.append(lbl)
                with cs_:
                    sc = st.number_input(
                        f"Score {j+1}", value=float(prev_score),
                        min_value=0.001, step=1.0, format="%.1f",
                        key=f"sc_{ci}_{j}", label_visibility="collapsed")
                    scores.append(sc)

            # Validate scores are increasing
            if scores != sorted(scores):
                st.markdown('<div class="callout warn">⚠️ Scores should increase from '
                            'Class 1 (lowest) to the last class (highest).</div>',
                            unsafe_allow_html=True)
                valid_criterion = False

            if any(l.strip()=='' for l in labels):
                st.markdown('<div class="callout warn">⚠️ All class labels must be filled in.'
                            '</div>', unsafe_allow_html=True)
                valid_criterion = False

            if valid_criterion:
                st.markdown(f'<div class="callout ok">✅ {n_classes} classes defined.</div>',
                            unsafe_allow_html=True)
            else:
                all_valid = False

            new_class_defs[cname] = {"n": n_classes, "labels": labels, "scores": scores}

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Back", key="back_s4"): S.step=3; st.rerun()
    with col_b:
        if st.button("Save class definitions and continue →",
                     type="primary", key="fwd_s4", disabled=not all_valid):
            S.class_defs = new_class_defs
            S.step = 5; st.rerun()

    if not all_valid:
        st.markdown('<div class="callout warn">⚠️ Fix the issues above before continuing.'
                    '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ASSIGNMENTS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 5:
    st.markdown("## Step 5 — Assign a class to each alternative on each criterion")
    st.markdown("""<div class="callout tip">
    For every combination of alternative and criterion, select the class that best
    describes that alternative's performance. This is where your domain expertise comes in.
    <br><br>
    <b>Example:</b> WCSB on Storage capacity → select "Very large" (because it has been
    proven to have very large storage potential).
    </div>""", unsafe_allow_html=True)

    cnames = S.criteria
    anames = S.alternatives
    class_defs = S.class_defs

    # Pre-fill example assignments
    EXAMPLE_ASSIGNMENTS = {
        "WCSB":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"Bedded","Reservoir–seal pairs":"Excellent",
            "Leakage via outcrops":"Very low","Storage capacity":"Very large",
            "Basin size":">200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Regional","Depleted reservoir potential":"Very large (>1 Gboe)",
            "Freshwater constraint":"Limited","Industry maturity":"Very mature",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Extensive","CO₂ source proximity":"<100 km"},
        "Williston Basin":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"Bedded","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Low","Storage capacity":"Large",
            "Basin size":">200,000 km²","Reservoir temperature":"70–100°C",
            "Hydrogeological confinement":"Regional","Depleted reservoir potential":"Large (500 Mboe–1 Gboe)",
            "Freshwater constraint":"Limited","Industry maturity":"Mature",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Extensive","CO₂ source proximity":"<200 km"},
        "Michigan Basin":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"Bedded","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Low","Storage capacity":"Medium",
            "Basin size":"<200,000 km²","Reservoir temperature":"40–70°C",
            "Hydrogeological confinement":"Regional","Depleted reservoir potential":"Large (500 Mboe–1 Gboe)",
            "Freshwater constraint":"Some","Industry maturity":"Mature",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Extensive","CO₂ source proximity":"<100 km"},
        "NL Offshore":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Intermediate","Storage capacity":"Large",
            "Basin size":"<200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Large (500 Mboe–1 Gboe)",
            "Freshwater constraint":"Abundant","Industry maturity":"Mature",
            "Onshore / offshore":"Shallow offshore","Accessibility":"Acceptable",
            "Infrastructure":"Moderate","CO₂ source proximity":"Moderate"},
        "Scotian Basin":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Intermediate","Storage capacity":"Large",
            "Basin size":"<200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Moderate (50–500 Mboe)",
            "Freshwater constraint":"Abundant","Industry maturity":"Mature",
            "Onshore / offshore":"Deep offshore","Accessibility":"Acceptable",
            "Infrastructure":"Moderate","CO₂ source proximity":"Minor"},
        "Flemish Pass":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Intermediate","Storage capacity":"Large",
            "Basin size":"<200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Minor (<50 Mboe)",
            "Freshwater constraint":"Abundant","Industry maturity":"Developing",
            "Onshore / offshore":"Deep offshore","Accessibility":"Difficult",
            "Infrastructure":"Minor","CO₂ source proximity":"Minor"},
        "Beaufort-Mackenzie":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Intermediate","Storage capacity":"Large",
            "Basin size":">200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Minor (<50 Mboe)",
            "Freshwater constraint":"Some","Industry maturity":"Developing",
            "Onshore / offshore":"Onshore","Accessibility":"Difficult",
            "Infrastructure":"Minor","CO₂ source proximity":"Minor"},
        "Hudson Bay":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"None","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Intermediate","Storage capacity":"Large",
            "Basin size":">200,000 km²","Reservoir temperature":"40–70°C",
            "Hydrogeological confinement":"Local","Depleted reservoir potential":"None",
            "Freshwater constraint":"All fresh","Industry maturity":"Unexplored",
            "Onshore / offshore":"Shallow offshore","Accessibility":"Difficult",
            "Infrastructure":"None","CO₂ source proximity":"None (>200 km)"},
        "St. Lawrence":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"Domes","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Intermediate","Storage capacity":"Small",
            "Basin size":"<2,000 km²","Reservoir temperature":"40–70°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Moderate (50–500 Mboe)",
            "Freshwater constraint":"All fresh","Industry maturity":"Developing",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Moderate","CO₂ source proximity":"<200 km"},
        "Nova Scotia":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"High","Storage capacity":"Small",
            "Basin size":"<1,000 km²","Reservoir temperature":"<40°C",
            "Hydrogeological confinement":"Local","Depleted reservoir potential":"Minor (<50 Mboe)",
            "Freshwater constraint":"Some","Industry maturity":"Developing",
            "Onshore / offshore":"Onshore","Accessibility":"Acceptable",
            "Infrastructure":"Moderate","CO₂ source proximity":"Moderate"},
        "Arctic Islands":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"None","Reservoir–seal pairs":"Poor",
            "Leakage via outcrops":"Intermediate","Storage capacity":"Medium",
            "Basin size":"<200,000 km²","Reservoir temperature":"<40°C",
            "Hydrogeological confinement":"Local","Depleted reservoir potential":"None",
            "Freshwater constraint":"All fresh","Industry maturity":"Unexplored",
            "Onshore / offshore":"Onshore","Accessibility":"Inaccessible",
            "Infrastructure":"None","CO₂ source proximity":"None (>200 km)"},
        "New Brunswick":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Poor",
            "Leakage via outcrops":"High","Storage capacity":"Small",
            "Basin size":"<600 km²","Reservoir temperature":"<40°C",
            "Hydrogeological confinement":"Local","Depleted reservoir potential":"None",
            "Freshwater constraint":"All fresh","Industry maturity":"Exploring",
            "Onshore / offshore":"Onshore","Accessibility":"Acceptable",
            "Infrastructure":"Minor","CO₂ source proximity":"Minor"},
        "Pacific Margin":{"Tectonic stability":">200 cm/s²","Fault and fracture intensity":"High",
            "Evaporites":"None","Reservoir–seal pairs":"Poor",
            "Leakage via outcrops":"Very high","Storage capacity":"Small",
            "Basin size":"<1,000 km²","Reservoir temperature":"70–100°C",
            "Hydrogeological confinement":"Local","Depleted reservoir potential":"None",
            "Freshwater constraint":"Abundant","Industry maturity":"Exploring",
            "Onshore / offshore":"Shallow offshore","Accessibility":"Difficult",
            "Infrastructure":"Minor","CO₂ source proximity":"Moderate"},
    }

    new_assignments = {}
    all_assigned = True

    st.markdown(f"Assigning classes for **{len(anames)} alternatives** "
                f"across **{len(cnames)} criteria**.")

    for ai, aname in enumerate(anames):
        prev_alt = S.assignments.get(aname, {})
        ex_alt = EXAMPLE_ASSIGNMENTS.get(aname, {})

        with st.expander(f"🏛️ {aname}", expanded=(ai==0)):
            new_assignments[aname] = {}
            cols = st.columns(min(3, len(cnames)))

            for ci, cname in enumerate(cnames):
                cdef = class_defs.get(cname, {})
                labels = cdef.get("labels", [])
                if not labels:
                    continue

                # Determine default selection
                prev_val = prev_alt.get(cname, None)
                ex_val = ex_alt.get(cname, None)
                default_label = prev_val or ex_val

                if default_label in labels:
                    default_idx = labels.index(default_label)
                else:
                    default_idx = 0

                with cols[ci % len(cols)]:
                    selected = st.selectbox(
                        cname,
                        options=labels,
                        index=default_idx,
                        key=f"assign_{ai}_{ci}",
                        help=f"Select the class that best describes {aname} on {cname}")
                    new_assignments[aname][cname] = selected

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Back", key="back_s5"): S.step=4; st.rerun()
    with col_b:
        if st.button("Calculate results →", type="primary", key="fwd_s5"):
            S.assignments = new_assignments

            # ── BUILD P MATRIX FROM ASSIGNMENTS ──────────────────────────────
            n_a = len(anames); n_c = len(cnames)
            P = np.zeros((n_a, n_c))

            for ci, cname in enumerate(cnames):
                cdef = class_defs.get(cname, {})
                labels = cdef.get("labels", [])
                scores = cdef.get("scores", [])
                if not labels or not scores:
                    continue
                all_scores = scores
                mn = min(all_scores); mx = max(all_scores)
                for ai, aname in enumerate(anames):
                    selected_label = new_assignments[aname].get(cname, labels[0])
                    if selected_label in labels:
                        raw = scores[labels.index(selected_label)]
                    else:
                        raw = scores[0]
                    P[ai, ci] = (raw - mn) / (mx - mn) if mx != mn else 1.0

            # ── COMPUTE COMPOSITE SCORES ──────────────────────────────────────
            weights = S.weights
            alt_scores = P @ weights
            n_tiers = S.n_tiers
            if n_a >= n_tiers:
                breaks, gvf = jenks(alt_scores, n_tiers)
                tiers = assign_tiers(alt_scores, breaks)
            else:
                breaks = np.array([]); gvf = 1.
                tiers = np.ones(n_a, dtype=int)

            S.P = P; S.scores = alt_scores
            S.tiers = tiers; S.breaks = breaks
            S.ran_mc = False; S.mc_w = None; S.mc_s = None
            S.step = 6; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 6:
    st.markdown("## Step 6 — Results")

    cnames = S.criteria; anames = S.alternatives
    weights = S.weights; scores = S.scores
    tiers = S.tiers; breaks = S.breaks; P = S.P
    n_tiers = S.n_tiers; n_a = len(anames); n_c = len(cnames)

    # ── SETTINGS INLINE ──────────────────────────────────────────────────────
    with st.expander("⚙️  Simulation settings", expanded=False):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            S.n_iter = st.select_slider("Simulations (N)",
                options=[1000,2000,5000,10000,20000,50000], value=S.n_iter)
        with sc2:
            S.p_perturb = st.slider("Perturbation probability", .10, .50, S.p_perturb, .05)
        with sc3:
            new_k = st.slider("Number of tiers (k)", 2, 6, S.n_tiers)
            if new_k != S.n_tiers:
                S.n_tiers = new_k
                if n_a >= new_k:
                    breaks_new, _ = jenks(scores, new_k)
                    S.breaks = breaks_new
                    S.tiers = assign_tiers(scores, breaks_new)
                    tiers = S.tiers; breaks = S.breaks
                st.rerun()

        if S.ran_mc and S.mc_w is not None:
            st.markdown("---")
            st.markdown(f"**Convergence from your last run ({S.n_iter:,} simulations):** "
                        "use this to judge whether N is already large enough, or whether "
                        "you can safely reduce it.")
            fig_cv_settings = plot_conv(S.mc_w, cnames, weights, top_n=min(4,n_c))
            st.pyplot(fig_cv_settings, use_container_width=True); plt.close(fig_cv_settings)
            st.caption("If the curves have flattened well before the right edge, you can "
                       "lower N above and re-run for a faster simulation with the same "
                       "level of confidence.")
        else:
            st.info("Run a simulation at least once to see a convergence preview here — "
                    "it'll help you judge whether N is larger than it needs to be.")

    # ── DETERMINISTIC RESULTS ─────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.15rem;font-weight:700;color:#1B3A5C;'
                'border-bottom:2.5px solid #D0DAE8;padding-bottom:.38rem;'
                'margin:1.2rem 0 .9rem;">Rankings and tier classification</div>',
                unsafe_allow_html=True)

    tier_labels = {1:"🟢 Tier 1 — Priority",2:"🟡 Tier 2 — Secondary",
                   3:"🟠 Tier 3 — Low priority",4:"⚫ Tier 4 — Marginal"}
    tier_css = {1:"result-tier-1",2:"result-tier-2",3:"result-tier-3",4:"result-tier-4"}

    # Score table
    df_s = pd.DataFrame({
        'Alternative': anames,
        'Composite Score (0–1)': [f"{s:.4f}" for s in scores],
        'Rank': pd.Series(scores).rank(ascending=False).astype(int).values,
        'Priority Tier': [tier_labels.get(t, f"Tier {t}") for t in tiers]
    }).sort_values('Rank').reset_index(drop=True)
    st.dataframe(df_s, use_container_width=True, hide_index=True)

    # Tier summary cards
    for t in range(1, S.n_tiers+1):
        members = [anames[i] for i,ti in enumerate(tiers) if ti==t]
        if members:
            st.markdown(f'<div class="{tier_css.get(t,"result-tier-4")}">'
                        f'<b>{tier_labels.get(t,f"Tier {t}")}</b>: '
                        f'{", ".join(members)}</div>', unsafe_allow_html=True)

    # Chart
    if n_a >= 2:
        jenks_breaks, gvf = jenks(scores, S.n_tiers)
        fig_s = plot_scores(np.array(anames), scores, tiers, breaks, S.n_tiers)
        st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)
        st.caption("Bar length = composite score. Colour = tier. "
                   "Dashed lines = Jenks-Fisher tier boundaries (data-driven).")
        st.metric("Classification quality (GVF)", f"{gvf:.4f}",
                  help="Above 0.90 = excellent tier separation.")

    # Weights chart
    st.markdown('<div style="font-size:1.15rem;font-weight:700;color:#1B3A5C;'
                'border-bottom:2.5px solid #D0DAE8;padding-bottom:.38rem;'
                'margin:1.2rem 0 .9rem;">Criterion weight breakdown</div>',
                unsafe_allow_html=True)
    fig_p = plot_pareto(weights, cnames)
    st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

    # ── MONTE CARLO ───────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.15rem;font-weight:700;color:#1B3A5C;'
                'border-bottom:2.5px solid #D0DAE8;padding-bottom:.38rem;'
                'margin:1.2rem 0 .9rem;">Monte Carlo robustness test</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<div class="callout">
    The app will rerun your entire analysis <b>{S.n_iter:,} times</b> with slightly
    different weights each time, simulating expert judgement uncertainty.
    If rankings and tier assignments stay stable across those runs, your results
    are <b>robust and trustworthy</b>.
    An alternative is considered <b>robustly classified</b> if it stays in its tier
    in ≥ 99.5% of simulations.
    </div>""", unsafe_allow_html=True)

    pm = S.pairwise_M if S.weight_mode == "expert" else None

    if st.button(f"▶  Run {S.n_iter:,} simulations", type="primary"):
        with st.spinner(f"Running {S.n_iter:,} simulations — please wait..."):
            aw, as_ = run_mc(weights, P, pm, S.n_iter, S.p_perturb)
            S.mc_w = aw; S.mc_s = as_; S.ran_mc = True
            S.sigma = aw.std(axis=0); S.mu = aw.mean(axis=0)
            S.stab = [(assign_tiers(as_[:,b], breaks)==tiers[b]).mean()*100
                      for b in range(n_a)]
        st.success(f"✅ {S.n_iter:,} simulations complete.")
        st.rerun()

    if S.ran_mc and S.mc_w is not None:
        aw = S.mc_w; as_ = S.mc_s
        sig = S.sigma; stab = S.stab

        # Weight uncertainty table
        st.markdown("**Weight variability across simulations:**")
        df_mc = pd.DataFrame({
            'Criterion': cnames,
            'Your weight': [f"{w:.4f}" for w in weights],
            'Avg across simulations': [f"{m:.4f}" for m in S.mu],
            'Variability (σᵢ)': [f"{s:.5f}" for s in sig],
            'Stable?': ['✅ Yes' if s<=0.010 else '⚠️ Check' for s in sig]
        })
        st.dataframe(df_mc, use_container_width=True, hide_index=True)

        r1,r2,r3 = st.columns(3)
        r1.metric("Max weight variability (σ)", f"{sig.max():.5f}")
        r2.metric("Mean weight variability (σ)", f"{sig.mean():.5f}")
        r3.metric("Simulations run", f"{S.n_iter:,}")

        # Tier stability table
        st.markdown("**Tier stability:**")
        df_st = pd.DataFrame({
            'Alternative': anames,
            'Score': [f"{s:.4f}" for s in scores],
            'Tier': [tier_labels.get(t, f"Tier {t}") for t in tiers],
            'Tier stability': [f"{p:.2f}%" for p in stab],
            'Verdict': ['✅ Robust' if p>=99.5 else '⚠️ Review' for p in stab]
        })
        st.dataframe(df_st, use_container_width=True, hide_index=True)

        if all(p>=99.5 for p in stab):
            st.markdown(f"""<div class="callout ok">
            ✅ <b>All {n_a} alternatives are robustly classified.</b>
            Every option retained its tier in ≥ 99.5% of {S.n_iter:,} simulations.
            </div>""", unsafe_allow_html=True)
        else:
            n_w = sum(p<99.5 for p in stab)
            st.markdown(f"""<div class="callout warn">
            ⚠️ <b>{n_w} alternative(s) are below 99.5% stability.</b>
            These are close to tier boundaries — review their criteria scores carefully.
            </div>""", unsafe_allow_html=True)

        # Convergence plots
        st.markdown("**Convergence charts (quality check):**")
        fig_cv = plot_conv(aw, cnames, weights, top_n=min(4,n_c))
        st.pyplot(fig_cv, use_container_width=True); plt.close(fig_cv)
        fig_st = plot_stability(as_, anames, tiers, breaks, S.n_iter)
        st.pyplot(fig_st, use_container_width=True); plt.close(fig_st)

    # ── EXPORT ────────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.15rem;font-weight:700;color:#1B3A5C;'
                'border-bottom:2.5px solid #D0DAE8;padding-bottom:.38rem;'
                'margin:1.2rem 0 .9rem;">Download results</div>',
                unsafe_allow_html=True)

    e1,e2 = st.columns(2)
    with e1:
        st.download_button("📥  Scores and tiers (CSV)",
            df_s.to_csv(index=False), "scores_tiers.csv", "text/csv",
            use_container_width=True)
    with e2:
        df_w_exp = pd.DataFrame({'Criterion':cnames,'Weight':weights,'Share_pct':weights*100})
        st.download_button("📥  Criterion weights (CSV)",
            df_w_exp.to_csv(index=False), "weights.csv", "text/csv",
            use_container_width=True)

    e3,e4 = st.columns(2)
    with e3:
        # Export class definitions
        class_export = []
        for cn in cnames:
            cd = S.class_defs.get(cn,{})
            for j,(lbl,sc) in enumerate(zip(cd.get('labels',[]),cd.get('scores',[])),1):
                class_export.append({'Criterion':cn,'Class':j,'Label':lbl,'Raw_score':sc})
        df_cls=pd.DataFrame(class_export)
        st.download_button("📥  Class definitions (CSV)",
            df_cls.to_csv(index=False), "class_definitions.csv", "text/csv",
            use_container_width=True)
    with e4:
        # Export assignments
        assign_export = []
        for aname in anames:
            row={'Alternative':aname}
            for cn in cnames:
                row[cn]=S.assignments.get(aname,{}).get(cn,'')
            assign_export.append(row)
        df_asgn=pd.DataFrame(assign_export)
        st.download_button("📥  Class assignments (CSV)",
            df_asgn.to_csv(index=False), "assignments.csv", "text/csv",
            use_container_width=True)

    if S.ran_mc and S.sigma is not None:
        e5,e6 = st.columns(2)
        with e5:
            st.download_button("📥  Monte Carlo weight stats (CSV)",
                df_mc.to_csv(index=False), "mc_weights.csv", "text/csv",
                use_container_width=True)
        with e6:
            st.download_button("📥  Tier stability results (CSV)",
                df_st.to_csv(index=False), "tier_stability.csv", "text/csv",
                use_container_width=True)

    # Full JSON export
    rd = {
        'criteria': cnames, 'alternatives': anames,
        'weights': {c:float(w) for c,w in zip(cnames,weights)},
        'class_definitions': {cn: S.class_defs.get(cn,{}) for cn in cnames},
        'assignments': S.assignments,
        'scores': {a:float(s) for a,s in zip(anames,scores)},
        'tiers': {a:int(t) for a,t in zip(anames,tiers)},
        'tier_breaks': [float(b) for b in breaks],
        'n_iterations': S.n_iter,
        'perturbation_probability': S.p_perturb
    }
    if S.sigma is not None:
        rd['mc_sigma']={c:float(s) for c,s in zip(cnames,S.sigma)}
        rd['mc_stability']={a:float(p) for a,p in zip(anames,S.stab)}
    st.download_button("📥  Full results (JSON)",
        json.dumps(rd,indent=2), "full_results.json",
        "application/json", use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Edit assignments (Step 5)"): S.step=5; st.rerun()
    with col_b:
        if st.button("🔄  Start over from scratch"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  AHP-MCDA Monte Carlo Simulator &nbsp;·&nbsp; Python 3 · NumPy · Matplotlib · Streamlit
  &nbsp;·&nbsp; Okwaraojimadu C.K. &amp; Ezekiel C.J., University of Calgary, 2025
  &nbsp;·&nbsp; chisom.okwaraojimadu@ucalgary.ca
</div>
""", unsafe_allow_html=True)
