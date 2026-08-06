import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import io
import json
from contextlib import nullcontext
def fig_to_png_bytes(fig, dpi=150):
    """Return a matplotlib figure as PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return buf.getvalue()

def df_to_csv_bytes(df):
    """Return a pandas DataFrame as UTF-8 CSV bytes with BOM.
    The BOM (utf-8-sig) tells Excel to read the file as UTF-8,
    preventing garbled characters like CO₂ appearing as COâ‚‚.
    """
    return df.to_csv(index=False).encode("utf-8-sig")

def fig_download_buttons(fig, stem, title="", csv_df=None):
    """Render PNG download button for a figure, and optional CSV button for chart data.

    Pass csv_df (a pandas DataFrame) to add a CSV download button alongside the PNG.
    The CSV contains the underlying chart data so users can rebuild the chart in Excel
    or any other tool with full control over colors, decimal places, and formatting.
    """
    png_bytes = fig_to_png_bytes(fig)
    if csv_df is not None:
        c1, c2, c3 = st.columns([1, 1, 4])
    else:
        c1, c3 = st.columns([1, 5])
        c2 = None
    with c1:
        st.download_button(
            "📥 Download PNG",
            data=png_bytes,
            file_name=f"{stem}.png",
            mime="image/png",
            use_container_width=True,
            key=f"dl_png_{stem}"
        )
    if c2 is not None and csv_df is not None:
        with c2:
            st.download_button(
                "📥 Download CSV",
                data=df_to_csv_bytes(csv_df),
                file_name=f"{stem}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_csv_{stem}"
            )

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

/* Step 1 — analysis mode cards */
.mode-head{text-align:center;margin:.6rem 0 1.3rem;}
.mode-head h2{font-size:1.35rem;font-weight:700;color:var(--navy);margin:0;}
.mode-head p{font-size:.86rem;color:#6B7280;margin:.35rem 0 0;}
.mode-card{background:white;border:1.5px solid var(--border);border-radius:14px;
  padding:1.1rem 1.05rem .9rem;position:relative;margin-bottom:.55rem;}
.mode-card.rec{border:2.5px solid var(--teal);padding:1.05rem 1rem .85rem;}
.mode-badge{position:absolute;top:-.62rem;left:1rem;background:#E1F5EE;
  color:#0F6E56;font-size:.65rem;font-weight:700;padding:.15rem .55rem;
  border-radius:8px;letter-spacing:.3px;text-transform:uppercase;}
.mode-ico{width:46px;height:46px;border-radius:11px;display:flex;
  align-items:center;justify-content:center;margin-bottom:.75rem;}
.mode-title{font-size:.98rem;font-weight:700;color:var(--navy);
  line-height:1.3;min-height:2.55rem;}
.mode-sub{font-size:.78rem;color:#6B7280;line-height:1.55;
  margin:.35rem 0 .75rem;min-height:4.2rem;}
.mode-spec{border-top:1px solid #E5EAF1;padding-top:.55rem;font-size:.74rem;
  color:#4B5563;line-height:1.85;min-height:4.4rem;}
.mode-chip{display:inline-flex;align-items:center;gap:.45rem;background:#EFF6FF;
  border:1px solid #C7DBF2;border-radius:999px;padding:.28rem .8rem;
  font-size:.76rem;font-weight:600;color:#1B3A5C;}
.mode-chip .dot{width:7px;height:7px;border-radius:50%;background:var(--teal);}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CORE MATH
# ══════════════════════════════════════════════════════════════════════════════

SAATY_SCALE = np.array([1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9])

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
    bi_sorted=sorted(bks)
    # Break points are placed at the MIDPOINT between the last member of the
    # lower class and the first member of the upper class, rather than at the
    # data value itself. Using the data value puts the class-defining
    # alternative exactly on the boundary, so any downward perturbation flips
    # its tier and its Monte Carlo stability collapses to ~50% purely as an
    # artifact of the convention. Midpoints leave tier assignments unchanged.
    bv=np.array([(sv[i]+sv[i-1])/2 if i>0 else sv[i] for i in bi_sorted])
    sdam=np.var(sv)*n
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

def run_mc(weights, P, pairwise_M, n_iter, p_perturb, w_sigma_pct=3.0):
    """Monte Carlo weight perturbation.

    If pairwise_M is supplied (AHP modes), judgements are perturbed one step
    along the Saaty scale and weights are re-derived from the perturbed matrix.
    Otherwise (direct-weight modes), each weight is perturbed with Gaussian
    noise whose standard deviation is w_sigma_pct percent of that weight.
    """
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
        sig=np.maximum(weights*(w_sigma_pct/100.0),0.002)
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

def plot_weights_box(aw, w_det, names, method_label='AHP'):
    """Horizontal box plot of MC weight distributions with deterministic weight markers."""
    n_c = len(names)
    # Sort criteria by deterministic weight descending (highest at top)
    order = np.argsort(w_det)[::-1]

    # Compute box statistics from actual MC data
    q1   = np.percentile(aw, 25, axis=0)
    q3   = np.percentile(aw, 75, axis=0)
    mu   = aw.mean(axis=0)
    sig  = aw.std(axis=0)
    wlo  = np.maximum(mu - 3*sig, 0)
    whi  = mu + 3*sig

    fig, ax = plt.subplots(figsize=(10, max(5, n_c * 0.42 + 1)), facecolor='white')
    ax.set_facecolor('#FAFBFC')

    BOX_CLR  = '#A8C0D6'
    LINE_CLR = '#7A9AB8'
    DIAM_CLR = '#E07030'

    y_positions = list(range(n_c))

    for rank, ci in enumerate(order):
        y = rank
        # Whisker line (full simulated range ±3σ)
        ax.plot([wlo[ci], whi[ci]], [y, y], color=LINE_CLR, lw=1.4, solid_capstyle='round')
        # Whisker end caps
        cap_h = 0.18
        ax.plot([wlo[ci], wlo[ci]], [y-cap_h, y+cap_h], color=LINE_CLR, lw=1.2)
        ax.plot([whi[ci], whi[ci]], [y-cap_h, y+cap_h], color=LINE_CLR, lw=1.2)
        # IQR box
        box_h = 0.40
        rect = plt.Rectangle((q1[ci], y - box_h/2), q3[ci]-q1[ci], box_h,
                              facecolor=BOX_CLR, edgecolor=LINE_CLR, lw=1.0, zorder=3)
        ax.add_patch(rect)
        # Deterministic weight diamond
        ax.plot(w_det[ci], y, marker='D', color=DIAM_CLR, markersize=7,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    # y-axis labels: criterion names sorted by weight
    y_labels = [f'{names[ci]}' for ci in order]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9, color='black', fontweight='bold')
    ax.invert_yaxis()

    ax.set_xlabel('Criterion weight wᵢ', fontsize=11, color='black', fontweight='bold')
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.tick_params(axis='x', labelsize=9, colors='black')
    ax.tick_params(axis='y', length=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_edgecolor('black')
    ax.grid(axis='x', color='#E0E0E0', lw=0.6, zorder=0)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=DIAM_CLR,
               markeredgecolor='white', markersize=8,
               label=f'Deterministic {method_label} weight'),
        Patch(facecolor=BOX_CLR, edgecolor=LINE_CLR, label='IQR (25th–75th percentile)'),
        Line2D([0], [0], color=LINE_CLR, lw=1.4, label='Full simulated range (±3σ)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, framealpha=0.95,
              loc='lower right', edgecolor='#CCCCCC')

    ax.set_title(f'Monte Carlo Perturbation Distributions — {method_label} Criterion Weights',
                 fontsize=11, fontweight='bold', color='black', pad=10)
    plt.tight_layout()
    return fig


def run_stress(weights, P, pairwise_M, breaks, tiers_det, n_iter=2000, p_perturb=0.30):
    """Stress test: how far can the judgements be pushed before a tier changes?

    Re-runs the analysis at increasing perturbation magnitude and returns the
    percentage of simulations in which each alternative retains its deterministic
    tier. Unlike stability-versus-N (which is flat once converged), this varies,
    so it shows where the classification actually breaks.
    Returns (labels, K) with K of shape (n_levels, n_alternatives).
    """
    if pairwise_M is None:
        return [], np.zeros((0, P.shape[0]))
    LEVELS = [(1, p_perturb, f"±1 step\np = {p_perturb:.2f}\n(adopted)"),
              (1, 0.60, "±1 step\np = 0.60"),
              (1, 1.00, "±1 step\np = 1.00\n(all judgements)"),
              (2, 1.00, "±2 steps\np = 1.00"),
              (3, 1.00, "±3 steps\np = 1.00"),
              (4, 1.00, "±4 steps\np = 1.00")]
    n_c = len(weights); n_a = P.shape[0]
    K = np.zeros((len(LEVELS), n_a)); labels = []
    for li, (step, p, lab) in enumerate(LEVELS):
        labels.append(lab)
        rng = np.random.default_rng(42)
        keep = np.zeros(n_a)
        for _ in range(n_iter):
            Mp = pairwise_M.copy()
            for i in range(n_c):
                for j in range(i+1, n_c):
                    if rng.random() < p:
                        idx = int(np.argmin(np.abs(SAATY_SCALE - Mp[i, j])))
                        idx = int(np.clip(idx + rng.choice([-step, step]), 0, 16))
                        Mp[i, j] = SAATY_SCALE[idx]; Mp[j, i] = 1/SAATY_SCALE[idx]
            wv, _, _, _ = compute_ahp(Mp)
            keep += (assign_tiers(P @ wv, breaks) == tiers_det)
        K[li] = keep / n_iter * 100
    return labels, K


def plot_stress(labels, K, anames, tiers, n_iter):
    """Tier stability against PERTURBATION MAGNITUDE (not against N)."""
    TC = {1:'#2E8B57', 2:'#C47A00', 3:'#8B4500', 4:'#777777'}
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('#FAFBFC')
    ax.axhspan(99.5, 100.6, color='#2E8B57', alpha=0.07, zorder=0)
    ax.axhline(99.5, color='#B22222', ls='--', lw=1.6, zorder=2,
               label='99.5% robustness threshold')
    worst = int(np.argmin(K[-1]))
    for i, a in enumerate(anames):
        if i == worst:
            ax.plot(x, K[:, i], '-o', color='#B22222', lw=2.2, ms=6, zorder=5,
                    label=f'{a} (first to move)')
        else:
            ax.plot(x, K[:, i], '-', color='#B8C4D0', lw=1.0, zorder=3)
    ax.plot([], [], '-', color='#B8C4D0', lw=1.0, label='other alternatives')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Magnitude of judgement perturbation applied to every pairwise comparison',
                  fontsize=10, labelpad=8)
    ax.set_ylabel('% of simulations in which the alternative retains its tier', fontsize=10)
    ax.set_title('Monte Carlo sensitivity: tier stability against perturbation magnitude',
                 fontsize=11.5, fontweight='bold', pad=12)
    lo = max(40, K.min() - 6)
    ax.set_ylim(lo, 101.5); ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    return fig


def plot_score_box(all_sim_scores, anames, scores, tiers, breaks, n_iter):
    """Composite score distributions against the tier boundaries.

    Deterministic score as a marker, the full Monte Carlo range as a bar.
    This is the figure that shows WHY tiers are stable: if no bar reaches a
    dashed boundary, no alternative can change tier.
    """
    TC = {1:'#2E8B57', 2:'#F0B860', 3:'#E8734A', 4:'#9E9E9E'}
    TN = {1:'Tier 1 — Priority', 2:'Tier 2 — Secondary',
          3:'Tier 3 — Low Priority', 4:'Tier 4 — Marginal'}
    order = np.argsort(scores)[::-1]          # best at the top
    n_a = len(anames)
    fig, ax = plt.subplots(figsize=(10, max(4.0, n_a*0.42 + 1.4)), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    y = np.arange(n_a)[::-1]

    for k, b in enumerate(breaks):
        ax.axvline(b, color='#555', ls='--', lw=1.0, zorder=1,
                   label='Tier boundary' if k == 0 else None)
        hi = len(breaks) - k        # breaks are ascending: b[0] = T3|T4 ... b[-1] = T1|T2
        ax.text(b, -0.9, f"T{hi} | T{hi+1}", fontsize=7, color='#555',
                ha='center', va='top')

    lo = all_sim_scores.min(axis=0); up = all_sim_scores.max(axis=0)
    for row, i in enumerate(order):
        yy = y[row]
        ax.barh(yy, up[i]-lo[i], left=lo[i], height=0.55, color=TC[tiers[i]],
                alpha=0.55, edgecolor=TC[tiers[i]], lw=0.8, zorder=3)
        ax.plot(scores[i], yy, 'o', color='#222', ms=5, zorder=5)
    ax.set_yticks(y[::-1][::-1]); ax.set_yticks(y)
    ax.set_yticklabels([anames[i] for i in order], fontsize=9)
    ax.set_xlabel('Composite suitability score  $R^k \\in [0, 1]$', fontsize=10)
    ax.set_xlim(max(0, lo.min()-0.08), min(1.06, up.max()+0.06))
    ax.set_ylim(-1.4, n_a-0.4)
    ax.grid(axis='x', alpha=0.25, zorder=0)
    h = [Patch(facecolor=TC[k], alpha=0.55, label=TN[k]) for k in sorted(set(tiers))]
    h.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#222',
                        markersize=5, label='Deterministic score'))
    h.append(Patch(facecolor='#BBB', alpha=0.55, label=f'Bar = full MC range (n = {n_iter:,})'))
    h.append(plt.Line2D([0],[0], color='#555', ls='--', lw=1.0, label='Tier boundary'))
    ax.legend(handles=h, loc='lower right', fontsize=7.5, framealpha=0.95, ncol=2)
    plt.tight_layout()
    return fig

def init_state():
    defaults = dict(
        step=1,
        # Step 1 — analysis mode ('mcda', 'mcda_mc', 'ahp', 'ahp_mc')
        analysis_mode=None,
        # Step 2 — criteria names
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
        all_sim_tiers=None,
        n_tiers=4, n_iter=10000, p_perturb=0.30,
        w_sigma_pct=3.0,
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v

init_state()
S = st.session_state


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS MODES
# ══════════════════════════════════════════════════════════════════════════════

ICO_SLIDERS = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
               'stroke="currentColor" stroke-width="1.8" stroke-linecap="round">'
               '<line x1="3" y1="6" x2="21" y2="6"/><circle cx="9" cy="6" r="2.4" fill="white"/>'
               '<line x1="3" y1="12" x2="21" y2="12"/><circle cx="16" cy="12" r="2.4" fill="white"/>'
               '<line x1="3" y1="18" x2="21" y2="18"/><circle cx="7" cy="18" r="2.4" fill="white"/>'
               '</svg>')

ICO_HIST = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.8" stroke-linecap="round">'
            '<line x1="3" y1="20" x2="21" y2="20"/>'
            '<rect x="4" y="14" width="3" height="6"/><rect x="8.5" y="9" width="3" height="11"/>'
            '<rect x="13" y="6" width="3" height="14"/><rect x="17.5" y="12" width="3" height="8"/>'
            '</svg>')

ICO_TREE = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.8" stroke-linecap="round">'
            '<rect x="9" y="2.5" width="6" height="4.5" rx="1"/>'
            '<rect x="2" y="16.5" width="6" height="4.5" rx="1"/>'
            '<rect x="16" y="16.5" width="6" height="4.5" rx="1"/>'
            '<path d="M12 7v4.5M5 16.5V12h14v4.5"/></svg>')

ICO_SHIELD = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
              'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
              'stroke-linejoin="round">'
              '<path d="M12 2.5 4 6v6c0 5 3.4 8.4 8 9.5 4.6-1.1 8-4.5 8-9.5V6z"/>'
              '<path d="M8.8 11.8 11.2 14.4 15.4 9.6"/></svg>')

# 15 suitability criteria. Storage capacity is deliberately NOT among them:
# it is reported separately for the highest-ranked basins, following the
# structure of Bachu (2003) and Ye et al. (2023). See Section 2.3.
EXAMPLE_CRITERIA = [
    "Tectonic stability","Fault and fracture intensity","Evaporites",
    "Reservoir–seal pairs","Leakage via outcrops",
    "Basin size","Reservoir temperature","Hydrogeological confinement",
    "Depleted reservoir potential","Freshwater constraint",
    "Industry maturity","Onshore / offshore","Accessibility",
    "Infrastructure","CO₂ source proximity",
]

# ── Example judgements for the Canadian CO2 basin screening ──────────────────
# The JUDGEMENT is the six-band importance hierarchy below. Weights are DERIVED
# from it by AHP, never the reverse. Band 1 is most important.
EXAMPLE_BANDS = {
    "CO₂ source proximity": 1, "Reservoir–seal pairs": 1,
    "Reservoir temperature": 2,
    "Fault and fracture intensity": 3, "Onshore / offshore": 3,
    "Depleted reservoir potential": 4, "Infrastructure": 4,
    "Tectonic stability": 4, "Industry maturity": 4,
    "Hydrogeological confinement": 5, "Basin size": 5, "Accessibility": 5,
    "Leakage via outcrops": 6, "Freshwater constraint": 6, "Evaporites": 6,
}
EXAMPLE_ALTS = [
    "WCSB","Williston Basin","Michigan Basin","NL Offshore",
    "Maritimes (onshore)","Maritimes (offshore)","St. Lawrence","Scotian Basin",
    "Beaufort-Mackenzie","Flemish Pass","Hudson Bay","Arctic Islands","Pacific Margin",
]

# Documented mapping rule: Saaty judgement from band distance.
BAND_RULE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7}

def example_matrix(cnames):
    """Build the example pairwise matrix from the band hierarchy."""
    n = len(cnames)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = EXAMPLE_BANDS[cnames[j]] - EXAMPLE_BANDS[cnames[i]]
            M[i, j] = BAND_RULE[abs(d)] if d > 0 else (
                1/BAND_RULE[abs(d)] if d < 0 else 1)
    return M

def example_weights():
    """Weights DERIVED from the example matrix (not hard-coded)."""
    return compute_ahp(example_matrix(EXAMPLE_CRITERIA))[0]

MODES = {
    "mcda": {
        "title": "Direct-weight MCDA",
        "sub": "Type importance values in directly. Weighted sum model, "
               "scores and Jenks-Fisher tiers.",
        "weights": "quick", "mc": False,
        "spec": ["Weights: typed", "Robustness: none", "Runtime: instant"],
        "spec_col": "#2E8B57",
        "ico": ICO_SLIDERS, "ico_bg": "#E6F1FB", "ico_fg": "#185FA5",
        "method": "Weighted sum model (WSM) with direct weights",
        "best": "Fast screening, teaching, or weights that come from an "
                "external source such as entropy weighting or a client mandate.",
    },
    "mcda_mc": {
        "title": "Direct-weight MCDA + Monte Carlo",
        "sub": "Same as above, plus a robustness test of weights you are "
               "not fully confident in.",
        "weights": "quick", "mc": True,
        "spec": ["Weights: typed", "Robustness: full suite", "Runtime: + about 1 min"],
        "spec_col": "#C47A00",
        "ico": ICO_HIST, "ico_bg": "#E1F5EE", "ico_fg": "#0F6E56",
        "method": "WSM with direct weights + Gaussian weight perturbation",
        "best": "You guessed the weights and want to know whether the ranking "
                "even depends on them.",
    },
    "ahp": {
        "title": "AHP-MCDA",
        "sub": "Saaty pairwise comparison with the consistency ratio held "
               "below 0.10, then scores and tiers.",
        "weights": "expert", "mc": False,
        "spec": ["Weights: pairwise", "Robustness: none", "Runtime: instant"],
        "spec_col": "#2E8B57",
        "ico": ICO_TREE, "ico_bg": "#EEEDFE", "ico_fg": "#534AB7",
        "method": "AHP (eigenvector weights, CR-verified) + WSM",
        "best": "Auditable, defensible weights where you do not need an "
                "uncertainty claim.",
    },
    "ahp_mc": {
        "title": "AHP-MCDA + Monte Carlo",
        "sub": "The full method. Adds Saaty-scale judgement perturbation, "
               "convergence and tier stability analysis.",
        "weights": "expert", "mc": True,
        "spec": ["Weights: pairwise", "Robustness: full suite", "Runtime: + about 1 min"],
        "spec_col": "#B22222",
        "ico": ICO_SHIELD, "ico_bg": "#FAEEDA", "ico_fg": "#854F0B",
        "method": "AHP (CR-verified) + WSM + Monte Carlo judgement perturbation",
        "best": "Publication-grade work. This is the method used in the "
                "Canadian CO\u2082 basin screening study.",
        "recommended": True,
    },
}

def mode_cfg():
    """Current mode config, defaulting to the full method if none chosen yet."""
    return MODES.get(S.analysis_mode, MODES["ahp_mc"])

def set_mode(key):
    """Select an analysis mode. Criteria, alternatives, class definitions and
    assignments are mode-independent and are preserved. Weights are discarded
    only when the weight derivation method itself changes."""
    old = MODES.get(S.analysis_mode)
    if old is not None and old["weights"] != MODES[key]["weights"]:
        S.weights = None; S.pairwise_M = None
        S.raw_weights = {}; S.cr_ok = True
    S.analysis_mode = key
    S.weight_mode = MODES[key]["weights"]
    if not MODES[key]["mc"]:
        S.ran_mc = False; S.mc_w = None; S.mc_s = None
        S.sigma = None; S.mu = None; S.stab = None; S.all_sim_tiers = None


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
    ("1","Analysis Type"),
    ("2","Criteria"),
    ("3","Weights"),
    ("4","Alternatives"),
    ("5","Class Definitions"),
    ("6","Assignments"),
    ("7","Results"),
]

bar_html = '<div class="wizard-bar">'
for i,(num,label) in enumerate(step_labels):
    s = i+1
    cls = "active" if S.step==s else ("done" if S.step>s else "")
    icon = "✓" if S.step>s else num
    bar_html += f'<div class="wstep {cls}"><span class="num">{icon}</span>{label}</div>'
bar_html += '</div>'
st.markdown(bar_html, unsafe_allow_html=True)

# Selected-mode chip with a change control (visible from Step 2 onwards)
if S.step > 1 and S.analysis_mode:
    ch1, ch2 = st.columns([5,1])
    with ch1:
        st.markdown(f'<div class="mode-chip"><span class="dot"></span>'
                    f'Analysis: {mode_cfg()["title"]}</div>', unsafe_allow_html=True)
    with ch2:
        if st.button("Change", key="chg_mode", use_container_width=True,
                     help="Return to Step 1. Your criteria, alternatives, classes "
                          "and assignments are kept."):
            S.step = 1; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CHOOSE THE ANALYSIS TYPE
# ══════════════════════════════════════════════════════════════════════════════

if S.step == 1:
    st.markdown("""<div class="mode-head">
      <h2>Step 1 — Which analysis do you want to run?</h2>
      <p>Pick how the criterion weights are derived, and whether uncertainty is
      propagated through the ranking. You can change this at any time without
      losing your data.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("📖  How to use this app — read before you start", expanded=False):
        st.markdown("""
**What does this app do?**

It ranks a list of options across multiple factors, sorts them into priority tiers, and
tells you how much confidence those rankings deserve. It implements AHP for weight
derivation, a weighted sum model for aggregation, Jenks-Fisher natural breaks for tier
classification, and Monte Carlo simulation for robustness validation. No programming needed.

---

**Step 1 — Analysis type:** Choose one of four modes. Two choices are crossed together:
whether weights are typed in directly or derived from AHP pairwise comparison, and whether
Monte Carlo uncertainty propagation is run. Everything downstream is identical across modes,
so you can switch at any time using the **Change** button and keep your data. If in doubt,
pick *AHP-MCDA + Monte Carlo*.

**Step 2 — Criteria:** List the factors you will evaluate on. Example: Storage capacity,
Tectonic stability, Infrastructure. Minimum 2, recommended 3 to 16. Tick *Load Canadian CO₂
basin example* here to fill the whole app with the published 16-criterion, 13-basin dataset.

**Step 3 — Weights:** How this looks depends on your Step 1 choice.
*Direct-weight modes* — type a number per criterion, higher meaning more important. They do
not need to sum to anything; the app rescales them to sum to 1.
*AHP modes* — compare every pair on Saaty's 1–9 scale. Weights come from the principal
eigenvector, and the Consistency Ratio must fall below 0.10 to continue. If it does not, the
app names your three most inconsistent pairs and shows what your other judgements imply each
one should have been.

**Step 4 — Alternatives:** List the options you want to rank. They do not have to be basins.
Minimum 2.

**Step 5 — Class definitions:** For each criterion, define 3 to 5 classes from least to most
favourable. Each class needs a plain English label (e.g. "Very large") and a raw score, higher
being better. Non-linear scores such as 1, 3, 7, 15, 21 are recommended: they amplify the gaps
between classes rather than pretending the steps are evenly spaced. Scores are min-max
normalised to [0, 1] automatically, so only the relative spacing matters. This is where you
encode your domain thresholds.

**Step 6 — Assignments:** For every alternative and criterion, pick the class that describes
that alternative's performance. Example: WCSB on Storage capacity → "Very large". The app
looks up the raw score, normalises it, and builds the scoring matrix for you.

**Step 7 — Results:** Everything comes out here.

---

**What you get in Step 7 depends on the mode you picked in Step 1.** Rankings, tiers, the
GVF metric, the Pareto weight chart and the CSV/JSON/PNG exports come out of every mode.
The Consistency Ratio and its diagnostics only exist in the AHP modes, and the robustness suite
(weight variability, tier stability, convergence charts, weight box plot) only exists in the
Monte Carlo modes. The panel below this one breaks down exactly what each of the four modes
produces.

---

**Tips:**
- Use the ← Back button to return to any previous step at any time. Nothing is lost.
- Load the Canadian CO₂ basin example first to see the full flow before entering your own data.
- For research use, an AHP mode is recommended: it produces auditable, CR-verified weights.
- Run a deterministic mode first to sanity-check your inputs, then switch to the matching
  Monte Carlo mode from Step 7 once the numbers look right. Simulating a mistake is expensive.
- If tier stability sits below 99.5% for an alternative, it is not necessarily wrong. It usually
  means the alternative sits close to a Jenks break, so re-read its class assignments rather
  than reaching for more simulations.
        """)

    order = ["mcda", "mcda_mc", "ahp", "ahp_mc"]
    cols = st.columns(4, gap="small")

    for col, key in zip(cols, order):
        m = MODES[key]
        with col:
            badge = ('<div class="mode-badge">Recommended</div>'
                     if m.get("recommended") else "")
            spec_html = "".join(
                f'<div style="color:{m["spec_col"]};">{t}</div>' if i == 2
                else f'<div>{t}</div>'
                for i, t in enumerate(m["spec"]))
            # Built as one unindented line: any leading whitespace or blank line
            # would make Streamlit's markdown parser treat this as a code block.
            st.markdown(
                f'<div class="mode-card{" rec" if m.get("recommended") else ""}">'
                f'{badge}'
                f'<div class="mode-ico" style="background:{m["ico_bg"]};'
                f'color:{m["ico_fg"]};">{m["ico"]}</div>'
                f'<div class="mode-title">{m["title"]}</div>'
                f'<div class="mode-sub">{m["sub"]}</div>'
                f'<div class="mode-spec">{spec_html}</div>'
                f'</div>',
                unsafe_allow_html=True)

            picked = (S.analysis_mode == key)
            if st.button("Selected ✓" if picked else "Select",
                         key=f"mode_{key}",
                         type="primary" if m.get("recommended") else "secondary",
                         use_container_width=True):
                set_mode(key)
                S.step = 2
                st.rerun()

    st.markdown("""<div class="callout tip">
    <b>Not sure?</b> Start with <b>AHP-MCDA + Monte Carlo</b>. It is the method used in the
    Canadian CO₂ basin screening study, and every other mode on this page is a subset of it.
    </div>""", unsafe_allow_html=True)

    with st.expander("🔍  What each analysis gives you — read this before choosing",
                     expanded=False):
        st.markdown("""
The four modes are two independent choices crossed together: **how weights are derived**
(you type them, or you derive them from pairwise comparisons) and **whether uncertainty is
propagated** (one deterministic answer, or a distribution of answers).

| | Deterministic | With Monte Carlo |
|---|---|---|
| **Direct weights** | 1. Direct-weight MCDA | 2. Direct-weight MCDA + MC |
| **AHP pairwise** | 3. AHP-MCDA | 4. AHP-MCDA + MC |

Everything not listed below is identical in all four modes: criteria, alternatives, class
definitions, class assignments, the weighted sum aggregation, Jenks-Fisher tier classification
and the GVF metric. So the mode only changes what you are asked for in Step 3 and what comes
out in Step 7.

---

### 1. Direct-weight MCDA

**What it does:** you type one importance number per criterion. The app rescales them to sum
to 1, multiplies them through your normalised class scores, and sorts the result into tiers.

**Results you get:**
- Composite score Rᵏ ∈ [0, 1] per alternative, with rank and priority tier
- Tier membership cards and a ranked bar chart with Jenks-Fisher boundaries drawn in
- GVF classification quality (above 0.90 = excellent separation), tier count k adjustable 2–6
- Pareto chart of criterion weights with the cumulative 80% line
- Downloads: both charts as PNG (with CSV data for editable chart rebuilding); 4 CSVs (scores and tiers, weights, class definitions,
  assignments); full JSON

**Results you do not get:** any check that your weights are internally consistent, and any
statement about whether the ranking would change if the weights were slightly different.
The ranking is exactly as defensible as the numbers you typed.

**Choose it when:** you want a fast screen, you are teaching the method, or the weights are
already fixed by something outside this app (entropy weighting, a literature value, a client
mandate).

---

### 2. Direct-weight MCDA + Monte Carlo

**What it does:** everything in mode 1, then re-runs the whole analysis N times, each time
redrawing every weight from a normal distribution centred on your typed value.

**Results you get:** everything from mode 1, plus:
- Weight variability table: your weight, mean across simulations, σᵢ, and a stability flag
- Tier stability table: percentage of simulations in which each alternative kept its tier,
  with a Robust or Review verdict against the 99.5% threshold
- Convergence charts for the four heaviest criteria, to show whether N was large enough
- Weight distribution box plot (IQR, ±3σ whiskers, deterministic weight marked)
- Tier stability chart: all alternatives, plus a panel isolating the five closest to a boundary
- 2 extra CSVs and 3 extra chart PNG+CSV exports; σ and iteration count recorded in the JSON

**Results you do not get:** a consistency check on the weights. And note the honest limit of
this mode: the uncertainty being tested is one you specified yourself with the σ slider, so the
robustness claim is only as good as that assumption. Report the σ you used.

**Choose it when:** you are not confident in the weights you typed and want to know whether
the ranking even depends on them.

---

### 3. AHP-MCDA

**What it does:** instead of typing weights, you compare every pair of criteria on Saaty's
1–9 scale (16 criteria = 120 comparisons). Weights are the principal eigenvector of that matrix.

**Results you get:** everything from mode 1, plus:
- Eigenvector-derived weights rather than asserted ones
- Consistency Ratio, Consistency Index and λmax, with CR < 0.10 enforced before you can continue
- An inconsistency diagnostic naming your three worst pairs and what your other judgements
  imply each one should have been

**Results you do not get:** any robustness or uncertainty analysis. You get one ranking from
one consistent matrix.

**Choose it when:** you need weights that are auditable and defensible, but you do not need to
make a claim about uncertainty.

---

### 4. AHP-MCDA + Monte Carlo (recommended)

**What it does:** everything in mode 3, then perturbs the *judgements themselves*: individual
pairwise entries shift one step along the Saaty scale and the weights are re-derived from each
perturbed matrix.

**Results you get:** the complete set. Everything in mode 3, plus every robustness output
listed in mode 2 (weight variability, tier stability against the 99.5% threshold, convergence
charts, weight box plot, tier stability chart, all exports).

**Why this one is stronger than mode 2:** it perturbs the judgements you actually made rather
than their downstream product, so the uncertainty being propagated is a real property of how
you filled the matrix, not a σ you had to invent. This is why it is the defensible construction
for publication, and it is the method behind the Canadian CO₂ basin screening study.

**Choose it when:** the result has to survive review.

---

**A practical order of operations:** run mode 3 (or 1) first to sanity-check your inputs, then
switch to mode 4 (or 2) from Step 7 once the numbers look sensible. Simulating a mistake 20,000
times is expensive and tells you nothing useful.
        """)

    if S.criteria or S.alternatives:
        st.markdown(f"""<div class="callout">
        Your existing work is still here: <b>{len(S.criteria)} criteria</b>,
        <b>{len(S.alternatives)} alternatives</b>. Selecting a mode keeps all of it.
        Weights are only cleared if you switch between direct and pairwise weighting.
        </div>""", unsafe_allow_html=True)
        if st.button("Continue without changing the analysis type →"):
            if S.analysis_mode is None:
                set_mode("ahp_mc")
            S.step = 2; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DEFINE CRITERIA
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 2:
    st.markdown("## Step 2 — What factors will you evaluate on?")


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
                S.criteria = list(EXAMPLE_CRITERIA)
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
            S.step = 3
            st.rerun()

    if S.criteria:
        st.markdown(f"**Currently defined:** {len(S.criteria)} criteria")
        for c in S.criteria:
            st.markdown(f"- {c}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 3:
    st.markdown("## Step 3 — How important is each criterion?")

    cnames = S.criteria
    n_c = len(cnames)

    M_CFG = mode_cfg()
    S.weight_mode = M_CFG["weights"]

    if S.weight_mode == "quick":
        st.markdown(f"""<div class="callout tip">
        You chose <b>{M_CFG['title']}</b>, so weights are entered directly.
        Need Saaty pairwise comparison and a Consistency Ratio instead? Use the
        <b>Change</b> button above to switch to an AHP mode.
        </div>""", unsafe_allow_html=True)
    else:
        n_pairs_preview = n_c*(n_c-1)//2
        st.markdown(f"""<div class="callout tip">
        You chose <b>{M_CFG['title']}</b>, so weights are derived from pairwise
        comparisons: <b>{n_pairs_preview} comparisons</b> for your {n_c} criteria.
        Want to just type weights instead? Use the <b>Change</b> button above to
        switch to a direct-weight mode.
        </div>""", unsafe_allow_html=True)

    if S.weight_mode == "quick":
        st.markdown("""<div class="callout">
        Type a number for each criterion. <b>Higher number = more important.</b>
        They do not need to add up to any particular total — the app rescales them to sum to 1.
        </div>""", unsafe_allow_html=True)

        with st.form("form_weights_quick"):
            # Pre-fill with example weights if example was loaded
            example_w = (list(example_weights())
                         if cnames==EXAMPLE_CRITERIA else [])
            wi = []
            cols = st.columns(min(4, n_c))
            for i, cn in enumerate(cnames):
                with cols[i % len(cols)]:
                    default_val = float(example_w[i]) if (
                        cnames==EXAMPLE_CRITERIA and i<len(example_w)) else round(1/n_c, 4)
                    wi.append(st.number_input(cn[:30], min_value=0.001,
                        max_value=100., value=default_val,
                        step=0.001, format="%.4f", key=f"wq_{i}"))

            col_a, col_b = st.columns(2)
            with col_a:
                back = st.form_submit_button("← Back")
            with col_b:
                fwd = st.form_submit_button("Save weights and continue →", type="primary")

            if back:
                S.step = 2; st.rerun()
            if fwd:
                raw = np.array(wi)
                S.raw_weights = {cn: float(raw[i]) for i,cn in enumerate(cnames)}
                S.weights = raw / raw.sum()
                S.pairwise_M = None; S.cr_ok = True
                S.step = 4; st.rerun()

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
        example_w = (example_weights() if cnames==EXAMPLE_CRITERIA
                     else np.full(n_c,1.0/n_c))

        def ns(v):
            opts=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
            return min(opts,key=lambda x:abs(x-v))

        if n_pairs <= 55:
            pairs=[(cnames[i],cnames[j]) for i in range(n_c) for j in range(i+1,n_c)]
            with st.form("form_weights_expert"):
                upper=[]
                cp=st.columns(min(3,n_pairs))
                for k_i,((a,b)) in enumerate(pairs):
                    if cnames==EXAMPLE_CRITERIA:
                        dv=float(ns(example_matrix(cnames)[cnames.index(a),cnames.index(b)]))
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
                if back: S.step=2; st.rerun()
                if fwd and cr_ok:
                    S.weights=weights_exp; S.pairwise_M=M
                    S.cr_ok=True; S.step=4; st.rerun()
        else:
            # Too many pairs for sliders: present the same judgements as a
            # scrollable, editable table. Not wrapped in a form, so the CR
            # updates live as each judgement is edited.
            SAATY=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
            def lbl(x): return f"1/{int(round(1/x))}" if x<1 else str(int(x))
            L2V={lbl(x):x for x in SAATY}
            OPTS=[lbl(x) for x in SAATY]

            st.markdown(f"""<div class="callout tip">
            <b>{n_pairs} comparisons</b> is too many for sliders, so they are laid out as an
            editable table. Edit the <b>A vs B</b> column only: <b>3</b> means A is moderately
            more important than B, <b>1/3</b> means B is moderately more important than A.
            The Consistency Ratio updates as you edit.
            </div>""", unsafe_allow_html=True)

            pairs=[(cnames[i],cnames[j]) for i in range(n_c) for j in range(i+1,n_c)]
            defaults=[]
            if cnames==EXAMPLE_CRITERIA:
                EM=example_matrix(cnames)
                for a,b in pairs:
                    defaults.append(lbl(ns(EM[cnames.index(a),cnames.index(b)])))
            else:
                defaults=["1"]*len(pairs)
            if cnames==EXAMPLE_CRITERIA:
                st.caption("Pre-filled with the published Canadian CO₂ basin screening "
                           "judgements. Edit any row to explore alternative elicitations.")

            df=pd.DataFrame({"Criterion A":[a for a,_ in pairs],
                             "Criterion B":[b for _,b in pairs],
                             "A vs B":defaults})
            ed_key=f"pair_editor_{n_c}"
            ed=st.data_editor(df, key=ed_key, hide_index=True, use_container_width=True,
                height=min(430, 40+35*n_pairs),
                disabled=["Criterion A","Criterion B"],
                column_config={"A vs B": st.column_config.SelectboxColumn(
                    "A vs B", options=OPTS, required=True, width="small",
                    help="Saaty 1–9 scale. >1 favours A, <1 favours B.")})

            upper=[L2V[v] for v in ed["A vs B"]]
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
                for ri,(ratio,na,nb,actual,ideal) in enumerate(
                        find_inconsistent_pairs(M,cnames,top_n=3),1):
                    st.markdown(f"**{ri}.** *{na}* vs *{nb}* — "
                                f"you said {actual:.2f}, weights suggest {ideal:.2f} "
                                f"(inconsistency: {ratio:.2f}×)")

            b1,b2,b3=st.columns([1,1,1])
            if b1.button("← Back", key="mx_back"): S.step=2; st.rerun()
            if b2.button("↺ Reset judgements", key="mx_reset"):
                S.pop(ed_key, None); st.rerun()
            if b3.button("Save and continue →", key="mx_fwd", type="primary",
                         disabled=not cr_ok):
                S.weights=weights_exp; S.pairwise_M=M
                S.cr_ok=True; S.step=4; st.rerun()

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
        with st.expander("📚  How the example weights were derived, and from what", expanded=False):
            st.markdown("""
The example weights are DERIVED, by AHP, from the six-band criterion importance
hierarchy in `EXAMPLE_BANDS`. The bands are the judgement; the weights are the output.
Band placement is grounded in the four studies that publish criterion weights for CO₂
storage screening:

- **Band 1 — CO₂ source proximity and reservoir–seal pairs.** Source proximity is the only
  criterion placed in the top three by both national-scale studies (Bachu 2003, w = 0.09,
  3rd of 15; Ye et al. 2023, w = 0.12, highest of 16). Reservoir–seal quality is the
  highest-weighted containment criterion in Ye et al. (w = 0.10), and the caprock cluster is
  the largest in Ma et al. (2025) at w = 0.236: containment failure, not injectivity or
  capacity, is the principal threat to permanence.
- **Band 2 — reservoir temperature.** Bachu assigns the geothermal regime w = 0.10, joint
  highest of 15, grounded in a physical argument about CO₂ density and buoyancy;
  Ye et al. independently assign w = 0.08.
- **Bands 3 to 6** follow the same four sources. Freshwater constraint is the sole criterion
  with no published weight anywhere and is placed by the authors' judgement.

Storage capacity is deliberately NOT a ranking criterion. Bachu excluded it because the data
burden exceeded screening scale, and Ye et al. keep the same separation, ranking on
suitability and quantifying capacity only for the basins that rank well. Published
basin-scale capacity exists for only 4 of the 13 Canadian basins, so scoring the other 9
would mean inventing values.

- Bachu, S. (2003). Screening and ranking of sedimentary basins for sequestration of CO₂
  in geological media in response to climate change. *Environmental Geology*, 44(3), 277–289.
  https://doi.org/10.1007/s00254-003-0762-9
- Ye, J., et al. (2023). Evaluation of geological CO₂ storage potential in Saudi Arabian
  sedimentary basins. *Earth-Science Reviews*, 244, 104539.
  https://doi.org/10.1016/j.earscirev.2023.104539
- Wei, N., et al. (2013). A preliminary sub-basin scale evaluation framework of site
  suitability for onshore aquifer-based CO₂ storage in China. *IJGGC*, 12, 231–246.
- Ma, Y., et al. (2025). Evaluation of suitability of CO₂ geologic storage in deep saline
  aquifers in Lindian area of Songliao Basin. *Hydrogeology & Engineering Geology*, 52(1), 238–248.
- Metz, B., et al. (Eds.) (2005). *IPCC Special Report on Carbon Dioxide Capture and Storage.*
- Celia, M.A., Bachu, S., Nordbotten, J.M., & Bandilla, K.W. (2015). Status of CO₂ storage
  in deep saline aquifers. *Water Resources Research*, 51(9), 6846–6892.

These weights are a starting reference only — adjust them to reflect your own judgement
or research context.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ALTERNATIVES
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 4:
    st.markdown("## Step 4 — What are you ranking?")
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
        if S.criteria == EXAMPLE_CRITERIA and not S.alternatives:
            default_alts = "\n".join(EXAMPLE_ALTS)

        alt_txt=st.text_area("Alternatives",value=default_alts,
                             height=220,label_visibility="collapsed")
        col_a,col_b=st.columns(2)
        with col_a: back=st.form_submit_button("← Back")
        with col_b: fwd=st.form_submit_button("Save alternatives and continue →",type="primary")

        if back: S.step=3; st.rerun()
        if fwd:
            anames=[a.strip() for a in alt_txt.strip().split("\n") if a.strip()]
            if len(anames)<2:
                st.error("Please enter at least 2 alternatives.")
                st.stop()
            S.alternatives=anames
            S.assignments={}  # reset assignments if alternatives changed
            S.step=5; st.rerun()

    if S.alternatives:
        st.markdown(f"**Currently defined:** {len(S.alternatives)} alternatives")
        cols=st.columns(min(4,len(S.alternatives)))
        for i,a in enumerate(S.alternatives):
            cols[i%len(cols)].markdown(f"• {a}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — CLASS DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 5:
    st.markdown("## Step 5 — Define the classes for each criterion")
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
        "Basin size":               {"n":5,"labels":["<600 km²","<1,000 km²","<2,000 km²","<200,000 km²",">200,000 km²"],"scores":[1,3,7,15,21]},
        "Reservoir temperature":    {"n":4,"labels":["<40°C","40–70°C","70–100°C",">100°C"],"scores":[1,3,7,15]},
        "Hydrogeological confinement":{"n":3,"labels":["Local","Intermediate","Regional"],"scores":[1,3,5]},
        "Depleted reservoir potential":{"n":5,"labels":["None","Minor (<50 Mboe)","Moderate (50–500 Mboe)","Large (500 Mboe–1 Gboe)","Very large (>1 Gboe)"],"scores":[1,3,5,15,21]},
        "Freshwater constraint":    {"n":4,"labels":["All fresh","Abundant","Some","Limited"],"scores":[1,3,5,7]},
        "Industry maturity":        {"n":5,"labels":["Unexplored","Exploring","Developing","Mature","Very mature"],"scores":[1,2,4,8,10]},
        "Onshore / offshore":       {"n":3,"labels":["Deep offshore","Shallow offshore","Onshore"],"scores":[1,6,10]},
        "Accessibility":            {"n":4,"labels":["Inaccessible","Difficult","Acceptable","Easy"],"scores":[1,3,6,10]},
        "Infrastructure":           {"n":4,"labels":["None","Minor","Moderate","Extensive"],"scores":[1,3,7,10]},
        "CO₂ source proximity":     {"n":5,"labels":["None","Minor","Moderate","Major <200 km","Major (within basin)"],"scores":[1,3,7,15,15]},
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
        if st.button("← Back", key="back_s4"): S.step=4; st.rerun()
    with col_b:
        if st.button("Save class definitions and continue →",
                     type="primary", key="fwd_s4", disabled=not all_valid):
            S.class_defs = new_class_defs
            S.step = 6; st.rerun()

    if not all_valid:
        st.markdown('<div class="callout warn">⚠️ Fix the issues above before continuing.'
                    '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — ASSIGNMENTS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 6:
    st.markdown("## Step 6 — Assign a class to each alternative on each criterion")
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
            "Leakage via outcrops":"Low",
            "Basin size":">200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Regional","Depleted reservoir potential":"Very large (>1 Gboe)",
            "Freshwater constraint":"Some","Industry maturity":"Very mature",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Extensive","CO₂ source proximity":"Major (within basin)"},
        "Williston Basin":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"Bedded","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Low",
            "Basin size":">200,000 km²","Reservoir temperature":"70–100°C",
            "Hydrogeological confinement":"Regional","Depleted reservoir potential":"Large (500 Mboe–1 Gboe)",
            "Freshwater constraint":"Limited","Industry maturity":"Mature",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Extensive","CO₂ source proximity":"Major (within basin)"},
        "Michigan Basin":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"Bedded","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Low",
            "Basin size":"<200,000 km²","Reservoir temperature":"40–70°C",
            "Hydrogeological confinement":"Regional","Depleted reservoir potential":"Large (500 Mboe–1 Gboe)",
            "Freshwater constraint":"Some","Industry maturity":"Mature",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Extensive","CO₂ source proximity":"Major (within basin)"},
        "NL Offshore":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Intermediate",
            "Basin size":"<200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Large (500 Mboe–1 Gboe)",
            "Freshwater constraint":"Abundant","Industry maturity":"Mature",
            "Onshore / offshore":"Shallow offshore","Accessibility":"Acceptable",
            "Infrastructure":"Moderate","CO₂ source proximity":"Major (within basin)"},
        "Scotian Basin":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Intermediate",
            "Basin size":">200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Moderate (50–500 Mboe)",
            "Freshwater constraint":"Abundant","Industry maturity":"Mature",
            "Onshore / offshore":"Deep offshore","Accessibility":"Acceptable",
            "Infrastructure":"Moderate","CO₂ source proximity":"Minor"},
        "Flemish Pass":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Good",
            "Leakage via outcrops":"Intermediate",
            "Basin size":"<200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Minor (<50 Mboe)",
            "Freshwater constraint":"Abundant","Industry maturity":"Developing",
            "Onshore / offshore":"Deep offshore","Accessibility":"Difficult",
            "Infrastructure":"Minor","CO₂ source proximity":"Minor"},
        "Beaufort-Mackenzie":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"None","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Intermediate",
            "Basin size":"<200,000 km²","Reservoir temperature":">100°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Minor (<50 Mboe)",
            "Freshwater constraint":"Some","Industry maturity":"Developing",
            "Onshore / offshore":"Onshore","Accessibility":"Difficult",
            "Infrastructure":"Minor","CO₂ source proximity":"None"},
        "Hudson Bay":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"None","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Intermediate",
            "Basin size":">200,000 km²","Reservoir temperature":"40–70°C",
            "Hydrogeological confinement":"Local","Depleted reservoir potential":"None",
            "Freshwater constraint":"All fresh","Industry maturity":"Unexplored",
            "Onshore / offshore":"Shallow offshore","Accessibility":"Difficult",
            "Infrastructure":"None","CO₂ source proximity":"Minor"},
        "St. Lawrence":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"Domes","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Intermediate",
            "Basin size":"<200,000 km²","Reservoir temperature":"40–70°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Moderate (50–500 Mboe)",
            "Freshwater constraint":"All fresh","Industry maturity":"Developing",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Moderate","CO₂ source proximity":"Major (within basin)"},
        "Maritimes (onshore)":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"Bedded","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Intermediate",
            "Basin size":"<200,000 km²","Reservoir temperature":"40–70°C",
            "Hydrogeological confinement":"Intermediate","Depleted reservoir potential":"Minor (<50 Mboe)",
            "Freshwater constraint":"Some","Industry maturity":"Exploring",
            "Onshore / offshore":"Onshore","Accessibility":"Easy",
            "Infrastructure":"Minor","CO₂ source proximity":"Major <200 km"},
        "Maritimes (offshore)":{"Tectonic stability":"50–100 cm/s²","Fault and fracture intensity":"Moderate",
            "Evaporites":"Bedded","Reservoir–seal pairs":"Intermediate",
            "Leakage via outcrops":"Low",
            "Basin size":"<200,000 km²","Reservoir temperature":"70–100°C",
            "Hydrogeological confinement":"Regional","Depleted reservoir potential":"None",
            "Freshwater constraint":"Abundant","Industry maturity":"Exploring",
            "Onshore / offshore":"Shallow offshore","Accessibility":"Difficult",
            "Infrastructure":"Minor","CO₂ source proximity":"Major <200 km"},
        "Arctic Islands":{"Tectonic stability":"<50 cm/s²","Fault and fracture intensity":"Low",
            "Evaporites":"None","Reservoir–seal pairs":"Poor",
            "Leakage via outcrops":"Intermediate",
            "Basin size":"<200,000 km²","Reservoir temperature":"<40°C",
            "Hydrogeological confinement":"Local","Depleted reservoir potential":"None",
            "Freshwater constraint":"All fresh","Industry maturity":"Unexplored",
            "Onshore / offshore":"Onshore","Accessibility":"Inaccessible",
            "Infrastructure":"None","CO₂ source proximity":"None"},

        "Pacific Margin":{"Tectonic stability":">200 cm/s²","Fault and fracture intensity":"High",
            "Evaporites":"None","Reservoir–seal pairs":"Poor",
            "Leakage via outcrops":"Very high",
            "Basin size":"<200,000 km²","Reservoir temperature":"70–100°C",
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
        if st.button("← Back", key="back_s5"): S.step=5; st.rerun()
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
            S.step = 7; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif S.step == 7:
    st.markdown("## Step 7 — Results")

    cnames = S.criteria; anames = S.alternatives
    weights = S.weights; scores = S.scores
    tiers = S.tiers; breaks = S.breaks; P = S.P
    n_tiers = S.n_tiers; n_a = len(anames); n_c = len(cnames)

    df_mc = None; df_st = None  # populated inside expander after MC runs

    tier_labels = {1:"🟢 Tier 1 — Priority",2:"🟡 Tier 2 — Secondary",
                   3:"🟠 Tier 3 — Low priority",4:"⚫ Tier 4 — Marginal"}
    tier_labels_plain = {1:"Tier 1 - Priority",2:"Tier 2 - Secondary",
                         3:"Tier 3 - Low priority",4:"Tier 4 - Marginal"}
    tier_css = {1:"result-tier-1",2:"result-tier-2",3:"result-tier-3",4:"result-tier-4"}

    M_CFG = mode_cfg()

    st.markdown(f'<div class="mode-chip"><span class="dot"></span>'
                f'{M_CFG["title"]} &nbsp;·&nbsp; {M_CFG["method"]}</div>',
                unsafe_allow_html=True)

    # ── TIER SETTINGS (always available) ──────────────────────────────────────
    def _tier_control(widget_key):
        new_k = st.slider("Number of tiers (k)", 2, 6, S.n_tiers, key=widget_key)
        if new_k != S.n_tiers:
            S.n_tiers = new_k
            if n_a >= new_k:
                breaks_new, _ = jenks(scores, new_k)
                S.breaks = breaks_new
                S.tiers = assign_tiers(scores, breaks_new)
            st.rerun()

    # ── DETERMINISTIC MODES — no Monte Carlo section ──────────────────────────
    if not M_CFG["mc"]:
        with st.expander("⚙️  Classification settings", expanded=False):
            _tier_control("k_det")
        st.markdown("""<div class="callout tip">
        <b>This is a deterministic run.</b> One set of weights in, one ranking out.
        It does not tell you whether the ranking would survive small changes to your
        weights. To find out, switch to the matching Monte Carlo mode — your criteria,
        weights, classes and assignments all carry over, so you will land straight
        back on this page.
        </div>""", unsafe_allow_html=True)
        target = "ahp_mc" if M_CFG["weights"] == "expert" else "mcda_mc"
        if st.button(f"🎲  Add robustness validation ({MODES[target]['title']})",
                     type="primary", key="upgrade_mc"):
            set_mode(target); st.rerun()

    # ── SETTINGS + ROBUSTNESS TEST (Monte Carlo modes only) ───────────────────
    mc_box = (st.expander("⚙️  Simulation settings & Monte Carlo robustness test",
                          expanded=False)
              if M_CFG["mc"] else nullcontext())
    with mc_box:
      if M_CFG["mc"]:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            S.n_iter = st.select_slider("Simulations (N)",
                options=[1000,2000,5000,10000,20000,50000], value=S.n_iter)
        with sc2:
            S.p_perturb = st.slider("Perturbation probability", .10, .50, S.p_perturb, .05,
                help="Share of judgements (AHP) or weights (direct) disturbed per simulation.")
        with sc3:
            _tier_control("k_mc")

        if S.weight_mode == "quick":
            S.w_sigma_pct = st.slider("Weight uncertainty σ (% of each weight)",
                1.0, 20.0, float(S.w_sigma_pct), 0.5,
                help="Standard deviation of the Gaussian noise applied to each weight, "
                     "as a percentage of that weight. This is a reportable assumption: "
                     "it is written into the JSON export.")
            st.markdown(f"""<div class="callout warn">
            You are perturbing <b>typed weights</b>, so the app has no pairwise judgements
            to disturb. Instead each weight is drawn from a normal distribution centred on
            your value with σ = <b>{S.w_sigma_pct:.1f}%</b> of that weight. This σ is your
            assumption, not a property of the data — state it in any write-up.
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        mech = ("individual pairwise judgements one step along the Saaty scale, "
                "re-deriving weights from each perturbed matrix"
                if S.weight_mode == "expert"
                else f"each weight with Gaussian noise (σ = {S.w_sigma_pct:.1f}% of the weight)")
        st.markdown(f"""<div class="callout">
        The app will rerun your entire analysis <b>{S.n_iter:,} times</b>, each time
        disturbing {mech}, simulating expert judgement uncertainty.
        If rankings and tier assignments stay stable across those runs, your results
        are <b>robust and trustworthy</b>.
        An alternative is considered <b>robustly classified</b> if it stays in its tier
        in &ge; 99.5% of simulations.
        </div>""", unsafe_allow_html=True)

        pm = S.pairwise_M if S.weight_mode == "expert" else None

        if st.button(f"▶  Run {S.n_iter:,} simulations", type="primary"):
            with st.spinner(f"Running {S.n_iter:,} simulations — please wait..."):
                aw, as_ = run_mc(weights, P, pm, S.n_iter, S.p_perturb, S.w_sigma_pct)
                S.mc_w = aw; S.mc_s = as_; S.ran_mc = True
                S.sigma = aw.std(axis=0); S.mu = aw.mean(axis=0)
                # Correctly compute tier stability: for each simulation run,
                # classify ALL alternatives together against the shared breaks,
                # then check each alternative's tier against its deterministic tier.
                all_sim_tiers = np.array([assign_tiers(as_[s,:], breaks)
                                          for s in range(S.n_iter)])
                S.all_sim_tiers = all_sim_tiers
                S.stab = [(all_sim_tiers[:,b]==tiers[b]).mean()*100
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
                'Your weight': weights,                    # float
                'Avg across simulations': S.mu,            # float
                'Variability (σᵢ)': sig,                   # float
                'Stable?': ['Yes' if s<=0.010 else 'Check' for s in sig]
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
                'Score': scores,                           # float
                'Tier': [tier_labels_plain.get(t, f"Tier {t}") for t in tiers],
                'Tier stability (%)': stab,                # float, column name clarifies unit
                'Verdict': ['Robust' if p>=99.5 else 'Review' for p in stab]
            })
            st.dataframe(df_st, use_container_width=True, hide_index=True)

            if all(p>=99.5 for p in stab):
                st.markdown(f"""<div class="callout ok">
                ✅ <b>All {n_a} alternatives are robustly classified.</b>
                Every option retained its tier in &ge; 99.5% of {S.n_iter:,} simulations.
                </div>""", unsafe_allow_html=True)
            else:
                n_w = sum(p<99.5 for p in stab)
                st.markdown(f"""<div class="callout warn">
                ⚠️ <b>{n_w} alternative(s) are below 99.5% stability.</b>
                These are close to tier boundaries — review their criteria scores carefully.
                </div>""", unsafe_allow_html=True)

            # Convergence plots
            st.markdown("**Convergence charts:**")
            st.caption("Use these to judge whether N is large enough. "
                       "If curves flatten well before the right edge, you can lower N and re-run.")
            fig_cv = plot_conv(aw, cnames, weights, top_n=min(4,n_c))
            st.pyplot(fig_cv, use_container_width=True)
            # Build CSV: running mean of each weight per iteration
            if aw is not None:
                _conv_arr = np.array(aw)
                _cum_mean = np.cumsum(_conv_arr, axis=0) / (np.arange(1, len(_conv_arr)+1)[:,None])
                _conv_df = pd.DataFrame(_cum_mean, columns=cnames)
                _conv_df.insert(0, "Iteration", range(1, len(_conv_arr)+1))
            else:
                _conv_df = None
            fig_download_buttons(fig_cv, "convergence_chart",
                                 "Monte Carlo Weight Convergence", csv_df=_conv_df)
            plt.close(fig_cv)
            wlab = "AHP" if S.weight_mode == "expert" else "Direct"
            fig_wb = plot_weights_box(aw, weights, cnames, method_label=wlab)
            st.pyplot(fig_wb, use_container_width=True)
            # Build CSV: all simulated weights per criterion
            if aw is not None:
                _wb_df = pd.DataFrame(np.array(aw), columns=cnames)
                _wb_df.insert(0, "Simulation", range(1, len(_wb_df)+1))
            else:
                _wb_df = None
            fig_download_buttons(fig_wb, "weights_box_plot",
                                 f"Monte Carlo {wlab} Weight Distributions",
                                 csv_df=_wb_df)
            plt.close(fig_wb)
            if S.mc_s is not None:
                fig_sb = plot_score_box(S.mc_s, anames, scores, tiers, breaks, S.n_iter)
                st.pyplot(fig_sb, use_container_width=True)
                # Build CSV: all simulated composite scores per alternative
                if S.mc_s is not None:
                    _sb_df = pd.DataFrame(np.array(S.mc_s), columns=anames)
                    _sb_df.insert(0, "Simulation", range(1, len(_sb_df)+1))
                else:
                    _sb_df = None
                fig_download_buttons(fig_sb, "score_distributions",
                                     "Composite Score Distributions vs Tier Boundaries",
                                     csv_df=_sb_df)
                plt.close(fig_sb)
        else:
            st.info("Run a simulation above to see robustness results and convergence charts here.")

    # ── DETERMINISTIC RESULTS ─────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.15rem;font-weight:700;color:#1B3A5C;'
                'border-bottom:2.5px solid #D0DAE8;padding-bottom:.38rem;'
                'margin:1.2rem 0 .9rem;">Rankings and tier classification</div>',
                unsafe_allow_html=True)

    # Score table
    df_s = pd.DataFrame({
        'Alternative': anames,
        'Composite Score (0-1)': [f"{s:.4f}" for s in scores],
        'Rank': pd.Series(scores).rank(ascending=False).astype(int).values,
        'Priority Tier': [tier_labels_plain.get(t, f"Tier {t}") for t in tiers]
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
        st.pyplot(fig_s, use_container_width=True)
        _scores_df = pd.DataFrame({
            "Alternative": anames,
            "Composite Score": scores,
            "Tier": tiers,
            "Rank": pd.Series(scores).rank(ascending=False).astype(int).values,
        }).sort_values("Rank").reset_index(drop=True)
        fig_download_buttons(fig_s, "basin_scores_chart",
                             "Basin Composite Scores and Tier Classification",
                             csv_df=_scores_df)
        plt.close(fig_s)
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
    st.pyplot(fig_p, use_container_width=True)
    _pareto_df = pd.DataFrame({
        "Criterion": cnames,
        "Weight": weights,
        "Weight Pct": weights * 100,
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    # Cumulative weight must be computed AFTER sorting so it matches the sorted order
    _pareto_df["Cumulative Weight"] = _pareto_df["Weight"].cumsum()
    _pareto_df = _pareto_df[["Criterion", "Weight", "Cumulative Weight", "Weight Pct"]]
    fig_download_buttons(fig_p, "pareto_weights_chart",
                         "AHP Criterion Weights — Pareto Chart",
                         csv_df=_pareto_df)
    plt.close(fig_p)

    # ── EXPORT ────────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.15rem;font-weight:700;color:#1B3A5C;'
                'border-bottom:2.5px solid #D0DAE8;padding-bottom:.38rem;'
                'margin:1.2rem 0 .9rem;">Download results</div>',
                unsafe_allow_html=True)

    e1,e2 = st.columns(2)
    with e1:
        # Build a float-score version for export so Excel can chart/sort numerically
        df_s_export = pd.DataFrame({
            'Alternative': anames,
            'Composite Score (0-1)': scores,          # float, not formatted string
            'Rank': pd.Series(scores).rank(ascending=False).astype(int).values,
            'Priority Tier': [tier_labels_plain.get(t, f"Tier {t}") for t in tiers],
        }).sort_values('Rank').reset_index(drop=True)
        st.download_button("📥  Scores and tiers (CSV)",
            df_s_export.to_csv(index=False).encode("utf-8-sig"), "scores_tiers.csv", "text/csv",
            use_container_width=True)
    with e2:
        df_w_exp = pd.DataFrame({'Criterion':cnames,'Weight':weights,'Share_pct':weights*100})
        st.download_button("📥  Criterion weights (CSV)",
            df_w_exp.to_csv(index=False).encode("utf-8-sig"), "weights.csv", "text/csv",
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
            df_cls.to_csv(index=False).encode("utf-8-sig"), "class_definitions.csv", "text/csv",
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
            df_asgn.to_csv(index=False).encode("utf-8-sig"), "assignments.csv", "text/csv",
            use_container_width=True)

    if S.ran_mc and S.sigma is not None and df_mc is not None and df_st is not None:
        e5,e6 = st.columns(2)
        with e5:
            st.download_button("📥  Monte Carlo weight stats (CSV)",
                df_mc.to_csv(index=False).encode("utf-8-sig"), "mc_weights.csv", "text/csv",
                use_container_width=True)
        with e6:
            st.download_button("📥  Tier stability results (CSV)",
                df_st.to_csv(index=False).encode("utf-8-sig"), "tier_stability.csv", "text/csv",
                use_container_width=True)

    # Full JSON export
    rd = {
        'analysis_mode': S.analysis_mode,
        'analysis_mode_label': M_CFG['title'],
        'method': M_CFG['method'],
        'weight_derivation': 'ahp_pairwise' if S.weight_mode=='expert' else 'direct',
        'monte_carlo': bool(M_CFG['mc'] and S.ran_mc),
        'criteria': cnames, 'alternatives': anames,
        'weights': {c:float(w) for c,w in zip(cnames,weights)},
        'class_definitions': {cn: S.class_defs.get(cn,{}) for cn in cnames},
        'assignments': S.assignments,
        'scores': {a:float(s) for a,s in zip(anames,scores)},
        'tiers': {a:int(t) for a,t in zip(anames,tiers)},
        'tier_breaks': [float(b) for b in breaks],
    }
    if M_CFG['mc'] and S.ran_mc:
        rd['n_iterations'] = S.n_iter
        rd['perturbation_probability'] = S.p_perturb
        rd['perturbation_mechanism'] = ('saaty_scale_step' if S.weight_mode=='expert'
                                        else 'gaussian_weight_noise')
        if S.weight_mode == 'quick':
            rd['weight_sigma_pct'] = float(S.w_sigma_pct)
    if S.sigma is not None:
        rd['mc_sigma']={c:float(s) for c,s in zip(cnames,S.sigma)}
        rd['mc_stability']={a:float(p) for a,p in zip(anames,S.stab)}
    st.download_button("📥  Full results (JSON)",
        json.dumps(rd,indent=2), "full_results.json",
        "application/json", use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Edit assignments (Step 6)"): S.step=6; st.rerun()
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
