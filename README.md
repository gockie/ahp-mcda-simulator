# AHP-MCDA Monte Carlo Simulator

**A free, open-access web app for Analytic Hierarchy Process (AHP) multi-criteria decision analysis with Monte Carlo weight uncertainty validation.**

Built alongside the research paper:

> Okwaraojimadu, C.K. & Ezekiel, C.J. (2025). *Canadian CO₂ Storage Basin Screening: An Advanced AHP-MCDA Approach with Monte Carlo Uncertainty Validation.* University of Calgary, MSc Research.

---

## What it does

The app implements a complete AHP-MCDA workflow in two modes:

**Quick Mode** — Enter criterion weights directly. Useful if you already have weights from your own AHP analysis or previous study.

**Expert Mode** — Build the full pairwise comparison matrix using Saaty's 1–9 scale. The app derives weights using the eigenvector method, computes the Consistency Ratio (CR), and warns you if CR > 0.10.

Both modes then:
- Compute composite suitability scores for all alternatives (Rᵏ = Σ wᵢ · Pᵢₖ)
- Classify alternatives into priority tiers using Jenks–Fisher statistical optimisation (GVF reported)
- Run a Monte Carlo weight perturbation analysis (N = 1,000 to 50,000 iterations)
- Report per-criterion weight standard deviations (σᵢ) and per-alternative tier stability percentages
- Generate convergence figures showing that results stabilise well before the chosen N
- Export all results as CSV, PNG figures, and JSON

---

## Live demo

**[Launch the app →](https://your-app-url.streamlit.app)**

The app loads the Canadian Basin Screening example by default (13 basins, 16 criteria, from the paper above). You can replace this with your own criteria, alternatives, and scores for any MCDA application.

---

## Run locally

**Requirements:** Python 3.10 or higher.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/ahp-mcda-simulator.git
cd ahp-mcda-simulator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud (free)

1. Fork or clone this repository to your own GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**
4. Select your repository, branch (`main`), and set the main file path to `app.py`
5. Click **Deploy** — your app is live in about 2 minutes

---

## File structure

```
ahp-mcda-simulator/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## How the Monte Carlo simulation works

For each of N iterations:

1. A perturbation noise value is drawn from N(0, σᵢ) for each criterion weight, where σᵢ ≈ 3% of the deterministic weight
2. Each weight is perturbed with probability p (default 0.30), mimicking the effect of ±1 Saaty scale unit variation
3. Negative weights are clamped to a small positive floor (0.005)
4. All weights are renormalised to sum to 1
5. Composite scores are recomputed for all alternatives using the perturbed weights
6. Jenks–Fisher tier assignments are reapplied
7. The fraction of iterations in which each alternative retains its deterministic tier is recorded as the tier stability percentage

A basin (or alternative) with ≥ 99.5% tier stability across all N iterations is considered robustly classified. Convergence of the running σᵢ statistic is plotted across checkpoints from N = 100 to N_max.

---

## Citation

If you use this tool in your research, please cite:

```
Okwaraojimadu, C.K. & Ezekiel, C.J. (2025). Canadian CO₂ Storage Basin Screening:
An Advanced AHP-MCDA Approach with Monte Carlo Uncertainty Validation.
University of Calgary, MSc Research.
```

**Code contact:** chisom.okwaraojimadu@ucalgary.ca

---

## Licence

MIT Licence — free to use, modify, and distribute with attribution.
