# Topological Recursion (Eynard–Orantin) Engine for ETFs

Applies the Eynard–Orantin topological recursion to the eigenvalue distribution of ETF correlation matrices. Extracts the first symplectic invariant (disc amplitude) and projects it onto each ETF to obtain a novel signal.

## Features
- Three ETF universes
- Seven rolling windows (63–4536 days)
- Spectral curve from eigenvalue density (smoothed KDE)
- Disc amplitude = derivative of log density
- Per‑ETF score = eigenvector-weighted sum of disc amplitude at each eigenvalue
- Best window automatically selected
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-topological-recursion-results`

## Usage

1. Set `HF_TOKEN`.
2. Run `python train.py`.
3. Run `streamlit run streamlit_app.py`.
4. GitHub Actions daily.

## Interpretation

- The topological recursion is a universal method to compute all correlation functions of a random matrix model.
- The disc amplitude indicates how sensitive the eigenvalue distribution is to small perturbations.
- ETFs with high score have eigenvectors aligning with eigenvalues in regions of steep spectral density – they are most responsive to regime shifts.

## Requirements

See `requirements.txt`.
