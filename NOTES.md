# polymarket-faa-attribution-01 entry

## Model
Per-market logit random walk with zero-inflated jump shape:
- Prices carried on logit scale u = log(p/(1-p)) → structural [0,1] bound
- Minute scale: EWMA (HL=100) of squared logit moves → sd; EWMA (HL=1440) of non-zero indicator → lam (capped 0.9)
- Global shrink 0.40, geometric blend with case median scale (weight 0.30)
- House prior: +0.25 * (0.0025 / sqrt(2*x*(1-x))) converted to logit units
- Horizon mixture: probability price moved by minute h is m(h) = 1-(1-lam)^h (capped 0.98)
  - Atom at 0 with weight 1-m(h) → inner quantiles sit on last price
  - Otherwise N(0,1) scaled by sd * h^0.45
- Five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) from mixture, mapped through sigmoid, clamped [0,1], rounded 6dp
- Markets with no prices: all quantiles = 0.5

## Verification (local scorer /root/makefun/work/pfa02/work/score.py)
- 6 practice cases: **loss 0.270813** (target 0.270813 ± 0.000002) ✓
- All 6 output files valid: correct market counts (19/19/19/23/24/24), 240 rows × 5 quantiles, monotone ✓
- Deterministic: two runs → byte-identical outputs ✓
- Time per case: < 0.5s (limit 8s) ✓
- Largest output file: ~265 KB (limit 1 MiB) ✓

## Build
Commit sha: (to be filled after commit)
