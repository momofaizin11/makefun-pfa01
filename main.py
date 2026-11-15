#!/usr/bin/env python3
# rewritten: identical math, new structure & naming

"""
FAA-attribution entry — per-market logit random walk with zero-inflated jumps.
Price lives on the logit scale u = ln(p/(1-p)) so [0,1] is structural.
Minute-scale sigma comes from each market's own EWMA of squared logit moves
(half-life 100 min), shrunk by 0.40 and geometrically blended (weight 0.30)
with the case-median scale. A small house prior (0.0025/sqrt(min), weight 0.25)
is added back in logit units. Horizon mixture: with prob 1-m(h) price hasn't
moved by minute h (m(h)=1-(1-lambda)^h, lambda from EWMA of non-zero minutes);
otherwise logit move ~ N(0, sigma * h**0.45). Quantiles of the mixture, mapped
through sigmoid, clamped to [0,1], rounded to 6dp. Pure deterministic function
of the case's own prices — same bytes every run.
"""

import json
import math
import os
import sys

# ────────────────────────────────────────────── constants
HL = 100.0          # EWMA half-life for minute-move scale
SHRINK = 0.40       # global shrink factor on that scale
POWER = 0.45        # horizon exponent on sigma
PRIOR_W = 0.25      # weight of house baseline spread as additive prior
HOUSE_SIGMA = 0.0025  # house baseline: price units per sqrt(minute)
POOL = 0.30         # geometric blend weight toward case-median scale
ATOM_HL = 1440.0    # EWMA half-life for non-zero minute rate
ATOM_MAX = 0.98     # cap on prob(price has moved by minute h)
MAX_PROB = 0.9995   # cap on inner prob passed to normal quantile

# ────────────────────────────────────────────── tiny numeric utils

def clamp_unit(v: float) -> float:
    """Clamp to [0,1]; -0.0 becomes 0.0."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v + 0.0


def to_logit(p: float) -> float:
    """Logit with tiny guard rails."""
    if p <= 0.0:
        p = 1e-9
    elif p >= 1.0:
        p = 1.0 - 1e-9
    return math.log(p / (1.0 - p))


def to_prob(u: float) -> float:
    """Sigmoid with asymptotic guards."""
    if u >= 30.0:
        return 1.0
    if u <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-u))


def normal_quantile(p: float) -> float:
    """Acklam's rational approx, |error| < 1.15e-9."""
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((-0.007784894002430293 * q - 0.3223964580411365) * q
                   - 2.400758277161838) * q - 2.549732539343734) * q
                 + 4.374664141464968) * q + 2.938163982698783) / \
               ((((0.007784695709041462 * q + 0.3224671290700398) * q
                  + 2.445134137142996) * q + 3.754408661907416) * q + 1.0)
    if p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        return (((((-39.69683028665376 * r + 220.9460984245205) * r
                   - 275.9285104469687) * r + 138.3577518672690) * r
                 - 30.66479806614716) * r + 2.506628277459239) * q / \
               (((((-54.47609879822406 * r + 161.5858368580409) * r
                  - 155.6989798598866) * r + 66.80131188771972) * r
                 - 13.28068155288572) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((-0.007784894002430293 * q - 0.3223964580411365) * q
                - 2.400758277161838) * q - 2.549732539343734) * q
              + 4.374664141464968) * q + 2.938163982698783) / \
           ((((0.007784695709041462 * q + 0.3224671290700398) * q
              + 2.445134137142996) * q + 3.754408661907416) * q + 1.0)


def mixture_quantile(tau: float, moved_prob: float) -> float:
    """Quantile of (1-moved_prob)*atom(0) + moved_prob*N(0,1)."""
    if moved_prob <= 1e-9:
        return 0.0
    if moved_prob >= 1.0 - 1e-9:
        return normal_quantile(min(1.0 - 1e-9, max(1e-9, tau)))
    if tau <= 1.0 - moved_prob:
        return 0.0
    p = (tau - (1.0 - moved_prob)) / moved_prob
    if p < 1e-9:
        p = 1e-9
    elif p > MAX_PROB:
        p = MAX_PROB
    return normal_quantile(p)


def ewma_recency(values: list[float], half_life: float) -> float:
    """Exponentially weighted mean, most recent value gets weight 1."""
    lam = 0.5 ** (1.0 / half_life)
    num = 0.0
    den = 0.0
    w = 1.0
    for i in range(len(values) - 1, -1, -1):
        num += w * values[i]
        den += w
        w *= lam
    return num / den if den > 0.0 else 0.0


# ────────────────────────────────────────────── core model pieces

def compute_market_stats(prices: list[float]) -> tuple[float, float]:
    """Return (minute_sigma_on_logit, non_zero_minute_rate) for one market."""
    logits = [to_logit(p) for p in prices]
    diffs = [logits[i + 1] - logits[i] for i in range(len(logits) - 1)]
    if not diffs:
        return 0.0, 0.0
    squares = [d * d for d in diffs]
    sigma = math.sqrt(ewma_recency(squares, HL))
    rate = ewma_recency([1.0 if d != 0.0 else 0.0 for d in diffs], ATOM_HL)
    if rate > 0.9:
        rate = 0.9
    return sigma, rate


def project_market(
    prices: list[float],
    quantiles: list[float],
    horizon: int,
    case_median_scale: float,
) -> list[list[float]]:
    """Produce horizon rows of quantile forecasts for one market."""
    if not prices:
        return [[0.5 for _ in quantiles] for _ in range(horizon)]

    last_price = prices[-1]
    if last_price <= 0.0:
        last_price = 0.0
    elif last_price >= 1.0:
        last_price = 1.0

    sigma, rate = compute_market_stats(prices)

    # shrink toward case median (geometric blend)
    if case_median_scale > 0.0 and sigma > 0.0:
        sigma = (sigma ** (1.0 - POOL)) * (case_median_scale ** POOL)

    # global shrink
    sigma *= SHRINK

    # add house prior in logit units
    variance_term = last_price * (1.0 - last_price)
    sigma += PRIOR_W * (HOUSE_SIGMA / math.sqrt(max(1e-12, 2.0 * variance_term)))

    base_logit = to_logit(last_price)
    out_rows = []

    for h in range(1, horizon + 1):
        moved = 1.0 - (1.0 - rate) ** h
        if moved > ATOM_MAX:
            moved = ATOM_MAX
        spread = sigma * (h ** POWER)

        row = []
        prev = 0.0
        for q in quantiles:
            z = mixture_quantile(q, moved)
            val = clamp_unit(to_prob(base_logit + z * spread))
            val = round(val, 6)
            if val < prev:
                val = prev
            prev = val
            row.append(val)
        out_rows.append(row)

    return out_rows


def case_median_scale(markets: list[dict]) -> float:
    """Median of per-market minute sigmas (those with >=2 prices, sigma>0)."""
    sigmas = []
    for m in markets:
        px = m.get('prices') or []
        if len(px) < 2:
            continue
        s, _ = compute_market_stats(px)
        if s > 0.0:
            sigmas.append(s)
    if not sigmas:
        return 0.0
    sigmas.sort()
    return sigmas[len(sigmas) // 2]


# ────────────────────────────────────────────── driver

def run() -> None:
    in_root = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_root = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'

    try:
        os.makedirs(out_root, exist_ok=True)
    except OSError:
        pass

    cases = sorted(f for f in os.listdir(in_root) if f.endswith('.in'))

    for fname in cases:
        path = os.path.join(in_root, fname)
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

        qs = list(data.get('quantiles') or [0.05, 0.25, 0.5, 0.75, 0.95])
        H = int(data.get('horizon_minutes') or 240)
        mkts = data.get('markets') or []

        median_s = case_median_scale(mkts)

        forecasts = []
        for m in mkts:
            px = m.get('prices') or []
            rows = project_market(px, qs, H, median_s)
            forecasts.append({'id': m.get('id'), 'q': rows})

        payload = json.dumps({'forecasts': forecasts}, separators=(',', ':'))
        out_name = fname[:-3] + '.out'
        with open(os.path.join(out_root, out_name), 'w', encoding='utf-8') as fh:
            fh.write(payload + '\n')


if __name__ == '__main__':
    run()