#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry: a per-market logit random walk with a zero-inflated
(atom at the current price) jump shape and a scale estimated from each market's own prices.

What it does, for every market of the case:

  * The price is carried on the logit scale, u = log(p / (1 - p)), so the [0, 1] bound is structural
    and a move is multiplicative in the odds (a market at 0.02 and one at 0.4 both get a band that
    never leaves [0, 1]).
  * The scale of one minute's move is an EWMA (half-life 100 minutes) of squared logit moves of that
    market's own prices, shrunk by a global factor (0.40, calibrated so the band matches what a
    market of that recent activity actually does over the next hours) and blended (geometric, weight
    0.30) with the case's median market scale, which steadies the estimate for quiet or very short
    histories.
  * The house baseline's flat-price spread (0.0025 per square-root minute) is added back as a small
    prior (weight 0.25), converted to logit units at that market's level, so a market that has not
    moved for hours still gets a sane band.
  * The horizon shape is a mixture: with probability 1 - m(h) the price has not moved at all by
    minute h (m(h) = 1 - (1 - lambda)^h, lambda the EWMA rate of non-zero one-minute moves), and
    otherwise the logit move is normal with standard deviation s * h**0.45.  Most minutes of these
    markets are exactly flat, so the atom is what makes the inner quantiles sit on the last price
    instead of paying for a band nobody trades into, while the tail still covers the jumps.
  * The five (or however many) quantiles the case asks for are the quantiles of that mixture,
    mapped back through the sigmoid, clamped to [0, 1] and rounded to 6 decimals.

Everything comes from the case's own prices: no network, no state, no randomness, so two runs of the
same case produce the same bytes.
"""

import json
import math
import os
import sys

# ---------------------------------------------------------------- model constants

HL = 100.0          # EWMA half-life (minutes) of the per-minute move scale
SHRINK = 0.40       # global shrink on that scale
POWER = 0.45        # horizon exponent
PRIOR_W = 0.25      # weight of the house baseline's spread as an additive prior
HOUSE_SIGMA = 0.0025  # the house baseline's spread, price units per square-root minute
POOL = 0.30         # geometric blend weight with the case's median market scale
ATOM_HL = 1440.0    # EWMA half-life (minutes) of the rate of non-zero minutes
ATOM_MAX = 0.98     # cap on the probability that the price has moved by minute h
MAX_PROB = 0.9995   # cap on the inner probability passed to the normal quantile


# ---------------------------------------------------------------- small numeric helpers

def clip01(v):
    """v clamped to [0, 1], with -0.0 turned into 0.0."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v + 0.0


def logit(p):
    if p <= 0.0:
        p = 1e-9
    elif p >= 1.0:
        p = 1.0 - 1e-9
    return math.log(p / (1.0 - p))


def sigmoid(u):
    if u >= 30.0:
        return 1.0
    if u <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-u))


def norm_q(p):
    """The standard normal's quantile (Acklam's rational approximation, |error| < 1.15e-9)."""
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((-0.007784894002430293 * q - 0.3223964580411365) * q - 2.400758277161838) * q
                  - 2.549732539343734) * q + 4.374664141464968) * q + 2.938163982698783) / \
               ((((0.007784695709041462 * q + 0.3224671290700398) * q + 2.445134137142996) * q
                 + 3.754408661907416) * q + 1.0)
    if p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        return (((((-39.69683028665376 * r + 220.9460984245205) * r - 275.9285104469687) * r
                  + 138.3577518672690) * r - 30.66479806614716) * r + 2.506628277459239) * q / \
               (((((-54.47609879822406 * r + 161.5858368580409) * r - 155.6989798598866) * r
                  + 66.80131188771972) * r - 13.28068155288572) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((-0.007784894002430293 * q - 0.3223964580411365) * q - 2.400758277161838) * q
               - 2.549732539343734) * q + 4.374664141464968) * q + 2.938163982698783) / \
        ((((0.007784695709041462 * q + 0.3224671290700398) * q + 2.445134137142996) * q
          + 3.754408661907416) * q + 1.0)


def mix_q(tau, m):
    """Quantile at tau of the mixture (1 - m) * atom(0) + m * N(0, 1)."""
    if m <= 1e-9:
        return 0.0
    if m >= 1.0 - 1e-9:
        return norm_q(min(1.0 - 1e-9, max(1e-9, tau)))
    if tau <= 1.0 - m:
        return 0.0
    p = (tau - (1.0 - m)) / m
    if p < 1e-9:
        p = 1e-9
    elif p > MAX_PROB:
        p = MAX_PROB
    return norm_q(p)


def ewma(values, hl):
    """Exponentially weighted mean of `values`, the last one weighted 1 (recency weighted)."""
    lam = 0.5 ** (1.0 / hl)
    num = 0.0
    den = 0.0
    w = 1.0
    for i in range(len(values) - 1, -1, -1):
        num += w * values[i]
        den += w
        w *= lam
    return num / den if den > 0.0 else 0.0


# ---------------------------------------------------------------- the model

def market_scale(prices):
    """(scale of one minute's logit move, rate of non-zero minutes) of one market's own prices."""
    u = [logit(p) for p in prices]
    deltas = [u[i + 1] - u[i] for i in range(len(u) - 1)]
    if not deltas:
        return 0.0, 0.0
    sq = [d * d for d in deltas]
    sd = math.sqrt(ewma(sq, HL))
    lam = ewma([1.0 if d != 0.0 else 0.0 for d in deltas], ATOM_HL)
    if lam > 0.9:
        lam = 0.9
    return sd, lam


def forecast_market(prices, quantiles, horizon, pooled):
    """The rows of one market: horizon rows, one value per quantile level."""
    if not prices:
        return [[0.5 for _ in quantiles] for _ in range(horizon)]
    x = prices[-1]
    if x <= 0.0:
        x = 0.0
    elif x >= 1.0:
        x = 1.0
    sd, lam = market_scale(prices)
    if pooled > 0.0 and sd > 0.0:
        sd = (sd ** (1.0 - POOL)) * (pooled ** POOL)
    sd *= SHRINK
    upl = x * (1.0 - x)
    sd += PRIOR_W * (HOUSE_SIGMA / math.sqrt(max(1e-12, 2.0 * upl)))
    base = logit(x)
    rows = []
    for h in range(1, horizon + 1):
        mh = 1.0 - (1.0 - lam) ** h
        if mh > ATOM_MAX:
            mh = ATOM_MAX
        spread = sd * (h ** POWER)
        row = []
        prev = 0.0
        for tau in quantiles:
            v = clip01(sigmoid(base + mix_q(tau, mh) * spread))
            v = round(v, 6)
            if v < prev:
                v = prev
            prev = v
            row.append(v)
        rows.append(row)
    return rows


def pooled_scale(markets):
    """The median of the case's market scales, used as a shrinkage target."""
    vals = []
    for m in markets:
        prices = m.get('prices') or []
        if len(prices) < 2:
            continue
        sd, _ = market_scale(prices)
        if sd > 0.0:
            vals.append(sd)
    if not vals:
        return 0.0
    vals.sort()
    return vals[len(vals) // 2]


# ---------------------------------------------------------------- the run

def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass
    names = sorted(f for f in os.listdir(in_dir) if f.endswith('.in'))
    for name in names:
        with open(os.path.join(in_dir, name), 'r', encoding='utf-8') as fh:
            case = json.load(fh)
        quantiles = list(case.get('quantiles') or [0.05, 0.25, 0.5, 0.75, 0.95])
        horizon = int(case.get('horizon_minutes') or 240)
        markets = case.get('markets') or []
        pooled = pooled_scale(markets)
        forecasts = []
        for m in markets:
            rows = forecast_market(m.get('prices') or [], quantiles, horizon, pooled)
            forecasts.append({'id': m.get('id'), 'q': rows})
        out = json.dumps({'forecasts': forecasts}, separators=(',', ':'))
        with open(os.path.join(out_dir, name[:-3] + '.out'), 'w', encoding='utf-8') as fh:
            fh.write(out + '\n')


if __name__ == '__main__':
    main()
