#!/usr/bin/env python3
"""Minute-by-minute quantile forecasts for Polymarket Yes prices.

Reads every *.in case in the input directory and writes one *.out forecast
per case. Each market's median is its last price (persistence: every drift
tried on the practice data made the loss worse, so the median does not
move). Its spread grows like the horizon to the power 0.4 and scales with
that market's own recent realised volatility, widening after jumps and
narrowing while flat. Standard library only, no randomness, byte-identical
on repeat runs.
"""

import json
import math
import os
from pathlib import Path

Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZON = 240

# Volatility windows and blend.
SHORT_MINUTES = 180
SHORT_HALF_LIFE = 90.0
LONG_MINUTES = 1440
LONG_HALF_LIFE = 720.0
SHORT_WEIGHT = 0.7
DIFF_CAP = 0.10
SIGMA_SCALE = 0.6
SIGMA_FLOOR = 0.0006
SIGMA_CAP = 0.006
SPREAD_EXPONENT = 0.4

# Jump widening: a move this large in the last half hour widens the spread.
JUMP_MINUTES = 30
JUMP_SIZE = 0.04
JUMP_FACTOR = 1.5

# Flat narrowing: almost no movement over the last two hours narrows it.
FLAT_MINUTES = 120
FLAT_TOTAL = 0.001
FLAT_FACTOR = 0.6


def clamp01(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def ewma_sigma(prices, window, half_life):
    """Root mean square of capped one-minute diffs, recent minutes heavier."""
    diffs = prices[-window - 1:]
    if len(diffs) < 2:
        return None
    total = 0.0
    weight_sum = 0.0
    for idx in range(len(diffs) - 1, 0, -1):
        age = len(diffs) - 1 - idx
        weight = 0.5 ** (age / half_life)
        step = diffs[idx] - diffs[idx - 1]
        if step > DIFF_CAP:
            step = DIFF_CAP
        elif step < -DIFF_CAP:
            step = -DIFF_CAP
        total += weight * step * step
        weight_sum += weight
    if weight_sum <= 0.0:
        return None
    return math.sqrt(total / weight_sum)


def recent_move_total(prices, window):
    """Total absolute movement over the last `window` minutes."""
    tail = prices[-window - 1:]
    total = 0.0
    for idx in range(1, len(tail)):
        total += abs(tail[idx] - tail[idx - 1])
    return total


def biggest_recent_jump(prices, window):
    """Largest single one-minute move over the last `window` minutes."""
    tail = prices[-window - 1:]
    biggest = 0.0
    for idx in range(1, len(tail)):
        move = abs(tail[idx] - tail[idx - 1])
        if move > biggest:
            biggest = move
    return biggest


def market_sigma(prices):
    """Per-market spread scale in price units per square-root minute."""
    short = ewma_sigma(prices, SHORT_MINUTES, SHORT_HALF_LIFE)
    long = ewma_sigma(prices, LONG_MINUTES, LONG_HALF_LIFE)
    if short is None and long is None:
        return 0.0025
    if short is None:
        sigma = long
    elif long is None:
        sigma = short
    else:
        sigma = SHORT_WEIGHT * short + (1.0 - SHORT_WEIGHT) * long
    sigma *= SIGMA_SCALE
    if biggest_recent_jump(prices, JUMP_MINUTES) >= JUMP_SIZE:
        sigma *= JUMP_FACTOR
    if recent_move_total(prices, FLAT_MINUTES) <= FLAT_TOTAL:
        sigma *= FLAT_FACTOR
    if sigma < SIGMA_FLOOR:
        sigma = SIGMA_FLOOR
    elif sigma > SIGMA_CAP:
        sigma = SIGMA_CAP
    return sigma


def forecast_market(prices):
    """240 rows of 5 quantiles for one market, median pinned at the last
    price, spread widening like horizon ** SPREAD_EXPONENT."""
    last = prices[-1]
    sigma = market_sigma(prices)
    rows = []
    for hour_step in range(1, HORIZON + 1):
        spread = sigma * (hour_step ** SPREAD_EXPONENT)
        row = []
        previous = -1.0
        for z in Z_SCORES:
            value = clamp01(last + z * spread)
            if value < previous:
                value = previous
            value = round(value, 6)
            if value < previous:
                value = previous
            previous = value
            row.append(value)
        rows.append(row)
    return rows


def run_case(case):
    forecasts = []
    for entry in case["markets"]:
        rows = forecast_market(entry["prices"])
        forecasts.append({"id": entry["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(in_dir.glob("*.in")):
        with open(in_path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = run_case(case)
        out_path = out_dir / (in_path.stem + ".out")
        text = json.dumps(result, sort_keys=True, separators=(",", ":"))
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
