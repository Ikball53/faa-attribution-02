"""FAA outage attribution forecaster.

Forecasts each market's Yes price for the 240 minutes after a checkpoint as
five quantiles per minute. Design, from the shape of Polymarket prediction
markets: prices sit still most minutes and then jump, so the forecast fan is
strongly asymmetric -- a long lower tail, a compressed upper inner quantile,
and a wide upper outer tail -- with the exact shape sliding with the market's
price level (big markets fall further and rise less than small ones). The
median drifts with the family: the markets are mutually exclusive, so when
their prices sum away from one the median is pulled back toward a sum of one,
gated to cases where the sum is clearly off. Young markets (short history)
move more, so their fan is wider. Volatility is a two-scale exponentially
weighted RMS of capped one-minute moves.
"""
import json
import math
import os


QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZON = 240
GROWTH_POWER = 0.4
SIGMA_SCALE = 0.9
SIGMA_FLOOR = 0.0002
SIGMA_CAP = 0.006
MOVE_CAP = 0.05
YOUNG_MINUTES = 500
YOUNG_WIDEN = 1.5
SUM_GATE = 0.05
SUM_TAU = 120.0

# Fan shape (quantile offsets in units of the spread) at the two ends of the
# price-level slide: small markets and big markets. The middle three entries
# are the median (0) and the inner quantiles; the outer entries are the tails.
SHAPE_SMALL = (-2.0, -0.8, 0.0, 0.1, 2.9)
SHAPE_BIG = (-2.8, -1.5, 0.0, 0.1, 1.3)
LEVEL_LOW = 0.03
LEVEL_HIGH = 0.25


def _clamp01(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _round6(value):
    return round(_clamp01(value) * 1000000.0) / 1000000.0 + 0.0


def _ewma_rms(moves, half_life):
    """Root-mean-square of moves with exponential weights (newest heaviest)."""
    alpha = 1.0 - 0.5 ** (1.0 / half_life)
    acc = 0.0
    for move in moves:
        acc = (1.0 - alpha) * acc + alpha * move * move
    return math.sqrt(acc)


def _volatility(prices):
    """Two-scale volatility of one-minute moves, capped and floored.

    The blend is scaled by 0.6, clamped to [floor, cap], then scaled by 0.9.
    """
    if len(prices) < 2:
        return SIGMA_FLOOR * SIGMA_SCALE
    steps = []
    for i in range(len(prices) - 1):
        step = prices[i + 1] - prices[i]
        if step > MOVE_CAP:
            step = MOVE_CAP
        elif step < -MOVE_CAP:
            step = -MOVE_CAP
        steps.append(step)
    near = _ewma_rms(steps[-180:], 90)
    far = _ewma_rms(steps[-1440:], 720)
    vol = 0.6 * (0.7 * near + 0.3 * far)
    if vol < SIGMA_FLOOR:
        vol = SIGMA_FLOOR
    if vol > SIGMA_CAP:
        vol = SIGMA_CAP
    return vol * SIGMA_SCALE


def _fan_shape(level):
    """Asymmetric fan offsets for a market at this checkpoint price."""
    slide = (level - LEVEL_LOW) / (LEVEL_HIGH - LEVEL_LOW)
    if slide < 0.0:
        slide = 0.0
    elif slide > 1.0:
        slide = 1.0
    return tuple(
        SHAPE_SMALL[i] + (SHAPE_BIG[i] - SHAPE_SMALL[i]) * slide
        for i in range(5)
    )


def _forecast_market(prices, family_sum, family_size):
    """240 rows of five quantiles for one market."""
    last = prices[-1]
    vol = _volatility(prices)
    if len(prices) < YOUNG_MINUTES:
        vol *= YOUNG_WIDEN
    offsets = _fan_shape(last)
    pull = 0.0
    if abs(family_sum - 1.0) >= SUM_GATE and family_sum > 0.0:
        pull = (1.0 - family_sum) / family_sum
    rows = []
    for minute in range(1, HORIZON + 1):
        if pull:
            drift = pull * (1.0 - math.exp(-minute / SUM_TAU))
            centre = last * (1.0 + drift)
        else:
            centre = last
        growth = vol * (minute ** GROWTH_POWER)
        row = [_round6(centre + offsets[k] * growth) for k in range(5)]
        row.sort()
        rows.append(row)
    return rows


def forecast_case(case):
    markets = case["markets"]
    total = 0.0
    for market in markets:
        total += market["prices"][-1]
    forecasts = []
    for market in markets:
        rows = _forecast_market(market["prices"], total, len(markets))
        forecasts.append({"id": market["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
