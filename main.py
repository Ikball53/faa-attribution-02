"""Quantile forecaster for Polymarket attribution markets.

The median path has two honest, checkpoint-only ingredients:

1. Case-sum reversion. The Yes prices of one family should add to about 1.
   When the checkpoint sum is off by 0.05 or more, each market's median is
   pulled multiplicatively so the sum drifts back to 1 with a 120-minute
   time constant.
2. Jump-hazard pull. After a large one-minute jump (at least 0.01), the price
   tends to give part of it back. The pull points against the last big jump,
   is strongest when the jump was recent (30-minute decay), and grows in over
   the first 30 minutes of the horizon. It is switched off when the case sum
   is far from 1 (sum reversion already dominates there) and for markets with
   under 120 minutes of history, where a jump statistic is just noise.

The fan around the median is asymmetric and level-dependent: small markets
get a wide upper tail, large markets a wide lower tail. Its width is a
two-scale EWMA volatility estimate (90- and 720-minute halves) times h**0.4,
with a 1.5x boost for young markets.

Standard library only; output is deterministic.
"""
import json, math, os, sys

HORIZON = 240


def _round6(v):
    v = min(1.0, max(0.0, v))
    return round(v * 1e6) / 1e6 + 0.0


def _ewma_rms(diffs, half_life):
    alpha = 1.0 - 0.5 ** (1.0 / half_life)
    acc = 0.0
    for d in diffs:
        acc = (1.0 - alpha) * acc + alpha * d * d
    return math.sqrt(acc)


def _volatility(prices):
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    capped = [max(-0.05, min(0.05, d)) for d in diffs]
    short = _ewma_rms(capped[-180:], 90)
    long_ = _ewma_rms(capped[-1440:], 720)
    raw = 0.6 * (0.7 * short + 0.3 * long_)
    return min(0.006, max(0.0002, raw)) * 0.9


def _fan(level):
    t = min(1.0, max(0.0, (level - 0.03) / 0.22))
    narrow = [-2.0, -0.8, 0.0, 0.1, 2.9]
    wide = [-2.8, -1.5, 0.0, 0.1, 1.3]
    return [narrow[i] + (wide[i] - narrow[i]) * t for i in range(5)]


def _hazard_pull(prices, case_sum):
    """Base amplitude of the jump-hazard pull. The pull at horizon h is
    base * (1 - exp(-h/30)). Returns 0 when the signal is switched off."""
    if abs(case_sum - 1.0) >= 0.1:
        return 0.0
    if len(prices) < 120:
        return 0.0
    lookback = min(1440, len(prices) - 1)
    for i in range(len(prices) - 1, len(prices) - 1 - lookback, -1):
        jump = prices[i] - prices[i - 1]
        if abs(jump) >= 0.01:
            age = len(prices) - 1 - i
            direction = 1.0 if jump > 0 else -1.0
            return -direction * 0.02 * math.exp(-age / 30.0) * 0.7
    return 0.0


def _forecast_case(case):
    markets = case["markets"]
    case_sum = sum(m["prices"][-1] for m in markets)
    revert = 1.0 if abs(case_sum - 1.0) >= 0.05 else 0.0
    out = []
    for m in markets:
        prices = m["prices"]
        anchor = prices[-1]
        vol = _volatility(prices)
        if len(prices) < 500:
            vol *= 1.5
        fan = _fan(anchor)
        pull = _hazard_pull(prices, case_sum)
        rows = []
        for h in range(1, HORIZON + 1):
            # Case sum at horizon h under the reversion; the median follows it.
            sum_at_h = case_sum + (1.0 - case_sum) * (1.0 - math.exp(-h / 120.0)) * revert
            med = anchor * (sum_at_h / case_sum if case_sum > 0 else 1.0)
            # Hazard pull grows in over the first ~30 minutes of the horizon.
            med += pull * (1.0 - math.exp(-h / 30.0))
            width = vol * (h ** 0.4)
            rows.append([_round6(med + z * width) for z in fan])
        out.append({"id": m["id"], "q": rows})
    return {"forecasts": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = _forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
