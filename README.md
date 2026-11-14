# FAA outage attribution forecaster

Forecasts the Yes price of every market in a Polymarket attribution family,
one minute at a time for the four hours after a checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per minute.

## How it works

The median path has two honest, checkpoint-only ingredients:

1. **Case-sum reversion.** The Yes prices of one family should add to about 1.
   When the checkpoint sum is off by 0.05 or more, each market's median is
   pulled multiplicatively so the sum drifts back to 1 with a 120-minute
   time constant.
2. **Jump-hazard pull.** After a large one-minute jump (at least 0.01), the
   price tends to give part of it back. The pull points against the last big
   jump, is strongest when the jump was recent (30-minute decay), and grows
   in over the first 30 minutes of the horizon. It is switched off when the
   case sum is far from 1 (sum reversion already dominates there) and for
   markets with under 120 minutes of history, where a jump statistic is
   just noise.

The fan around the median is asymmetric and level-dependent: small markets
get a wide upper tail, large markets a wide lower tail. Its width is a
two-scale EWMA volatility estimate (90- and 720-minute halves) times
h^0.4, with a 1.5x boost for young markets.

## Running it

`python3 main.py` reads every `*.in` case in `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output`). Standard library only;
deterministic.
