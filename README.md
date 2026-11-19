[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bayuanwar8/trophy-rarity-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bayuanwar8/trophy-rarity-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts each Grand Theft Auto VI PS5 trophy's Exophase earn rate at
2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%), for
[make.fun's gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## How it works

`main.ts` (Node 22, standard library only) reads a case file and writes one
forecast file. The model, in brief:

- **Logit-rate trend, refresh-aware.** Each trophy's logit earn rate is fitted
  against log-time using only within-refresh-segment variation, so the slope
  measures pure unlock growth — dilution steps from tracker refreshes are
  removed by demeaning inside each constant-player segment. Slopes are
  precision-shrunk toward their tier mean (platinum/gold/silver/bronze),
  tier means toward the global mean. Forecasts anchor at the last read.
- **Refresh-kappa scenario mixture.** A refresh multiplies the tracked-player
  count, diluting every rate. The forecast is a mixture over scenarios: no
  refresh before the target (probability from a hazard model fitted to the
  observed refreshes, with a prior when history is flat), or a refresh with a
  total dilution factor drawn from a lognormal fitted to observed jumps (or a
  prior). Each scenario shifts the logit forecast by the exact dilution shift,
  with an earner-import correction estimated from past refreshes — saturated
  trophies (rate ≥ 99.5%) stay saturated, since new players hold them at once.
- **Honest quantiles from one distribution.** All five quantiles are read off
  the mixture CDF on a logit grid. Pinball loss on the logit scale is a proper
  scoring rule, so the forecast reports the quantiles it believes — it does not
  bet everything on one branch.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Deterministic: two runs on the same case write byte-identical output.
