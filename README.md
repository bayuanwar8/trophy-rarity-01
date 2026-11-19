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

- **Rate-dependent refresh shift.** A refresh jumps the tracked-player
  count and shifts every rate down — but not by a flat factor: rare trophies
  dilute almost purely, common ones barely fall (new players earn common
  trophies fast). Each refresh shifts logit(rate) by
  `import(L) * (kappa-1)/5.06 - ln(kappa)`, where kappa is that refresh's
  player jump and `import(L) = a + b*L` is the earner import from the new
  players. (a, b) are estimated from the refreshes observed in the case's
  own history (corrected weighted regression), shrunk toward a prior
  fitted from our own scored entry (a=1.302, b=0.24 — at a 6.06x jump this
  reproduces the fitted shift law s = -0.50 + 0.24*logit(rate)). Future
  refreshes are counted by a renewal-style rule (next refresh due from
  staleness, then Poisson), total player growth is extrapolated by a
  power law, and shifts are compounded step by step.
- **Logit-rate trend.** Each trophy's logit earn rate is fitted against
  log-time using only within-refresh-segment variation, so the slope
  measures pure unlock growth. Slopes are shrunk toward their tier mean
  (platinum/gold/silver/bronze) and clamped >= 0. On launch night the
  history is flat so the trend is 0 and the refresh shift does the work.
- **Narrow asymmetric quantiles.** All five quantiles come from a
  Normal-in-logit fan around the median, wider below than above (refreshes
  only push rates down), with sigma from trend + refresh-count + parameter
  + partition uncertainty — about 0.5 logit wide (q05-q95) on a 90-minute
  horizon, wider on the 4.5-day final.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Deterministic: two runs on the same case write byte-identical output.
