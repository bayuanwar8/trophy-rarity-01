// gta6-trophy-rarity-forecast-01 — account 1 (@bayuanwar8)
// LOGIT-RATE TREND + REFRESH-KAPPA mixture forecaster (Node 22, stdlib only).
//
// Model, built fresh (reasoning in README.md):
//  * Per trophy, fit logit(rate) against log-time using ONLY within-refresh-segment
//    variation, so the slope is pure unlock growth: refresh dilution steps are
//    removed by segment demeaning. Slopes are precision-shrunk toward tier means,
//    tier means toward the global mean. The forecast anchors at the last read.
//  * Refreshes are a scenario mixture: with probability q0 no refresh lands before
//    the target; otherwise the tracked-player count grows by a total factor D
//    (lognormal, fitted from observed jumps or a prior). Each scenario shifts the
//    logit forecast by the EXACT dilution shift, with an earner-import correction
//    estimated from past refreshes (saturated trophies stay saturated).
//  * Quantiles come from the mixture CDF, inverted by bisection — one honest
//    distribution, not a median plus an ad-hoc fan. Pinball loss on the logit
//    scoring rule, so honest quantiles are the right target, not a bet on one branch.
//  * Deterministic: fixed quadrature, no randomness, no clock, no network. The
//    program is a pure function of the case file.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Random-walk spread of a logit's move, in logit units per square-root day. */
const SIGMA_RW = 0.4;
/** Prior refresh hazard: one refresh per HAZARD_PRIOR_HOURS, blended with history. */
const HAZARD_PRIOR_HOURS = 8;
/** Prior refresh size when no jump was ever observed: lognormal median. */
const KAPPA_PRIOR_MEDIAN = 4;
/** Prior refresh size spread (log sd) when no jump was ever observed. */
const KAPPA_PRIOR_LOGSD = 0.7;
/** Quadrature points for the refresh-size distribution (equal weights). */
const KAPPA_GRID_N = 9;

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** logit of a percent, clamped to the scorer's 0.01..99.99 range first. */
const logit = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};
/** Inverse logit: logit units back to percent. */
const logistic = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
const normCdf = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-(x * x) / 2);
  const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x > 0 ? 1 - p : p;
};
/** Fixed standard-normal quantiles for the equal-weight quadrature grid. */
const normQuantile = (p: number): number => {
  // Acklam's approximation (good to ~1e-9 relative); deterministic, no tables.
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pLow) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
};

interface TrophyInfo { id: number; tier: string }
interface ReadInfo { atMs: number; players: number; rates: Array<number | null> }
interface Case {
  id: string;
  horizonMinutes: number;
  trophies: TrophyInfo[];
  history: ReadInfo[];
}

/** log-time coordinate: ln(1 + hours since the case's first read). */
const logTime = (atMs: number, firstMs: number): number =>
  Math.log(1 + Math.max(0, (atMs - firstMs) / 3_600_000));

/**
 * Within-refresh-segment slope of logit(rate) vs log-time.
 * Reads are grouped into segments of constant players_tracked; demeaning inside
 * each segment removes the dilution steps, so the pooled slope is pure unlock
 * growth. Returns the slope, its standard error, and the information Sxx.
 */
const withinSegmentSlope = (
  xs: number[], ys: Array<number | null>, seg: number[],
): { slope: number; se: number; sxx: number; n: number } => {
  const bySeg = new Map<number, number[]>();
  xs.forEach((_, j) => {
    if (ys[j] === null || ys[j] === undefined) return;
    const list = bySeg.get(seg[j]) ?? [];
    list.push(j);
    bySeg.set(seg[j], list);
  });
  let sxx = 0;
  let sxy = 0;
  let n = 0;
  const segMean = new Map<number, { mx: number; my: number }>();
  for (const idxs of bySeg.values()) {
    if (idxs.length < 2) continue;
    const mx = idxs.reduce((a, j) => a + xs[j], 0) / idxs.length;
    const my = idxs.reduce((a, j) => a + (ys[j] as number), 0) / idxs.length;
    segMean.set(seg[idxs[0]], { mx, my });
    for (const j of idxs) {
      sxx += (xs[j] - mx) * (xs[j] - mx);
      sxy += (xs[j] - mx) * ((ys[j] as number) - my);
    }
    n += idxs.length;
  }
  if (sxx <= 1e-9 || n < 3) return { slope: 0, se: Infinity, sxx: 0, n };
  const slope = sxy / sxx;
  // Residual variance around the segment-demeaned fit.
  let rss = 0;
  let dof = 0;
  for (const [s, idxs] of bySeg.entries()) {
    if (idxs.length < 2) continue;
    const { mx, my } = segMean.get(s) as { mx: number; my: number };
    for (const j of idxs) {
      const r = (ys[j] as number) - (my + slope * (xs[j] - mx));
      rss += r * r;
      dof += 1;
    }
    dof -= 1; // one segment mean estimated
  }
  dof -= 1; // the slope itself
  const sigma2 = dof > 0 ? rss / dof : 0.25;
  return { slope, se: Math.sqrt(Math.max(sigma2, 1e-4) / sxx), sxx, n };
};

/**
 * Precision-weighted mean of values with standard errors; each value also gets a
 * prior variance TAU2 so a very precise estimate cannot dominate completely.
 */
const precisionMean = (vals: number[], ses: number[], tau2 = 0.25): number => {
  let wSum = 0;
  let vSum = 0;
  vals.forEach((v, i) => {
    const w = 1 / (ses[i] * ses[i] + tau2);
    wSum += w;
    vSum += w * v;
  });
  return wSum > 0 ? vSum / wSum : 0;
};

/**
 * Shrink per-trophy slopes toward tier means, tier means toward the global mean.
 * Weight depends on each slope's information (Sxx): a trophy with no within-segment
 * variation borrows its tier mean entirely. Slopes are clamped at 0 only AFTER
 * shrinkage, so the tier means are not biased upward by pre-clamping.
 */
const pooledSlopes = (
  fits: Array<{ slope: number; se: number; sxx: number }>,
  tiers: string[],
): number[] => {
  const global = precisionMean(
    fits.map((f) => f.slope),
    fits.map((f) => f.se),
  );
  const tierMean = new Map<string, number>();
  for (const tier of new Set(tiers)) {
    const idxs = tiers.map((t, i) => (t === tier ? i : -1)).filter((i) => i >= 0);
    const raw = precisionMean(
      idxs.map((i) => fits[i].slope),
      idxs.map((i) => fits[i].se),
    );
    // Small tiers (platinum n=1, gold n=2) borrow from the global mean.
    const k = 3; // pseudo-count
    tierMean.set(tier, (idxs.length * raw + k * global) / (idxs.length + k));
  }
  return fits.map((f, i) => {
    const w = f.sxx / (f.sxx + 0.5); // information weight, 0 when no signal
    const shrunk = w * f.slope + (1 - w) * (tierMean.get(tiers[i]) ?? global);
    return Math.max(0, shrunk);
  });
};

/** Player-count refresh analysis over the case history. */
const analysePlayers = (
  reads: ReadInfo[],
  xs: number[],
  xLast: number,
  xTarget: number,
  horizonHours: number,
): {
  q0: number; // P(no refresh before the target)
  muLogD: number; // centre of the total log-dilution distribution
  sdLogD: number; // spread of the total log-dilution distribution
  jumps: Array<{ kappa: number; before: number; after: number }>;
} => {
  const jumps: Array<{ kappa: number; before: number; after: number }> = [];
  for (let j = 0; j + 1 < reads.length; j++) {
    const before = reads[j].players;
    const after = reads[j + 1].players;
    if (before > 0 && after > before * 1.0001) {
      jumps.push({ kappa: after / before, before, after });
    }
  }
  const spanHours = reads.length > 1
    ? (reads[reads.length - 1].atMs - reads[0].atMs) / 3_600_000
    : 0;
  // Refresh hazard: observed jumps plus a prior of one refresh per
  // HAZARD_PRIOR_HOURS, so a flat history still expects refreshes eventually.
  const lambda = (jumps.length + 1) / (Math.max(0, spanHours) + HAZARD_PRIOR_HOURS);
  const q0 = Math.exp(-lambda * horizonHours);
  if (jumps.length === 0) {
    // No jump ever seen: dilution comes from the prior alone.
    return { q0, muLogD: Math.log(KAPPA_PRIOR_MEDIAN), sdLogD: KAPPA_PRIOR_LOGSD, jumps };
  }
  // Players trend: OLS of log(players) on log-time; extrapolate to the target.
  const n = reads.length;
  const mx = xs.reduce((a, x) => a + x, 0) / n;
  const my = reads.reduce((a, r) => a + Math.log(Math.max(1, r.players)), 0) / n;
  let sxx = 0;
  let sxy = 0;
  reads.forEach((r, j) => {
    sxx += (xs[j] - mx) * (xs[j] - mx);
    sxy += (xs[j] - mx) * (Math.log(Math.max(1, r.players)) - my);
  });
  const slopeP = sxx > 1e-9 ? Math.max(0, sxy / sxx) : 0;
  const dTrend = Math.exp(slopeP * (xTarget - xLast));
  // Scatter of log(players) around its trend line: the dilution uncertainty.
  let rss = 0;
  reads.forEach((r, j) => {
    const fit = my + slopeP * (xs[j] - mx);
    const res = Math.log(Math.max(1, r.players)) - fit;
    rss += res * res;
  });
  const sdLogD = clamp(Math.sqrt(rss / Math.max(1, n - 2)), 0.3, 1.5);
  return { q0, muLogD: Math.log(Math.max(1, dTrend)), sdLogD, jumps };
};

/**
 * Earner-import rate per trophy: the share of the new players a refresh brings
 * who already hold the trophy. Estimated from observed refreshes; 0 when none
 * were seen (new players start fresh), except a saturated trophy (rate >= 99.5%),
 * which new players hold immediately too.
 */
const earnerImport = (
  c: Case,
  jumps: Array<{ kappa: number; before: number; after: number }>,
  reads: ReadInfo[],
): number[] => {
  return c.trophies.map((_, i) => {
    const last = lastNonNullRate(reads, i);
    if (last >= 99.5) return 1;
    if (jumps.length === 0) return 0;
    const est: number[] = [];
    for (let j = 0; j + 1 < reads.length; j++) {
      const before = reads[j].players;
      const after = reads[j + 1].players;
      if (!(before > 0 && after > before * 1.0001)) continue;
      const pBefore = reads[j].rates[i];
      const pAfter = reads[j + 1].rates[i];
      if (pBefore === null || pAfter === null) continue;
      const kappa = after / before;
      est.push(clamp(((pAfter / 100) * kappa - pBefore / 100) / (kappa - 1), 0, 1));
    }
    if (est.length === 0) return 0;
    est.sort((a, b) => a - b);
    return est[Math.floor(est.length / 2)];
  });
};

/** Most recent non-null rate for a trophy (scanning back), else 0.01. */
const lastNonNullRate = (reads: ReadInfo[], i: number): number => {
  for (let j = reads.length - 1; j >= 0; j--) {
    const r = reads[j].rates[i];
    if (r !== null && r !== undefined) return r;
  }
  return 0.01;
};

/**
 * The five quantiles of one trophy's rate at the target, from the refresh
 * scenario mixture. Components: a no-refresh branch (weight q0) and refresh
 * sub-branches over a quadrature of the total log-dilution (weight 1-q0).
 * Each component is Normal(mu, sigmaRw) in logit space; the mixture CDF is
 * inverted on a fixed logit grid. Fully deterministic.
 */
const forecastTrophy = (
  trendLogit: number,
  eImport: number,
  q0: number,
  muLogD: number,
  sdLogD: number,
  sigmaRw: number,
): number[] => {
  interface Comp { w: number; mu: number }
  const comps: Comp[] = [{ w: q0, mu: trendLogit }];
  // The dilution applies to the rate AT the target (after trend growth), not
  // to the last rate: p' = (p_trend + e*(D-1)) / D, in exact logit terms.
  const pTrend = clamp(logistic(trendLogit), 0.01, 99.99) / 100;
  for (let k = 0; k < KAPPA_GRID_N; k++) {
    const z = normQuantile((k + 0.5) / KAPPA_GRID_N);
    const d = Math.exp(muLogD + sdLogD * z);
    const pNew = clamp((pTrend + eImport * (d - 1)) / d, 0.0001, 0.9999);
    const mu = logit(pNew * 100);
    comps.push({ w: (1 - q0) / KAPPA_GRID_N, mu });
  }
  const sd = Math.max(sigmaRw, 0.05);
  // Mixture CDF, inverted by bisection: no fixed grid, so a quantile can never
  // silently collapse onto a grid endpoint. Brackets widen until they straddle
  // the target, then 80 bisection steps pin it to ~1e-5 logit. Deterministic.
  const mixCdf = (x: number): number => {
    let s = 0;
    for (const comp of comps) s += comp.w * normCdf((x - comp.mu) / sd);
    return s;
  };
  return LEVELS.map((tau) => {
    let lo = trendLogit - 12;
    let hi = trendLogit + 6;
    while (mixCdf(lo) > tau) lo -= 6;
    while (mixCdf(hi) < tau) hi += 6;
    for (let it = 0; it < 80; it++) {
      const mid = (lo + hi) / 2;
      if (mixCdf(mid) < tau) lo = mid; else hi = mid;
    }
    return round4(clamp(logistic((lo + hi) / 2), 0.01, 99.99));
  });
};

/** Forecast one case: five quantiles per trophy. */
const forecastCase = (c: Case): Array<{ id: number; q: number[] }> => {
  const firstMs = c.history[0].atMs;
  const lastMs = c.history[c.history.length - 1].atMs;
  const xs = c.history.map((r) => logTime(r.atMs, firstMs));
  const xLast = xs[xs.length - 1];
  const xTarget = logTime(lastMs + c.horizonMinutes * 60_000, firstMs);
  const horizonHours = c.horizonMinutes / 60;
  const horizonDays = c.horizonMinutes / 1440;

  // Refresh segments: a new segment starts whenever players_tracked changes.
  const seg: number[] = [];
  let s = 0;
  c.history.forEach((r, j) => {
    if (j > 0 && r.players !== c.history[j - 1].players) s++;
    seg.push(s);
  });

  // Per-trophy within-segment logit trend, pooled by tier.
  const fits = c.trophies.map((_, i) => {
    const ys = c.history.map((r) => {
      const v = r.rates[i];
      return v === null || v === undefined ? null : logit(v);
    });
    return withinSegmentSlope(xs, ys, seg);
  });
  const slopes = pooledSlopes(fits, c.trophies.map((t) => t.tier));

  const players = analysePlayers(c.history, xs, xLast, xTarget, horizonHours);
  const eImports = earnerImport(c, players.jumps, c.history);
  const sigmaRw = SIGMA_RW * Math.sqrt(Math.max(0, horizonDays));

  return c.trophies.map((t, i) => {
    const lastRate = lastNonNullRate(c.history, i);
    const trendLogit = logit(lastRate) + slopes[i] * (xTarget - xLast);
    let q = forecastTrophy(trendLogit, eImports[i], players.q0, players.muLogD, players.sdLogD, sigmaRw);
    // Enforce non-decreasing quantiles after rounding.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
};

const main = (): void => {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
  for (const file of inputs) {
    const raw = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as {
      id: string;
      horizon_minutes: number;
      levels: number[];
      trophies: Array<{ id: number; tier: string }>;
      history: Array<{ at: string; players_tracked: number; rates: Array<number | null> }>;
    };
    if (raw.levels.length !== LEVELS.length || raw.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(raw.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    if (!raw.history || raw.history.length === 0) throw new Error(`${file} has no history`);
    if (!Number.isFinite(raw.horizon_minutes) || raw.horizon_minutes <= 0) {
      throw new Error(`${file} has a non-positive horizon`);
    }
    for (const r of raw.history) {
      if (!Number.isFinite(r.players_tracked) || r.players_tracked <= 0) {
        throw new Error(`${file} has a bad players_tracked value`);
      }
      for (const v of r.rates) {
        if (v !== null && v !== undefined && (!Number.isFinite(v) || v < 0 || v > 100)) {
          throw new Error(`${file} has a rate outside 0..100`);
        }
      }
    }
    const c: Case = {
      id: raw.id,
      horizonMinutes: raw.horizon_minutes,
      trophies: raw.trophies.map((t) => ({ id: t.id, tier: t.tier })),
      history: raw.history.map((r) => ({
        atMs: Date.parse(r.at),
        players: r.players_tracked,
        rates: r.rates,
      })),
    };
    if (c.history.some((r) => Number.isNaN(r.atMs))) {
      throw new Error(`${file} has an unreadable timestamp in its history`);
    }
    const trophies = forecastCase(c);
    writeFileSync(join(OUTPUT_DIR, `${c.id}.out`), `${JSON.stringify({ trophies })}\n`);
  }
};

main();
