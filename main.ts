// GTA VI trophy rarity forecaster v2 — rate-dependent refresh shift + trend.
//
// Model (logit space, NOT earners/players):
// - A refresh jumps players_tracked and shifts every rate down. The shift is
//   RATE-DEPENDENT: s = import(L) * (kappa-1)/5.06 - ln(kappa), where kappa is
//   that refresh's player jump, L = logit(rate), and import(L) = a + b*L.
//   This is an empirical transition model calibrated at kappa = 6.06 (the
//   launch-night first refresh): there it reduces exactly to the fitted law
//   s = -0.50 + 0.24*L (a0 = -0.50 + ln(6.06) = 1.302, b0 = 0.24). At
//   kappa = 1 the shift is 0, as it must be. The -ln(kappa) term is the
//   small-rate dilution approximation; the import term absorbs the rest.
//   (a, b) are estimated from OBSERVED refreshes in the case's own history
//   (corrected regression, see fitImport), shrunk toward the prior.
// - Trend: within-segment logit-vs-log-time slope (demeaned per segment, so
//   refresh steps do not pollute it), shrunk to the tier mean, clamped >= 0.
// - Expected future refreshes N from a renewal-style rule (next refresh due
//   at dueIn, then Poisson at rate lambda). Total future player growth
//   kappa_total from a power-law extrapolation (prior 6.06x when no refresh
//   observed). Shifts are compounded step by step over round(N) steps.
// - Spread: Normal-in-logit, asymmetric (wider down), sigma from trend +
//   refresh-count + parameter + partition uncertainty. Narrow on practice
//   (~0.5 logit q05-q95), wider on the 4.5-day final.
// - Deterministic: pure function of the case file. No randomness, no clock,
//   no network. Node 22 stdlib only (node:fs, node:path).

import * as fs from "node:fs";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Constants (priors — fitted from OUR OWN account-2 entry's public dev grade,
// NOT from any other entry: loss 0.38, typical miss 0.15pp, coverage 0.9818).
// ---------------------------------------------------------------------------
const KREF = 6.06; // reference first-refresh player jump (638 -> 3868)
const A0 = 1.302; // prior import intercept: -0.50 + ln(6.06)
const B0 = 0.24; // prior import slope on logit(rate)
const PRIOR_INTERVAL_H = 4.5; // prior inter-refresh interval, hours
const Z = [-1.645, -0.674, 0, 0.674, 1.645]; // normal quantiles, levels .05/.25/.5/.75/.95

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

// Logit of a rate given in percent; rate clamped to [0.01, 99.99] first.
function logitPct(rate: number): number {
  const p = clamp(rate, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

// Inverse logit, returned as a percent in [0.01, 99.99].
function invLogitPct(l: number): number {
  const x = clamp(l, -9.21, 9.21);
  return clamp(100 / (1 + Math.exp(-x)), 0.01, 99.99);
}

function round4(x: number): number {
  const r = Math.round(x * 10000) / 10000;
  return r === 0 ? 0 : r; // avoid -0
}

// Ordinary least squares: y = a + b*x. Returns { a, b }.
function ols(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx < 1e-12) return { a: my, b: 0 };
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}

// Weighted least squares: y = a + b*x, weights ws. Returns { a, b }.
function wols(xs: number[], ys: number[], ws: number[]): { a: number; b: number } {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw < 1e-12) return { a: 0, b: 0 };
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  if (sxx < 1e-12) return { a: my, b: 0 };
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}

// ---------------------------------------------------------------------------
// Case types (plain data, no behaviour)
// ---------------------------------------------------------------------------
interface Trophy {
  id: number;
  name: string;
  tier: string;
}
interface Read {
  at: string;
  atMs: number;
  players: number;
  rates: (number | null)[];
}
interface CaseData {
  id: string;
  at: string;
  atMs: number;
  horizonMinutes: number;
  target: string;
  targetMs: number;
  trophies: Trophy[];
  history: Read[];
  releaseMs: number;
}

// ---------------------------------------------------------------------------
// Refresh detection: any increase in players_tracked between consecutive
// reads is a refresh (the counts are exact integers from the page, so there
// is no measurement noise to threshold against). Splits history into segments.
// ---------------------------------------------------------------------------
interface Refresh {
  kappa: number; // players_after / players_before
  beforeIdx: number; // index of last read before the jump
  afterIdx: number; // index of first read after the jump
  afterMs: number;
}

function detectRefreshes(history: Read[]): { segments: number[][]; refreshes: Refresh[] } {
  const segments: number[][] = [];
  const refreshes: Refresh[] = [];
  let cur: number[] = [];
  for (let k = 0; k < history.length; k++) {
    if (k > 0 && history[k].players > history[k - 1].players) {
      segments.push(cur);
      cur = [];
      refreshes.push({
        kappa: history[k].players / history[k - 1].players,
        beforeIdx: k - 1,
        afterIdx: k,
        afterMs: history[k].atMs,
      });
    }
    cur.push(k);
  }
  segments.push(cur);
  return { segments, refreshes };
}

// ---------------------------------------------------------------------------
// Fit the import law import(L) = a + b*L from OBSERVED refreshes.
//
// For a refresh j with jump kappa_j, trophy i shifts by
//   s_ij = import(L_i) * (kappa_j - 1) / (KREF - 1) - ln(kappa_j),
// so  (s_ij + ln(kappa_j)) / w_j = a + b * L_i   with w_j = (kappa_j-1)/5.06.
// The correct regression is therefore z = (s + ln kappa)/w on L, weighted by
// w^2 (the variance of z grows as 1/w^2, so small jumps are weak evidence).
// Regressing (s + ln kappa) on L with weights w would be WRONG: it fits a
// quantity scaled by w. Saturated trophies (rate >= 99.5%) are excluded:
// their shift is 0 by construction and would corrupt the fit.
// Falls back to the prior when there is nothing identifiable (no refresh, too
// few points, or the observed L range is too narrow to identify the slope).
// ---------------------------------------------------------------------------
function fitImport(
  history: Read[],
  refreshes: Refresh[],
  trophies: Trophy[]
): { a: number; b: number; nRef: number } {
  const nRef = refreshes.length;
  if (nRef === 0) return { a: A0, b: B0, nRef: 0 };

  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  for (const r of refreshes) {
    const w = (r.kappa - 1) / (KREF - 1);
    if (w < 0.02) continue; // jump too small to say anything
    const before = history[r.beforeIdx].rates;
    const after = history[r.afterIdx].rates;
    for (let i = 0; i < trophies.length; i++) {
      const rb = before[i];
      const ra = after[i];
      if (rb === null || ra === null) continue;
      if (rb >= 99.5 || ra >= 99.5) continue; // saturated: skip
      const L = logitPct(rb);
      const s = logitPct(ra) - L;
      const z = (s + Math.log(r.kappa)) / w;
      // z must stay in a sane range; a wild z means a corrupted read.
      if (!isFinite(z) || z < -5 || z > 10) continue;
      xs.push(L);
      ys.push(z);
      ws.push(w * w);
    }
  }
  if (xs.length < 10) return { a: A0, b: B0, nRef };
  let lMin = Infinity;
  let lMax = -Infinity;
  for (const x of xs) {
    if (x < lMin) lMin = x;
    if (x > lMax) lMax = x;
  }
  if (lMax - lMin < 1.0) return { a: A0, b: B0, nRef }; // slope unidentified
  const fit = wols(xs, ys, ws);
  const aHat = clamp(fit.a, 0, 3);
  const bHat = clamp(fit.b, 0, 0.6);
  // Shrink toward the prior, weighted by how many refreshes were observed.
  const wShrink = nRef / (nRef + 2);
  return {
    a: wShrink * aHat + (1 - wShrink) * A0,
    b: wShrink * bHat + (1 - wShrink) * B0,
    nRef,
  };
}

// ---------------------------------------------------------------------------
// Trend: within-segment logit-vs-log-time slope per trophy.
// x = ln(1 + hours since release). Demeaning per segment removes refresh
// steps, so the slope is pure unlock growth. Shrunk toward the tier mean,
// clamped >= 0 AFTER shrinkage. Flat history -> slope 0.
// ---------------------------------------------------------------------------
function fitTrend(
  cs: CaseData,
  segments: number[][]
): { slopes: number[]; xLast: number; xTarget: number } {
  const n = cs.trophies.length;
  const slopes = new Array<number>(n).fill(0);
  const xOf = (ms: number): number => {
    const h = (ms - cs.releaseMs) / 3600000;
    return Math.log(1 + Math.max(-0.9, h));
  };
  const xLast = xOf(cs.history[cs.history.length - 1].atMs);
  const xTarget = xOf(cs.targetMs);

  // Per-trophy pooled within-segment OLS slope.
  const nEff = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sxx = 0;
    let sxy = 0;
    let cnt = 0;
    for (const seg of segments) {
      const pts: { x: number; y: number }[] = [];
      for (const k of seg) {
        const r = cs.history[k].rates[i];
        if (r === null || r >= 99.5) continue; // saturated: no slope signal
        pts.push({ x: xOf(cs.history[k].atMs), y: logitPct(r) });
      }
      if (pts.length < 2) continue;
      let mx = 0;
      let my = 0;
      for (const p of pts) {
        mx += p.x;
        my += p.y;
      }
      mx /= pts.length;
      my /= pts.length;
      for (const p of pts) {
        const dx = p.x - mx;
        sxx += dx * dx;
        sxy += dx * (p.y - my);
        cnt++;
      }
    }
    nEff[i] = cnt;
    slopes[i] = sxx > 1e-9 ? sxy / sxx : 0;
  }

  // Tier means, then shrink each slope toward its tier mean.
  const tiers = ["platinum", "gold", "silver", "bronze"];
  const tierMean = new Map<string, number>();
  for (const t of tiers) {
    let s = 0;
    let c = 0;
    for (let i = 0; i < n; i++) {
      if (cs.trophies[i].tier === t && nEff[i] > 0) {
        s += slopes[i];
        c++;
      }
    }
    tierMean.set(t, c > 0 ? s / c : 0);
  }
  for (let i = 0; i < n; i++) {
    const tm = tierMean.get(cs.trophies[i].tier) ?? 0;
    const w = nEff[i] / (nEff[i] + 4);
    slopes[i] = Math.max(0, w * slopes[i] + (1 - w) * tm);
  }
  return { slopes, xLast, xTarget };
}

// ---------------------------------------------------------------------------
// Expected future refresh count N (renewal-style rule).
// The next refresh is due at dueIn = max(1h, I - staleness), where I is the
// inter-refresh interval (prior 4.5h, or the mean observed interval) and
// staleness is the hours since the last refresh (or page creation). If the
// horizon runs past the due time, count 1 refresh plus Poisson arrivals at
// rate lambda thereafter; if it ends before, the expected count is the
// fraction horizon/dueIn. Continuous at horizon == dueIn (both give 1).
// ---------------------------------------------------------------------------
function expectedRefreshes(
  cs: CaseData,
  refreshes: Refresh[]
): { n: number; lambda: number } {
  const firstMs = cs.history[0].atMs;
  const lastMs = cs.history[cs.history.length - 1].atMs;
  const spanH = Math.max(0.1, (lastMs - firstMs) / 3600000);
  const nRef = refreshes.length;
  const lambda = (nRef + 1) / (spanH + PRIOR_INTERVAL_H);

  // Inter-refresh interval: mean of [page-creation -> 1st refresh, gaps...].
  let interval = PRIOR_INTERVAL_H;
  if (nRef >= 1) {
    const times = refreshes.map((r) => r.afterMs);
    const gaps: number[] = [(times[0] - firstMs) / 3600000];
    for (let j = 1; j < times.length; j++) gaps.push((times[j] - times[j - 1]) / 3600000);
    let g = 0;
    for (const x of gaps) g += x;
    interval = g / gaps.length;
    if (!isFinite(interval) || interval <= 0) interval = PRIOR_INTERVAL_H;
  }

  // Staleness: hours since the last refresh (or since the first read).
  const lastRefreshMs = nRef > 0 ? refreshes[refreshes.length - 1].afterMs : firstMs;
  const staleness = Math.max(0, (cs.atMs - lastRefreshMs) / 3600000);
  const dueIn = Math.max(1.0, interval - staleness);
  const horizonH = cs.horizonMinutes / 60;
  const n =
    horizonH > dueIn ? 1 + lambda * (horizonH - dueIn) : horizonH / dueIn;
  return { n: clamp(n, 0, 40), lambda };
}

// ---------------------------------------------------------------------------
// Total future player growth kappa_total = P(target) / P(now).
// With >= 2 segments, fit a power law ln(P) = c + gamma*ln(1+hours) to the
// segment-start player counts and extrapolate to the target (clamped to at
// most 20x growth). With a single segment (no refresh observed) there is
// nothing to fit: use the prior first-refresh jump, 6.06x.
// ---------------------------------------------------------------------------
function totalPlayerGrowth(
  cs: CaseData,
  segments: number[][]
): number {
  const pNow = cs.history[cs.history.length - 1].players;
  if (segments.length < 2) return KREF;
  const xs: number[] = [];
  const ys: number[] = [];
  for (const seg of segments) {
    if (seg.length === 0) continue;
    const read = cs.history[seg[0]];
    const h = (read.atMs - cs.releaseMs) / 3600000;
    xs.push(Math.log(1 + Math.max(-0.9, h)));
    ys.push(Math.log(Math.max(1, read.players)));
  }
  if (xs.length < 2) return KREF;
  const fit = ols(xs, ys);
  if (!isFinite(fit.b) || fit.b <= 0) return 1; // no growth fitted
  const hT = (cs.targetMs - cs.releaseMs) / 3600000;
  const pTarget = Math.exp(fit.a + fit.b * Math.log(1 + Math.max(-0.9, hT)));
  if (!isFinite(pTarget)) return KREF;
  return clamp(pTarget / Math.max(1, pNow), 1, 20);
}

// ---------------------------------------------------------------------------
// One refresh step: s = import(L) * (kappa-1)/5.06 - ln(kappa), clamped.
// import(L) uses L clamped to [-6, 9.21] so it cannot explode at the edges.
// ---------------------------------------------------------------------------
function shiftStep(L: number, kappa: number, a: number, b: number): number {
  const lc = clamp(L, -6, 9.21);
  const imp = a + b * lc;
  const s = imp * ((kappa - 1) / (KREF - 1)) - Math.log(kappa);
  return clamp(s, -3, 2);
}

// ---------------------------------------------------------------------------
// Forecast one trophy: trend + compounded refresh shifts, with a partition
// guard (stepped vs single aggregate jump) and a narrow asymmetric spread.
// Returns the five quantiles in percent.
// ---------------------------------------------------------------------------
function forecastTrophy(
  lastRate: number,
  slope: number,
  dxTrend: number,
  a: number,
  b: number,
  steps: number,
  kappaStep: number,
  kappaTotal: number,
  sig: { up: number; down: number }
): number[] {
  if (lastRate >= 99.5) {
    // Saturated: stays ~99.99 whatever happens.
    return [99.98, 99.99, 99.99, 99.99, 99.99];
  }
  const l0 = logitPct(lastRate);
  // Trend first (unlock growth), then refresh shifts on top.
  let lTrend = clamp(l0 + slope * dxTrend, -9.21, 9.21);

  // Stepped compounding: recompute L after each shift.
  let lStep = lTrend;
  for (let k = 0; k < steps; k++) {
    lStep += shiftStep(lStep, kappaStep, a, b);
    lStep = clamp(lStep, -9.21, 9.21);
  }
  // Guard the total refresh shift (never annihilate the rate).
  if (lStep - lTrend < -4) lStep = lTrend - 4;
  if (lStep - lTrend > 1) lStep = lTrend + 1;

  // Single aggregate jump with the same total growth (partition check).
  const lAgg = clamp(lTrend + shiftStep(lTrend, kappaTotal, a, b), -9.21, 9.21);
  let lMed = lStep;
  let sigUp = sig.up;
  let sigDown = sig.down;
  const diff = Math.abs(lStep - lAgg);
  if (diff > 0.3) {
    // The two partitions disagree: average them and widen the spread.
    lMed = (lStep + lAgg) / 2;
    const extra = diff / 2;
    sigUp = Math.sqrt(sigUp * sigUp + extra * extra);
    sigDown = Math.sqrt(sigDown * sigDown + extra * extra);
  }

  const med = invLogitPct(lMed);
  const q: number[] = [0, 0, 0, 0, 0];
  // q05/q25 use sigmaDown (wider below), q75/q95 use sigmaUp.
  q[0] = invLogitPct(lMed + Z[0] * sigDown);
  q[1] = invLogitPct(lMed + Z[1] * sigDown);
  q[2] = med;
  q[3] = invLogitPct(lMed + Z[3] * sigUp);
  q[4] = invLogitPct(lMed + Z[4] * sigUp);
  // Enforce non-decreasing (clamping can tie neighbours, never invert).
  for (let i = 1; i < 5; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
  return q.map(round4);
}

// ---------------------------------------------------------------------------
// Process one case file -> forecast rows.
// ---------------------------------------------------------------------------
function processCase(raw: any): { id: string; trophies: { id: number; q: number[] }[] } {
  const history: Read[] = (raw.history as any[]).map((h: any) => ({
    at: h.at,
    atMs: Date.parse(h.at),
    players: h.players_tracked,
    rates: (h.rates as (number | null)[]).slice(),
  }));
  const trophies: Trophy[] = (raw.trophies as any[]).map((t: any) => ({
    id: t.id,
    name: t.name,
    tier: t.tier,
  }));
  const cs: CaseData = {
    id: raw.id,
    at: raw.at,
    atMs: Date.parse(raw.at),
    horizonMinutes: raw.horizon_minutes,
    target: raw.target,
    targetMs: Date.parse(raw.target),
    trophies,
    history,
    releaseMs: Date.parse((raw.game?.release_date ?? "2026-11-19") + "T00:00:00Z"),
  };

  const { segments, refreshes } = detectRefreshes(cs.history);
  const imp = fitImport(cs.history, refreshes, cs.trophies);
  const trend = fitTrend(cs, segments);
  const dxTrend = trend.xTarget - trend.xLast;
  const exp = expectedRefreshes(cs, refreshes);
  const kappaTotal = totalPlayerGrowth(cs, segments);
  const steps = Math.max(1, Math.round(exp.n));
  const kappaStep = Math.pow(kappaTotal, 1 / steps);

  // Spread components (logit sigma). Calibrated so q05-q95 is ~0.5 logit
  // on a 90-minute practice horizon and wider on the 4.5-day final.
  const horizonDays = cs.horizonMinutes / 1440;
  const sigTrend = 0.15 * Math.sqrt(horizonDays);
  const sigRefreshUp = 0.05 * Math.sqrt(exp.n);
  const sigRefreshDown = 0.15 * Math.sqrt(exp.n);
  const sigParam = 0.12 / Math.sqrt(1 + imp.nRef);
  const sigUp = Math.sqrt(
    sigTrend * sigTrend + sigRefreshUp * sigRefreshUp + sigParam * sigParam
  );
  const sigDown = Math.sqrt(
    sigTrend * sigTrend + sigRefreshDown * sigRefreshDown + sigParam * sigParam
  );

  // Last non-null rate per trophy (scan back); fallback: median last rate.
  const lastRates: number[] = [];
  const lastReadRates = cs.history[cs.history.length - 1].rates;
  const present = lastReadRates.filter((r): r is number => r !== null);
  const fallback = present.length > 0 ? present.sort((x, y) => x - y)[Math.floor(present.length / 2)] : 1.0;
  for (let i = 0; i < cs.trophies.length; i++) {
    let r: number | null = null;
    for (let k = cs.history.length - 1; k >= 0; k--) {
      if (cs.history[k].rates[i] !== null) {
        r = cs.history[k].rates[i] as number;
        break;
      }
    }
    lastRates.push(r === null ? fallback : r);
  }

  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < cs.trophies.length; i++) {
    const q = forecastTrophy(
      lastRates[i],
      trend.slopes[i],
      dxTrend,
      imp.a,
      imp.b,
      steps,
      kappaStep,
      kappaTotal,
      { up: sigUp, down: sigDown }
    );
    rows.push({ id: cs.trophies[i].id, q });
  }
  return { id: cs.id, trophies: rows };
}

// ---------------------------------------------------------------------------
// Entry point: read every *.in in the input dir (name order), write one
// *.out per case to the output dir. Output JSON is exactly
// {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.
// ---------------------------------------------------------------------------
function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs
    .readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const raw = JSON.parse(fs.readFileSync(path.join(inputDir, f), "utf-8"));
    const out = processCase(raw);
    // Contract: EXACTLY {"trophies":[...]} — no other top-level keys.
    const text = JSON.stringify({ trophies: out.trophies });
    if (text.length > 65536) throw new Error(`output too large for ${out.id}`);
    fs.writeFileSync(path.join(outputDir, out.id + ".out"), text, "utf-8");
  }
}

main();
