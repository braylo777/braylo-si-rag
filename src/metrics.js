// Retrieval and faithfulness metrics. Pure functions, no model calls.
export const STOP = new Set("a an and are as at be by can for from how in is it its of on or so that the their them then this to was what when where which who why with does do into than such each name two own other".split(" "));
export const tokens = (s) => String(s).toLowerCase().match(/[a-z0-9]+/g)?.filter((t) => t.length > 2 && !STOP.has(t)).map((t) => t.replace(/(ing|es|s)$/, "")) ?? [];
export function precisionAtK(ranked, relevant, k) { const R = new Set(relevant); return ranked.slice(0, k).filter((id) => R.has(id)).length / k; }
export function recallAtK(ranked, relevant, k) { const R = new Set(relevant); return R.size ? ranked.slice(0, k).filter((id) => R.has(id)).length / R.size : 0; }
export function reciprocalRank(ranked, relevant) { const R = new Set(relevant); const i = ranked.findIndex((id) => R.has(id)); return i < 0 ? 0 : 1 / (i + 1); }
export const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
export const r3 = (x) => Math.round(x * 1000) / 1000;
// Citation-overlap faithfulness: share of the answer's content tokens that appear in the chunks it cites.
export function citationOverlap(answer, citedTexts) {
  const a = tokens(answer), src = new Set(citedTexts.flatMap(tokens));
  return a.length ? a.filter((t) => src.has(t)).length / a.length : 0;
}
// ROC AUC via pairwise comparison (ties count half).
export function auc(scores, labels) {
  const pos = scores.filter((_, i) => labels[i] === 1), neg = scores.filter((_, i) => labels[i] === 0);
  let s = 0; for (const p of pos) for (const n of neg) s += p > n ? 1 : p === n ? 0.5 : 0;
  return pos.length && neg.length ? s / (pos.length * neg.length) : 0;
}
