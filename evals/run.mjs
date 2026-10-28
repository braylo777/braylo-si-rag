// Offline evals (no model calls). node evals/run.mjs -> evals/results.json
import fs from "node:fs";
import { labelled, SET } from "./lib.mjs";
import { bm25Ids, rrf } from "../src/retrieve.js";
import { precisionAtK, recallAtK, reciprocalRank, mean, r3, citationOverlap } from "../src/metrics.js";
import { inputGuard } from "../src/guards.js";
import { onTopic, redactPII, answer, SCOPE_MIN } from "../src/agent.js";
import { routeIntent } from "../src/tools.js";
import { pct } from "../src/obs.js";
import { KB } from "../src/kb.js";
const L = labelled(), empty = L.filter((q) => !q.relevant.length).map((q) => q.id);
if (empty.length) throw new Error("queries with no relevant chunk: " + empty);
function score(qs, rank) {
  const o = { n: qs.length };
  for (const k of [1, 3, 5]) { o[`precision@${k}`] = r3(mean(qs.map((q) => precisionAtK(rank(q), q.relevant, k)))); o[`recall@${k}`] = r3(mean(qs.map((q) => recallAtK(rank(q), q.relevant, k)))); }
  o.hit_rate_at_5 = r3(mean(qs.map((q) => (rank(q).slice(0, 5).some((id) => q.relevant.includes(id)) ? 1 : 0)))); o.mrr = r3(mean(qs.map((q) => reciprocalRank(rank(q), q.relevant)))); return o;
}
const bm = (q) => bm25Ids(q.q, 50);
const retrieval = { bm25: { all: score(L, bm), dev: score(L.filter((q) => q.split === "dev"), bm), test: score(L.filter((q) => q.split === "test"), bm) },
  relevant_per_query: { min: Math.min(...L.map((q) => q.relevant.length)), max: Math.max(...L.map((q) => q.relevant.length)), mean: r3(mean(L.map((q) => q.relevant.length))) },
  vector_bge_base: "not yet run (needs Workers AI embeddings; quota resets ~7 PM CT Oct 28)", hybrid_rrf: "not yet run (needs vector side)" };
// Scope (off-topic) guard: off-topic set + all 24 on-topic retrieval queries.
const scopeCases = [...SET.scope, ...L.map((q) => ({ q: q.q, on: true, split: q.split }))];
const sc = (split) => { const c = scopeCases.filter((x) => x.split === split), ok = c.filter((x) => onTopic(x.q).on === x.on).length; const off = c.filter((x) => !x.on), on = c.filter((x) => x.on);
  return { n: c.length, accuracy: r3(ok / c.length), off_topic_blocked: `${off.filter((x) => !onTopic(x.q).on).length}/${off.length}`, on_topic_false_blocks: `${on.filter((x) => !onTopic(x.q).on).length}/${on.length}` }; };
// Injection guard: agent-14's red-team set (guard was written against it, so not independent) + Braylo on-topic queries as extra benign cases.
const cases = JSON.parse(fs.readFileSync(new URL("./guard-cases.agent14.json", import.meta.url)));
const att = cases.filter((c) => c.label === "attack"), ben = cases.filter((c) => c.label !== "attack");
const blocked = (c) => inputGuard(c.prompt + " " + (c.doc || "")).blocked;
const injection = { source: "agent-14 evals/cases.json", attacks_blocked: `${att.filter(blocked).length}/${att.length}`, benign_false_positives: `${ben.filter(blocked).length}/${ben.length}`,
  braylo_queries_false_positives: `${L.filter((q) => inputGuard(q.q).blocked).length}/${L.length}`, note: "Rules were tuned on the agent-14 set, so its numbers are optimistic." };
const pii = { n: SET.pii.length, passed: SET.pii.filter((p) => { const o = redactPII(p.in); return p.same ? o === p.in : !o.includes(p.must_not); }).length };
const tools = { router: "rules (fallback path)", n: SET.tools.length, accuracy: r3(SET.tools.filter((t) => (routeIntent(t.q)?.tool ?? null) === t.tool).length / SET.tools.length),
  misses: SET.tools.filter((t) => (routeIntent(t.q)?.tool ?? null) !== t.tool).map((t) => t.q), llm_function_calling: "not yet run (needs Workers AI)" };
// End-to-end extractive pipeline (no model): does the answer contain a gold phrase, and how grounded is it in its citations?
const e2e = [], spans = {}, totals = [];
for (const q of L) {
  const r = await answer({}, q.q, { noModel: true });
  const cited = r.citations?.map((c) => KB.chunks.find((k) => k.id === c.id).text) || [];
  e2e.push({ id: q.id, contains_gold: q.must.every((m) => (r.answer || "").toLowerCase().includes(m.toLowerCase())) ? 1 : 0, overlap: citationOverlap((r.answer || "").replace(/\[\d+\]/g, ""), cited), blocked: r.blocked || null });
  totals.push(r.duration_ms); for (const s of r.spans) (spans[s.name] ||= []).push(s.duration_ms);
}
const extractive = { n: e2e.length, answer_contains_gold_phrase: r3(mean(e2e.map((e) => e.contains_gold))), citation_overlap_mean: r3(mean(e2e.map((e) => e.overlap))), wrongly_blocked: e2e.filter((e) => e.blocked).map((e) => e.id),
  note: "Extractive answers copy sentences from cited chunks, so citation overlap is near 1 by construction; the gold-phrase rate is the informative number." };
const latency = { where: "box CPU, Node " + process.version + ", extractive path, 24 queries (not Workers edge latency)", total_ms: { p50: pct(totals, 50), p95: pct(totals, 95) },
  per_span_ms: Object.fromEntries(Object.entries(spans).map(([k, v]) => [k, { p50: pct(v, 50), p95: pct(v, 95) }])) };
const out = { project: "braylo-si-rag", run_at: new Date().toISOString(), kb: { files: KB.built_from.length, chunks: KB.chunks.length },
  retrieval, scope_guard: { threshold_bm25: SCOPE_MIN, dev: sc("dev"), test: sc("test") }, injection_guard: injection, pii_redaction: pii, tool_routing: tools, extractive_answers: extractive, latency,
  not_yet_run: { reason: "Workers AI free daily quota exhausted until ~7 PM CT Oct 28, 2026", items: ["vector (bge-base) and hybrid RRF retrieval metrics", "LLM function-calling accuracy", "LLM answer faithfulness (agent-11 judge)", "live Workers latency p50/p95 and token counts"] } };
fs.writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(out, null, 2));
console.log(JSON.stringify(out, null, 1));
