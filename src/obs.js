// Observability: trace IDs, spans, token counting, published-price cost estimate, percentiles, KV trace store.
// Pricing source: developers.cloudflare.com/workers-ai/platform/pricing (read Oct 27, 2026). ESTIMATE only.
export const PRICING = { "@cf/meta/llama-3.3-70b-instruct-fp8-fast": { in_neurons_per_m: 26668, out_neurons_per_m: 204805 }, usd_per_1k_neurons: 0.011 };
export function newTraceId() { const a = new Uint8Array(16); crypto.getRandomValues(a); return [...a].map((b) => b.toString(16).padStart(2, "0")).join(""); }
export function tracer(route, now = () => performance.now()) {
  const t0 = now(), spans = [], trace_id = newTraceId();
  return {
    trace_id,
    async span(name, fn, attrs = {}) {
      const s = { name, start_ms: Math.round((now() - t0) * 100) / 100, attrs };
      try { return await fn(s); } catch (e) { s.error = String(e.message || e); throw e; } finally { s.duration_ms = Math.round((now() - t0 - s.start_ms) * 100) / 100; spans.push(s); }
    },
    finish(status, extra = {}) { return { trace_id, route, status, ts: Date.now(), duration_ms: Math.round((now() - t0) * 100) / 100, spans, ...extra }; },
  };
}
// Rough token estimate (~4 chars/token) used only when the model returns no usage block.
export const estTokens = (s) => Math.ceil(String(s || "").length / 4);
export function costEstimate(model, tin, tout) {
  const p = PRICING[model]; if (!p) return null;
  const neurons = (tin * p.in_neurons_per_m + tout * p.out_neurons_per_m) / 1e6;
  return { neurons: +neurons.toFixed(3), usd: +(neurons / 1000 * PRICING.usd_per_1k_neurons).toFixed(8), label: "estimate from published neuron pricing; free tier pays $0" };
}
export function pct(xs, p) { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.ceil((p / 100) * s.length) - 1)]; }
export function summarize(traces) {
  const by = {};
  for (const t of traces) (by[t.route] ||= []).push(t);
  const out = {};
  for (const [r, ts] of Object.entries(by)) {
    const d = ts.map((t) => t.duration_ms), ok = ts.filter((t) => t.status < 400);
    const cost = ts.reduce((a, t) => a + (t.cost?.usd || 0), 0), tok = ts.filter((t) => t.tokens);
    out[r] = { n: ts.length, errors: ts.length - ok.length, p50_ms: pct(d, 50), p95_ms: pct(d, 95),
      avg_tokens_in: tok.length ? +(tok.reduce((a, t) => a + t.tokens.in, 0) / tok.length).toFixed(1) : null,
      avg_tokens_out: tok.length ? +(tok.reduce((a, t) => a + t.tokens.out, 0) / tok.length).toFixed(1) : null,
      est_cost_usd_per_req: tok.length ? +(cost / tok.length).toFixed(8) : null };
  }
  return out;
}
const IDX = "idx", MAX = 200;
export async function store(env, trace) {
  if (!env.TRACES) return;
  await env.TRACES.put("t:" + trace.trace_id, JSON.stringify(trace));
  // Optimistic index update; concurrent writers may drop a few ids under load — acceptable for a demo window.
  const cur = (await env.TRACES.get(IDX, "json")) || [];
  cur.unshift(trace.trace_id);
  const next = [...new Set(cur)].slice(0, MAX);
  await env.TRACES.put(IDX, JSON.stringify(next));
  for (const drop of cur.slice(MAX)) await env.TRACES.delete("t:" + drop).catch(() => {});
}
export async function recent(env) {
  if (!env.TRACES) return [];
  const ids = (await env.TRACES.get(IDX, "json")) || [];
  const rows = await Promise.all(ids.map((id) => env.TRACES.get("t:" + id, "json")));
  return rows.filter(Boolean);
}
