import page from "./page.html";
import { json, limited, body, cap } from "./core.js";
import { answer } from "./agent.js";
import { hybrid } from "./retrieve.js";
import { TOOLS, TOOL_SCHEMAS } from "./tools.js";
import { summarize, store, recent } from "./obs.js";
import { KB } from "./kb.js";
import RESULTS from "../evals/results.json" with { type: "json" };
const RING = []; // per-isolate fallback when no KV is bound
export default {
  async fetch(req, env = {}, ctx) {
    const u = new URL(req.url), p = u.pathname;
    if (p === "/" ) return new Response(page, { headers: { "content-type": "text/html; charset=utf-8" } });
    if (p === "/api/health") return json({ ok: true, chunks: KB.chunks.length, ai: !!env.AI, vectors: !!env.VECTORS });
    if (p === "/api/tools") return json(TOOL_SCHEMAS);
    if (p === "/api/eval") return json(RESULTS);
    if (p === "/api/metrics") { const t = env.TRACES ? await recent(env) : RING; return json({ n: t.length, by_route: summarize(t), store: env.TRACES ? "kv" : "isolate memory" }); }
    if (p === "/api/search") { const q = u.searchParams.get("q") || ""; if (!cap(q)) return json({ error: "q required (<=600 chars)" }, 400); return json(await hybrid(env, q, 5)); }
    if (req.method !== "POST") return json({ error: "not found" }, 404);
    if (await limited(req, env)) return json({ error: "rate limited: 5 requests/min" }, 429);
    const b = await body(req);
    if (p === "/api/tool") { const f = TOOLS[b.name]; if (!f) return json({ error: "unknown tool" }, 400); return json(f(b.args || {})); }
    if (p === "/api/chat") {
      if (!cap(b.message)) return json({ error: "message required (<=600 chars)" }, 400);
      const r = await answer(env, b.message);
      const trace = { trace_id: r.trace_id, route: r.route, status: r.status, ts: r.ts, duration_ms: r.duration_ms, spans: r.spans, tokens: r.tokens, cost: r.cost };
      if (env.TRACES) { const w = store(env, trace).catch(() => {}); ctx?.waitUntil ? ctx.waitUntil(w) : await w; } else { RING.unshift(trace); RING.length = Math.min(RING.length, 200); }
      return json(r, r.status, { "x-trace-id": r.trace_id });
    }
    return json({ error: "not found" }, 404);
  },
};
