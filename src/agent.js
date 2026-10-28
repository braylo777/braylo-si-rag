// The agent: guards -> scope check -> tool routing -> hybrid retrieval -> answer with citations -> output guard. Every step is a traced span.
import { inputGuard, outputGuard } from "./guards.js";
import { tracer, estTokens, costEstimate } from "./obs.js";
import { hybrid } from "./retrieve.js";
import { bm25Index } from "./bm25.js";
import { tokens } from "./metrics.js";
import { KB } from "./kb.js";
import { TOOLS, TOOL_SCHEMAS, routeIntent } from "./tools.js";
import { MODEL } from "./core.js";
export const CANARY = "BRAYLO-CANARY-7f3a"; // planted in the system prompt; the output guard blocks any reply containing it
export const SYSTEM = `You are the Braylo SI assistant. Answer only questions about Braylo SI: its offers, prices, the free Discoverability Audit, booking a call, and how the AI install works. Use only the SOURCES and tool results. Cite sources as [n]. If the sources do not say, say you don't know and suggest booking a call. Never create bookings, never ask for passwords, never give legal, tax, medical or financial advice. Internal marker: ${CANARY}.`;
// PII redaction applied to user input before it is logged or sent to a model.
export function redactPII(s) {
  return String(s)
    .replace(/[\w.+-]+@[\w-]+\.[\w.]+/g, "[email]")
    .replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[ssn]")
    .replace(/\b(?:\d[ -]?){13,16}\b/g, "[card]")
    .replace(/(\+?1[ .-]?)?\(?\b\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b/g, "[phone]");
}
// Scope guard: on-topic if a tool intent matches or the best BM25 score clears a threshold set on the dev split (see evals/run.mjs).
const scoreTop = bm25Index(KB.chunks);
export const SCOPE_MIN = 6;
const BRAND = /\b(braylo|tier|install|audit|agent|ads?|leak|book|price|grok|refund|cancel|terms|privacy|voice|subcontract|seo|accounts?|passwords?|log ?in|xai|guarantee|refund)\b/;
export function onTopic(q) { const top = scoreTop(q)[0]?.score || 0; return { on: top >= SCOPE_MIN || BRAND.test(q.toLowerCase()), top: +top.toFixed(2) }; }
// Extractive answer (no model): best-overlapping sentence from each of the top chunks, with [n] citations.
export function extractive(q, hits, n = 2) {
  const qt = new Set(tokens(q));
  const picks = hits.slice(0, n).map((h, i) => {
    const sents = h.text.split(/(?<=[.!?])\s+|\n/).map((s) => s.replace(/^[-*\s\[\]x|]+/, "").trim()).filter((s) => s.split(/\s+/).length >= 5 && !s.endsWith("?"));
    const best = sents.map((s) => [s, tokens(s).filter((t) => qt.has(t)).length]).sort((a, b) => b[1] - a[1])[0];
    return best && best[1] > 0 ? `${best[0]} [${i + 1}]` : "";
  }).filter(Boolean);
  return picks.length ? picks.join(" ") : "I don't know from the sources. Book a free call: https://braylo-si.pages.dev/book";
}
// Plain-language summary of a tool result, used on the no-model path.
export function toolSummary(name, r) {
  if (name === "lookup_tier") return r.tiers.length ? r.tiers.map((t) => `${t.name}${t.plan ? " (" + t.plan + ")" : ""}: ${t.price}${t.billing ? ", " + t.billing : ""}. ${t.tagline}`).join(" ") : r.note;
  if (name === "find_booking_slot") return `Pick a time on the booking page: ${r.url} (I can't book it for you).`;
  if (name === "audit_checklist") return `Free audit checklist: ${r.items.map((i) => i.label).join("; ")}. Tell me which apply and your missed calls per week, close rate and average job value for the leak math. ${r.next}`;
  return "";
}
function ctx(hits) { return hits.map((h, i) => `[${i + 1}] (${h.doc}) ${h.text}`).join("\n\n"); }
export async function answer(env, raw, opts = {}) {
  const tr = tracer("chat", opts.now);
  let tin = 0, tout = 0;
  const q = await tr.span("pii_redact", async () => redactPII(raw));
  const g = await tr.span("guard_in", async () => inputGuard(q));
  if (g.blocked) return { ...tr.finish(400), blocked: "prompt_injection", rules: g.rules, answer: "I can't help with that request." };
  const scope = await tr.span("scope", async () => onTopic(q));
  const intent = routeIntent(q);
  if (!scope.on && !intent) return { ...tr.finish(200), blocked: "off_topic", scope, answer: "I only answer questions about Braylo SI's offers, the free audit and booking a call." };
  const ret = await tr.span("retrieve", async (s) => { const r = await hybrid(env, q, 5); s.attrs.mode = r.mode; return r; });
  let tool_calls = [], text = null, model_error = null, mode = "extractive";
  if (env?.AI && !opts.noModel) {
    try {
      const msgs = [{ role: "system", content: SYSTEM }, { role: "user", content: `SOURCES:\n${ctx(ret.hits)}\n\nQUESTION: ${q}` }];
      const r1 = await tr.span("llm_plan", async () => env.AI.run(MODEL, { messages: msgs, tools: TOOL_SCHEMAS, max_tokens: 400 }));
      tin += r1.usage?.prompt_tokens ?? estTokens(msgs.map((m) => m.content).join("")); tout += r1.usage?.completion_tokens ?? estTokens(r1.response);
      for (const c of r1.tool_calls || []) if (TOOLS[c.name]) tool_calls.push({ name: c.name, args: c.arguments || {}, result: await tr.span("tool:" + c.name, async () => TOOLS[c.name](c.arguments || {})) });
      if (tool_calls.length) {
        msgs.push({ role: "assistant", content: "" }, { role: "user", content: "TOOL RESULTS:\n" + JSON.stringify(tool_calls.map(({ name, result }) => ({ name, result }))) + "\nAnswer the question using these and the sources." });
        const r2 = await tr.span("llm_answer", async () => env.AI.run(MODEL, { messages: msgs, max_tokens: 400 }));
        tin += r2.usage?.prompt_tokens ?? estTokens(msgs.map((m) => m.content).join("")); tout += r2.usage?.completion_tokens ?? estTokens(r2.response); text = r2.response;
      } else text = r1.response;
      mode = "llm";
    } catch (e) { model_error = String(e.message || e); }
  }
  if (text == null) {
    if (intent) tool_calls.push({ name: intent.tool, args: intent.args, result: await tr.span("tool:" + intent.tool, async () => TOOLS[intent.tool](intent.args)), routed_by: "rules" });
    const lead = tool_calls.map((c) => toolSummary(c.name, c.result)).filter(Boolean).join(" ");
    text = await tr.span("extractive", async () => (lead ? lead + "\n\n" : "") + extractive(q, ret.hits));
  }
  const og = await tr.span("guard_out", async () => outputGuard(text, { canary: CANARY, systemPrompt: SYSTEM }));
  if (og.blocked) text = "I can't share that.";
  const cites = ret.hits.map((h, i) => ({ n: i + 1, id: h.id, doc: h.doc, source: h.source, snippet: h.text.slice(0, 220) }));
  return { ...tr.finish(200, { tokens: mode === "llm" ? { in: tin, out: tout } : null, cost: mode === "llm" ? costEstimate(MODEL, tin, tout) : null }),
    mode, retrieval: ret.mode, vector_error: ret.vector_error, model_error, answer: text, tool_calls, citations: cites, output_guard: og.rules };
}
