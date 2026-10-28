// Agent tools (function calling). All deterministic: no model, no side effects, no bookings created.
import { SITE } from "./core.js";
// Source of truth: How We Help section of the site preview (worker v22v).
export const TIERS = [
  { step: 1, name: "Discoverability Audit", es: "Auditoría de Visibilidad", price: "Free", tagline: "See where you're leaking.", includes: ["Your monthly revenue leak, in dollars, and where it's coming from.", "A step-by-step fix plan your AI agents can follow."] },
  { step: 2, name: "Agent Infrastructure Install", price: "$497", billing: "one-time", tagline: "Lay a digital foundation.", includes: ["Your own AI agent team, set up and linked to your accounts.", "Your Google, Facebook, X and Grok accounts set up, with your SEO fixed.", "Phone, email and security cleaned up, and you own all of it."] },
  { step: 3, name: "Autonomous Ads Campaign", plan: "Basic", price: "$500/wk", tagline: "Stay in front of buyers.", includes: ["3 ready-to-post ads or clips every week, delivered to your Drive.", "Copy written for you, posted from your own ad accounts."] },
  { step: 4, name: "Inbound Closing Engine", plan: "Pro", price: "$1,500/wk", tagline: "Turn leads into booked jobs.", includes: ["A voice AI agent that answers, qualifies and books every lead.", "Automatic follow-up so no lead goes cold."] },
  { step: 5, name: "Subcontractor Coordination", plan: "VIP", price: "$2,000/wk", tagline: "Step into a broker role.", includes: ["Subs recruited, onboarded and scheduled for you.", "Our agents run day-to-day communication with your crews."] },
];
const SRC = "sources/site_index.txt";
const fold = (s) => String(s || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
export function lookup_tier({ query = "" } = {}) {
  const q = fold(query);
  if (/\b(all|every|ladder|tiers|plans|pricing|prices|precios|planes)\b/.test(q)) return { tiers: TIERS, source: SRC };
  const hit = TIERS.filter((t) => [t.name, t.es, t.plan, t.price, String(t.step)].filter(Boolean).some((k) => q.includes(fold(k))) ||
    (/audit|auditoria|free|gratis/.test(q) && t.step === 1) || (/install|497|instala/.test(q) && t.step === 2) || (/\bads?\b|anuncio|basic/.test(q) && t.step === 3) ||
    (/voice|closing|pro\b|leads/.test(q) && t.step === 4) || (/vip|sub(contractor)?s?\b|crew/.test(q) && t.step === 5));
  return hit.length ? { tiers: hit, source: SRC } : { tiers: [], note: "No matching tier. Ask about: " + TIERS.map((t) => t.name).join(", "), source: SRC };
}
export function find_booking_slot({ lang = "en" } = {}) {
  const url = SITE + (lang === "es" ? "/book.es" : "/book");
  return { url, note: "Pick an open time on the booking page. This assistant does not create bookings.", source: "sources/site_book.txt" };
}
// Free Discoverability Audit checklist. Leak math only uses numbers the owner gives; unknowns are "not checked" (never invent numbers).
export const CHECKS = [
  ["has_website", "Website that loads on mobile and states your service area"],
  ["google_profile", "Google Business Profile claimed, with hours, photos and services"],
  ["reviews_30plus", "30+ recent reviews, with replies"],
  ["listings_consistent", "Same name, address and phone on Google, Facebook, Yelp and directories"],
  ["answers_missed_calls", "Missed calls get an answer or a text back within 5 minutes"],
  ["fast_web_reply", "Web leads get a reply within 5 minutes"],
  ["quote_followup", "Every open quote gets a follow-up"],
];
export function audit_checklist(a = {}) {
  const items = CHECKS.map(([k, label]) => ({ check: k, label, status: a[k] === true ? "pass" : a[k] === false ? "gap" : "not checked" }));
  const n = (k) => (typeof a[k] === "number" && isFinite(a[k]) && a[k] >= 0 ? a[k] : null);
  const calls = n("missed_calls_per_week"), job = n("avg_job_value"), rate = n("close_rate");
  const leak = calls != null && job != null && rate != null && rate <= 1
    ? { monthly_usd: Math.round(calls * rate * job * 52 / 12), math: `${calls} missed calls/wk × ${rate} close rate × $${job} avg job × 52/12` }
    : { monthly_usd: null, math: "not checked: needs missed_calls_per_week, close_rate (0–1) and avg_job_value" };
  return { items, gaps: items.filter((i) => i.status === "gap").length, leak, next: "Book a free call to get the full audit: " + SITE + "/book", source: "sources/site_index.txt" };
}
export const TOOLS = { lookup_tier, find_booking_slot, audit_checklist };
// JSON schemas in the Workers AI / OpenAI function-calling format.
export const TOOL_SCHEMAS = [
  { name: "lookup_tier", description: "Look up a Braylo SI tier: name, price, tagline and what it includes.", parameters: { type: "object", properties: { query: { type: "string", description: "Tier name, number, price, or 'all'" } }, required: ["query"] } },
  { name: "find_booking_slot", description: "Return the link to the site's booking page where the user picks a time. Never books.", parameters: { type: "object", properties: { lang: { type: "string", enum: ["en", "es"] } } } },
  { name: "audit_checklist", description: "Run the free Discoverability Audit checklist and leak math from facts the owner gives.", parameters: { type: "object", properties: Object.fromEntries([...CHECKS.map(([k, l]) => [k, { type: "boolean", description: l }]), ["missed_calls_per_week", { type: "number" }], ["close_rate", { type: "number", description: "0-1" }], ["avg_job_value", { type: "number", description: "USD" }]]) } },
];
// Deterministic router used when the model is unavailable (and as an eval baseline for model routing).
export function routeIntent(q) {
  const s = fold(q);
  if (/\b(book|booking|schedule|appointment|call with|pick a time|reservar|cita|agendar)\b/.test(s)) return { tool: "find_booking_slot", args: { lang: /reservar|cita|agendar/.test(s) ? "es" : "en" } };
  if (/\b(checklist|audit me|run (the|an|my)? ?audit|check my|am i leaking|leak math)\b/.test(s)) return { tool: "audit_checklist", args: {} };
  if (/\b(price|cost|how much|tier|plan|pricing|\$|precio|cuanto|cuesta)\b/.test(s)) return { tool: "lookup_tier", args: { query: q } };
  return null;
}
