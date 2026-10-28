// Shared helpers: Workers AI (free tier), rate limiting, JSON responses.
export const MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
export const EMBED = "@cf/baai/bge-base-en-v1.5"; // 768-dim, matches the Vectorize index
export const SITE = "https://braylo-si.pages.dev";
export const json = (o, status = 200, h = {}) => Response.json(o, { status, headers: h });
export async function limited(req, env) {
  const ip = req.headers.get("cf-connecting-ip") || "anon";
  return env.RATE_LIMITER && !(await env.RATE_LIMITER.limit({ key: ip })).success;
}
export async function body(req) { return req.json().catch(() => ({})); }
export const cap = (s, n = 600) => typeof s === "string" && s.trim() && s.length <= n;
