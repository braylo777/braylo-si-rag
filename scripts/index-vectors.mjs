// After the quota reset: embeds every chunk with bge-base-en-v1.5 and upserts into Vectorize, via the Cloudflare REST API.
// Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the environment (never committed). ~68 chunks, well inside the free tier.
import { KB } from "../src/kb.js";
const { CLOUDFLARE_ACCOUNT_ID: A, CLOUDFLARE_API_TOKEN: T } = process.env;
if (!A || !T) { console.error("set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN"); process.exit(1); }
const api = (p, init) => fetch(`https://api.cloudflare.com/client/v4/accounts/${A}${p}`, { ...init, headers: { authorization: `Bearer ${T}`, ...(init?.headers || {}) } }).then((r) => r.json());
for (let i = 0; i < KB.chunks.length; i += 50) {
  const batch = KB.chunks.slice(i, i + 50);
  const e = await api("/ai/run/@cf/baai/bge-base-en-v1.5", { method: "POST", body: JSON.stringify({ text: batch.map((c) => c.doc + "\n" + c.text) }) });
  if (!e.success) throw new Error(JSON.stringify(e.errors));
  const nd = batch.map((c, j) => JSON.stringify({ id: c.id, values: e.result.data[j], metadata: { doc: c.doc } })).join("\n");
  const u = await api("/vectorize/v2/indexes/braylo-si-rag-kb/upsert", { method: "POST", body: nd, headers: { "content-type": "application/x-ndjson" } });
  console.log(i, u.success ? "ok" : JSON.stringify(u.errors));
}
