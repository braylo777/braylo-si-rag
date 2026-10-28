// Hybrid retrieval: BM25 (in-Worker) + Workers AI embeddings in Vectorize, fused with reciprocal rank fusion (RRF, k=60).
// If the vector side is unavailable (no binding, quota, error) it degrades to BM25 and says so.
import { bm25Index } from "./bm25.js";
import { KB } from "./kb.js";
import { EMBED } from "./core.js";
const bm25 = bm25Index(KB.chunks);
export const byId = new Map(KB.chunks.map((c) => [c.id, c]));
export function rrf(lists, k = 60) {
  const s = new Map();
  for (const l of lists) l.forEach((id, i) => s.set(id, (s.get(id) || 0) + 1 / (k + i + 1)));
  return [...s.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([id, score]) => ({ id, score }));
}
export const bm25Ids = (q, n = 20) => bm25(q).filter((h) => h.score > 0).slice(0, n).map((h) => h.id);
export async function vectorIds(env, q, n = 20) {
  const e = await env.AI.run(EMBED, { text: [q] });
  const r = await env.VECTORS.query(e.data[0], { topK: n });
  return r.matches.map((m) => m.id);
}
export async function hybrid(env, q, k = 5) {
  const lex = bm25Ids(q); let vec = null, vector_error = null;
  if (env?.AI && env?.VECTORS) { try { vec = await vectorIds(env, q); } catch (e) { vector_error = String(e.message || e); } }
  const ranked = vec ? rrf([lex, vec]) : lex.map((id, i) => ({ id, score: 1 / (60 + i + 1) }));
  return { mode: vec ? "hybrid" : "bm25", vector_error, hits: ranked.slice(0, k).map((h) => ({ ...byId.get(h.id), score: +h.score.toFixed(5) })) };
}
