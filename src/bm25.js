// BM25 lexical retriever (k1=1.5, b=0.75). Runs inside the Worker with no model calls.
import { tokens } from "./metrics.js";
export function bm25Index(chunks, k1 = 1.5, b = 0.75) {
  const docs = chunks.map((c) => tokens(c.text)), N = docs.length, avg = docs.reduce((a, d) => a + d.length, 0) / N, df = new Map();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) || 0) + 1);
  return (query) => {
    const q = tokens(query);
    return docs.map((d, i) => {
      let s = 0;
      for (const t of q) { const f = d.filter((x) => x === t).length; if (!f) continue; const n = df.get(t); s += Math.log(1 + (N - n + 0.5) / (n + 0.5)) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.length / avg)); }
      return { id: chunks[i].id, score: s };
    }).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  };
}
