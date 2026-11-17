//! Knowledge base (public site copy only, built by scripts/ingest.py into data/kb.json) and a cached BM25 index over it.
use crate::metrics::tokens;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

pub const MODEL: &str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
pub const EMBED: &str = "@cf/baai/bge-base-en-v1.5"; // 768-dim, matches the Vectorize index
pub const SITE: &str = "https://braylo-si.pages.dev";
const KB_JSON: &str = include_str!("../data/kb.json");

#[derive(Clone, Debug)]
pub struct Chunk {
    pub id: String,
    pub doc: String,
    pub source: String,
    pub lang: String,
    pub text: String,
}

pub fn chunks() -> &'static Vec<Chunk> {
    static C: OnceLock<Vec<Chunk>> = OnceLock::new();
    C.get_or_init(|| {
        let v: Value = serde_json::from_str(KB_JSON).unwrap();
        v["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Chunk {
                id: c["id"].as_str().unwrap().into(),
                doc: c["doc"].as_str().unwrap().into(),
                source: c["source"].as_str().unwrap().into(),
                lang: c["lang"].as_str().unwrap().into(),
                text: c["text"].as_str().unwrap().into(),
            })
            .collect()
    })
}
pub fn by_id(id: &str) -> Option<&'static Chunk> {
    static M: OnceLock<HashMap<String, usize>> = OnceLock::new();
    M.get_or_init(|| {
        chunks()
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.clone(), i))
            .collect()
    })
    .get(id)
    .map(|i| &chunks()[*i])
}

pub fn cap(s: Option<&str>, n: usize) -> bool {
    matches!(s, Some(s) if !s.trim().is_empty() && s.encode_utf16().count() <= n)
}

struct Bm25 {
    docs: Vec<Vec<String>>,
    df: HashMap<String, f64>,
    n: f64,
    avg: f64,
}
fn bm25() -> &'static Bm25 {
    static B: OnceLock<Bm25> = OnceLock::new();
    B.get_or_init(|| {
        let docs: Vec<Vec<String>> = chunks().iter().map(|c| tokens(&c.text)).collect();
        let n = docs.len() as f64;
        let avg = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / n;
        let mut df: HashMap<String, f64> = HashMap::new();
        for d in &docs {
            for t in d.iter().collect::<HashSet<_>>() {
                *df.entry(t.clone()).or_insert(0.0) += 1.0;
            }
        }
        Bm25 { docs, df, n, avg }
    })
}

/// BM25 (k1=1.5, b=0.75) over the KB. Returns (chunk id, score) sorted by score desc then id.
pub fn bm25_search(query: &str) -> Vec<(String, f64)> {
    let (k1, b) = (1.5f64, 0.75f64);
    let ix = bm25();
    let q = tokens(query);
    let mut out: Vec<(String, f64)> = ix
        .docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut s = 0.0;
            for t in &q {
                let f = d.iter().filter(|x| *x == t).count() as f64;
                if f == 0.0 {
                    continue;
                }
                let n = ix.df[t.as_str()];
                s += (1.0 + (ix.n - n + 0.5) / (n + 0.5)).ln() * (f * (k1 + 1.0))
                    / (f + k1 * (1.0 - b + b * d.len() as f64 / ix.avg));
            }
            (chunks()[i].id.clone(), s)
        })
        .collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}
