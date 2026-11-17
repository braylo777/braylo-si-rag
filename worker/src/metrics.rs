//! Retrieval and faithfulness metrics. Pure functions, no model calls.
use regex::Regex;
use std::collections::HashSet;

const STOP: &str = "a an and are as at be by can for from how in is it its of on or so that the their them then this to was what when where which who why with does do into than such each name two own other";

pub fn tokens(s: &str) -> Vec<String> {
    let stop: HashSet<&str> = STOP.split(' ').collect();
    let tok = Regex::new(r"[a-z0-9]+").unwrap();
    let stem = Regex::new(r"(ing|es|s)$").unwrap();
    tok.find_iter(&s.to_lowercase()).map(|m| m.as_str()).filter(|t| t.len() > 2 && !stop.contains(t)).map(|t| stem.replace(t, "").into_owned()).collect()
}

pub fn precision_at_k(ranked: &[String], relevant: &[String], k: usize) -> f64 {
    let r: HashSet<&String> = relevant.iter().collect();
    ranked.iter().take(k).filter(|i| r.contains(i)).count() as f64 / k as f64
}
pub fn recall_at_k(ranked: &[String], relevant: &[String], k: usize) -> f64 {
    let r: HashSet<&String> = relevant.iter().collect();
    if r.is_empty() { 0.0 } else { ranked.iter().take(k).filter(|i| r.contains(i)).count() as f64 / r.len() as f64 }
}
pub fn reciprocal_rank(ranked: &[String], relevant: &[String]) -> f64 {
    let r: HashSet<&String> = relevant.iter().collect();
    ranked.iter().position(|i| r.contains(i)).map_or(0.0, |i| 1.0 / (i as f64 + 1.0))
}
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}
/// JS Math.round(x * 1000) / 1000 (half rounds up).
pub fn r3(x: f64) -> f64 {
    (x * 1000.0 + 0.5).floor() / 1000.0
}
/// Share of the answer's content tokens that appear in the chunks it cites.
pub fn citation_overlap(answer: &str, cited: &[String]) -> f64 {
    let a = tokens(answer);
    let src: HashSet<String> = cited.iter().flat_map(|c| tokens(c)).collect();
    if a.is_empty() { 0.0 } else { a.iter().filter(|t| src.contains(*t)).count() as f64 / a.len() as f64 }
}
/// ROC AUC via pairwise comparison (ties count half).
pub fn auc(scores: &[f64], labels: &[i64]) -> f64 {
    let pos: Vec<f64> = scores.iter().zip(labels).filter(|(_, l)| **l == 1).map(|(s, _)| *s).collect();
    let neg: Vec<f64> = scores.iter().zip(labels).filter(|(_, l)| **l == 0).map(|(s, _)| *s).collect();
    let mut s = 0.0;
    for p in &pos {
        for n in &neg {
            s += if p > n { 1.0 } else if p == n { 0.5 } else { 0.0 };
        }
    }
    if pos.is_empty() || neg.is_empty() { 0.0 } else { s / (pos.len() * neg.len()) as f64 }
}
