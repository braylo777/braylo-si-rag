//! Rule-based input/output guards (no model needed). Each rule has an id so blocks are explainable.
//! Regexes follow JavaScript semantics: ASCII word boundaries (\b, \w) and the JS whitespace set.
use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

const JS_WS: &str = r"[\t\n\x0B\x0C\r \x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}\x{feff}]+";

/// Write rules with JS-style `\b` and `\w`; they are compiled as ASCII-only assertions/classes.
fn js(p: &str) -> Regex {
    Regex::new(&p.replace(r"\b", r"(?-u:\b)").replace(r"\w", "[A-Za-z0-9_]")).expect("rule regex")
}

pub fn norm(s: &str) -> String {
    static ZERO: OnceLock<Regex> = OnceLock::new();
    static WS: OnceLock<Regex> = OnceLock::new();
    let zero = ZERO.get_or_init(|| Regex::new(r"[\x{200b}-\x{200f}\x{2060}\x{feff}]").unwrap());
    let ws = WS.get_or_init(|| Regex::new(JS_WS).unwrap());
    let n: String = s.nfkc().collect();
    ws.replace_all(&zero.replace_all(&n, ""), " ").to_lowercase()
}

const RULES: [(&str, &str); 10] = [
    ("override", r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|system|the)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|directives?)\b"),
    ("new_instructions", r"\b(new|updated|real|actual)\s+(system\s+)?(instructions?|directive|prompt)s?\s*[:\-]"),
    ("role_hijack", r"\b(you are now|from now on you|act as|pretend (to be|you are)|roleplay as)\b.{0,60}\b(dan|unfiltered|uncensored|jailbr\w*|no (rules|restrictions|limits)|evil|developer mode)\b"),
    ("jailbreak_mode", r"\b(developer mode|dan mode|jailbreak|do anything now|god mode|unrestricted mode)\b"),
    ("prompt_leak", r"\b(reveal|print|show|repeat|output|tell me|what (is|are)|leak|dump)\b.{0,40}\b(system prompt|your (instructions|prompt|rules)|hidden (prompt|instructions)|initial prompt|the text above)\b"),
    ("secret_ask", r"\b(reveal|print|dump|exfiltrate|send me|give me|tell me|output|leak)\b.{0,50}\b(api[_ ]?key|secret key|password|access token|the canary|credentials?|env(ironment)? var\w*)\b|\b(api[_ ]?key|secret key|the canary|access token)\b.{0,40}\b(reveal|print|dump|exfiltrate|send|give|tell|output|leak)\b"),
    ("exfil_channel", r"(!\[[^\]]*\]\(https?://|<img[^>]+src=|\bfetch\(|\b(send|post|forward|upload|exfiltrate|email)\b.{0,40}\b(to|at)\b.{0,20}(https?://|\b[\w.-]+@[\w-]+\.\w+|webhook))"),
    ("fake_delimiter", r"(</?\s*(system|assistant|im_start|im_end)\s*>|\[/?(system|inst)\]|###\s*(system|instruction)|<\|im_(start|end)\|>)"),
    ("hidden_instr", r"\b(ai|assistant|model|llm|chatbot)s?\b.{0,30}\b(reading|processing|summari[sz]ing) this\b|\bnote to (the )?(ai|assistant|model)\b"),
    ("encoding_smuggle", r"\b(decode|base64|rot13)\b.{0,40}\b(then|and)\b.{0,20}\b(follow|execute|obey|run)\b"),
];

fn rules() -> &'static Vec<(&'static str, Regex)> {
    static R: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    R.get_or_init(|| RULES.iter().map(|(i, p)| (*i, js(p))).collect())
}

pub fn input_guard(text: &str) -> Value {
    let n = norm(text);
    let hits: Vec<&str> = rules().iter().filter(|(_, re)| re.is_match(&n)).map(|(i, _)| *i).collect();
    json!({"blocked": !hits.is_empty(), "rules": hits})
}

/// Output guard: canary/secret leak, system-prompt echo, PII, exfil links.
pub fn output_guard(text: &str, canary: Option<&str>, system_prompt: Option<&str>) -> Value {
    static RX: OnceLock<[Regex; 4]> = OnceLock::new();
    let rx = RX.get_or_init(|| [js(r"\b(sk|pk|ghp|xox[bp])[-_][A-Za-z0-9_-]{12,}\b"), js(r"\b\d{3}-\d{2}-\d{4}\b"), js(r"\b(?:\d[ -]?){13,16}\b"), js(r"!\[[^\]]*\]\(https?://[^)]*[?&][^)]*=")]);
    let n = norm(text);
    let mut hits: Vec<&str> = vec![];
    if canary.map_or(false, |c| !c.is_empty() && text.contains(c)) {
        hits.push("canary_leak");
    }
    if let Some(sp) = system_prompt.filter(|s| !s.is_empty()) {
        let head: String = norm(sp).encode_utf16().take(60).collect::<Vec<u16>>().pipe(|u| String::from_utf16_lossy(&u));
        if n.contains(&head) {
            hits.push("system_prompt_echo");
        }
    }
    if rx[0].is_match(text) {
        hits.push("secret_pattern");
    }
    if rx[1].is_match(text) || rx[2].is_match(text) {
        hits.push("pii_pattern");
    }
    if rx[3].is_match(text) {
        hits.push("exfil_link");
    }
    json!({"blocked": !hits.is_empty(), "rules": hits})
}

trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}
