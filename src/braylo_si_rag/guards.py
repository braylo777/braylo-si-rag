"""Rule-based input/output guards (no model needed). Each rule has an id so blocks are explainable.
Regexes follow JavaScript semantics: ASCII word boundaries (\\b, \\w) and the JS whitespace set."""
import re, unicodedata

_JS_WS = "[\\t\\n\\v\\f\\r \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff]+"
_ZERO = re.compile("[\\u200b-\\u200f\\u2060\\ufeff]")
_WS = re.compile(_JS_WS)
A = re.ASCII


def norm(s):
    s = "" if s is None or s == "" else str(s)
    return _WS.sub(" ", _ZERO.sub("", unicodedata.normalize("NFKC", s))).lower()


INPUT_RULES = [(i, re.compile(p, A)) for i, p in [
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
]]


def input_guard(text):
    n = norm(text)
    hits = [i for i, rx in INPUT_RULES if rx.search(n)]
    return {"blocked": bool(hits), "rules": hits}


_SECRET = re.compile(r"\b(sk|pk|ghp|xox[bp])[-_][A-Za-z0-9_-]{12,}\b", A)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b", A)
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b", A)
_EXFIL = re.compile(r"!\[[^\]]*\]\(https?://[^)]*[?&][^)]*=")


def output_guard(text, canary=None, system_prompt=None):
    """Output guard: canary/secret leak, system-prompt echo, PII, exfil links."""
    t = "" if text is None or text == "" else str(text)
    n = norm(t)
    hits = []
    if canary and canary in t:
        hits.append("canary_leak")
    if system_prompt and norm(system_prompt)[:60] in n:
        hits.append("system_prompt_echo")
    if _SECRET.search(t):
        hits.append("secret_pattern")
    if _SSN.search(t) or _CARD.search(t):
        hits.append("pii_pattern")
    if _EXFIL.search(t):
        hits.append("exfil_link")
    return {"blocked": bool(hits), "rules": hits}
