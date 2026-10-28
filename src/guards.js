// Rule-based input/output guards (no model needed). Each rule has an id so blocks are explainable.
const norm = (s) => String(s || "").normalize("NFKC").replace(/[\u200b-\u200f\u2060\ufeff]/g, "").replace(/\s+/g, " ").toLowerCase();
export const INPUT_RULES = [
  ["override", /\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|system|the)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|directives?)\b/],
  ["new_instructions", /\b(new|updated|real|actual)\s+(system\s+)?(instructions?|directive|prompt)s?\s*[:\-]/],
  ["role_hijack", /\b(you are now|from now on you|act as|pretend (to be|you are)|roleplay as)\b.{0,60}\b(dan|unfiltered|uncensored|jailbr\w*|no (rules|restrictions|limits)|evil|developer mode)\b/],
  ["jailbreak_mode", /\b(developer mode|dan mode|jailbreak|do anything now|god mode|unrestricted mode)\b/],
  ["prompt_leak", /\b(reveal|print|show|repeat|output|tell me|what (is|are)|leak|dump)\b.{0,40}\b(system prompt|your (instructions|prompt|rules)|hidden (prompt|instructions)|initial prompt|the text above)\b/],
  ["secret_ask", /\b(reveal|print|dump|exfiltrate|send me|give me|tell me|output|leak)\b.{0,50}\b(api[_ ]?key|secret key|password|access token|the canary|credentials?|env(ironment)? var\w*)\b|\b(api[_ ]?key|secret key|the canary|access token)\b.{0,40}\b(reveal|print|dump|exfiltrate|send|give|tell|output|leak)\b/],
  ["exfil_channel", /(!\[[^\]]*\]\(https?:\/\/|<img[^>]+src=|\bfetch\(|\b(send|post|forward|upload|exfiltrate|email)\b.{0,40}\b(to|at)\b.{0,20}(https?:\/\/|\b[\w.-]+@[\w-]+\.\w+|webhook))/],
  ["fake_delimiter", /(<\/?\s*(system|assistant|im_start|im_end)\s*>|\[\/?(system|inst)\]|###\s*(system|instruction)|<\|im_(start|end)\|>)/],
  ["hidden_instr", /\b(ai|assistant|model|llm|chatbot)s?\b.{0,30}\b(reading|processing|summari[sz]ing) this\b|\bnote to (the )?(ai|assistant|model)\b/],
  ["encoding_smuggle", /\b(decode|base64|rot13)\b.{0,40}\b(then|and)\b.{0,20}\b(follow|execute|obey|run)\b/],
];
export function inputGuard(text) {
  const n = norm(text), hits = INPUT_RULES.filter(([, re]) => re.test(n)).map(([id]) => id);
  return { blocked: hits.length > 0, rules: hits };
}
// Output guard: canary/secret leak, system-prompt echo, PII, exfil links.
export function outputGuard(text, { canary, systemPrompt } = {}) {
  const t = String(text || ""), n = norm(t), hits = [];
  if (canary && t.includes(canary)) hits.push("canary_leak");
  if (systemPrompt && n.includes(norm(systemPrompt).slice(0, 60))) hits.push("system_prompt_echo");
  if (/\b(sk|pk|ghp|xox[bp])[-_][A-Za-z0-9_-]{12,}\b/.test(t)) hits.push("secret_pattern");
  if (/\b\d{3}-\d{2}-\d{4}\b/.test(t) || /\b(?:\d[ -]?){13,16}\b/.test(t)) hits.push("pii_pattern");
  if (/!\[[^\]]*\]\(https?:\/\/[^)]*[?&][^)]*=/.test(t)) hits.push("exfil_link");
  return { blocked: hits.length > 0, rules: hits };
}
