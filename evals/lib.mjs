import fs from "node:fs";
import { KB } from "../src/kb.js";
export const SET = JSON.parse(fs.readFileSync(new URL("./queries.json", import.meta.url)));
export const relevant = (must) => KB.chunks.filter((c) => must.every((m) => c.text.toLowerCase().includes(m.toLowerCase()))).map((c) => c.id);
export const labelled = () => SET.retrieval.map((r) => ({ ...r, relevant: relevant(r.must) }));
