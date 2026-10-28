// Bundles src/ into one ES module (dist/worker.js) for API upload; wrangler users can deploy src/index.js directly.
import fs from "node:fs";
const order = ["core", "metrics", "bm25", "guards", "obs", "kb", "tools", "retrieve", "agent"];
const strip = (f) => fs.readFileSync(f, "utf8").replace(/^import .*\n/gm, "");
const out = order.map((f) => strip(`src/${f}.js`)).join("\n") + `\nconst page = ${JSON.stringify(fs.readFileSync("src/page.html", "utf8"))};\nconst RESULTS = ${fs.readFileSync("evals/results.json", "utf8")};\n` + strip("src/index.js");
fs.mkdirSync("dist", { recursive: true }); fs.writeFileSync("dist/worker.js", out); console.log("dist/worker.js", out.length, "bytes");
