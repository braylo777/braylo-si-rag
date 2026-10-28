// Lets Node import .html as a string, like the Workers "Text" rule in wrangler.toml.
import fs from "node:fs";
export async function load(url, ctx, next) {
  if (url.endsWith(".html")) return { format: "module", shortCircuit: true, source: `export default ${JSON.stringify(fs.readFileSync(new URL(url), "utf8"))};` };
  return next(url, ctx);
}
