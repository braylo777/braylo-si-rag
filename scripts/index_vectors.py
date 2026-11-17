"""After the Workers AI quota reset: embed every chunk with bge-base-en-v1.5 and upsert into Vectorize via the Cloudflare REST API.
Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the environment (never committed). ~68 chunks, well inside the free tier."""

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = json.loads((ROOT / "data" / "kb.json").read_text(encoding="utf-8"))
A, T = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
if not A or not T:
    sys.exit("set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")


def api(path, body, ctype="application/json"):
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{A}{path}",
        data=body.encode("utf-8"),
        method="POST",
        headers={"authorization": f"Bearer {T}", "content-type": ctype},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


chunks = KB["chunks"]
for i in range(0, len(chunks), 50):
    batch = chunks[i : i + 50]
    e = api("/ai/run/@cf/baai/bge-base-en-v1.5", json.dumps({"text": [c["doc"] + "\n" + c["text"] for c in batch]}))
    if not e.get("success"):
        raise SystemExit(json.dumps(e.get("errors")))
    nd = "\n".join(
        json.dumps({"id": c["id"], "values": e["result"]["data"][j], "metadata": {"doc": c["doc"]}})
        for j, c in enumerate(batch)
    )
    u = api("/vectorize/v2/indexes/braylo-si-rag-kb/upsert", nd, "application/x-ndjson")
    print(i, "ok" if u.get("success") else json.dumps(u.get("errors")))
