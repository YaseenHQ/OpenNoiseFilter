// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/** Zero-dependency static server for the demo: `npm run demo`, open
 *  http://localhost:8080/demo/ */
import { createServer } from "http";
import { readFileSync, existsSync, statSync } from "fs";
import { join, extname, resolve } from "path";

const ROOT = resolve(import.meta.dirname, "..");
const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css",
};
const server = createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p.endsWith("/")) p += "index.html";
  const f = join(ROOT, p);
  if (!resolve(f).startsWith(ROOT) || !existsSync(f) || !statSync(f).isFile()) {
    res.writeHead(404); res.end("not found"); return;
  }
  res.writeHead(200, { "content-type": MIME[extname(f)] || "application/octet-stream" });
  res.end(readFileSync(f));
});
server.listen(8080, () => console.log("demo at http://localhost:8080/demo/"));
