// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

test("README LiveKit example typechecks against the installed client", () => {
  const root = join(dirname(fileURLToPath(import.meta.url)), "..");
  const readme = readFileSync(join(root, "README.md"), "utf8");
  const section = readme.split("## LiveKit usage")[1].split("## Options")[0];
  const snippet = section.match(/```ts\n([\s\S]*?)```/)[1];
  const dir = mkdtempSync(join(root, "test", ".docs-"));
  try {
    const file = join(dir, "example.ts");
    writeFileSync(file, 'import type { Room } from "livekit-client";\ndeclare const room: Room;\n' + snippet);
    const options = {
      strict: true, noEmit: true, skipLibCheck: true,
      target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      baseUrl: root,
      paths: { "open-noise-filter/livekit": ["src/livekit.ts"] },
    };
    const diagnostics = ts.getPreEmitDiagnostics(ts.createProgram([file], options));
    assert.equal(diagnostics.length, 0, ts.formatDiagnosticsWithColorAndContext(diagnostics, {
      getCanonicalFileName: (f) => f, getCurrentDirectory: () => root, getNewLine: () => "\n",
    }));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
