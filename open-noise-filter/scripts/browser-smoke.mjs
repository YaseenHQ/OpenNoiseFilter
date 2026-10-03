// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/** Packed-package consumer test: npm run test:browser.
 * Install Chromium once with `npx playwright install chromium`.
 * Optional BROWSER_EXECUTABLE selects an installed Chromium/Chrome/Edge.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build, createServer, preview } from "vite";
import { chromium } from "playwright";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const temp = mkdtempSync(join(tmpdir(), "noise-filter-consumer-"));
let browser, server;
try {
  // Use npm's JS entry on Windows to avoid cmd.exe command quoting.
  const npmCli = process.env.npm_execpath;
  assert.ok(npmCli, "Run this script through npm run test:browser");
  console.log("Building and packing the consumer package...");
  execFileSync(process.execPath, [npmCli, "pack", "--pack-destination", temp], { cwd: root, stdio: "pipe", timeout: 120000 });
  const { name, version } = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  writeFileSync(join(temp, "package.json"), JSON.stringify({ private: true, type: "module" }));
  execFileSync(process.execPath, [npmCli, "install", "--ignore-scripts", "--legacy-peer-deps", "--no-audit", "--no-fund", join(temp, `${name.replace("@", "").replace("/", "-")}-${version}.tgz`)], { cwd: temp, stdio: "pipe", timeout: 120000 });
  cpSync(join(root, "scripts", "browser-fixture"), temp, { recursive: true });
  mkdirSync(join(temp, "public"));
  cpSync(join(root, "test", "reference_x.f32"), join(temp, "public", "reference.f32"));
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.BROWSER_EXECUTABLE || undefined,
    args: ["--autoplay-policy=no-user-gesture-required"],
  });
  for (const mode of ["development", "production"]) {
    const config = { root: temp, configFile: false, logLevel: "warn", server: { host: "127.0.0.1", port: 0 }, preview: { host: "127.0.0.1", port: 0 } };
    if (mode === "production") {
      await build(config);
      server = await preview(config);
    } else {
      server = await createServer(config);
      await server.listen();
    }
    const address = server.httpServer.address();
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("requestfailed", (request) => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
    page.on("response", (response) => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
    await page.goto(`http://127.0.0.1:${address.port}/`);
    await page.waitForFunction(() => typeof window.runFilterSmoke === "function");
    const cases = [
      { quality: "low" }, { quality: "medium" }, { quality: "high" }, { quality: "gate" },
      { quality: "medium", sampleRate: 44100, maxChannels: 2 },
      { quality: "medium", sampleRate: 16000 },
      { quality: "medium", scalar: true },
    ];
    for (const options of cases) {
      const result = await page.evaluate((o) => window.runFilterSmoke(o), options);
      assert.equal(result.thread, options.quality === "low" || options.quality === "gate" ? "audio" : "worker");
      assert.ok(result.peak > 0);
      console.log(`${mode}: ${JSON.stringify(result)}`);
    }
    console.log(`${mode} LiveKit: ${JSON.stringify(await page.evaluate(() => window.runLivekitSmoke()))}`);
    assert.deepEqual(errors, [], "Browser errors or missing assets");
    await page.close();
    await server.close();
    server = undefined;
  }
  console.log("Packed Vite consumer passed in development and production.");
} finally {
  await server?.close();
  await browser?.close();
  rmSync(temp, { recursive: true, force: true });
}
