#!/usr/bin/env node
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";

// cargo-workspaces preserves dependency order and skips versions already in the
// registry. Retry only a registry quota refusal with an explicit retry time.
export function retryDelay(output, now = Date.now()) {
  const quota = /You have published too many (?:new crates|updates to existing crates) in a short period of time/i;
  if (!quota.test(output)) return null;
  const date = output.match(/Please try again after ([^\r\n]+? GMT)/i)?.[1];
  const timestamp = date ? Date.parse(date) : NaN;
  if (!Number.isFinite(timestamp)) return null;
  const delay = Math.max(2_000, timestamp - now + 2_000);
  return delay <= 24 * 60 * 60 * 1_000 ? delay : null;
}

export async function publishWithBackoff({ run, wait = sleep, now = Date.now, log = console.log, maxAttempts = 200 }) {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const result = await run();
    if (result.code === 0) return;
    const delay = retryDelay(result.output, now());
    if (delay === null || attempt === maxAttempts) {
      throw new Error(`Crate publication stopped (exit ${result.code}); inspect the preceding Cargo error.`);
    }
    log(`Registry quota reached; resuming after ${Math.ceil(delay / 1_000)} seconds. Published versions will be skipped.`);
    await wait(delay);
  }
}

function publishWorkspace() {
  return new Promise((resolve, reject) => {
    // Cargo reads CARGO_REGISTRY_TOKEN directly. Never put a token in argv.
    const child = spawn("cargo", ["workspaces", "publish", "--all", "--yes", "--allow-dirty",
      "--skip-published", "--publish-as-is", "--publish-interval", "30", "--no-verify"],
    { stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    for (const [stream, destination] of [[child.stdout, process.stdout], [child.stderr, process.stderr]]) {
      stream.on("data", chunk => {
        destination.write(chunk);
        output = (output + chunk.toString()).slice(-256 * 1024);
      });
    }
    child.on("error", reject);
    child.on("close", (code, signal) => resolve({ code: code ?? signal ?? "unknown", output }));
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.env.CARGO_REGISTRY_TOKEN) throw new Error("CARGO_REGISTRY_TOKEN is required for release publication.");
  await publishWithBackoff({ run: publishWorkspace });
}
