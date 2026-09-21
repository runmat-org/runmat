#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const expected = process.argv[2];
if (!expected || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(expected)) {
  console.error("Usage: scripts/verify-release-version.mjs <version>");
  process.exit(1);
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const metadata = JSON.parse(execFileSync(
  "cargo",
  ["metadata", "--locked", "--format-version", "1", "--no-deps"],
  { cwd: root, encoding: "utf8" },
));
const workspaceMembers = new Set(metadata.workspace_members);
const mismatchedCrates = metadata.packages
  .filter((pkg) => workspaceMembers.has(pkg.id) && pkg.version !== expected)
  .map((pkg) => `${pkg.name}@${pkg.version}`)
  .sort();
const publishableCratesMissingDescriptions = metadata.packages
  .filter((pkg) => workspaceMembers.has(pkg.id))
  .filter((pkg) => pkg.publish === null || pkg.publish.length > 0)
  .filter((pkg) => typeof pkg.description !== "string" || pkg.description.trim().length === 0)
  .map((pkg) => `${pkg.name}: missing package.description`)
  .sort();

const bindingsPackage = JSON.parse(fs.readFileSync(path.join(root, "bindings/ts/package.json"), "utf8"));
const bindingsLock = JSON.parse(fs.readFileSync(path.join(root, "bindings/ts/package-lock.json"), "utf8"));
const mismatches = [...mismatchedCrates, ...publishableCratesMissingDescriptions];
if (bindingsPackage.version !== expected) {
  mismatches.push(`bindings/ts/package.json=${bindingsPackage.version}`);
}
if (bindingsLock.version !== expected) {
  mismatches.push(`bindings/ts/package-lock.json=${bindingsLock.version}`);
}
if (bindingsLock.packages?.[""]?.version !== expected) {
  mismatches.push(`bindings/ts/package-lock.json packages[\"\"].version=${bindingsLock.packages?.[""]?.version ?? "missing"}`);
}

if (mismatches.length > 0) {
  console.error(`Release state does not match ${expected}:`);
  for (const mismatch of mismatches) console.error(`- ${mismatch}`);
  process.exit(1);
}

console.log(`Verified ${metadata.workspace_members.length} workspace crates and the TypeScript package at ${expected}.`);
