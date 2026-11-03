#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const expected = process.argv[2];
if (!expected || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(expected)) {
  console.error("Usage: scripts/verify-release-version.mjs <version>");
  process.exit(1);
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const licensePins = {
  "LICENSE.md": "f923b86930cc30ee506959d2e026ae4cef0b29f74c00947540dd80f28e4eae51",
  "FREE-USE.md": "0ce48a3efdb6408518cafb1749bcf7a04ff8b07439c254ec7f04aa46768012f6",
};
for (const [name, expectedHash] of Object.entries(licensePins)) {
  const filename = path.join(root, name), stat = fs.lstatSync(filename);
  if (!stat.isFile() || stat.isSymbolicLink()
      || createHash("sha256").update(fs.readFileSync(filename)).digest("hex") !== expectedHash) {
    throw new Error(`Release license differs from the reviewed bytes: ${name}`);
  }
}
if (fs.existsSync(path.join(root, "bindings/LICENSE.md"))) throw new Error("Redundant bindings license must be absent");
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
for (const pkg of metadata.packages.filter(pkg => workspaceMembers.has(pkg.id))) {
  if (pkg.name === "runmat-clarabel") {
    if (pkg.license !== "Apache-2.0") mismatches.push("Vendored Clarabel must retain Apache-2.0");
    continue;
  }
  if (pkg.license !== null || !pkg.license_file
      || path.resolve(path.dirname(pkg.manifest_path), pkg.license_file) !== path.join(root, "LICENSE.md")) {
    mismatches.push(`${pkg.name}: first-party license metadata must resolve to root LICENSE.md`);
  }
}
if (bindingsPackage.license !== "SEE LICENSE IN LICENSE.md"
    || bindingsLock.packages?.[""]?.license !== bindingsPackage.license
    || !Object.keys(licensePins).every(name => bindingsPackage.files?.includes(name))) {
  mismatches.push("TypeScript package must declare and include both reviewed runtime license files");
}
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
