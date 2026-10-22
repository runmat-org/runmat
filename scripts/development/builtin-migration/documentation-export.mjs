import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export const DOCUMENTATION_EXPORT_ARGUMENTS = Object.freeze([
  "run", "--quiet", "-p", "runmat-builtins", "--bin", "export_builtin_documentation",
  "--", "--transition",
]);

export function runDocumentationExport(spawn = spawnSync, { repository, binary, transition = true } = {}) {
  const argumentsList = binary
    ? (transition ? ["--transition"] : [])
    : (transition ? DOCUMENTATION_EXPORT_ARGUMENTS : DOCUMENTATION_EXPORT_ARGUMENTS.slice(0, -1));
  const result = spawn(binary ? resolve(repository ?? ".", binary) : "cargo", argumentsList, {
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
    ...(repository === undefined ? {} : { cwd: repository }),
  });
  if (result.error) throw new Error(`could not execute the catalog documentation exporter: ${result.error.message}`);
  if (result.status === null) throw new Error(`catalog documentation exporter terminated by signal ${result.signal ?? "unknown"}`);
  return {
    status: result.status,
    signal: result.signal ?? null,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

/** Read the aggregate exporter product; never reconstruct authority from sidecars. */
export function readDocumentationExport({
  repository,
  exportPath = process.env.RUNMAT_BUILTIN_DOCUMENTATION_EXPORT,
  binary = process.env.RUNMAT_BUILTIN_DOCUMENTATION_BINARY,
  mode = process.env.RUNMAT_BUILTIN_DOCUMENTATION_MODE ?? "strict",
}, spawn = spawnSync) {
  if (!["strict", "transition"].includes(mode)) throw new Error("Builtin documentation mode must be strict or transition");
  if (exportPath && binary) throw new Error("Select a builtin documentation export file or binary, not both");
  let encoded;
  if (exportPath) {
    encoded = readFileSync(resolve(repository, exportPath), "utf8");
  } else {
    const result = runDocumentationExport(spawn, { repository, binary, transition: mode === "transition" });
    if (result.status !== 0) throw new Error(`catalog documentation exporter exited ${result.status}: ${result.stderr.trim()}`);
    encoded = result.stdout;
  }
  const payload = JSON.parse(encoded);
  if (![1, 2].includes(payload?.schema_version) || !Array.isArray(payload.builtins)
      || payload.preview !== undefined || !payload.inventory
      || payload.inventory.documents !== payload.builtins.length
      || !Number.isSafeInteger(payload.inventory.catalog_identities) || payload.inventory.catalog_identities < 0
      || !Number.isSafeInteger(payload.inventory.legacy_sidecars) || payload.inventory.legacy_sidecars < 0
      || !Array.isArray(payload.inventory.missing_catalog_documentation)
      || !payload.inventory.missing_catalog_documentation.every(name => typeof name === "string" && name.length > 0)) {
    throw new Error("Unsupported or incomplete aggregate builtin documentation export");
  }
  const identities = new Set();
  for (const document of payload.builtins) {
    if (!document || typeof document !== "object" || Array.isArray(document)
        || typeof document.key !== "string" || !document.key
        || !["catalog", "legacy_sidecar"].includes(document.authority)) {
      throw new Error("Invalid builtin documentation identity or authority");
    }
    const identity = document.key.toLowerCase();
    if (identities.has(identity)) throw new Error(`Duplicate builtin documentation identity: ${document.key}`);
    identities.add(identity);
  }
  if (mode === "strict" && (payload.inventory.legacy_sidecars !== 0
      || payload.inventory.missing_catalog_documentation.length !== 0
      || payload.builtins.some(document => document.authority !== "catalog"))) {
    throw new Error("Strict builtin documentation consumption requires completed canonical documentation; select transition mode explicitly during migration");
  }
  return payload;
}
