import { spawnSync } from "node:child_process";

export const DOCUMENTATION_EXPORT_ARGUMENTS = Object.freeze([
  "run", "--quiet", "-p", "runmat-builtins", "--bin", "export_builtin_documentation",
  "--", "--transition",
]);

export function runDocumentationExport(spawn = spawnSync) {
  const result = spawn("cargo", DOCUMENTATION_EXPORT_ARGUMENTS, {
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
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
