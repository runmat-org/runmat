import { spawnSync } from "node:child_process";
import { isDeepStrictEqual } from "node:util";
import { readDocumentationExport } from "../development/builtin-migration/documentation-export.mjs";

/** Bind the public docs product to this checkout's complete compiled catalog. */
export function productionDocumentation(repository, {
  exportPath = process.env.RUNMAT_BUILTIN_DOCUMENTATION_EXPORT,
  binary = process.env.RUNMAT_BUILTIN_DOCUMENTATION_BINARY,
  mode = process.env.RUNMAT_BUILTIN_DOCUMENTATION_MODE ?? "strict",
} = {}, spawn = spawnSync) {
  if (mode !== "strict") {
    throw new Error("Public docs production requires strict canonical builtin documentation");
  }
  // Null, rather than undefined, disables environment defaults in the shared
  // adapter. No --identity selector is passed to this current-source exporter.
  const current = readDocumentationExport({ repository, exportPath: null, binary: null, mode }, spawn);
  if (exportPath || binary) {
    const supplied = readDocumentationExport({ repository, exportPath, binary, mode }, spawn);
    if (!isDeepStrictEqual(supplied, current)) {
      throw new Error("Supplied builtin documentation differs from the current full catalog export");
    }
  }
  return current;
}
