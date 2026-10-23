import { readDocumentationExport } from "./builtin-migration/documentation-export.mjs";

// Reference remains string|null, but denotes the actual exported implementation
// source URL, not a guessed physical documentation sidecar.
try {
  if (process.argv.length !== 3) throw new Error("expected repository path");
  const payload = readDocumentationExport({ repository: process.argv[2] });
  const references = Object.fromEntries(payload.builtins.map(document => [
    document.key.toLowerCase(),
    typeof document.source?.url === "string" && document.source.url.length > 0
      ? document.source.url : null,
  ]));
  process.stdout.write(`${JSON.stringify(references)}\n`);
} catch (error) {
  process.stderr.write(`integer-capability-audit: ${error.message}\n`);
  process.exitCode = 1;
}
