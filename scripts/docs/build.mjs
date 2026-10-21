/** Produce one complete source-owned docs artifact. The website consumes this
 * artifact through prepare-docs.mjs; it never compiles or runs FEA itself. */
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildExampleSnapshots } from "./build-example-snapshots.mjs";
import { buildSourceDocs } from "./export.mjs";
const args = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  if (!process.argv[index + 1] || args.has(process.argv[index])) throw new Error("Invalid docs build arguments");
  args.set(process.argv[index], resolve(process.argv[index + 1]));
}
for (const key of ["--native-executable", "--graphics-runtime", "--requirements", "--builtin-descriptors", "--builtin-documentation", "--out"]) {
  if (!args.has(key)) throw new Error(`Missing ${key}`);
}
const source = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const snapshotsRoot = buildExampleSnapshots(source, args.get("--native-executable"));
const manifest = buildSourceDocs({ source, snapshotsRoot, output: args.get("--out"), graphicsRoot: args.get("--graphics-runtime"),
  requirementsPath: args.get("--requirements"), builtinDescriptorsPath: args.get("--builtin-descriptors"), builtinDocumentationPath: args.get("--builtin-documentation"), evidencePath: args.get("--evidence-manifest") });
console.log(`Source docs: ${manifest.content_hash} (${manifest.files.length} files)`);
