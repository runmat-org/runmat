#!/usr/bin/env node
// Check the public field association against Rust, without rewriting generated files.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

const scriptRoot = __dirname;
const generatedPath = path.resolve(scriptRoot, "../src/generated/fea-analysis-contracts.ts");
let emitted;
const generator = fs.readFileSync(path.join(scriptRoot, "generate-fea-contracts.cjs"), "utf8");
vm.runInNewContext(generator, {
  __dirname: scriptRoot,
  require: (name) => name === "fs" ? {
    ...fs,
    mkdirSync: () => {},
    writeFileSync: (file, contents) => {
      assert.equal(path.resolve(file), generatedPath);
      emitted = contents;
    },
  } : require(name),
  console: { log: () => {} },
});
const generated = fs.readFileSync(generatedPath, "utf8");
assert.ok(generated.trimEnd() === emitted.trimEnd(), "generated FEA contracts differ from their generator");
const location = generated.match(/^export type FeaFieldLocationContract = .+;$/m)?.[0];
const shape = generated.match(/^export type FeaSignalShape = .+;$/m)?.[0];
assert.ok(location && shape, "field and signal shape contracts must exist");
const validLocations = [...location.matchAll(/"([a-z_]+)"/g)].map((match) => match[1]);
const source = [location, shape,
  'const scalar: FeaSignalShape = "scalar";',
  'const vector: FeaSignalShape = "vector3";',
  'const tensor: FeaSignalShape = { tensor: [2, 3] };',
  ...validLocations.map((value, index) =>
    `const field${index}: FeaSignalShape = { topology_field: { topology_id: "mesh", location: "${value}", component_count: 3 } };`),
  '// @ts-expect-error association is required, not silently defaulted',
  'const missing: FeaSignalShape = { topology_field: { topology_id: "mesh", component_count: 3 } };',
  '// @ts-expect-error arbitrary strings do not name field locations',
  'const invalid: FeaSignalShape = { topology_field: { topology_id: "mesh", location: "unknown", component_count: 3 } };',
].join("\n");
const virtualPath = path.resolve(scriptRoot, "field-shape-check.ts");
const options = { strict: true, noEmit: true, types: [], target: ts.ScriptTarget.ES2020 };
const host = ts.createCompilerHost(options);
const readSource = host.getSourceFile.bind(host);
host.getSourceFile = (file, language, ...rest) => path.resolve(file) === virtualPath
  ? ts.createSourceFile(file, source, language, true)
  : readSource(file, language, ...rest);
const program = ts.createProgram([virtualPath], options, host);
const diagnostics = ts.getPreEmitDiagnostics(program);
assert.equal(diagnostics.length, 0, diagnostics.map((diagnostic) =>
  ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")).join("\n"));
console.log(`FEA signal shapes: generator parity and ${validLocations.length} typed locations pass; missing/invalid association rejects.`);
