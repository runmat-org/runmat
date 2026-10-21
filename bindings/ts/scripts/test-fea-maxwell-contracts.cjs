#!/usr/bin/env node
// Non-mutating check of genuine Maxwell options and public execution contracts.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");
const generatedPath = path.resolve(__dirname, "../src/generated/fea-analysis-contracts.ts");
let emitted;
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "generate-fea-contracts.cjs"), "utf8"), {
  __dirname,
  require: (name) => name === "fs" ? {
    ...fs, mkdirSync: () => {},
    writeFileSync: (file, contents) => {
      assert.equal(path.resolve(file), generatedPath);
      emitted = contents;
    },
  } : require(name),
  console: { log: () => {} },
});
const generated = fs.readFileSync(generatedPath, "utf8");
const optionsPattern = /export interface FeaMaxwellHarmonicOptions \{[\s\S]*?\n\}/;
const executionPattern = /^export type FeaExecutionInput = .+;$/m;
const options = generated.match(optionsPattern)?.[0];
const execution = generated.match(executionPattern)?.[0];
assert.ok(options && execution);
assert.equal(options, emitted.match(optionsPattern)?.[0]);
assert.equal(execution, emitted.match(executionPattern)?.[0]);
assert.deepEqual([...options.matchAll(/^  ([a-z_]+):/gm)].map((m) => m[1]),
  ["precision_mode", "residual_target"]);
for (const retired of ["FeaElectromagneticPotentialOptions", "electromagnetic_potential",
  "reduced_edge_graph_curl_curl", "reduced_electromagnetic_potential",
  "electromagnetic_reduced_edge_graph_harmonic_solve", "reduced_frequency_material_response",
  "electromagnetic_potential_profile", "reduced_edge_graph_flux", "canonical_edge_projection",
  "canonical_ground_with_ordinal_insulation_profile"]) {
  assert.ok(!generated.includes(retired), `retired proxy contract survives: ${retired}`);
}
assert.match(fs.readFileSync(path.resolve(__dirname, "../src/fea-contracts.ts"), "utf8"),
  /export \* from "\.\/generated\/fea-analysis-contracts\.js"/);
assert.match(fs.readFileSync(path.resolve(__dirname, "../src/index.ts"), "utf8"), /\bFeaMaxwellHarmonicOptions,/);
const aliases = ["FeaPrecisionMode", "FeaQualityPolicy", "FeaComputeBackend"]
  .map((name) => generated.match(new RegExp(`^export type ${name} = .+;$`, "m"))?.[0]);
assert.ok(aliases.every(Boolean));
const otherOptions = [...execution.matchAll(/options: (Fea\w+Options)/g)]
  .map((m) => m[1]).filter((name) => name !== "FeaMaxwellHarmonicOptions");
const source = [...aliases, ...otherOptions.map((name) => `type ${name} = {};`), options, execution,
  'const options: FeaMaxwellHarmonicOptions = { precision_mode: "fp64", residual_target: 1e-8 };',
  'const valid: FeaExecutionInput = { step: "maxwell", solver: "maxwell_harmonic", options };',
  '// @ts-expect-error retired proxy is not execution authority',
  'const proxy: FeaExecutionInput = { step: "maxwell", solver: "electromagnetic_potential", options };',
  '// @ts-expect-error sweeps are not Maxwell publication controls',
  'const sweep: FeaMaxwellHarmonicOptions = { ...options, sweep_enabled: true };',
  '// @ts-expect-error inert deterministic controls are not published',
  'const deterministic: FeaMaxwellHarmonicOptions = { ...options, deterministic_mode: true };',
  '// @ts-expect-error inert quality policy is not a Maxwell solve control',
  'const quality: FeaMaxwellHarmonicOptions = { ...options, quality_policy: "balanced" };',
  '// @ts-expect-error exact solver options are required',
  'const missing: FeaExecutionInput = { step: "maxwell", solver: "maxwell_harmonic" };',
].join("\n");
const virtualPath = path.resolve(__dirname, "maxwell-contract-check.ts");
const compilerOptions = { strict: true, noEmit: true, types: [], target: ts.ScriptTarget.ES2020 };
const host = ts.createCompilerHost(compilerOptions);
const getSource = host.getSourceFile.bind(host);
host.getSourceFile = (file, language, ...rest) => path.resolve(file) === virtualPath
  ? ts.createSourceFile(file, source, language, true) : getSource(file, language, ...rest);
const diagnostics = ts.getPreEmitDiagnostics(ts.createProgram([virtualPath], compilerOptions, host));
assert.equal(diagnostics.length, 0, diagnostics.map((d) =>
  ts.flattenDiagnosticMessageText(d.messageText, "\n")).join("\n"));
console.log("Maxwell TS: Rust-derived options, typed execution, public exports and proxy/sweep/missing-options rejection pass.");
