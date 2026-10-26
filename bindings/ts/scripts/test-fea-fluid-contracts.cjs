#!/usr/bin/env node
// Public fluid controls must preserve the explicit Rust numerical/geometry authority.
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
assert.equal(generated.trimEnd(), emitted.trimEnd(), "fluid contract generator parity");
const names = ["FeaCanonicalFluidPolicy", "FeaAndersonOnePolicy", "FeaFiniteVolumeProjectionPolicy"];
const interfaces = names.map((name) => {
  const declaration = generated.match(new RegExp(`export interface ${name} \\{[\\s\\S]*?\\n\\}`))?.[0];
  assert.ok(declaration, `missing ${name}`);
  return declaration;
});
const coupling = generated.match(/export type FeaPressureVelocityCoupling = [^;]+;/)?.[0];
assert.ok(coupling, "missing Rust-derived pressure/velocity coupling choices");
assert.match(coupling, /"coupled" \| "segregated"/);
interfaces.unshift(coupling);
const source = [...interfaces,
  'declare const ordinary: Omit<FeaCanonicalFluidPolicy, "fixed_point_acceleration">;',
  'const disabled: FeaCanonicalFluidPolicy = { ...ordinary, fixed_point_acceleration: null };',
  'const enabled: FeaCanonicalFluidPolicy = { ...ordinary, fixed_point_acceleration: { maximum_absolute_coefficient: 32 } };',
  'const segregated: FeaCanonicalFluidPolicy = { ...disabled, pressure_velocity_coupling: "segregated" };',
  '// @ts-expect-error coupling choices must match Rust',
  'const invalidCoupling: FeaCanonicalFluidPolicy = { ...disabled, pressure_velocity_coupling: "automatic" };',
  '// @ts-expect-error acceleration authority must not silently default',
  'const missing: FeaCanonicalFluidPolicy = ordinary;',
  '// @ts-expect-error numerical policy is a typed object, not a switch',
  'const invalid: FeaCanonicalFluidPolicy = { ...ordinary, fixed_point_acceleration: true };',
  '// @ts-expect-error coefficient authority must be present',
  'const incomplete: FeaCanonicalFluidPolicy = { ...ordinary, fixed_point_acceleration: {} };',
].join("\n");
const virtualPath = path.resolve(__dirname, "fluid-contract-check.ts");
const options = { strict: true, noEmit: true, types: [], target: ts.ScriptTarget.ES2020 };
const host = ts.createCompilerHost(options);
const original = host.getSourceFile.bind(host);
host.getSourceFile = (file, language, ...rest) => path.resolve(file) === virtualPath
  ? ts.createSourceFile(file, source, language, true)
  : original(file, language, ...rest);
const diagnostics = ts.getPreEmitDiagnostics(ts.createProgram([virtualPath], options, host));
assert.equal(diagnostics.length, 0, diagnostics.map((item) =>
  ts.flattenDiagnosticMessageText(item.messageText, "\n")).join("\n"));
console.log("FEA fluid contracts: generator parity, explicit nullable acceleration and closed coefficient shape pass.");
