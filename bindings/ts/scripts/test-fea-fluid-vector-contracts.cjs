#!/usr/bin/env node
// Non-mutating parity and compilation checks for shared fluid vector boundaries.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

const generatedPath = path.resolve(__dirname, "../src/generated/fea-analysis-contracts.ts");
let emitted;
const generator = fs.readFileSync(path.join(__dirname, "generate-fea-contracts.cjs"), "utf8");
vm.runInNewContext(generator, {
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
assert.equal(generated, emitted, "generated FEA contracts differ from their generator");
assert.match(generated, /"volume_flow_rate"/, "native oriented-face volume flux quantity is absent");
const semanticPath = path.resolve(__dirname, "../src/generated/fea-semantic-contracts.ts");
const assessmentGuardsPath = path.resolve(__dirname, "../src/generated/fea-assessment-guards.ts");
const physicalUnitsPath = path.resolve(__dirname, "../src/generated/physical-unit-catalog.ts");
let semanticEmitted;
let assessmentGuardsEmitted;
let physicalUnitsEmitted;
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "generate-fea-semantic-contracts.cjs"), "utf8"), {
  __dirname,
  require: (name) => name === "fs" ? {
    ...fs,
    mkdirSync: () => {},
    writeFileSync: (file, contents) => {
      const output = path.resolve(file);
      if (output === semanticPath) semanticEmitted = contents;
      else if (output === physicalUnitsPath) physicalUnitsEmitted = contents;
      else {
        assert.equal(output, assessmentGuardsPath, "unexpected semantic-generator output");
        assessmentGuardsEmitted = contents;
      }
    },
  } : require(name),
  console: { log: () => {} },
});
const semantic = fs.readFileSync(semanticPath, "utf8");
assert.equal(semantic, semanticEmitted, "authored semantic contracts differ from Rust-derived generator");
assert.equal(fs.readFileSync(physicalUnitsPath, "utf8"), physicalUnitsEmitted,
  "physical unit catalog differs from Rust-derived generator");
assert.match(semantic, /"volume_flow_rate"/, "authored volume flux quantity is absent");
if (assessmentGuardsEmitted !== undefined) {
  assert.equal(fs.readFileSync(assessmentGuardsPath, "utf8"), assessmentGuardsEmitted,
    "assessment guards differ from Rust-derived generator");
}
assert.doesNotMatch(semantic, /:\s*(?:unknown|any)\b|\b(?:unknown|any)\[\]/);
// Compile the actual exported constraint and its complete dependency declarations;
// this focused test does not require unrelated semantic-program artifacts.
const parsed = ts.createSourceFile(generatedPath, generated, ts.ScriptTarget.ES2020, true);
const declarations = ["FeaPosition", "FeaCoordinateFrame", "FeaPrescribedValue",
  "FeaRegionTarget", "FeaPairedConstraint", "FeaStructuralConstraintKind", "FeaConstraintInput"]
  .map((name) => {
    const declaration = parsed.statements.find((statement) => statement.name?.text === name)?.getText(parsed);
    assert.ok(declaration, `missing generated dependency: ${name}`);
    return declaration;
  });
const source = [
  'import type { AuthoredConstraintKind, MaterialModel, MaterialFluidModel } from "../src/generated/fea-semantic-contracts.js";',
  ...declarations,
  'type Fluid = Extract<FeaConstraintInput, { fluid: unknown }>["fluid"];',
  'type Inlet = Extract<Fluid["kind"], { prescribed_velocity: unknown }>["prescribed_velocity"];',
  'type Wall = Extract<Fluid["kind"], { moving_wall: unknown }>["moving_wall"];',
  'type Gauge = Extract<Fluid["kind"], { pressure_reference: unknown }>["pressure_reference"];',
  'declare const target: FeaRegionTarget;',
  'const inlet: Inlet = { target, value: { constant: [{ meters_per_second: -2 }, { meters_per_second: 3 }, { meters_per_second: -4 }] } };',
  'const wall: Wall = { target, velocity: { constant: [{ meters_per_second: 0 }, { meters_per_second: -2 }, { meters_per_second: 0 }] } };',
  'const inletSignal: Inlet = { target, value: { signal: "original-signed-vector" } };',
  'const wallSignal: Wall = { target, velocity: { signal: "original-rotated-table" } };',
  'const gauge: Gauge = { target, value: { constant: { pascals: 0 } } };',
  '// @ts-expect-error a scalar speed cannot supply inlet direction',
  'const scalarInlet: Inlet = { target, value: { constant: { meters_per_second: 2 } } };',
  '// @ts-expect-error a scalar speed cannot supply moving-wall direction',
  'const scalarWall: Wall = { target, velocity: { constant: { meters_per_second: 2 } } };',
  '// @ts-expect-error exact three-component velocity is required',
  'const short: Inlet = { target, value: { constant: [{ meters_per_second: 1 }, { meters_per_second: 2 }] } };',
  '// @ts-expect-error extra components are not silently discarded',
  'const long: Wall = { target, velocity: { constant: [{ meters_per_second: 1 }, { meters_per_second: 2 }, { meters_per_second: 3 }, { meters_per_second: 4 }] } };',
  '// @ts-expect-error pressure cannot masquerade as a velocity component',
  'const wrongUnit: Inlet = { target, value: { constant: [{ pascals: 1 }, { meters_per_second: 2 }, { meters_per_second: 3 }] } };',
  '// @ts-expect-error numeric vectors must retain typed component quantities',
  'const untyped: Wall = { target, velocity: { constant: [1, 2, 3] } };',
  'type AuthoredInlet = Extract<AuthoredConstraintKind, { kind: "fluid_velocity" }>["value"];',
  'type AuthoredWall = Extract<AuthoredConstraintKind, { kind: "fluid_moving_wall" }>["velocity"];',
  'const authoredInlet: AuthoredInlet = { constant: [{ meters_per_second: -2 }, { meters_per_second: 3 }, { meters_per_second: -4 }] };',
  'const authoredWall: AuthoredWall = { constant: [{ meters_per_second: 0 }, { meters_per_second: -2 }, { meters_per_second: 0 }] };',
  'const authoredSignal: AuthoredInlet = { signal: "original-frame-and-table" };',
  'const authoredWallSignal: AuthoredWall = { signal: "original-wall-frame-and-table" };',
  '// @ts-expect-error authored inlet must not retain the old scalar shape',
  'const authoredScalar: AuthoredInlet = { constant: { meters_per_second: 2 } };',
  '// @ts-expect-error authored moving wall must not retain the old scalar shape',
  'const authoredScalarWall: AuthoredWall = { constant: { meters_per_second: 2 } };',
  '// @ts-expect-error authored inlet requires exactly three typed components',
  'const authoredShort: AuthoredInlet = { constant: [{ meters_per_second: 1 }, { meters_per_second: 2 }] };',
  'const newtonian: MaterialFluidModel = { kind: "incompressible_newtonian", dynamic_viscosity: { pascal_seconds: 0.001 } };',
  'const fluidMaterial: MaterialModel = { material_id: "water", name: "Water", provenance: { kind: "authored" }, mass_density_kg_per_m3: 1000, fluid: newtonian, elasticity: null, damping: null, thermal: null, acoustic: null, electrical: null, plastic: null };',
  '// @ts-expect-error viscosity is a typed physical quantity, not a raw speed',
  'const untypedViscosity: MaterialFluidModel = { kind: "incompressible_newtonian", dynamic_viscosity: 1 };',
  '// @ts-expect-error no undocumented turbulence law is admitted',
  'const unknownLaw: MaterialFluidModel = { kind: "turbulent", dynamic_viscosity: { pascal_seconds: 1 } };',
  '// @ts-expect-error a duplicate fluid-specific density is not an authority',
  'const duplicateDensity: MaterialFluidModel = { ...newtonian, density: { kilograms_per_cubic_meter: 1000 } };',
].join("\n");
const virtualPath = path.resolve(__dirname, "fluid-vector-contract-check.ts");
const options = { strict: true, noEmit: true, types: [], target: ts.ScriptTarget.ES2020 };
const host = ts.createCompilerHost(options);
const readSource = host.getSourceFile.bind(host);
host.getSourceFile = (file, language, ...rest) => path.resolve(file) === virtualPath
  ? ts.createSourceFile(file, source, language, true)
  : readSource(file, language, ...rest);
const diagnostics = ts.getPreEmitDiagnostics(ts.createProgram([virtualPath], options, host));
assert.equal(diagnostics.length, 0, diagnostics.map((diagnostic) =>
  ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")).join("\n"));
console.log("Fluid TS: public and Rust-derived authored generator parity, signed three-vector constants and original signal references pass; scalar, length and unit mismatches reject; no unknown/any semantic fields.");
