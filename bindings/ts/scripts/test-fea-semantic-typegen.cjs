#!/usr/bin/env node

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const { parseDeclarations, rustFiles } = require("./fea-semantic-typegen/syntax.cjs");
const { renderReachable } = require("./fea-semantic-typegen/render.cjs");
const { renderGuards } = require("./fea-semantic-typegen/guards.cjs");

const directory = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-fea-typegen-"));
const fixture = path.join(directory, "fixture.rs");
fs.writeFileSync(fixture, `
#[derive(Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Root {
    pub required_value: String,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub canonical_modulus: YoungsModulus,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub canonical_position: runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub omitted_when_none: Option<u32>,
    pub explicit_null: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub omitted_when_empty: Vec<String>,
    #[serde(skip_serializing_if = "custom_predicate")]
    pub conditionally_omitted: Option<bool>,
    pub color: [u8; 3],
    pub ceiling: Ceiling,
    #[serde(with = "serde_bytes")]
    pub binary: Vec<u8>,
    pub state: State,
    pub source: Source,
    pub spectrum: Spectrum,
    pub transparent_series: PhysicalSeries,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "domain", content = "samples")]
pub enum Spectrum {
    FrequencyHertz(#[serde(with = "runmat_types::physical::canonical_storage")] Quantity<Frequency, Vec<f64>>),
    ErrorAndSamples(
        #[serde(with = "runmat_types::physical::canonical_scalar::optional")] Option<YoungsModulus>,
        #[serde(with = "runmat_types::physical::canonical_storage")] Quantity<Frequency, [f64; 2]>,
    ),
}

#[derive(Serialize)]
#[serde(transparent)]
pub struct PhysicalSeries(#[serde(with = "runmat_types::physical::canonical_storage")] pub Quantity<Frequency, Vec<f64>>);

#[derive(Serialize)]
pub struct YoungsModulus { pub pascals: f64 }

#[derive(Serialize, Deserialize)]
pub struct Ceiling {
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_samples: usize,
    #[serde(with = "runmat_types::wire_integer::option")]
    pub maximum_resident_bytes: Option<u64>,
    #[serde(with = "runmat_types::wire_integer::vec")]
    pub extents: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "runmat_types::wire_integer::option")]
    pub face_id: Option<u64>,
    pub offset: Offset,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Offset {
    Signed(#[serde(with = "runmat_types::wire_integer")] i64),
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum State {
    Ready,
    Running { worker_count: u32 },
}

#[derive(Serialize)]
pub struct Reference {
    pub module_path: String,
    pub declaration_id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum Source {
    Declaration(Reference),
}
`);

const registry = parseDeclarations(rustFiles(directory));
const output = renderReachable(registry, ["Root"]);
assert.match(output, /requiredValue: string/);
assert.match(output, /canonicalModulus: number/);
assert.match(output, /canonicalPosition: \[number, number, number\]/);
assert.match(output, /domain: "frequency_hertz"; samples: number\[\]/);
assert.match(output, /domain: "error_and_samples"; samples: \[number \| null, \[number, number\]\]/);
assert.match(output, /export type PhysicalSeries = number\[\]/);
assert.match(output, /color: \[WireU8, WireU8, WireU8\]/);
assert.match(output, /maximum_samples: WireU64;/);
assert.match(output, /maximum_resident_bytes: WireU64 \| null;/);
assert.match(output, /extents: WireU64\[\];/);
assert.match(output, /face_id\?: WireU64;/);
assert.match(output, /\{ signed: WireI64 \}/);
assert.match(output, /binary: Uint8Array \| number\[\]/);
assert.match(output, /omittedWhenNone\?: WireU32/);
assert.match(output, /explicitNull: boolean \| null/);
assert.match(output, /omittedWhenEmpty\?: string\[\]/);
assert.match(output, /conditionallyOmitted\?: boolean \| null/);
assert.match(output, /kind: "running"/);
assert.match(output, /worker_count: WireU32/);
assert.match(output, /Reference & \{ source: "declaration" \}/);
assert.doesNotMatch(output, /source: "declaration"; value:/);
assert.doesNotMatch(output, /:\s*(?:unknown|any)\b|\b(?:unknown|any)\[\]/);

const guardBody = `export interface Evidence { digest: [number, number]; value: number }
export type Report = { kind: "passed"; evidence: Evidence | null } | { kind: "indeterminate"; reason?: string };`;
const compiledGuards = require("typescript").transpileModule(renderGuards(guardBody, ["Report"]), {
  compilerOptions: { module: require("typescript").ModuleKind.CommonJS },
}).outputText;
const exportsObject = {};
require("node:vm").runInNewContext(compiledGuards, { exports: exportsObject });
assert.equal(exportsObject.isReport({ kind: "passed", evidence: { digest: [1, 2], value: 0.001 } }), true);
assert.equal(exportsObject.isReport({ kind: "passed", evidence: null }), true);
assert.equal(exportsObject.isReport({ kind: "indeterminate" }), true);
for (const malformed of [
  { kind: "unknown", evidence: null },
  { kind: "passed", evidence: { digest: [1], value: 0.001 } },
  { kind: "passed", evidence: { digest: [1, 2], value: Infinity } },
  { kind: "indeterminate", reason: 3 },
]) assert.equal(exportsObject.isReport(malformed), false);
assert.throws(() => renderGuards("export type Unsupported = unknown;", ["Unsupported"]), /Unsupported guard type/);
const omissionGuards = require("typescript").transpileModule(renderGuards(output, ["Root"]), {
  compilerOptions: { module: require("typescript").ModuleKind.CommonJS },
}).outputText;
const omissions = {};
require("node:vm").runInNewContext(omissionGuards, { exports: omissions, Uint8Array });
const root = { spectrum: { domain: "frequency_hertz", samples: [1, 2] }, transparentSeries: [1, 2], requiredValue: "checked", canonicalModulus: 210e9, canonicalPosition: [0.0, 0.01, 0.02], explicitNull: null, color: [1, 2, 3], ceiling: { maximum_samples: 200, maximum_resident_bytes: null, extents: [3], offset: { signed: -4 } }, binary: [], state: { kind: "ready" }, source: { source: "declaration", module_path: "project/main.fea", declaration_id: "main" } };
assert.equal(omissions.isRoot(root), true);
assert.equal(omissions.isRoot({ ...root, spectrum: { domain: "error_and_samples", samples: [null, [1, 2]] } }), true);
assert.equal(omissions.isRoot({ ...root, spectrum: { domain: "error_and_samples", samples: [0.1, [1]] } }), false);
assert.equal(omissions.isRoot({ ...root, spectrum: { domain: "frequency_hertz", samples: [Infinity] } }), false);
assert.equal(omissions.isRoot({ ...root, transparentSeries: ["1"] }), false);
assert.equal(omissions.isRoot({ ...root, binary: new Uint8Array([1, 2, 3]) }), true);
assert.equal(omissions.isRoot({ ...root, binary: "bytes" }), false);
assert.equal(omissions.isRoot({ ...root, omittedWhenEmpty: ["body"], conditionallyOmitted: null }), true);
assert.equal(omissions.isRoot({ ...root, omittedWhenEmpty: [3] }), false);
assert.equal(omissions.isRoot({ ...root, omittedWhenEmpty: null }), false);
assert.equal(omissions.isRoot({ ...root, conditionallyOmitted: 3 }), false);
// Wire-integer bound shared with runmat_types::wire_integer (2^53 - 1).
const MAX = Number.MAX_SAFE_INTEGER;
const withCeiling = (patch) => ({ ...root, ceiling: { ...root.ceiling, ...patch } });
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: null })), true, "null means no authored limit");
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: 4294967296 })), true);
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: MAX })), true, "2^53 - 1 accepted");
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: MAX + 1 })), false, "2^53 rejected");
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: 2 ** 64 })), false);
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: -1 })), false, "negative rejected");
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: 1.5 })), false, "fractional rejected");
assert.equal(omissions.isRoot(withCeiling({ maximum_resident_bytes: undefined })), false, "omission rejected");
assert.equal(omissions.isRoot(withCeiling({ maximum_samples: 0 })), true, "0 accepted for a count that allows it");
assert.equal(omissions.isRoot(withCeiling({ maximum_samples: MAX + 1 })), false);
assert.equal(omissions.isRoot(withCeiling({ extents: [3, MAX] })), true);
assert.equal(omissions.isRoot(withCeiling({ extents: [3, MAX + 1] })), false);
assert.equal(omissions.isRoot(withCeiling({ face_id: MAX })), true);
assert.equal(omissions.isRoot(withCeiling({ face_id: -1 })), false);
assert.equal(omissions.isRoot(withCeiling({ offset: { signed: -MAX } })), true);
assert.equal(omissions.isRoot(withCeiling({ offset: { signed: -MAX - 1 } })), false);
assert.equal(omissions.isRoot({ ...root, color: [255, 0, 0] }), true);
assert.equal(omissions.isRoot({ ...root, color: [256, 0, 0] }), false, "narrow integers keep their own range");
// A 64-bit field without the adapter is refused, as is an adapter on the wrong shape.
for (const [field, message, jsonType] of [
  ["pub unbounded: u64,", /must carry #\[serde\(with = "runmat_types::wire_integer/, "number"],
  ["pub unbounded: Option<usize>,", /must carry/, "number | null"],
  ["pub unbounded: Vec<i64>,", /must carry/, "number[]"],
  ["#[serde(with = \"runmat_types::wire_integer\")]\n    pub unbounded: Option<u64>,", /does not match field type/],
]) {
  const wide = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-fea-typegen-wide-"));
  fs.writeFileSync(path.join(wide, "wide.rs"), `#[derive(Serialize)]\npub struct Wide {\n    ${field}\n}\n`);
  const declarations = parseDeclarations(rustFiles(wide));
  assert.throws(() => renderReachable(declarations, ["Wide"]), message, field);
  if (jsonType) {
    const scalarTypes = new Map(["u64", "usize", "i64"].map((name) => [name, "number"]));
    assert.ok(renderReachable(declarations, ["Wide"], { scalarTypes }).includes(`unbounded: ${jsonType};`));
    assert.throws(() => renderReachable(declarations, ["Wide"]), message, "consumer policy must not leak");
  }
  fs.rmSync(wide, { recursive: true, force: true });
}
const jsonScalars = renderReachable(registry, ["Root"], {
  scalarTypes: new Map([["u8", "number"], ["u32", "number"]]),
});
assert.match(jsonScalars, /color: \[number, number, number\]/);
assert.match(jsonScalars, /maximum_samples: WireU64;/, "explicit Serde adapters retain their wire representation");
const composedGuards = require("typescript").transpileModule(renderGuards(`
export type Value<T> = { constant: T } | { values: T[] };
export interface Point { x: number }
export type Composed = { entries: Record<string, Value<Point>> } & { source: "checked" };
`, ["Composed"]), { compilerOptions: { module: require("typescript").ModuleKind.CommonJS } }).outputText;
const composed = {};
require("node:vm").runInNewContext(composedGuards, { exports: composed });
assert.equal(composed.isComposed({ entries: { a: { constant: { x: 1 } }, b: { values: [{ x: 2 }] } }, source: "checked" }), true);
assert.equal(composed.isComposed({ entries: { a: { constant: { x: "1" } } }, source: "checked" }), false);
assert.equal(composed.isComposed({ entries: { a: { values: [{ x: Infinity }] } }, source: "checked" }), false);
assert.equal(composed.isComposed({ entries: {}, source: "draft" }), false);

fs.rmSync(directory, { recursive: true, force: true });
console.log("FEA semantic type generator fixtures passed.");
