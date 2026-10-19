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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub omitted_when_none: Option<u32>,
    pub explicit_null: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub omitted_when_empty: Vec<String>,
    #[serde(skip_serializing_if = "custom_predicate")]
    pub conditionally_omitted: Option<bool>,
    pub color: [u8; 3],
    #[serde(with = "serde_bytes")]
    pub binary: Vec<u8>,
    pub state: State,
    pub source: Source,
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
assert.match(output, /color: \[number, number, number\]/);
assert.match(output, /binary: Uint8Array \| number\[\]/);
assert.match(output, /omittedWhenNone\?: number/);
assert.match(output, /explicitNull: boolean \| null/);
assert.match(output, /omittedWhenEmpty\?: string\[\]/);
assert.match(output, /conditionallyOmitted\?: boolean \| null/);
assert.match(output, /kind: "running"/);
assert.match(output, /worker_count: number/);
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
const root = { requiredValue: "checked", explicitNull: null, color: [1, 2, 3], binary: [], state: { kind: "ready" }, source: { source: "declaration", module_path: "project/main.fea", declaration_id: "main" } };
assert.equal(omissions.isRoot(root), true);
assert.equal(omissions.isRoot({ ...root, binary: new Uint8Array([1, 2, 3]) }), true);
assert.equal(omissions.isRoot({ ...root, binary: "bytes" }), false);
assert.equal(omissions.isRoot({ ...root, omittedWhenEmpty: ["body"], conditionallyOmitted: null }), true);
assert.equal(omissions.isRoot({ ...root, omittedWhenEmpty: [3] }), false);
assert.equal(omissions.isRoot({ ...root, omittedWhenEmpty: null }), false);
assert.equal(omissions.isRoot({ ...root, conditionallyOmitted: 3 }), false);
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
