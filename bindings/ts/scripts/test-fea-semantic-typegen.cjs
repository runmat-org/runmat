#!/usr/bin/env node

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const { parseDeclarations, rustFiles } = require("./fea-semantic-typegen/syntax.cjs");
const { renderReachable } = require("./fea-semantic-typegen/render.cjs");

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
    pub state: State,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum State {
    Ready,
    Running { worker_count: u32 },
}
`);

const registry = parseDeclarations(rustFiles(directory));
const output = renderReachable(registry, ["Root"]);
assert.match(output, /requiredValue: string/);
assert.match(output, /omittedWhenNone\?: number/);
assert.match(output, /explicitNull: boolean \| null/);
assert.match(output, /kind: "running"/);
assert.match(output, /worker_count: number/);
assert.doesNotMatch(output, /:\s*(?:unknown|any)\b|\b(?:unknown|any)\[\]/);

fs.rmSync(directory, { recursive: true, force: true });
console.log("FEA semantic type generator fixtures passed.");
