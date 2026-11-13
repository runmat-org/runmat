#!/usr/bin/env node
// Exercise the transport validators against the canonical generated wire union.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const contracts = fs.readFileSync(path.resolve(__dirname, '../src/generated/fea-workspace-contracts.ts'), 'utf8');
const source = fs.readFileSync(path.resolve(__dirname, '../src/fs/workspace-contracts.ts'), 'utf8');
const constants = Object.fromEntries([...contracts.matchAll(/export const (\w+) = (\d+);/g)].map((match) => [match[1], Number(match[2])]));
const emitted = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
const checked = {};
vm.runInNewContext(emitted, { exports: checked, Uint8Array, require: (name) => name === '../generated/fea-workspace-contracts.js' ? constants : require(name) });
const receipt = { id: Array(16).fill(1), revision: Array(32).fill(2) };
const outcomes = [
  ...['published', 'committed_pending_recovery', 'durability_indeterminate'].map((type) => ({ type, receipt })),
  { type: 'conflict', current: receipt.revision }, { type: 'id_reused' },
];
const declared = [...contracts.slice(contracts.indexOf('export type WorkspacePublicationOutcome ='), contracts.indexOf('export interface WorkspacePublicationReceipt')).matchAll(/type: "([^"]+)"/g)].map((match) => match[1]);
assert.deepEqual(outcomes.map((value) => value.type).sort(), declared.sort(), 'Every canonical publication outcome must be covered');
for (const value of outcomes) assert.ok(checked.workspacePublicationOutcomeSchema.safeParse(value).success, value.type);
for (const value of [
  { type: 'published', receipt: { ...receipt, id: [1] } },
  { type: 'durability_indeterminate' }, { type: 'unknown' },
  { type: 'published', receipt, extra: true },
]) assert.equal(checked.workspacePublicationOutcomeSchema.safeParse(value).success, false);
const file = { path: 'model.fea', version: 'original', bytes: new Uint8Array([1, 2]), readonly: false };
const snapshot = { root: '/admitted', excluded_subtrees: [], revision: receipt.revision, files: [file] };
assert.ok(checked.workspaceSnapshotSchema.safeParse(snapshot).success);
assert.equal(checked.workspaceSnapshotSchema.safeParse({ ...snapshot, files: [file, file] }).success, false);
assert.equal(checked.workspaceSnapshotSchema.safeParse({ ...snapshot, files: [{ ...file, bytes: [256] }] }).success, false);
const maximum = constants.MAX_WORKSPACE_PUBLICATION_BYTES;
const half = new Uint8Array(Math.floor(maximum / 2) + 1);
assert.equal(checked.workspaceSnapshotSchema.safeParse({ ...snapshot, files: [{ ...file, bytes: half }, { ...file, path: 'other.fea', bytes: half }] }).success, false);
console.log('FEA workspace contracts: all publication outcomes, strict receipts, unique paths and cumulative byte bounds pass.');
