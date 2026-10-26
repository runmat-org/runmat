#!/usr/bin/env node
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");
const root = path.resolve(__dirname,"..");
const source = fs.readFileSync(path.join(root,"src/index.ts"),"utf8");
const start = source.indexOf("function admitFeaExecutionChoices(");
assert.ok(start >= 0);
const context = {};
vm.runInNewContext(ts.transpileModule(source.slice(start), {compilerOptions:{target:ts.ScriptTarget.ES2020}}).outputText,context);
const admit = context.admitFeaExecutionChoices;
const candidate = {mode:"single_core",cores:1,memoryBytes:1024,estimatedMillis:1000,durationEvidence:{lowerMillis:800,upperMillis:1200,sampleCount:3},readiness:"ready",reason:null};
const plan = value => ({schemaVersion:1,planId:"checked",revision:Array(32).fill(1),candidates:[value]});
assert.equal(admit(plan(candidate)).candidates[0],candidate);
for (const evidence of [
 {lowerMillis:1100,upperMillis:1200,sampleCount:3},
 {lowerMillis:800,upperMillis:900,sampleCount:3},
 {lowerMillis:800,upperMillis:1200,sampleCount:2},
 {lowerMillis:NaN,upperMillis:1200,sampleCount:3},
 {lowerMillis:800,upperMillis:Infinity,sampleCount:3},
 {lowerMillis:800,upperMillis:1200,sampleCount:3.5},
]) assert.throws(()=>admit(plan({...candidate,durationEvidence:evidence})),/duration evidence/);
assert.throws(()=>admit(plan({...candidate,estimatedMillis:null})),/duration evidence/);
assert.doesNotThrow(()=>admit(plan({...candidate,estimatedMillis:null,durationEvidence:undefined})));
const generated=fs.readFileSync(path.join(root,"src/generated/fea-analysis-contracts.ts"),"utf8");
assert.match(generated,/durationEvidence\?: FeaExecutionDuration;/);
assert.match(generated,/export interface FeaExecutionDuration \{/);
console.log("Execution duration evidence: valid observed range accepted; sparse, non-finite, fractional and inconsistent evidence rejected; unknown remains absent.");
