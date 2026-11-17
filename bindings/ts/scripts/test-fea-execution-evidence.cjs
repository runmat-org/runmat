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
const estimated = {...candidate,estimatedElementCount:124318,estimatedNodeCount:182210,resourceLimit:null};
assert.doesNotThrow(()=>admit(plan(estimated)));
const limit = {code:"RM.FEA.RUN_STUDY.BROWSER_RESOURCE_LIMIT",term:"estimated_memory",requiredBytes:5000000000,maximumBytes:4000000000,nextAction:"Continue in the desktop app."};
assert.doesNotThrow(()=>admit(plan({...estimated,readiness:"desktop_required",durationEvidence:undefined,estimatedMillis:null,resourceLimit:limit})));
assert.throws(()=>admit(plan({...estimated,resourceLimit:limit})),/resource limit/);
assert.throws(()=>admit(plan({...estimated,readiness:"unavailable",resourceLimit:{...limit,requiredBytes:1}})),/resource limit/);
assert.throws(()=>admit(plan({...estimated,estimatedElementCount:-1})),/mesh estimate/);
const operations = {meshing:"0",assembly:"22100000000",factorization:"11400000000",solve:"340000000",other:"1000",total:"33840001000",solveIterations:null};
assert.doesNotThrow(()=>admit(plan({...estimated,estimatedOperations:operations})));
assert.doesNotThrow(()=>admit(plan({...estimated,estimatedOperations:{...operations,solveIterations:500}})));
for (const invalid of [{...operations,total:61e9},{...operations,solve:"-1"},{...operations,assembly:"01"},{...operations,solveIterations:0}]) assert.throws(()=>admit(plan({...estimated,estimatedOperations:invalid})),/operations estimate/);
const generated=fs.readFileSync(path.join(root,"src/generated/fea-analysis-contracts.ts"),"utf8");
assert.match(generated,/durationEvidence\?: FeaExecutionDuration;/);
assert.match(generated,/export interface FeaExecutionDuration \{/);
assert.match(generated,/resourceLimit: FeaResourceLimit \| null;/);
console.log("Execution duration evidence: valid observed range accepted; sparse, non-finite, fractional and inconsistent evidence rejected; unknown remains absent. Mesh estimates and resource limits: a limit is never ready and always exceeds its maximum. Operations estimates are decimal counts.");
