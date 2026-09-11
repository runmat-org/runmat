// @ts-check

import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";
import { buildArtifactManifest, buildArtifactManifestFromFiles, buildArtifactManifestFromTree, validateArtifactManifest } from "./artifacts.mjs";
import { digestObject, partitionBucket, sha256 } from "./identity.mjs";
import { buildInventory, validateInventory } from "./inventory.mjs";
import { buildExecutionMatrix, validateExecutionMatrix } from "./execution-matrix.mjs";
import { buildPlan, PLAN_SCHEMA, TOPOLOGY_SCHEMA, validatePlan } from "./plan.mjs";
import { buildProductProbe, DESKTOP_HOST_PROTOCOL_PROBE, NATIVE_EMBEDDED_AOT_PROBE, runDesktopHostProtocolProbe, runNativeEmbeddedAotProbe, validateProductProbe } from "./product-probe.mjs";
import { stageNativeProduct } from "./product-staging.mjs";
import { reconcileShardResults, validateReconciliation } from "./reconcile.mjs";
import { buildShardResult, validateShardResult } from "./result-manifest.mjs";
import { executeShard, normalizeRunnerRecords } from "./runner-process.mjs";

const SOURCE = "0123456789abcdef0123456789abcdef01234567";
const RUNNER = sha256("runner");

test("inventory identities are order-independent and retain documentation-only examples", () => {
    const first = buildInventory(exportFixture(), inventoryOptions());
    const reversed = buildInventory({ schema_version: 1, builtins: [...exportFixture().builtins].reverse() }, inventoryOptions());
    assert.deepEqual(first.examples, reversed.examples);
    assert.deepEqual(first.executionUnits, reversed.executionUnits);
    assert.equal(first.inventoryDigest, reversed.inventoryDigest);
    assert.equal(first.counts.documentationOnly, 1);
    assert.equal(first.examples.find((entry) => entry.exampleId === "notes")?.admission.kind, "documentation-only");
    assert.deepEqual(first.examples.find((entry) => entry.exampleId === "portable")?.requiredLanes, ["native-host", "browser-host"]);
});

test("catalog examples cannot silently become documentation-only", () => {
    const exported = { schema_version: 1, builtins: [{ key: "bad", authority: "catalog", category: "math", examples: [{ id: "notes", input: "", output: "% note", verification: "Succeeds" }] }] };
    assert.throws(() => buildInventory(exported, inventoryOptions()), /catalog example.*not executable/i);
});

test("catalog examples require an explicit verification policy", () => {
    const exported = { schema_version: 1, builtins: [{ key: "bad", authority: "catalog", category: "math", examples: [{ id: "missing", input: "disp(1)", output: "1", harness: "Portable" }] }] };
    assert.throws(() => buildInventory(exported, inventoryOptions()), /no verification policy/);
});

test("fixture and requirement contracts participate in example definition digests", () => {
    const exported = structuredClone(exportFixture());
    exported.schema_version = 2;
    for (const document of exported.builtins) {
        for (const example of document.examples) {
            example.fixture = "None";
            example.requirements = { host: "Any", engine: "Default", compiler: [], runtime: [], toolchain: [] };
        }
    }
    const first = buildInventory(exported, inventoryOptions());
    const changed = structuredClone(exported);
    changed.builtins[2].examples[0].requirements.engine = "Interpreter";
    changed.builtins[2].examples[0].requirements.host = "NativeOnly";
    const second = buildInventory(changed, inventoryOptions());
    const firstForeign = first.examples.find((example) => example.builtinKey === "foreign");
    const secondForeign = second.examples.find((example) => example.builtinKey === "foreign");
    assert.notEqual(firstForeign.definitionDigest, secondForeign.definitionDigest);
    assert.equal(validateInventory(first), first);
});

test("documentation schema v2 requires explicit fixture declarations", () => {
    const exported = structuredClone(exportFixture());
    exported.schema_version = 2;
    assert.throws(() => buildInventory(exported, inventoryOptions()), /lacks fixture requirements/);
});

test("development limits recompute counts and cannot masquerade as closure inventory", () => {
    const inventory = buildInventory(exportFixture(), { ...inventoryOptions(), scope: { limit: 1 } });
    assert.equal(inventory.counts.selectedExamples, 1);
    assert.equal(inventory.counts.executableExamples + inventory.counts.documentationOnly, 1);
    const plan = buildPlan(inventory);
    assert.throws(() => reconcileShardResults(inventory, plan, completeResults(inventory, plan), { closure: true }), /complete inventory/);
});

test("plans use stable hash buckets and cover every execution identity exactly once", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    const topology = {
        schema: TOPOLOGY_SCHEMA,
        products: {},
        lanes: Object.fromEntries([...new Set(inventory.executionUnits.map((unit) => unit.lane))].map((lane) => [lane, { shards: 3 }]))
    };
    const plan = buildPlan(inventory, topology);
    validatePlan(plan, inventory);
    assert.equal(plan.schema, PLAN_SCHEMA);
    assert.deepEqual(plan.products.find((product) => product.kind === "native-cli")?.requiredProbes, [NATIVE_EMBEDDED_AOT_PROBE]);
    assert.deepEqual(plan.products.find((product) => product.kind === "browser-wasm")?.requiredProbes, []);
    for (const lane of plan.lanes) {
        for (const shard of lane.shards) {
            for (const id of shard.executionIdentities) assert.equal(partitionBucket(id, lane.shardCount), shard.index);
        }
    }
    const reordered = buildPlan(buildInventory({ schema_version: 1, builtins: [...exportFixture().builtins].reverse() }, inventoryOptions()), topology);
    assert.equal(plan.planDigest, reordered.planDigest);
});

test("product-scoped plans contain only that product and cannot claim full closure", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    const topology = {
        schema: TOPOLOGY_SCHEMA,
        products: { "browser-wasm": { artifactProfile: "web" }, "native-cli": { artifactProfile: "embedded-aot" } },
        lanes: Object.fromEntries([...new Set(inventory.executionUnits.map((unit) => unit.lane))].map((lane) => [lane, { shards: 2 }]))
    };
    const native = buildPlan(inventory, topology, { product: "native-cli" });
    validatePlan(native, inventory);
    assert.equal(native.productScope, "native-cli");
    assert.deepEqual(native.products.map((product) => product.kind), ["native-cli"]);
    assert.ok(native.lanes.every((lane) => lane.product === "native-cli"));
    assert.ok(native.lanes.every((lane) => lane.shardCount === 2));
    const report = reconcileShardResults(inventory, native, completeResults(inventory, native));
    assert.equal(report.productScope, "native-cli");
    validateReconciliation(report);
    assert.throws(() => reconcileShardResults(inventory, native, completeResults(inventory, native), { closure: true }), /all-product execution plan/);
});

test("public-product plans exclude private Desktop evidence without claiming full closure", () => {
    const exported = exportFixture();
    exported.builtins.push({
        key: "desktop-dialog",
        authority: "catalog",
        category: "io/dialog",
        examples: [{
            id: "open",
            input: "input('Name: ', 's');",
            harness: "InteractiveHost",
            compatibility: "RunMat",
            verification: "Succeeds",
            requirements: { host: "DesktopHostOnly", engine: "Default", compiler: [], runtime: [], toolchain: [] },
            fixture: {
                DesktopHostOnly: {
                    id: { local_name: "desktop-dialog" },
                    entries: [],
                    interactions: [{ LineInput: { prompt: "Name: ", echo: true, outcome: { Line: "Ada" } } }]
                }
            }
        }]
    });
    const inventory = buildInventory(exported, inventoryOptions());
    const complete = buildPlan(inventory);
    assert.ok(complete.products.some((product) => product.kind === "desktop-native"));
    const publicProducts = buildPlan(inventory, null, { product: "public-products" });
    assert.deepEqual(publicProducts.products.map((product) => product.kind), ["browser-wasm", "native-cli"]);
    assert.ok(publicProducts.lanes.every((lane) => lane.product !== "desktop-native"));
    assert.throws(
        () => reconcileShardResults(inventory, publicProducts, completeResults(inventory, publicProducts), { closure: true }),
        /all-product execution plan/u
    );
});

test("execution matrices exactly bind every frozen lane and shard assignment", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    const plan = buildPlan(inventory);
    const matrix = buildExecutionMatrix(plan);
    validateExecutionMatrix(matrix, plan);
    assert.equal(matrix.include.length, plan.lanes.reduce((count, lane) => count + lane.shardCount, 0));

    const omitted = structuredClone(matrix);
    omitted.include.pop();
    omitted.executionMatrixDigest = digestObject(omitted, ["executionMatrixDigest"]);
    assert.throws(() => validateExecutionMatrix(omitted, plan), /does not exactly match/);

    const duplicate = structuredClone(matrix);
    duplicate.include.push(structuredClone(duplicate.include[0]));
    duplicate.executionMatrixDigest = digestObject(duplicate, ["executionMatrixDigest"]);
    assert.throws(() => validateExecutionMatrix(duplicate, plan), /does not exactly match/);

    const wrongAssignment = structuredClone(matrix);
    wrongAssignment.include[0].assignmentDigest = sha256("wrong assignment");
    wrongAssignment.executionMatrixDigest = digestObject(wrongAssignment, ["executionMatrixDigest"]);
    assert.throws(() => validateExecutionMatrix(wrongAssignment, plan), /does not exactly match/);
});

test("plan validation rejects omission, duplication, wrong buckets, and stale digests", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    assert.throws(() => buildPlan(inventory, { schema: TOPOLOGY_SCHEMA, lanes: {}, products: {} }), /omits required lane/);
    const plan = buildPlan(inventory);
    const omitted = structuredClone(plan);
    omitted.lanes[0].shards[0].executionIdentities.pop();
    refreshPlan(omitted);
    assert.throws(() => validatePlan(omitted, inventory), /Invalid shard declaration|assignment digest|cover/);
    const duplicate = structuredClone(plan);
    duplicate.lanes[0].shards[0].executionIdentities.push(duplicate.lanes[0].shards[0].executionIdentities[0]);
    duplicate.lanes[0].shards[0].expectedCount += 1;
    refreshShard(duplicate.lanes[0]);
    refreshPlan(duplicate);
    assert.throws(() => validatePlan(duplicate, inventory), /duplicate planned/);
    const stale = structuredClone(plan);
    stale.sourceRevision = "abcdef0123456789abcdef0123456789abcdef01";
    refreshPlan(stale);
    assert.throws(() => validatePlan(stale, inventory), /does not match inventory/);
});

test("native artifact manifests close over the staged tree and reject traversal, adjacent dependency changes, and extras", () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-artifacts-"));
    try {
        const product = join(directory, "product");
        mkdirSync(join(product, "bin"), { recursive: true });
        writeFileSync(join(product, "bin", "runmat.exe"), "binary", "utf8");
        writeFileSync(join(product, "runmat_runtime.dll"), "runtime", "utf8");
        const path = join(directory, "manifest.json");
        const manifest = buildArtifactManifestFromTree({ product: "native-cli", artifactProfile: "embedded-aot", sourceRevision: SOURCE, manifestPath: path, root: product, entrypoints: { "runmat-binary": "bin/runmat.exe" } });
        assert.deepEqual(manifest.files.map((file) => file.path), ["bin/runmat.exe", "runmat_runtime.dll"]);
        validateArtifactManifest(manifest, { sourceRevision: SOURCE, verifyFiles: true, manifestPath: path });
        writeFileSync(join(product, "runmat_runtime.dll"), "changed", "utf8");
        assert.throws(() => validateArtifactManifest(manifest, { verifyFiles: true, manifestPath: path }), /recursive tree/);
        writeFileSync(join(product, "runmat_runtime.dll"), "runtime", "utf8");
        writeFileSync(join(product, "unexpected.dll"), "extra", "utf8");
        assert.throws(() => validateArtifactManifest(manifest, { verifyFiles: true, manifestPath: path }), /recursive tree/);
        const escaping = buildArtifactManifest({
            product: "native-cli",
            artifactProfile: "embedded-aot",
            sourceRevision: SOURCE,
            layout: { kind: "recursive-tree", root: "../elsewhere" },
            entrypoints: [{ role: "runmat-binary", path: "runmat" }],
            files: [{ path: "runmat", size: 1, sha256: sha256("x") }]
        });
        assert.throws(() => validateArtifactManifest(escaping, { verifyFiles: true, manifestPath: path }), /canonical relative path/);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("browser artifact producer preserves its exact-file profile", () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-artifact-producer-"));
    try {
        const javascript = join(directory, "runmat.js");
        const wasm = join(directory, "runmat.wasm");
        const manifestPath = join(directory, "artifacts.json");
        writeFileSync(javascript, "js");
        writeFileSync(wasm, "wasm");
        const manifest = buildArtifactManifestFromFiles({
            product: "browser-wasm",
            artifactProfile: "web",
            sourceRevision: SOURCE,
            manifestPath,
            files: { "wasm-js": javascript, "wasm-binary": wasm }
        });
        assert.deepEqual(manifest.entrypoints.map((entrypoint) => entrypoint.role), ["wasm-binary", "wasm-js"]);
        assert.equal(manifest.files.find((file) => file.path === "runmat.js").sha256, sha256("js"));
        const outside = `${directory}-outside`;
        writeFileSync(outside, "outside");
        assert.throws(() => buildArtifactManifestFromFiles({
            product: "browser-wasm", artifactProfile: "web", sourceRevision: SOURCE,
            manifestPath, files: { "wasm-js": javascript, "wasm-binary": outside }
        }), /canonical relative path|escapes/);
        rmSync(outside, { force: true });
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("Desktop artifact manifests bind Runtime and Desktop revisions", () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-desktop-artifacts-"));
    try {
        const product = join(directory, "desktop-product");
        mkdirSync(product);
        writeFileSync(join(product, "runmat-desktop"), "desktop");
        const manifestPath = join(directory, "artifacts.json");
        const producerRevision = "abcdef0123456789abcdef0123456789abcdef01";
        const manifest = buildArtifactManifestFromTree({
            product: "desktop-native",
            artifactProfile: "desktop-host",
            sourceRevision: SOURCE,
            producerRevision,
            manifestPath,
            root: product,
            entrypoints: { "runmat-desktop-binary": "runmat-desktop" }
        });
        assert.equal(manifest.producerRevision, producerRevision);
        validateArtifactManifest(manifest, {
            sourceRevision: SOURCE,
            producerRevision,
            verifyFiles: true,
            manifestPath
        });
        assert.throws(() => buildArtifactManifestFromTree({
            product: "desktop-native",
            artifactProfile: "desktop-host",
            sourceRevision: SOURCE,
            manifestPath,
            root: product,
            entrypoints: { "runmat-desktop-binary": "runmat-desktop" }
        }), /producer revision/u);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("native product staging preserves the executable and closes Windows dependency collisions", () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-product-stage-"));
    try {
        const binary = join(directory, "runmat.exe");
        const first = join(directory, "first");
        const second = join(directory, "second");
        writeFileSync(binary, "binary");
        mkdirSync(first);
        mkdirSync(second);
        writeFileSync(join(first, "runtime.dll"), "same");
        writeFileSync(join(second, "RUNTIME.DLL"), "different");
        const staged = stageNativeProduct({ binary, destination: join(directory, "product"), dependencyDirectories: [first] });
        assert.equal(staged.fileCount, 2);
        assert.equal(readFileSync(join(staged.destination, "runtime.dll"), "utf8"), "same");
        assert.throws(() => stageNativeProduct({ binary, destination: join(directory, "conflict"), dependencyDirectories: [first, second] }), /Conflicting native dependency bytes/);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("reconciliation rejects missing, duplicate, stale, extra, and inconsistent shards", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    const plan = buildPlan(inventory);
    const manifests = completeResults(inventory, plan);
    const report = reconcileShardResults(inventory, plan, manifests);
    assert.equal(report.status, "failed");
    assert.ok(report.summary.byStatus.unavailable > 0);
    assert.throws(() => reconcileShardResults(inventory, plan, manifests.slice(1)), /Missing shard/);
    assert.throws(() => reconcileShardResults(inventory, plan, [...manifests, manifests[0]]), /Duplicate shard/);
    const stale = structuredClone(manifests);
    stale[0].runnerDigest = sha256("old runner");
    refreshResult(stale[0]);
    assert.throws(() => reconcileShardResults(inventory, plan, stale), /runnerDigest/);
    const wrongDefinition = structuredClone(manifests);
    wrongDefinition.find((manifest) => manifest.results.length).results[0].definitionDigest = sha256("old definition");
    refreshResult(wrongDefinition.find((manifest) => manifest.results.length));
    assert.throws(() => reconcileShardResults(inventory, plan, wrongDefinition), /execution result/);
    const mixedArtifacts = structuredClone(manifests);
    const comparable = mixedArtifacts.filter((manifest) => manifest.product === "browser-wasm" && manifest.artifactManifestDigest);
    if (comparable.length > 1) {
        comparable[1].artifactManifestDigest = sha256("different artifact");
        refreshResult(comparable[1]);
        assert.throws(() => reconcileShardResults(inventory, plan, mixedArtifacts), /not identical/);
    }
});

test("closure requires exact verified products and rejects unavailable or infrastructure evidence", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    const plan = buildPlan(inventory);
    const manifests = completeResults(inventory, plan);
    assert.throws(() => reconcileShardResults(inventory, plan, manifests, { closure: true }), /verified artifact manifest/);
    withVerifiedArtifacts((artifactManifests, productProbes, directory) => {
        assert.throws(() => reconcileShardResults(inventory, plan, manifests, { closure: true, artifactManifests, productProbes }), /unavailable shard adapters|evidence is incomplete/);

        const portableExport = { schema_version: 1, builtins: [exportFixture().builtins[0]] };
        const portableInventory = buildInventory(portableExport, inventoryOptions());
        const portablePlan = buildPlan(portableInventory);
        const portableResults = completeResults(portableInventory, portablePlan);
        assert.throws(() => reconcileShardResults(portableInventory, portablePlan, portableResults, { closure: true, artifactManifests }), /requires product probe/);
        const report = reconcileShardResults(portableInventory, portablePlan, portableResults, { closure: true, artifactManifests, productProbes });
        assert.equal(report.status, "passed");
        writeFileSync(join(directory, "native-product", "runmat"), "mutated");
        assert.throws(() => reconcileShardResults(portableInventory, portablePlan, portableResults, { closure: true, artifactManifests, productProbes }), /recursive tree/);
    });
});

test("product probes bind the fixed vector, product bytes, and successful process result", () => {
    const artifact = artifactDefinitions()[0];
    const probe = successfulNativeProbe(artifact);
    validateProductProbe(probe, { sourceRevision: SOURCE, artifactManifest: artifact });
    const wrongArtifact = structuredClone(probe);
    wrongArtifact.artifactManifestDigest = sha256("wrong artifact");
    wrongArtifact.productProbeDigest = digestObject(wrongArtifact, ["productProbeDigest"]);
    assert.throws(() => validateProductProbe(wrongArtifact, { artifactManifest: artifact }), /artifact manifest digest mismatch/);
    const tampered = structuredClone(probe);
    tampered.result.status = "failed";
    assert.throws(() => validateProductProbe(tampered), /Product probe digest mismatch/);

    const impossibleNativeFailure = structuredClone(probe);
    impossibleNativeFailure.result.status = "failed";
    impossibleNativeFailure.result.failureKind = "protocol";
    impossibleNativeFailure.productProbeDigest = digestObject(impossibleNativeFailure, ["productProbeDigest"]);
    assert.throws(() => validateProductProbe(impossibleNativeFailure), /native product probe failure kind/u);

    const inconsistentNativeFailure = structuredClone(probe);
    inconsistentNativeFailure.result.status = "failed";
    inconsistentNativeFailure.result.failureKind = "prepare-process";
    inconsistentNativeFailure.productProbeDigest = digestObject(inconsistentNativeFailure, ["productProbeDigest"]);
    assert.throws(() => validateProductProbe(inconsistentNativeFailure), /prepare-process evidence is inconsistent/u);
});

test("native product probes execute the manifested entrypoint and reject bytes changed during the probe", { skip: process.platform === "win32" }, () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-product-probe-test-"));
    try {
        const product = join(directory, "product");
        mkdirSync(product);
        const binary = join(product, "runmat");
        const dependency = join(product, "runmat.runtime");
        const manifestPath = join(directory, "artifacts.json");
        writeFakeCompiler(binary, false);
        writeFileSync(dependency, "runtime");
        const manifest = buildArtifactManifestFromTree({
            product: "native-cli", artifactProfile: "embedded-aot", sourceRevision: SOURCE,
            manifestPath, root: product, entrypoints: { "runmat-binary": "runmat" }
        });
        writeFileSync(manifestPath, JSON.stringify(manifest));
        const passed = runNativeEmbeddedAotProbe({ artifactManifest: manifest, artifactManifestPath: manifestPath, sourceRevision: SOURCE });
        assert.equal(passed.result.status, "passed");

        writeFakeCompiler(binary, true);
        const mutatingManifest = buildArtifactManifestFromTree({
            product: "native-cli", artifactProfile: "embedded-aot", sourceRevision: SOURCE,
            manifestPath, root: product, entrypoints: { "runmat-binary": "runmat" }
        });
        writeFileSync(manifestPath, JSON.stringify(mutatingManifest));
        assert.throws(() => runNativeEmbeddedAotProbe({
            artifactManifest: mutatingManifest,
            artifactManifestPath: manifestPath,
            sourceRevision: SOURCE
        }), /recursive tree/);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("Desktop product probes execute the closed hidden-host protocol", { skip: process.platform === "win32" }, async () => {
    const directory = mkdtempSync(join(tmpdir(), "runmat-desktop-product-probe-test-"));
    try {
        const product = join(directory, "product");
        mkdirSync(product);
        const binary = join(product, "runmat-desktop");
        writeFakeDesktopHost(binary);
        const manifestPath = join(directory, "artifacts.json");
        const producerRevision = "abcdef0123456789abcdef0123456789abcdef01";
        const manifest = buildArtifactManifestFromTree({
            product: "desktop-native",
            artifactProfile: "desktop-host",
            sourceRevision: SOURCE,
            producerRevision,
            manifestPath,
            root: product,
            entrypoints: { "runmat-desktop-binary": "runmat-desktop" }
        });
        writeFileSync(manifestPath, JSON.stringify(manifest));
        const probe = await runDesktopHostProtocolProbe({
            artifactManifest: manifest,
            artifactManifestPath: manifestPath,
            sourceRevision: SOURCE,
            producerRevision
        });
        assert.equal(probe.probeKind, DESKTOP_HOST_PROTOCOL_PROBE);
        assert.equal(probe.result.status, "passed");
        validateProductProbe(probe, { artifactManifest: manifest, artifactManifestPath: manifestPath, verifyFiles: true });

        const impossibleDesktopFailure = structuredClone(probe);
        impossibleDesktopFailure.result.status = "failed";
        impossibleDesktopFailure.result.failureKind = "prepare-process";
        impossibleDesktopFailure.productProbeDigest = digestObject(impossibleDesktopFailure, ["productProbeDigest"]);
        assert.throws(() => validateProductProbe(impossibleDesktopFailure), /Desktop product probe failure kind/u);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("closure rejects duplicate and failed product probes", () => {
    const portableInventory = buildInventory({ schema_version: 1, builtins: [exportFixture().builtins[0]] }, inventoryOptions());
    const portablePlan = buildPlan(portableInventory);
    const results = completeResults(portableInventory, portablePlan);
    withVerifiedArtifacts((artifactManifests, productProbes) => {
        assert.throws(() => reconcileShardResults(portableInventory, portablePlan, results, { closure: true, artifactManifests, productProbes: [...productProbes, ...productProbes] }), /Duplicate product probe/);
        const failed = structuredClone(productProbes[0]);
        failed.result.status = "failed";
        failed.result.failureKind = "stdout-mismatch";
        failed.productProbeDigest = digestObject(failed, ["productProbeDigest"]);
        assert.throws(() => reconcileShardResults(portableInventory, portablePlan, results, { closure: true, artifactManifests, productProbes: [failed] }), /product probe failed/);
    });
});

test("an unavailable adapter writes explicit isolated shard evidence without product execution", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    const plan = buildPlan(inventory);
    const lanePlan = plan.lanes.find((lane) => lane.lane === "native-foreign-runtime");
    assert.ok(lanePlan);
    const directory = mkdtempSync(join(tmpdir(), "runmat-unavailable-shard-"));
    try {
        const path = join(directory, "foreign.shard-result.json");
        const result = executeShard({
            inventory,
            plan,
            lanePlan,
            shard: lanePlan.shards[0],
            artifactManifest: null,
            artifactManifestPath: null,
            resultPath: path,
            resolveAdapterAvailability: () => ({ available: false, reason: "test capability is absent" })
        });
        assert.equal(result.status, "unavailable");
        assert.ok(result.results.every((record) => record.status === "unavailable"));
        assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), result);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

test("all protocol schemas are closed against unknown and missing fields", () => {
    const inventory = buildInventory(exportFixture(), inventoryOptions());
    const unknownInventory = structuredClone(inventory);
    unknownInventory.surprise = true;
    unknownInventory.inventoryDigest = digestObject(unknownInventory, ["inventoryDigest"]);
    assert.throws(() => validateInventory(unknownInventory), /unknown: surprise/);
    const missingExampleField = structuredClone(inventory);
    delete missingExampleField.examples[0].runnerKey;
    missingExampleField.inventoryDigest = digestObject(missingExampleField, ["inventoryDigest"]);
    assert.throws(() => validateInventory(missingExampleField), /missing: runnerKey/);

    const plan = buildPlan(inventory);
    const unknownPlan = structuredClone(plan);
    unknownPlan.lanes[0].surprise = true;
    refreshPlan(unknownPlan);
    assert.throws(() => validatePlan(unknownPlan, inventory), /unknown: surprise/);
    assert.throws(() => buildPlan(inventory, { schema: TOPOLOGY_SCHEMA, products: {}, lanes: { "browser-host": { surprise: true } } }), /Unknown topology field/);

    const matrix = buildExecutionMatrix(plan);
    const unknownMatrix = structuredClone(matrix);
    unknownMatrix.include[0].surprise = true;
    unknownMatrix.executionMatrixDigest = digestObject(unknownMatrix, ["executionMatrixDigest"]);
    assert.throws(() => validateExecutionMatrix(unknownMatrix, plan), /unknown: surprise/);

    const artifact = artifactDefinitions()[0];
    const unknownArtifact = structuredClone(artifact);
    unknownArtifact.surprise = true;
    unknownArtifact.artifactManifestDigest = digestObject(unknownArtifact, ["artifactManifestDigest"]);
    assert.throws(() => validateArtifactManifest(unknownArtifact), /unknown: surprise/);
    const unknownArtifactFile = structuredClone(artifact);
    unknownArtifactFile.files[0].surprise = true;
    unknownArtifactFile.artifactManifestDigest = digestObject(unknownArtifactFile, ["artifactManifestDigest"]);
    assert.throws(() => validateArtifactManifest(unknownArtifactFile), /unknown: surprise/);

    const unknownProbe = successfulNativeProbe(artifactDefinitions()[0]);
    unknownProbe.result.surprise = true;
    unknownProbe.productProbeDigest = digestObject(unknownProbe, ["productProbeDigest"]);
    assert.throws(() => validateProductProbe(unknownProbe), /unknown: surprise/);

    const shard = completeResults(inventory, plan)[0];
    const unknownShard = structuredClone(shard);
    unknownShard.surprise = true;
    refreshResult(unknownShard);
    assert.throws(() => validateShardResult(unknownShard), /unknown: surprise/);

    const portableInventory = buildInventory({ schema_version: 1, builtins: [exportFixture().builtins[0]] }, inventoryOptions());
    const portablePlan = buildPlan(portableInventory);
    withVerifiedArtifacts((artifactManifests, productProbes) => {
        const report = reconcileShardResults(portableInventory, portablePlan, completeResults(portableInventory, portablePlan), { closure: true, artifactManifests, productProbes });
        const unknownReport = structuredClone(report);
        unknownReport.surprise = true;
        unknownReport.reconciliationDigest = digestObject(unknownReport, ["reconciliationDigest"]);
        assert.throws(() => validateReconciliation(unknownReport), /unknown: surprise/);
    });
});

test("runner record normalization rejects duplicate, unplanned, missing, oversized, and escaping outputs", () => {
    const inventory = buildInventory({ schema_version: 1, builtins: [exportFixture().builtins[0]] }, inventoryOptions());
    const unit = inventory.executionUnits[0];
    const expected = [unit];
    const limits = { maxOutputBytes: 8, maxFigureBytes: 8 };
    const valid = { exampleKey: unit.runnerKey, matches: true, normalizedExpected: "1", normalizedActual: "1", imageRelPath: null, result: { errorText: "", errorIdentifier: "" } };
    assert.equal(normalizeRunnerRecords(expected, [valid], "/private/tmp", limits)[0].status, "passed");
    assert.equal(normalizeRunnerRecords(expected, [valid, valid], "/private/tmp", limits)[0].status, "infra-error");
    assert.equal(normalizeRunnerRecords(expected, [{ ...valid, exampleKey: "unplanned" }], "/private/tmp", limits)[0].status, "infra-error");
    assert.equal(normalizeRunnerRecords(expected, [], "/private/tmp", limits)[0].status, "infra-error");
    assert.equal(normalizeRunnerRecords(expected, [{ ...valid, normalizedActual: "0123456789" }], "/private/tmp", limits)[0].status, "infra-error");
    assert.equal(normalizeRunnerRecords(expected, [{ ...valid, imageRelPath: "../escape.png" }], "/private/tmp", limits)[0].status, "infra-error");
});

function exportFixture() {
    return {
        schema_version: 1,
        builtins: [
            { key: "alpha", authority: "catalog", category: "math", examples: [
                { id: "portable", input: "disp(1)", output: "1", harness: "Portable", compatibility: "RunMat", verification: "Succeeds" }
            ] },
            { key: "notes", authority: "legacy_sidecar", category: "math", examples: [
                { id: "notes", input: "% explanation", output: "% presentation only", harness: "LegacyBrowser" }
            ] },
            { key: "foreign", authority: "catalog", category: "language/foreign", examples: [
                { id: "python", input: "disp(2)", output: "2", harness: "NativeForeignRuntime", compatibility: "RunMat", verification: "Succeeds" }
            ] },
            { key: "network", authority: "catalog", category: "io/net", examples: [
                { id: "loopback", input: "disp(3)", output: "3", harness: "NativeLoopbackNetwork", compatibility: "RunMat", verification: "Succeeds" }
            ] }
        ]
    };
}

function artifactDefinitions() {
    return [
        buildArtifactManifest({
            product: "native-cli", artifactProfile: "embedded-aot", sourceRevision: SOURCE,
            layout: { kind: "recursive-tree", root: "native-product" },
            entrypoints: [{ role: "runmat-binary", path: "runmat" }],
            files: [
                { path: "runmat", size: 6, sha256: sha256("native") },
                { path: "runmat_runtime.dll", size: 7, sha256: sha256("runtime") }
            ]
        }),
        buildArtifactManifest({
            product: "browser-wasm", artifactProfile: "web", sourceRevision: SOURCE,
            layout: { kind: "exact-files" },
            entrypoints: [{ role: "wasm-js", path: "runmat.js" }, { role: "wasm-binary", path: "runmat.wasm" }],
            files: [{ path: "runmat.js", size: 2, sha256: sha256("js") }, { path: "runmat.wasm", size: 4, sha256: sha256("wasm") }]
        })
    ];
}

function withVerifiedArtifacts(callback) {
    const directory = mkdtempSync(join(tmpdir(), "runmat-closure-artifacts-"));
    try {
        const definitions = artifactDefinitions();
        mkdirSync(join(directory, "native-product"));
        writeFileSync(join(directory, "native-product", "runmat"), "native");
        writeFileSync(join(directory, "native-product", "runmat_runtime.dll"), "runtime");
        writeFileSync(join(directory, "runmat.js"), "js");
        writeFileSync(join(directory, "runmat.wasm"), "wasm");
        const artifactManifests = definitions.map((manifest) => {
            const manifestPath = join(directory, `${manifest.product}.json`);
            writeFileSync(manifestPath, JSON.stringify(manifest));
            return { manifest, manifestPath };
        });
        const productProbes = [successfulNativeProbe(definitions[0])];
        callback(artifactManifests, productProbes, directory);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
}

function successfulNativeProbe(artifact) {
    return buildProductProbe({
        sourceRevision: SOURCE,
        product: "native-cli",
        artifactProfile: "embedded-aot",
        artifactManifestDigest: artifact.artifactManifestDigest,
        probeKind: NATIVE_EMBEDDED_AOT_PROBE,
        vectorDigest: sha256("disp(2 + 3);\n"),
        result: {
            kind: "compile-and-execute",
            status: "passed",
            failureKind: "none",
            prepareExitCode: 0,
            executeExitCode: 0
        },
        stdoutDigest: sha256("5\n")
    });
}

function writeFakeCompiler(path, mutateAdjacentDependency) {
    const mutation = mutateAdjacentDependency ? "printf changed > \"$0.runtime\"\n" : "";
    writeFileSync(path, `#!/bin/sh\nset -eu\n${mutation}output=\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = -o ]; then output=$2; shift 2; else shift; fi\ndone\nprintf '#!/bin/sh\\nprintf \"5\\\\n\"\\n' > \"$output\"\nchmod +x \"$output\"\n`, "utf8");
    chmodSync(path, 0o700);
}

function writeFakeDesktopHost(path) {
    writeFileSync(path, `#!/usr/bin/env node
const fs = require("node:fs");
const request = JSON.parse(fs.readFileSync(process.env.RUNMAT_BUILTIN_EXAMPLE_HOST_REQUEST, "utf8"));
fs.writeFileSync(process.env.RUNMAT_BUILTIN_EXAMPLE_HOST_RESULT, JSON.stringify({
  schema: "runmat.builtin-examples.desktop-host-result.v1",
  exampleKey: request.exampleKey,
  stdoutText: "5\\n",
  valueText: "",
  errorText: "",
  errorIdentifier: "",
  interactions: { expectedCount: 1, observedCount: 1, expectedFigureCount: 0, observedFigureCount: 0, matched: true, figurePresentationsMatched: true }
}));
`, "utf8");
    chmodSync(path, 0o700);
}

function inventoryOptions() {
    return { sourceRevision: SOURCE, sourceState: "clean", runnerDigest: RUNNER, scope: {} };
}

function completeResults(inventory, plan) {
    const units = new Map(inventory.executionUnits.map((unit) => [unit.executionIdentity, unit]));
    return plan.lanes.flatMap((lanePlan) => lanePlan.shards.map((shard) => {
        const unavailable = ["native-loopback-network", "native-foreign-runtime", "interactive-host"].includes(lanePlan.lane);
        return buildShardResult({
            sourceRevision: plan.sourceRevision,
            sourceState: plan.sourceState,
            inventoryDigest: plan.inventoryDigest,
            planDigest: plan.planDigest,
            runnerDigest: plan.runnerDigest,
            lane: lanePlan.lane,
            shardIndex: shard.index,
            shardCount: lanePlan.shardCount,
            assignmentDigest: shard.assignmentDigest,
            product: lanePlan.product,
            artifactProfile: plan.products.find((product) => product.kind === lanePlan.product).artifactProfile,
            artifactManifestDigest: unavailable ? null : artifactDefinitions().find((artifact) => artifact.product === lanePlan.product).artifactManifestDigest,
            environment: { platform: "test", architecture: "test", node: "test" },
            limits: lanePlan.limits,
            adapter: { available: !unavailable, reason: unavailable ? "adapter is not implemented" : "" },
            results: shard.executionIdentities.map((id) => {
                const unit = units.get(id);
                return {
                    executionIdentity: id,
                    exampleIdentity: unit.exampleIdentity,
                    definitionDigest: unit.definitionDigest,
                    builtinKey: unit.builtinKey,
                    exampleId: unit.exampleId,
                    lane: unit.lane,
                    status: unavailable ? "unavailable" : "passed",
                    errorIdentifier: "",
                    errorText: unavailable ? "adapter is not implemented" : "",
                    normalizedExpected: unit.exampleId === "portable" ? "1" : "",
                    normalizedActual: unit.exampleId === "portable" ? "1" : "",
                    imageRelativePath: null
                };
            })
        });
    }));
}

function refreshShard(lane) {
    for (const shard of lane.shards) {
        shard.assignmentDigest = digestObject({ lane: lane.lane, index: shard.index, shardCount: lane.shardCount, executionIdentities: shard.executionIdentities });
    }
}

function refreshPlan(plan) {
    plan.planDigest = digestObject(plan, ["planDigest"]);
}

function refreshResult(result) {
    result.shardResultDigest = digestObject(result, ["shardResultDigest"]);
}
