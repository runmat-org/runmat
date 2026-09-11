// @ts-check

import { compareUtf8, digestObject } from "./identity.mjs";
import { validateInventory } from "./inventory.mjs";
import { validatePlan } from "./plan.mjs";
import { validateProductProbe } from "./product-probe.mjs";
import { requiredProductProbes } from "./product-contracts.mjs";
import { RESULT_STATUSES, validateShardResult } from "./result-manifest.mjs";
import { validateArtifactManifest } from "./artifacts.mjs";
import { digest, enumValue, exactKeys, gitRevision, integer } from "./schema.mjs";

export const RECONCILIATION_SCHEMA = "runmat.builtin-examples.reconciliation.v1";

export function reconcileShardResults(inventory, plan, manifests, options = {}) {
    validateInventory(inventory);
    validatePlan(plan, inventory);
    if (options.closure && (inventory.scope.kind !== "complete" || inventory.sourceState !== "clean" || plan.sourceState !== "clean")) {
        throw new Error("Closure reconciliation requires a complete inventory from a clean source state");
    }
    if (options.closure && plan.productScope !== "all") throw new Error("Closure reconciliation requires an all-product execution plan");
    const units = new Map(inventory.executionUnits.map((unit) => [unit.executionIdentity, unit]));
    const productPlans = new Map(plan.products.map((product) => [product.kind, product]));
    const verifiedArtifacts = new Map();
    for (const record of options.artifactManifests ?? []) {
        if (!record || typeof record.manifestPath !== "string") throw new Error("Artifact verification requires a manifest path");
        const artifact = record.manifest;
        validateArtifactManifest(artifact, { sourceRevision: plan.sourceRevision, verifyFiles: true, manifestPath: record.manifestPath });
        if (verifiedArtifacts.has(artifact.product)) throw new Error(`Duplicate verified artifact manifest for ${artifact.product}`);
        const productPlan = productPlans.get(artifact.product);
        if (!productPlan || artifact.artifactProfile !== productPlan.artifactProfile) throw new Error(`Artifact profile does not match the plan for ${artifact.product}`);
        verifiedArtifacts.set(artifact.product, artifact);
    }
    if (options.closure) {
        for (const product of plan.products) {
            if (!verifiedArtifacts.has(product.kind)) throw new Error(`Closure reconciliation requires a verified artifact manifest for ${product.kind}`);
        }
    }
    const verifiedProbes = new Map();
    for (const probe of options.productProbes ?? []) {
        const artifactRecord = (options.artifactManifests ?? []).find((record) => record.manifest?.product === probe?.product);
        if (!artifactRecord) throw new Error(`Product probe has no verified artifact manifest for ${probe?.product ?? "unknown"}`);
        validateProductProbe(probe, {
            sourceRevision: plan.sourceRevision,
            artifactManifest: artifactRecord.manifest,
            artifactManifestPath: artifactRecord.manifestPath,
            verifyFiles: true
        });
        const productPlan = productPlans.get(probe.product);
        if (!productPlan || probe.artifactProfile !== productPlan.artifactProfile || !productPlan.requiredProbes.includes(probe.probeKind)) {
            throw new Error(`Product probe is outside the plan: ${probe.product}/${probe.probeKind}`);
        }
        const key = probeKey(probe.product, probe.probeKind);
        if (verifiedProbes.has(key)) throw new Error(`Duplicate product probe: ${key}`);
        verifiedProbes.set(key, probe);
    }
    if (options.closure) {
        for (const product of plan.products) {
            for (const probeKind of product.requiredProbes) {
                const probe = verifiedProbes.get(probeKey(product.kind, probeKind));
                if (!probe) throw new Error(`Closure reconciliation requires product probe ${product.kind}/${probeKind}`);
                if (probe.result.status !== "passed") throw new Error(`Closure product probe failed: ${product.kind}/${probeKind}`);
            }
        }
    }
    const expectedShards = new Map();
    for (const lanePlan of plan.lanes) {
        for (const shard of lanePlan.shards) expectedShards.set(shardKey(lanePlan.lane, shard.index), { lanePlan, shard });
    }
    const received = new Map();
    const productArtifacts = new Map();
    for (const manifest of manifests) {
        validateShardResult(manifest);
        const key = shardKey(manifest.lane, manifest.shardIndex);
        if (!expectedShards.has(key)) throw new Error(`Unexpected shard result: ${key}`);
        if (received.has(key)) throw new Error(`Duplicate shard result: ${key}`);
        received.set(key, manifest);
        const { lanePlan, shard } = expectedShards.get(key);
        for (const field of ["sourceRevision", "sourceState", "inventoryDigest", "planDigest", "runnerDigest"]) {
            if (manifest[field] !== plan[field]) throw new Error(`Stale or inconsistent ${field} in shard ${key}`);
        }
        if (manifest.shardCount !== lanePlan.shardCount || manifest.assignmentDigest !== shard.assignmentDigest || manifest.product !== lanePlan.product) {
            throw new Error(`Shard provenance does not match plan: ${key}`);
        }
        if (manifest.artifactProfile !== productPlans.get(manifest.product)?.artifactProfile) throw new Error(`Shard artifact profile does not match plan: ${key}`);
        if (JSON.stringify(manifest.limits) !== JSON.stringify(lanePlan.limits)) throw new Error(`Shard limits do not match plan: ${key}`);
        if (manifest.adapter?.available === true && !manifest.artifactManifestDigest) throw new Error(`Available shard has no artifact identity: ${key}`);
        if (manifest.adapter?.available === false && manifest.results.some((result) => result.status !== "unavailable")) {
            throw new Error(`Unavailable adapter emitted executable results: ${key}`);
        }
        const expectedIds = shard.executionIdentities;
        const actualIds = manifest.results.map((result) => result.executionIdentity);
        if (JSON.stringify(expectedIds) !== JSON.stringify(actualIds)) throw new Error(`Shard ${key} does not contain its exact planned execution identities`);
        for (const result of manifest.results) {
            const unit = units.get(result.executionIdentity);
            if (!unit || unit.lane !== manifest.lane || unit.definitionDigest !== result.definitionDigest
                || unit.exampleIdentity !== result.exampleIdentity || unit.builtinKey !== result.builtinKey
                || unit.exampleId !== result.exampleId) {
                throw new Error(`Stale or inconsistent execution result: ${result.executionIdentity}`);
            }
            const example = inventory.examples.find((candidate) => candidate.identity === unit.exampleIdentity);
            const expectedIdentifier = example?.verification?.ExpectedError?.identifier;
            if (result.status === "passed" && typeof expectedIdentifier === "string" && result.errorIdentifier !== expectedIdentifier) {
                throw new Error(`Expected-error result has the wrong stable identifier: ${result.executionIdentity}`);
            }
        }
        if (manifest.artifactManifestDigest) {
            const existing = productArtifacts.get(manifest.product);
            if (existing && existing !== manifest.artifactManifestDigest) throw new Error(`Product ${manifest.product} was not identical across shards`);
            productArtifacts.set(manifest.product, manifest.artifactManifestDigest);
        }
    }
    const missing = [...expectedShards.keys()].filter((key) => !received.has(key));
    if (missing.length) throw new Error(`Missing shard results: ${missing.join(", ")}`);
    const allResults = [...received.values()].flatMap((manifest) => manifest.results).sort((left, right) => compareUtf8(left.executionIdentity, right.executionIdentity));
    if (options.closure) {
        const unavailableAdapters = [...received.values()].filter((manifest) => manifest.adapter.available === false);
        if (unavailableAdapters.length) throw new Error(`Closure evidence contains ${unavailableAdapters.length} unavailable shard adapters`);
        for (const [product, artifact] of verifiedArtifacts) {
            if (productArtifacts.get(product) !== artifact.artifactManifestDigest) throw new Error(`Closure artifact identity for ${product} does not match shard evidence`);
        }
        const incomplete = allResults.filter((result) => ["unavailable", "infra-error", "timed-out"].includes(result.status));
        if (incomplete.length) throw new Error(`Closure evidence is incomplete: ${incomplete.length} execution units are unavailable or infrastructure-failed`);
    } else {
        for (const [product, artifact] of verifiedArtifacts) {
            const observed = productArtifacts.get(product);
            if (observed && observed !== artifact.artifactManifestDigest) throw new Error(`Verified artifact identity for ${product} does not match shard evidence`);
        }
    }
    validatePortablePairs(inventory, allResults, new Set(plan.lanes.map((lane) => lane.lane)));
    const byStatus = Object.fromEntries(RESULT_STATUSES.map((status) => [status, 0]));
    for (const result of allResults) byStatus[result.status] += 1;
    const reconciliation = {
        schema: RECONCILIATION_SCHEMA,
        sourceRevision: plan.sourceRevision,
        inventoryDigest: plan.inventoryDigest,
        planDigest: plan.planDigest,
        runnerDigest: plan.runnerDigest,
        productScope: plan.productScope,
        closure: Boolean(options.closure),
        status: allResults.every((result) => result.status === "passed") ? "passed" : "failed",
        summary: { shards: received.size, executionUnits: allResults.length, byStatus },
        productArtifactDigests: Object.fromEntries([...productArtifacts.entries()].sort(([a], [b]) => compareUtf8(a, b))),
        productProbeDigests: [...verifiedProbes.entries()].sort(([a], [b]) => compareUtf8(a, b)).map(([key, probe]) => ({ key, digest: probe.productProbeDigest })),
        shardResultDigests: [...received.entries()].sort(([a], [b]) => compareUtf8(a, b)).map(([key, manifest]) => ({ key, digest: manifest.shardResultDigest })),
        reconciliationDigest: ""
    };
    reconciliation.reconciliationDigest = digestObject(reconciliation, ["reconciliationDigest"]);
    return reconciliation;
}

export function validateReconciliation(report) {
    if (!report || report.schema !== RECONCILIATION_SCHEMA) throw new Error("Unsupported builtin example reconciliation schema");
    exactKeys(report, ["schema", "sourceRevision", "inventoryDigest", "planDigest", "runnerDigest", "productScope", "closure", "status", "summary", "productArtifactDigests", "productProbeDigests", "shardResultDigests", "reconciliationDigest"], "reconciliation");
    gitRevision(report.sourceRevision);
    for (const [value, label] of [[report.inventoryDigest, "reconciliation inventory digest"], [report.planDigest, "reconciliation plan digest"], [report.runnerDigest, "reconciliation runner digest"], [report.reconciliationDigest, "reconciliation digest"]]) digest(value, label);
    enumValue(report.productScope, ["all", "native-cli", "browser-wasm"], "reconciliation product scope");
    if (typeof report.closure !== "boolean") throw new Error("Reconciliation closure must be a boolean");
    enumValue(report.status, ["passed", "failed"], "reconciliation status");
    exactKeys(report.summary, ["shards", "executionUnits", "byStatus"], "reconciliation summary");
    integer(report.summary.shards, "reconciliation shard count");
    integer(report.summary.executionUnits, "reconciliation execution unit count");
    exactKeys(report.summary.byStatus, RESULT_STATUSES, "reconciliation status counts");
    for (const [status, count] of Object.entries(report.summary.byStatus)) integer(count, `reconciliation ${status} count`);
    if (Object.values(report.summary.byStatus).reduce((sum, count) => sum + count, 0) !== report.summary.executionUnits) throw new Error("Reconciliation status counts do not match execution unit count");
    const expectedStatus = report.summary.byStatus.failed === 0 && report.summary.byStatus.unavailable === 0 && report.summary.byStatus["infra-error"] === 0 && report.summary.byStatus["timed-out"] === 0 ? "passed" : "failed";
    if (report.status !== expectedStatus) throw new Error("Reconciliation status does not match its counts");
    if (!Array.isArray(report.shardResultDigests)) throw new Error("Reconciliation shard digests must be an array");
    const shardKeys = new Set();
    for (const entry of report.shardResultDigests) {
        exactKeys(entry, ["key", "digest"], "reconciliation shard digest");
        if (typeof entry.key !== "string" || !entry.key || shardKeys.has(entry.key)) throw new Error(`Invalid or duplicate reconciliation shard key: ${entry.key}`);
        shardKeys.add(entry.key);
        digest(entry.digest, `reconciliation shard digest ${entry.key}`);
    }
    const orderedShards = [...report.shardResultDigests].sort((left, right) => compareUtf8(left.key, right.key));
    if (JSON.stringify(orderedShards) !== JSON.stringify(report.shardResultDigests)) throw new Error("Reconciliation shard digests are not in canonical order");
    if (!Array.isArray(report.productProbeDigests)) throw new Error("Reconciliation product probe digests must be an array");
    const probeKeys = new Set();
    for (const entry of report.productProbeDigests) {
        exactKeys(entry, ["key", "digest"], "reconciliation product probe digest");
        if (typeof entry.key !== "string" || !entry.key || probeKeys.has(entry.key)) throw new Error(`Invalid or duplicate reconciliation product probe key: ${entry.key}`);
        probeKeys.add(entry.key);
        const separator = entry.key.indexOf("/");
        const product = entry.key.slice(0, separator);
        const kind = entry.key.slice(separator + 1);
        if (separator <= 0 || !requiredProductProbes(product).includes(kind)) throw new Error(`Invalid reconciliation product probe key: ${entry.key}`);
        digest(entry.digest, `reconciliation product probe digest ${entry.key}`);
    }
    const orderedProbes = [...report.productProbeDigests].sort((left, right) => compareUtf8(left.key, right.key));
    if (JSON.stringify(orderedProbes) !== JSON.stringify(report.productProbeDigests)) throw new Error("Reconciliation product probe digests are not in canonical order");
    if (!report.productArtifactDigests || typeof report.productArtifactDigests !== "object" || Array.isArray(report.productArtifactDigests)) throw new Error("Reconciliation product artifact digests must be an object");
    for (const [product, value] of Object.entries(report.productArtifactDigests)) {
        enumValue(product, ["native-cli", "browser-wasm"], "reconciliation product");
        digest(value, `reconciliation product digest ${product}`);
    }
    const orderedProducts = Object.keys(report.productArtifactDigests).sort(compareUtf8);
    if (JSON.stringify(orderedProducts) !== JSON.stringify(Object.keys(report.productArtifactDigests))) throw new Error("Reconciliation product artifacts are not in canonical order");
    if (digestObject(report, ["reconciliationDigest"]) !== report.reconciliationDigest) throw new Error("Reconciliation digest mismatch");
    return report;
}

function validatePortablePairs(inventory, results, plannedLanes) {
    const resultById = new Map(results.map((result) => [result.executionIdentity, result]));
    for (const example of inventory.examples) {
        if (example.harness !== "Portable" || example.admission.kind !== "executable") continue;
        const units = inventory.executionUnits.filter((unit) => unit.exampleIdentity === example.identity && plannedLanes.has(unit.lane));
        const laneResults = units.map((unit) => resultById.get(unit.executionIdentity));
        if (laneResults.some((result) => !result)) throw new Error(`Portable example is missing a lane result: ${example.identity}`);
        const expectedIdentifier = example.verification?.ExpectedError?.identifier;
        if (typeof expectedIdentifier === "string") {
            for (const result of laneResults) {
                if (result.errorIdentifier !== expectedIdentifier) throw new Error(`Portable expected-error identifiers disagree: ${example.identity}`);
            }
        } else if (laneResults.every((result) => result.status === "passed")) {
            const actual = new Set(laneResults.map((result) => result.normalizedActual));
            if (actual.size !== 1) throw new Error(`Portable lane outputs disagree: ${example.identity}`);
        }
    }
}

function shardKey(lane, index) {
    return `${lane}/${index}`;
}

function probeKey(product, kind) {
    return `${product}/${kind}`;
}
