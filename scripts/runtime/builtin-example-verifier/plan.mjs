// @ts-check

import { compareUtf8, digestObject, isDigest, partitionBucket } from "./identity.mjs";
import { EXECUTION_LANES, isExecutionLane, laneProduct } from "./lanes.mjs";
import { validateInventory } from "./inventory.mjs";
import { defaultArtifactProfile, requiredArtifactRoles, requiredProductProbes } from "./product-contracts.mjs";
import { digest, enumValue, exactKeys, gitRevision, integer } from "./schema.mjs";
import { validateInventoryScope } from "./scope.mjs";

export { defaultArtifactProfile, requiredArtifactRoles, requiredProductProbes } from "./product-contracts.mjs";

export const TOPOLOGY_SCHEMA = "runmat.builtin-examples.topology.v1";
export const PLAN_SCHEMA = "runmat.builtin-examples.plan.v2";
export const PARTITION_ALGORITHM = "sha256-execution-identity-u64be-mod-v1";

const DEFAULT_LIMITS = Object.freeze({
    perCaseTimeoutMs: 30_000,
    shardWallTimeoutMs: 30 * 60_000,
    concurrency: 4,
    maxOutputBytes: 16 * 1024 * 1024,
    maxResultBytes: 32 * 1024 * 1024,
    maxFigureBytes: 8 * 1024 * 1024
});

export function buildPlan(inventory, topology = null, options = {}) {
    validateInventory(inventory);
    const productScope = options.product ?? "all";
    enumValue(productScope, ["all", "public-products", "native-cli", "browser-wasm", "desktop-native"], "plan product scope");
    const selectedUnits = inventory.executionUnits.filter((unit) => productInScope(laneProduct(unit.lane), productScope));
    if (selectedUnits.length === 0 && options.product) throw new Error(`Inventory contains no execution units for product scope ${productScope}`);
    const requiredLanes = [...new Set(selectedUnits.map((unit) => unit.lane))];
    const normalizedTopology = normalizeTopology(topology, requiredLanes);
    const unitsByLane = new Map(EXECUTION_LANES.map((lane) => [lane, []]));
    for (const unit of selectedUnits) unitsByLane.get(unit.lane)?.push(unit);
    const lanes = [];
    for (const lane of EXECUTION_LANES) {
        const units = unitsByLane.get(lane) ?? [];
        if (units.length === 0) continue;
        const config = normalizedTopology.lanes[lane];
        if (!config) throw new Error(`Execution topology omits required lane: ${lane}`);
        const shards = Array.from({ length: config.shards }, (_, index) => {
            const executionIdentities = units
                .filter((unit) => partitionBucket(unit.executionIdentity, config.shards) === index)
                .map((unit) => unit.executionIdentity)
                .sort(compareUtf8);
            return {
                index,
                expectedCount: executionIdentities.length,
                executionIdentities,
                assignmentDigest: digestObject({ lane, index, shardCount: config.shards, executionIdentities })
            };
        });
        lanes.push({
            lane,
            product: laneProduct(lane),
            shardCount: config.shards,
            requiredCapabilities: config.requiredCapabilities,
            limits: config.limits,
            shards
        });
    }
    const products = [...new Set(lanes.map((lane) => lane.product))]
        .sort(compareUtf8)
        .map((kind) => {
            const artifactProfile = normalizedTopology.products[kind].artifactProfile;
            return {
                kind,
                artifactProfile,
                requiredRoles: requiredArtifactRoles(kind, artifactProfile),
                requiredProbes: requiredProductProbes(kind, artifactProfile)
            };
        });
    const plan = {
        schema: PLAN_SCHEMA,
        sourceRevision: inventory.sourceRevision,
        sourceState: inventory.sourceState,
        inventoryDigest: inventory.inventoryDigest,
        runnerDigest: inventory.runnerDigest,
        partitionAlgorithm: PARTITION_ALGORITHM,
        productScope,
        scope: inventory.scope,
        products,
        lanes,
        planDigest: ""
    };
    plan.planDigest = digestObject(plan, ["planDigest"]);
    return plan;
}

export function validatePlan(plan, inventory = null) {
    if (!plan || plan.schema !== PLAN_SCHEMA) throw new Error("Unsupported builtin example plan schema");
    exactKeys(plan, ["schema", "sourceRevision", "sourceState", "inventoryDigest", "runnerDigest", "partitionAlgorithm", "productScope", "scope", "products", "lanes", "planDigest"], "execution plan");
    gitRevision(plan.sourceRevision);
    enumValue(plan.sourceState, ["clean", "dirty"], "plan source state");
    for (const [value, label] of [[plan.inventoryDigest, "plan inventory digest"], [plan.runnerDigest, "plan runner digest"], [plan.planDigest, "plan digest"]]) digest(value, label);
    if (digestObject(plan, ["planDigest"]) !== plan.planDigest) throw new Error("Builtin example plan digest mismatch");
    if (plan.partitionAlgorithm !== PARTITION_ALGORITHM) throw new Error("Unsupported builtin example partition algorithm");
    enumValue(plan.productScope, ["all", "public-products", "native-cli", "browser-wasm", "desktop-native"], "plan product scope");
    validateInventoryScope(plan.scope, "plan scope");
    if (!Array.isArray(plan.lanes) || !Array.isArray(plan.products)) throw new Error("Invalid builtin example plan records");
    const seenLanes = new Set();
    const seenUnits = new Set();
    const productKinds = new Set();
    for (const product of plan.products) {
        exactKeys(product, ["kind", "artifactProfile", "requiredRoles", "requiredProbes"], "planned product");
        enumValue(product.kind, ["native-cli", "browser-wasm", "desktop-native"], "planned product kind");
        if (productKinds.has(product.kind)) throw new Error(`Duplicate planned product: ${product.kind}`);
        productKinds.add(product.kind);
        if (JSON.stringify(product.requiredRoles) !== JSON.stringify(requiredArtifactRoles(product.kind, product.artifactProfile))) throw new Error(`Invalid artifact profile or required roles for ${product.kind}`);
        if (JSON.stringify(product.requiredProbes) !== JSON.stringify(requiredProductProbes(product.kind, product.artifactProfile))) throw new Error(`Invalid required product probes for ${product.kind}`);
    }
    const orderedProducts = [...plan.products].sort((left, right) => compareUtf8(left.kind, right.kind));
    if (JSON.stringify(orderedProducts) !== JSON.stringify(plan.products)) throw new Error("Planned products are not in canonical order");
    for (const lanePlan of plan.lanes) {
        exactKeys(lanePlan, ["lane", "product", "shardCount", "requiredCapabilities", "limits", "shards"], `lane plan ${lanePlan.lane}`);
        if (!isExecutionLane(lanePlan.lane) || seenLanes.has(lanePlan.lane)) throw new Error(`Invalid or duplicate lane plan: ${lanePlan.lane}`);
        seenLanes.add(lanePlan.lane);
        if (lanePlan.product !== laneProduct(lanePlan.lane)) throw new Error(`Wrong product for lane ${lanePlan.lane}`);
        positiveInteger(lanePlan.shardCount, `${lanePlan.lane} shard count`);
        if (!productKinds.has(lanePlan.product)) throw new Error(`Lane ${lanePlan.lane} references an undeclared product`);
        if (!Array.isArray(lanePlan.requiredCapabilities) || lanePlan.requiredCapabilities.some((value) => typeof value !== "string" || !value.trim()) || new Set(lanePlan.requiredCapabilities).size !== lanePlan.requiredCapabilities.length) throw new Error(`Invalid required capabilities for ${lanePlan.lane}`);
        if (JSON.stringify([...lanePlan.requiredCapabilities].sort(compareUtf8)) !== JSON.stringify(lanePlan.requiredCapabilities)) throw new Error(`Required capabilities are not in canonical order for ${lanePlan.lane}`);
        exactKeys(lanePlan.limits, Object.keys(DEFAULT_LIMITS), `limits for ${lanePlan.lane}`);
        for (const [name, value] of Object.entries(lanePlan.limits)) integer(value, `${lanePlan.lane} ${name}`, 1);
        if (!Array.isArray(lanePlan.shards) || lanePlan.shards.length !== lanePlan.shardCount) throw new Error(`Incomplete shard topology for ${lanePlan.lane}`);
        for (let index = 0; index < lanePlan.shards.length; index += 1) {
            const shard = lanePlan.shards[index];
            exactKeys(shard, ["index", "expectedCount", "executionIdentities", "assignmentDigest"], `shard declaration ${lanePlan.lane}/${index}`);
            if (shard.index !== index || shard.expectedCount !== shard.executionIdentities?.length) throw new Error(`Invalid shard declaration for ${lanePlan.lane}/${index}`);
            const ordered = [...shard.executionIdentities].sort(compareUtf8);
            if (JSON.stringify(ordered) !== JSON.stringify(shard.executionIdentities)) throw new Error(`Unsorted shard assignment for ${lanePlan.lane}/${index}`);
            const expectedDigest = digestObject({ lane: lanePlan.lane, index, shardCount: lanePlan.shardCount, executionIdentities: ordered });
            if (expectedDigest !== shard.assignmentDigest) throw new Error(`Shard assignment digest mismatch for ${lanePlan.lane}/${index}`);
            for (const id of ordered) {
                if (!isDigest(id) || seenUnits.has(id)) throw new Error(`Invalid or duplicate planned execution identity: ${id}`);
                if (partitionBucket(id, lanePlan.shardCount) !== index) throw new Error(`Execution identity is assigned to the wrong shard: ${id}`);
                seenUnits.add(id);
            }
        }
    }
    const orderedLanes = [...plan.lanes].sort((left, right) => EXECUTION_LANES.indexOf(left.lane) - EXECUTION_LANES.indexOf(right.lane));
    if (JSON.stringify(orderedLanes) !== JSON.stringify(plan.lanes)) throw new Error("Planned lanes are not in canonical order");
    const usedProducts = [...new Set(plan.lanes.map((lane) => lane.product))].sort(compareUtf8);
    if (JSON.stringify(usedProducts) !== JSON.stringify([...productKinds].sort(compareUtf8))) throw new Error("Planned product declarations do not exactly match lane products");
    if (inventory) {
        validateInventory(inventory);
        for (const key of ["sourceRevision", "sourceState", "inventoryDigest", "runnerDigest"]) {
            if (plan[key] !== inventory[key]) throw new Error(`Plan ${key} does not match inventory`);
        }
        if (JSON.stringify(plan.scope) !== JSON.stringify(inventory.scope)) throw new Error("Plan scope does not match inventory");
        const scopedUnits = inventory.executionUnits.filter((unit) => productInScope(laneProduct(unit.lane), plan.productScope));
        const expectedLanes = [...new Set(scopedUnits.map((unit) => unit.lane))].sort((left, right) => EXECUTION_LANES.indexOf(left) - EXECUTION_LANES.indexOf(right));
        if (JSON.stringify(plan.lanes.map((lane) => lane.lane)) !== JSON.stringify(expectedLanes)) throw new Error("Plan lanes do not exactly match the inventory");
        const expected = scopedUnits.map((unit) => unit.executionIdentity).sort(compareUtf8);
        const actual = [...seenUnits].sort(compareUtf8);
        if (JSON.stringify(expected) !== JSON.stringify(actual)) throw new Error("Plan does not cover the inventory exactly once");
    }
    return plan;
}

function productInScope(product, scope) {
    if (scope === "all") return true;
    if (scope === "public-products") return product === "native-cli" || product === "browser-wasm";
    return product === scope;
}

function normalizeTopology(topology, requiredLanes) {
    if (topology === null || topology === undefined) {
        topology = {
            schema: TOPOLOGY_SCHEMA,
            lanes: Object.fromEntries(requiredLanes.map((lane) => [lane, {}])),
            products: {}
        };
    }
    if (!topology || topology.schema !== TOPOLOGY_SCHEMA || !topology.lanes || typeof topology.lanes !== "object") {
        throw new Error("Unsupported builtin example topology schema");
    }
    exactKeys(topology, ["schema", "lanes", "products"], "execution topology");
    if (!topology.products || typeof topology.products !== "object" || Array.isArray(topology.products)) throw new Error("Execution topology products must be an object");
    const lanes = {};
    for (const [lane, raw] of Object.entries(topology.lanes)) {
        if (!isExecutionLane(lane)) throw new Error(`Unknown lane in execution topology: ${lane}`);
        if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`Invalid topology for lane ${lane}`);
        for (const key of Object.keys(raw)) {
            if (!["shards", "requiredCapabilities", "limits"].includes(key)) throw new Error(`Unknown topology field for ${lane}: ${key}`);
        }
        const shards = raw.shards ?? 1;
        positiveInteger(shards, `${lane} shards`);
        const requiredCapabilities = Array.isArray(raw.requiredCapabilities)
            ? raw.requiredCapabilities.map((value) => String(value).trim()).filter(Boolean).sort(compareUtf8)
            : [];
        if (new Set(requiredCapabilities).size !== requiredCapabilities.length) throw new Error(`Duplicate required capability for lane ${lane}`);
        lanes[lane] = { shards, requiredCapabilities, limits: normalizeLimits(raw.limits, lane) };
    }
    const configuredProducts = {};
    for (const [product, raw] of Object.entries(topology.products)) {
        enumValue(product, ["native-cli", "browser-wasm", "desktop-native"], "topology product");
        if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`Invalid topology product ${product}`);
        exactKeys(raw, ["artifactProfile"], `topology product ${product}`);
        requiredArtifactRoles(product, raw.artifactProfile);
        configuredProducts[product] = { artifactProfile: raw.artifactProfile };
    }
    const products = {};
    for (const product of [...new Set(requiredLanes.map(laneProduct))]) {
        products[product] = configuredProducts[product] ?? { artifactProfile: defaultArtifactProfile(product) };
        requiredArtifactRoles(product, products[product].artifactProfile);
    }
    return { schema: TOPOLOGY_SCHEMA, lanes, products };
}

function normalizeLimits(raw = {}, lane) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`Invalid limits for lane ${lane}`);
    for (const key of Object.keys(raw)) if (!(key in DEFAULT_LIMITS)) throw new Error(`Unknown limit for ${lane}: ${key}`);
    const result = { ...DEFAULT_LIMITS, ...raw };
    for (const [name, value] of Object.entries(result)) positiveInteger(value, `${lane} ${name}`);
    return result;
}

function positiveInteger(value, label) {
    if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${label} must be a positive safe integer`);
}
