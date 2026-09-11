// @ts-check

import { mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { validateArtifactManifest } from "./artifacts.mjs";
import { validateInventory } from "./inventory.mjs";
import { validatePlan } from "./plan.mjs";
import { executeShard } from "./runner-process.mjs";

export function executeProductPlan({ inventory, plan, artifactManifest, artifactManifestPath, outputDirectory }) {
    validateInventory(inventory);
    validatePlan(plan, inventory);
    if (plan.productScope === "all") throw new Error("Product execution requires a product-scoped plan");
    if (plan.products.length !== 1 || plan.products[0].kind !== plan.productScope) throw new Error("Product-scoped plan has inconsistent product declarations");
    validateArtifactManifest(artifactManifest, {
        sourceRevision: plan.sourceRevision,
        verifyFiles: true,
        manifestPath: artifactManifestPath
    });
    if (artifactManifest.product !== plan.productScope || artifactManifest.artifactProfile !== plan.products[0].artifactProfile) {
        throw new Error("Artifact manifest does not match the product-scoped plan");
    }
    const root = resolve(outputDirectory);
    mkdirSync(root, { recursive: true });
    const results = [];
    for (const lanePlan of plan.lanes) {
        for (const shard of lanePlan.shards) {
            results.push(executeShard({
                inventory,
                plan,
                lanePlan,
                shard,
                artifactManifest,
                artifactManifestPath,
                resultPath: join(root, `${lanePlan.lane}-${shard.index}.shard-result.json`),
                workDirectory: join(root, "work", lanePlan.lane, String(shard.index))
            }));
        }
    }
    return results;
}
