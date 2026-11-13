// @ts-check

import { buildInventory, validateInventory } from "./inventory.mjs";
import { productionDocumentation } from "../../docs/production-documentation.mjs";

/** Production inventory shares the strict, complete current-source docs owner.
 * Explicit file inputs remain useful for modeled/scoped inventory preparation;
 * closure must independently compare them with this current compiled product. */
export function productionExampleDocumentation(repository, spawn) {
    return productionDocumentation(repository, {
        binary: process.env.RUNMAT_EXAMPLE_DOCUMENTATION_BINARY
            ?? process.env.RUNMAT_BUILTIN_DOCUMENTATION_BINARY,
    }, spawn);
}

/** Pure comparison, not a claim that supplied bytes were compiled. The caller
 * supplies the independently observed current exporter, Git and runner inputs. */
export function validateCurrentExampleInventory(inventory, documentExport, source) {
    validateInventory(inventory);
    if (inventory.scope.kind !== "complete" || inventory.sourceState !== "clean"
        || source.sourceState !== "clean") {
        throw new Error("Closure requires a complete inventory and current clean source state");
    }
    const current = buildInventory(documentExport, source);
    if (inventory.inventoryDigest !== current.inventoryDigest) {
        throw new Error("Closure inventory differs from the current full catalog, source or runner");
    }
    return inventory;
}
