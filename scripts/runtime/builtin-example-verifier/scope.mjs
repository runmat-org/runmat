// @ts-check

import { compareUtf8 } from "./identity.mjs";
import { exactKeys, integer } from "./schema.mjs";

export function normalizeInventoryScope(scope = {}) {
    const builtins = normalizedBuiltins(scope.builtins ?? (scope.builtin ? [scope.builtin] : []));
    const filter = normalizedOptional(scope.filter, "inventory text filter");
    if (builtins.length && filter) throw new Error("Builtin-set and text filters cannot be combined");
    const limit = scope.limit === undefined || scope.limit === null ? null : Number(scope.limit);
    if (limit !== null && (!Number.isSafeInteger(limit) || limit < 1)) throw new Error("Inventory limit must be a positive integer");
    return { kind: builtins.length || filter || limit !== null ? "development" : "complete", builtins, filter, limit };
}

export function validateInventoryScope(scope, label = "inventory scope") {
    exactKeys(scope, ["kind", "builtins", "filter", "limit"], label);
    if (!["complete", "development"].includes(scope.kind)) throw new Error(`${label} has an invalid kind`);
    const builtins = normalizedBuiltins(scope.builtins);
    if (JSON.stringify(builtins) !== JSON.stringify(scope.builtins)) throw new Error(`${label} builtins are not normalized and canonically ordered`);
    const filter = normalizedOptional(scope.filter, `${label} filter`);
    if (filter !== scope.filter) throw new Error(`${label} filter is not normalized`);
    if (builtins.length && filter) throw new Error(`${label} cannot combine builtin-set and text filters`);
    if (scope.limit !== null) integer(scope.limit, `${label} limit`, 1);
    const development = builtins.length > 0 || filter !== null || scope.limit !== null;
    if ((scope.kind === "development") !== development) throw new Error(`${label} kind does not match its selectors`);
    return scope;
}

export function matchesInventoryScope(example, scope) {
    if (scope.builtins.length && !scope.builtins.includes(example.builtinKey)) return false;
    if (!scope.filter) return true;
    return [example.builtinKey, example.description, example.program, example.category]
        .join("\n").toLowerCase().includes(scope.filter);
}

function normalizedBuiltins(value) {
    if (!Array.isArray(value)) throw new Error("Inventory builtin scope must be an array");
    const result = value.map((entry) => {
        const normalized = String(entry ?? "").trim().toLowerCase();
        if (!normalized) throw new Error("Inventory builtin scope contains an empty identity");
        return normalized;
    }).sort(compareUtf8);
    if (new Set(result).size !== result.length) throw new Error("Inventory builtin scope contains duplicate identities");
    return result;
}

function normalizedOptional(value, label) {
    if (value === undefined || value === null || value === "") return null;
    const normalized = String(value).trim().toLowerCase();
    if (!normalized) throw new Error(`${label} must be nonempty`);
    return normalized;
}
