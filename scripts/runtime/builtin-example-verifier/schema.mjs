// @ts-check

import { isDigest } from "./identity.mjs";

export function exactKeys(value, required, label) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
    const expected = [...required].sort();
    const actual = Object.keys(value).sort();
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        const missing = expected.filter((key) => !actual.includes(key));
        const unknown = actual.filter((key) => !expected.includes(key));
        throw new Error(`${label} has invalid fields (missing: ${missing.join(", ") || "none"}; unknown: ${unknown.join(", ") || "none"})`);
    }
}

export function gitRevision(value, label = "source revision") {
    if (typeof value !== "string" || !/^[a-f0-9]{40}$/.test(value)) throw new Error(`${label} must be a full lowercase Git object id`);
}

export function digest(value, label) {
    if (!isDigest(value)) throw new Error(`${label} must be a SHA-256 digest`);
}

export function enumValue(value, allowed, label) {
    if (!allowed.includes(value)) throw new Error(`${label} must be one of ${allowed.join(", ")}`);
}

export function integer(value, label, minimum = 0) {
    if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${label} must be a safe integer at least ${minimum}`);
}
