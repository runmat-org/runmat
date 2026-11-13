// @ts-check

import { createHash } from "node:crypto";

const DIGEST = /^sha256:[a-f0-9]{64}$/;

export function canonicalJson(value) {
    return JSON.stringify(canonicalValue(value));
}

export function sha256(value) {
    const source = typeof value === "string" || value instanceof Uint8Array ? value : canonicalJson(value);
    return `sha256:${createHash("sha256").update(source).digest("hex")}`;
}

export function digestObject(value, omitted = []) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("digestObject requires an object");
    }
    const copy = { ...value };
    for (const key of omitted) delete copy[key];
    return sha256(copy);
}

export function isDigest(value) {
    return typeof value === "string" && DIGEST.test(value);
}

export function compareUtf8(left, right) {
    return Buffer.compare(Buffer.from(String(left), "utf8"), Buffer.from(String(right), "utf8"));
}

export function exampleIdentity(builtinKey, exampleId) {
    return sha256(["runmat.builtin-example.identity.v1", normalize(builtinKey, "builtin key"), normalize(exampleId, "example id")]);
}

export function executionIdentity(builtinKey, exampleId, lane) {
    return sha256([
        "runmat.builtin-example.execution.v1",
        normalize(builtinKey, "builtin key"),
        normalize(exampleId, "example id"),
        normalize(lane, "execution lane")
    ]);
}

export function legacyExampleId(definition) {
    return `legacy-${sha256(["runmat.builtin-example.legacy.v1", definition]).slice("sha256:".length)}`;
}

export function partitionBucket(executionId, count) {
    if (!isDigest(executionId)) throw new Error("partition identity must be a sha256 digest");
    if (!Number.isSafeInteger(count) || count < 1) throw new Error("partition count must be a positive safe integer");
    const word = BigInt(`0x${executionId.slice("sha256:".length, "sha256:".length + 16)}`);
    return Number(word % BigInt(count));
}

function canonicalValue(value) {
    if (value === null || typeof value === "string" || typeof value === "boolean") return value;
    if (typeof value === "number") {
        if (!Number.isFinite(value)) throw new Error("canonical JSON does not admit non-finite numbers");
        return value;
    }
    if (Array.isArray(value)) return value.map(canonicalValue);
    if (typeof value === "object") {
        return Object.fromEntries(
            Object.keys(value)
                .sort(compareUtf8)
                .map((key) => [key, canonicalValue(value[key])])
        );
    }
    throw new Error(`canonical JSON does not admit ${typeof value}`);
}

function normalize(value, label) {
    const normalized = String(value ?? "").trim().toLowerCase();
    if (!normalized) throw new Error(`Missing ${label}`);
    return normalized;
}
