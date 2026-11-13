import assert from "node:assert/strict";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { REGISTRIES, registryCoverage, registryNames } from "./registry-coverage.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

test("every registry anchor extracts names from its source", () => {
    for (const registry of REGISTRIES) assert.ok(registryNames(repo, registry).length > 5, `${registry.id} extracted too few names`);
});

test("helper-registered operations resolve to their registered names", () => {
    const operations = registryNames(repo, REGISTRIES.find(registry => registry.id === "output operations"));
    for (const name of ["magnitude", "recover", "interpolate", "sample", "mean", "rms", "von_mises"]) assert.ok(operations.includes(name), name);
});

test("the docs document every registry name", () => {
    for (const entry of registryCoverage(repo)) assert.deepEqual(entry.missing, [], entry.id);
});
