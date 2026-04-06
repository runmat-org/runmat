import { describe, expect, it } from "vitest";

import {
  categoryPathFromCategory,
  getBuiltinManifest,
  getBuiltinManifestEntry,
  loadAllBuiltinDocs,
  loadBuiltinExamplesCatalog,
  loadBuiltinDoc,
  normalizeBuiltinKey,
  slugFromBuiltinTitle
} from "./builtins.js";

describe("builtins helpers", () => {
  it("normalizes builtin keys and slugs", () => {
    expect(normalizeBuiltinKey("  containers.Map ")).toBe("containers.map");
    expect(slugFromBuiltinTitle("hist")).toBe("hist");
    expect(categoryPathFromCategory("array/creation")).toEqual(["array", "creation"]);
  });

  it("exposes manifest metadata for builtins", () => {
    const manifest = getBuiltinManifest();
    const hist = getBuiltinManifestEntry("hist");

    expect(manifest.length).toBeGreaterThan(100);
    expect(hist).toMatchObject({
      key: "hist",
      title: "hist",
      category: "plotting",
      exampleCount: 3
    });
  });

  it("loads an individual builtin doc lazily", async () => {
    const hist = await loadBuiltinDoc("hist");

    expect(hist).not.toBeNull();
    expect(hist).toMatchObject({
      key: "hist",
      title: "hist",
      category: "plotting"
    });
    expect(hist?.examples?.[1]?.description).toBe("Pass explicit bin centers");
  });

  it("loads the full builtin docs corpus on demand", async () => {
    const docs = await loadAllBuiltinDocs();

    expect(docs.length).toBe(getBuiltinManifest().length);
    expect(docs.some((doc) => doc.key === "hist")).toBe(true);
  });

  it("loads the builtin examples catalog in one lazy step", async () => {
    const examples = await loadBuiltinExamplesCatalog();
    const histExample = examples.find((example) => example.id === "hist:2");

    expect(examples.length).toBeGreaterThan(100);
    expect(histExample).toMatchObject({
      builtinKey: "hist",
      exampleTitle: "Pass explicit bin centers",
      suggestedPath: "/hist-2.m"
    });
  });
});
