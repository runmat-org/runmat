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
  it("retains typed Desktop document fixtures in the generated documentation", async () => {
    const doc = await loadBuiltinDoc("opentoline");
    const fixtures = (doc?.examples ?? []).flatMap((example) => {
      const fixture = example.fixture;
      return fixture && typeof fixture === "object" && "DesktopHostOnly" in fixture
        ? [fixture.DesktopHostOnly] : [];
    });
    expect(fixtures.length).toBeGreaterThan(0);
    const selections = new Set<string>();
    for (const fixture of fixtures) {
      expect(fixture.entries.length).toBeGreaterThan(0);
      expect(JSON.parse(JSON.stringify(fixture))).toEqual(fixture);
      for (const interaction of fixture.interactions) {
        if (!("Document" in interaction) || !("Navigate" in interaction.Document)) continue;
        const navigation = interaction.Document.Navigate;
        expect(navigation.line).toBeGreaterThan(0);
        expect(navigation.column).toBeGreaterThan(0);
        expect(navigation.relative_path.length).toBeGreaterThan(0);
        selections.add(navigation.selection);
      }
    }
    expect(selections).toEqual(new Set(["Caret", "Select"]));
  });

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
    const histExample = examples.find((example) => example.id === "hist:retained-2");

    expect(examples.length).toBeGreaterThan(100);
    expect(histExample).toMatchObject({
      builtinKey: "hist",
      exampleTitle: "Pass explicit bin centers",
      suggestedPath: "/hist-retained-2.m"
    });
  });
});
