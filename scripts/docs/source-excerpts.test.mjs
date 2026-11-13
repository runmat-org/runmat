import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { sourceExcerpts } from "./source-excerpts.mjs";

test("cantilever excerpts retain exact packaged study clauses", () => {
  const root = new URL("../../docs/simulation/examples/", import.meta.url);
  const catalog = JSON.parse(readFileSync(new URL("catalog.json", root)));
  const example = catalog.examples.find(item => item.id === "cantilever-static");
  const source = readFileSync(new URL(example.entry, root), "utf8");
  const excerpts = sourceExcerpts(source, example.source_sections);
  assert.equal(excerpts.length, 4);
  for (const excerpt of excerpts) assert.ok(source.includes(excerpt.content));
  assert.ok(excerpts[2].content.includes("[0, 0, -100000] Pa"));
  assert.ok(excerpts[3].content.startsWith("study cantilever_static {"));
  assert.ok(excerpts[3].content.includes("analysis static { structural static {} }"));
});

test("missing or ambiguous excerpt anchors fail instead of publishing stale instructions", () => {
  const section = { title: "Material", start: "material", end: null };
  assert.throws(() => sourceExcerpts("model", [section]), /canonical source/);
  assert.throws(() => sourceExcerpts("material one\nmaterial two", [section]), /canonical source/);
});


test("every snapshotted guide exposes complete canonical source sections", () => {
  const root = new URL("../../docs/simulation/examples/", import.meta.url);
  const catalog = JSON.parse(readFileSync(new URL("catalog.json", root)));
  for (const example of catalog.examples.filter(example => example.snapshot)) {
    const source = readFileSync(new URL(example.entry, root), "utf8");
    const excerpts = sourceExcerpts(source, example.source_sections);
    assert.equal(excerpts.length, 4, example.id);
    for (const excerpt of excerpts) assert.ok(source.includes(excerpt.content), example.id);
    assert.ok(excerpts[3].content.startsWith("study "), example.id);
    assert.ok(/\banalysis\s+\w+\s*\{/.test(excerpts[3].content), example.id);
    assert.ok(!/\bformulation\s*=/.test(excerpts[3].content), example.id);
    const geometryFiles = [...source.matchAll(/file\s*=\s*"\.\/([^"]+\.step)"/g)].map(match => match[1]);
    assert.ok(geometryFiles.length > 0, example.id);
    for (const path of geometryFiles) {
      assert.ok(example.files.includes(path), `${example.id}: ${path}`);
    }
    if (example.model_illustration) {
      assert.ok(example.files.includes(example.model_illustration.path), example.id);
      assert.ok(example.model_illustration.caption.length > 0, example.id);
      assert.ok(example.model_illustration.width > 0 && example.model_illustration.height > 0, example.id);
    }
  }
});
