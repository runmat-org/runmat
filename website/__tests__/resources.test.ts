import assert from "node:assert/strict";
import test from "node:test";

import {
  getAllResources,
  getFeaturedResources,
  getResourcesByType,
  getAvailableResourceTypes,
  getRoutableResourceTypes,
  getResourceTypeLink,
  getDisplayResourceTypes,
  getGuidesCollection,
  RESOURCE_TYPES,
} from "@/lib/resources";
import * as curated from "@/content/resources";

test("curated docs and benchmarks resolve", () => {
  const resources = getAllResources();

  for (const entry of curated.curatedDocs) {
    const found = resources.find(
      (r) => r.source === "doc" && r.href.endsWith(entry.slug.replace(/^\/docs\//, ""))
    );
    assert.ok(found, `doc not resolved: ${entry.slug}`);
  }

  for (const entry of curated.curatedBenchmarks) {
    const found = resources.find((r) => r.source === "benchmark" && r.slug === entry.slug);
    assert.ok(found, `benchmark not resolved: ${entry.slug}`);
  }
});

test("resources return for all types without throwing", () => {
  for (const type of RESOURCE_TYPES) {
    assert.doesNotThrow(() => getResourcesByType(type));
  }
});

test("featured resources resolve or warn", () => {
  const featured = getFeaturedResources();
  assert.ok(featured.length > 0, "no featured resources resolved");
});

test("all resources have canonical hrefs", () => {
  const resources = getAllResources();
  for (const item of resources) {
    assert.ok(item.href && item.href.startsWith("/"), `href missing or invalid for ${item.id}`);
    if (item.source === "doc") {
      assert.ok(item.href.startsWith("/docs/"), `doc href must be /docs for ${item.id}`);
    }
    if (item.source === "blog") {
      assert.ok(item.href.startsWith("/blog/"), `blog href must be /blog for ${item.id}`);
    }
    if (item.source === "benchmark") {
      assert.ok(item.href.startsWith("/benchmarks/"), `benchmark href must be /benchmarks for ${item.id}`);
    }
    if (item.source === "resource") {
      assert.ok(item.href.startsWith("/resources/"), `resource href must be /resources for ${item.id}`);
    }
  }
});

test("available types are filtered to types with items", () => {
  const available = getAvailableResourceTypes();
  const all = getAllResources();
  for (const t of available) {
    assert.ok(all.some((r) => r.type === t), `available type ${t} has no items`);
  }
});

test("blogs tile points to canonical blog index", () => {
  const href = getResourceTypeLink("blogs");
  assert.equal(href, "/blog");
});

test("docs and benchmarks tiles point to canonical sections", () => {
  assert.equal(getResourceTypeLink("docs"), "/docs");
  assert.equal(getResourceTypeLink("benchmarks"), "/benchmarks");
});

test("guides tile points to resources when native", () => {
  assert.equal(getResourceTypeLink("guides"), "/resources/guides");
});

test("blog items link to canonical blog posts", () => {
  const all = getAllResources();
  const blogs = all.filter((r) => r.type === "blogs");
  for (const b of blogs) {
    assert.ok(b.href.startsWith("/blog/"), `blog href must be /blog/: ${b.href}`);
    assert.ok(!b.href.startsWith("/resources/"), `blog href must not be /resources: ${b.href}`);
  }
});

test("routable resource types exclude blogs", () => {
  const routable = getRoutableResourceTypes();
  assert.ok(!routable.includes("blogs"), "blogs should not be routable under /resources");
  assert.ok(!routable.includes("docs"), "docs should not be routable under /resources");
  assert.ok(!routable.includes("benchmarks"), "benchmarks should not be routable under /resources");
});

test("display resource types hide case studies and webinars when no destinations", () => {
  const display = getDisplayResourceTypes();
  assert.ok(!display.includes("case-studies"), "case-studies should be hidden until a destination exists");
  assert.ok(!display.includes("webinars"), "webinars should be hidden until a destination exists");
});

test("guides collection includes native guide, blog, and doc entries with canonical hrefs", () => {
  const guides = getGuidesCollection();
  const native = guides.find((g) => g.source === "resource" && g.slug === "quick-start-runmat");
  assert.ok(native, "native guide missing");
  assert.ok(native?.href.startsWith("/resources/guides/"), "native guide href should be /resources/guides");

  const blog = guides.find((g) => g.source === "blog" && g.slug === "in-defense-of-matlab-whiteboard-style-code");
  assert.ok(blog, "guide blog missing");
  assert.ok(blog?.href.startsWith("/blog/"), "guide blog href must be /blog/");

  const doc = guides.find((g) => g.source === "doc" && g.href.startsWith("/docs/desktop-browser-guide"));
  assert.ok(doc, "guide doc missing");
  assert.ok(doc?.href.startsWith("/docs/"), "guide doc href must be /docs/");
});

test("featured resources include featured-flagged items", () => {
  const featured = getFeaturedResources();
  const hasBlog = featured.some((f) => f.source === "blog" && f.featured);
  const hasGuide = featured.some((f) => f.source === "resource" && f.featured);
  assert.ok(hasBlog, "expected a featured blog to appear");
  assert.ok(hasGuide, "expected a featured native guide to appear");
});

test("curated featured entries skip blogs", () => {
  const fromCurated = getFeaturedResources().filter((f) => f.source === "blog");
  // Any blog in featured must come from frontmatter flag
  assert.ok(
    fromCurated.every((f) => f.featured),
    "blogs should only be featured via frontmatter flag, not curated list"
  );
});

