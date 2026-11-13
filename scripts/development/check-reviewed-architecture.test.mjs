import assert from "node:assert/strict";
import test from "node:test";
import { gitBlobDigest, matchesReviewedDebt } from "./check-reviewed-architecture.mjs";

test("reviewed architecture matching preserves exact multiplicity and scope", () => {
  assert.equal(gitBlobDigest("abc"), "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f");
  const a = "architecture boundary check failed: crates/runmat-analysis/fea/src/a.rs has existing debt";
  const b = "architecture boundary check failed: crates/runmat-runtime/src/analysis/b.rs has existing debt";
  const expected = [...Array(106).fill(a), b];
  assert(matchesReviewedDebt([b, ...Array(106).fill(a)], expected));
  assert(!matchesReviewedDebt(Array(107).fill(a), expected));
  assert(!matchesReviewedDebt(expected.slice(1), expected));
  assert(!matchesReviewedDebt([...expected, a], expected));
  assert(!matchesReviewedDebt([], expected));
  const nonFea = "architecture boundary check failed: crates/runmat-runtime/src/builtins/foo.rs exceeds boundary";
  const unapproved = [...expected.slice(1), nonFea];
  assert(!matchesReviewedDebt(unapproved, expected));
  assert(!matchesReviewedDebt(unapproved, unapproved));
  assert(!matchesReviewedDebt(expected, null));
  const refreshed = [...expected, ...Array(4).fill(b)];
  assert(matchesReviewedDebt(refreshed, refreshed));
  assert(!matchesReviewedDebt(refreshed.slice(1), refreshed));
  assert(!matchesReviewedDebt([], []));
});
