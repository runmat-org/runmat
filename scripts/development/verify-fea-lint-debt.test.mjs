import assert from "node:assert/strict";
import test from "node:test";
import { diagnosticInventory, matchesDebt } from "./verify-fea-lint-debt.mjs";

test("debt matching retains source anchors and emission counts across line shifts", () => {
  const record = { reason: "compiler-message", message: { level: "error",
    code: { code: "clippy::needless_borrow" }, message: "needless borrow",
    spans: [{ is_primary: true, file_name: "crates/runmat-analysis/fea/src/example.rs",
      line_start: 10, text: [{ text: "    &value" }] }] } };
  const original = diagnosticInventory(JSON.stringify(record));
  record.message.spans[0].line_start = 20;
  const shifted = JSON.stringify(record);
  assert(matchesDebt(diagnosticInventory(shifted), original));
  assert(!matchesDebt(diagnosticInventory(`${shifted}\n${shifted}`), original));
  record.message.spans[0].text[0].text = "&other";
  assert(!matchesDebt(diagnosticInventory(JSON.stringify(record)), original));
  record.message.code.code = "E0308";
  assert.throws(() => diagnosticInventory(JSON.stringify(record)), /Unreviewed/);
  assert.throws(() => diagnosticInventory("truncated JSON"));
});
