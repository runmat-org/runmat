import assert from "node:assert/strict";
import test from "node:test";

import { moduleCompositionProductRegistry } from "../registry.mjs";
import { parseTransactionJournal } from "../transaction-journal-schema.mjs";
import { legacyTransactionRegistryDigest, transactionRegistryDigest } from "../transaction-plan.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;
const ROOT = moduleCompositionProductRegistry().find((entry) => entry.product_id === "catalog-root");
const COMPANION = {
  product_id: "catalog-root-reexports",
  path: "crates/runmat-builtins/src/catalog/entries/exports.rs",
};

function entry(product, companionSections = [], priorSections = []) {
  return {
    product_id: product.product_id, path: product.path, existed: false,
    before_digest: null, before_file_identity: null,
    desired_state: "present", new_digest: DIGEST,
    companion_sections: companionSections, prior_companion_sections: priorSections,
  };
}

function journal(entries) {
  return {
    schema_version: 3, kind: "runmat-module-composition-transaction",
    phase: "installing", registry_digest: transactionRegistryDigest(),
    token: `123-${"b".repeat(24)}`, entries,
  };
}

test("recovery journal binds each required companion to its parent transition", () => {
  const complete = journal([entry(COMPANION), entry(ROOT, ["reexports"])]);
  assert.doesNotThrow(() => parseTransactionJournal(complete));
  assert.throws(() => parseTransactionJournal(journal([entry(ROOT, ["reexports"]), entry(COMPANION)])), /unsafe companion publication order/);
  assert.throws(() => parseTransactionJournal(journal([entry(ROOT, ["reexports"])])), /omits or misstates/);
  assert.throws(() => parseTransactionJournal(journal([entry(COMPANION), entry(ROOT)])), /omits or misstates/);
  assert.throws(() => parseTransactionJournal(journal([entry(COMPANION)])), /journaled with its parent/);
  const wrongPath = structuredClone(complete);
  wrongPath.entries[0].path = "crates/runmat-builtins/src/catalog/entries/unreviewed.rs";
  assert.throws(() => parseTransactionJournal(wrongPath), /fixed registry/);
  const wrongSection = structuredClone(complete);
  wrongSection.entries[1].companion_sections = ["declarations"];
  assert.throws(() => parseTransactionJournal(wrongSection), /unauthorized companion sections/);
  const nested = structuredClone(complete);
  nested.entries[0].companion_sections = ["declarations"];
  assert.throws(() => parseTransactionJournal(nested), /cannot own nested sections/);
  const nestedPrior = structuredClone(complete);
  nestedPrior.entries[0].prior_companion_sections = ["reexports"];
  assert.throws(() => parseTransactionJournal(nestedPrior), /cannot own nested sections/);
  const priorOmitted = journal([{
    ...entry(ROOT, [], ["reexports"]), existed: true,
    before_digest: DIGEST, before_file_identity: { device: "1", inode: "1" },
  }]);
  assert.throws(() => parseTransactionJournal(priorOmitted), /omits or misstates a prior companion/);
});

test("legacy recovery admits only the original no-companion journal grammar", () => {
  const old = journal([entry(ROOT)]);
  old.schema_version = 2;
  old.registry_digest = legacyTransactionRegistryDigest();
  delete old.entries[0].companion_sections;
  delete old.entries[0].prior_companion_sections;
  assert.doesNotThrow(() => parseTransactionJournal(old));
  old.entries.push({ ...entry(COMPANION) });
  delete old.entries[1].companion_sections;
  delete old.entries[1].prior_companion_sections;
  assert.throws(() => parseTransactionJournal(old), /fixed registry/);
});
