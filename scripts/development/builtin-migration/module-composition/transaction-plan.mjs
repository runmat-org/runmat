import crypto from "node:crypto";

import { evidenceDigest } from "../evidence.mjs";
import { moduleCompositionProductRegistry } from "./registry.mjs";
import {
  assertProductStateUnchanged,
  resolveRepositoryProduct,
} from "./repository-state.mjs";

export function planTransactionEntry(root, token, entry, before) {
  assertProductStateUnchanged(root, before);
  const target = resolveRepositoryProduct(root, entry.product.path);
  const desiredState = entry.content === null ? "absent" : "present";
  if (desiredState !== entry.product.state) {
    throw new Error(`${entry.product.product_id}: desired content differs from product state`);
  }
  return {
    ...entry,
    before,
    desired_state: desiredState,
    target,
    temporary: `${target}.runmat-stage-${token}`,
    backup: `${target}.runmat-backup-${token}`,
    new_digest: entry.content === null ? null : digest(entry.content),
  };
}

export function transactionJournalEntry(entry) {
  return {
    product_id: entry.product.product_id,
    path: entry.product.path,
    existed: entry.before.state === "present",
    before_digest: entry.before.content_digest ?? null,
    before_file_identity: entry.before.file_identity ?? null,
    desired_state: entry.desired_state,
    new_digest: entry.new_digest,
  };
}

export function transactionRegistryDigest() {
  return evidenceDigest(moduleCompositionProductRegistry());
}

function digest(value) {
  return `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`;
}
