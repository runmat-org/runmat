import { evidenceDigest } from "../evidence.mjs";
import { moduleCompositionProductRegistry } from "./registry.mjs";

export function parseTransactionJournal(value) {
  const keys = ["entries", "kind", "phase", "registry_digest", "schema_version", "token"];
  if (value?.schema_version !== 2 || value.kind !== "runmat-module-composition-transaction"
    || value.phase !== "installing" || value.registry_digest !== registryDigest()
    || !/^[0-9]+-[a-f0-9]{24}$/.test(value.token) || !Array.isArray(value.entries)
    || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(keys)) {
    throw new Error("module composition recovery journal is invalid");
  }
  validateRegistryEntries(value.entries);
  return value;
}

export function validateJournalEntry(entry) {
  const keys = ["before_digest", "before_file_identity", "desired_state", "existed", "new_digest", "path", "product_id"];
  const identityValid = entry?.before_file_identity
    && typeof entry.before_file_identity.device === "string"
    && typeof entry.before_file_identity.inode === "string"
    && JSON.stringify(Object.keys(entry.before_file_identity).sort())
      === JSON.stringify(["device", "inode"]);
  const desiredValid = entry?.desired_state === "present"
    ? digestValue(entry.new_digest)
    : entry?.desired_state === "absent" && entry.new_digest === null;
  if (!entry || JSON.stringify(Object.keys(entry).sort()) !== JSON.stringify(keys)
    || typeof entry.product_id !== "string" || typeof entry.path !== "string"
    || typeof entry.existed !== "boolean" || !desiredValid
    || (entry.existed && (!digestValue(entry.before_digest) || !identityValid))
    || (!entry.existed && (entry.before_digest !== null || entry.before_file_identity !== null))
    || (entry.desired_state === "absent" && !entry.existed)) {
    throw new Error("module composition recovery journal entry is invalid");
  }
}

function validateRegistryEntries(entries) {
  const registry = new Map(moduleCompositionProductRegistry()
    .map((entry) => [entry.product_id, entry.path]));
  const productIds = new Set();
  const paths = new Set();
  for (const entry of entries) {
    validateJournalEntry(entry);
    if (registry.get(entry.product_id) !== entry.path) {
      throw new Error("module composition recovery journal product differs from the fixed registry");
    }
    if (productIds.has(entry.product_id) || paths.has(entry.path)) {
      throw new Error("module composition recovery journal contains duplicate products");
    }
    productIds.add(entry.product_id);
    paths.add(entry.path);
  }
}

function digestValue(value) {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}

function registryDigest() {
  return evidenceDigest(moduleCompositionProductRegistry());
}
