import { moduleCompositionProductRegistry } from "./registry.mjs";
import { companionDefinitions } from "./companions.mjs";
import { legacyTransactionRegistryDigest, transactionRegistryDigest } from "./transaction-plan.mjs";

export function parseTransactionJournal(value) {
  const keys = ["entries", "kind", "phase", "registry_digest", "schema_version", "token"];
  const version = value?.schema_version;
  const registryDigest = version === 3 ? transactionRegistryDigest()
    : version === 2 ? legacyTransactionRegistryDigest() : null;
  if (registryDigest === null || value.kind !== "runmat-module-composition-transaction"
    || value.phase !== "installing" || value.registry_digest !== registryDigest
    || !/^[0-9]+-[a-f0-9]{24}$/.test(value.token) || !Array.isArray(value.entries)
    || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(keys)) {
    throw new Error("module composition recovery journal is invalid");
  }
  validateRegistryEntries(value.entries, version);
  return value;
}

export function validateJournalEntry(entry) {
  const legacyKeys = ["before_digest", "before_file_identity", "desired_state", "existed", "new_digest", "path", "product_id"];
  const sections = entry?.companion_sections;
  const priorSections = entry?.prior_companion_sections;
  const keys = sections === undefined ? legacyKeys
    : [...legacyKeys, "companion_sections", "prior_companion_sections"].sort();
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
    || (sections !== undefined && (!Array.isArray(sections) || !Array.isArray(priorSections)
      || sections.some((section) => !["declarations", "reexports"].includes(section))
      || priorSections.some((section) => !["declarations", "reexports"].includes(section))
      || new Set(sections).size !== sections.length
      || new Set(priorSections).size !== priorSections.length
      || JSON.stringify(sections) !== JSON.stringify([...sections].sort())
      || JSON.stringify(priorSections) !== JSON.stringify([...priorSections].sort())))
    || (entry.desired_state === "absent" && !entry.existed)) {
    throw new Error("module composition recovery journal entry is invalid");
  }
}

function validateRegistryEntries(entries, version) {
  const registry = new Map(moduleCompositionProductRegistry()
    .map((entry) => [entry.product_id, entry.path]));
  if (version === 3) for (const companion of companionDefinitions()) {
    registry.set(`${companion.parent_id}-${companion.section}`, companion.path);
  }
  const productIds = new Set();
  const paths = new Set();
  for (const entry of entries) {
    validateJournalEntry(entry);
    if ((version === 3) !== Object.hasOwn(entry, "companion_sections")) {
      throw new Error("module composition journal entry has the wrong section schema");
    }
    if (registry.get(entry.product_id) !== entry.path) {
      throw new Error("module composition recovery journal product differs from the fixed registry");
    }
    if (productIds.has(entry.product_id) || paths.has(entry.path)) {
      throw new Error("module composition recovery journal contains duplicate products");
    }
    productIds.add(entry.product_id);
    paths.add(entry.path);
  }
  if (version !== 3) return;
  const byId = new Map(entries.map((entry) => [entry.product_id, entry]));
  const companionIds = new Set(companionDefinitions().map((entry) => `${entry.parent_id}-${entry.section}`));
  for (const entry of entries.filter((item) => companionIds.has(item.product_id))) {
    if (entry.companion_sections.length !== 0 || entry.prior_companion_sections.length !== 0) {
      throw new Error("module composition companion cannot own nested sections");
    }
  }
  for (const entry of entries.filter((item) => !companionIds.has(item.product_id))) {
    const allowed = companionDefinitions().filter((companion) => companion.parent_id === entry.product_id)
      .map((companion) => companion.section);
    if (entry.companion_sections.some((section) => !allowed.includes(section))
      || entry.prior_companion_sections.some((section) => !allowed.includes(section))
      || (entry.desired_state === "absent" && entry.companion_sections.length)) {
      throw new Error("module composition parent has unauthorized companion sections");
    }
  }
  for (const companion of companionDefinitions()) {
    const child = entries.find((entry) => entry.product_id === `${companion.parent_id}-${companion.section}`);
    const parent = byId.get(companion.parent_id);
    if (child && !parent) throw new Error("module composition companion must be journaled with its parent");
    if (!parent) continue;
    const required = parent.companion_sections.includes(companion.section);
    const wasPresent = parent.prior_companion_sections.includes(companion.section);
    if (required !== (child?.desired_state === "present")) {
      throw new Error("module composition journal omits or misstates a required companion");
    }
    if (wasPresent !== (child?.existed === true)) {
      throw new Error("module composition journal omits or misstates a prior companion");
    }
    if (child) {
      const childIndex = entries.indexOf(child);
      const parentIndex = entries.indexOf(parent);
      if (child.desired_state === "present" ? childIndex > parentIndex : childIndex < parentIndex) {
        throw new Error("module composition journal has an unsafe companion publication order");
      }
    }
  }
}

function digestValue(value) {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}
