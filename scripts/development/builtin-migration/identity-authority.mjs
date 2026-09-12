import { compareCodePoint } from "./constants.mjs";
import {
  array, enumValue, exact, identity, nonempty, repositoryPath, rustIdentifier, rustModulePath,
} from "./schema.mjs";

export function parsePublicIdentity(value, id) {
  const normalized = identity(id, "public identity control id").toLowerCase();
  if (value?.kind === "primary") {
    exact(value, ["kind", "primary_spelling"], `${id} primary public identity`);
    const primary = parseSpelling(value.primary_spelling, `${id} primary spelling`);
    if (primary.identity !== normalized) throw new Error(`${id}: primary spelling identity must equal its control identity`);
  } else if (value?.kind === "alias") {
    exact(value, ["kind", "alias_spelling", "canonical_identity"], `${id} alias public identity`);
    const alias = parseSpelling(value.alias_spelling, `${id} alias spelling`);
    if (alias.identity !== normalized) throw new Error(`${id}: alias spelling identity must equal its control identity`);
    const target = identity(value.canonical_identity, `${id} canonical identity`).toLowerCase();
    if (target === normalized) throw new Error(`${id}: alias canonical identity cannot be self`);
  } else if (value?.kind === "internal") {
    exact(value, ["kind", "reason", "evidence"], `${id} internal public identity`);
    nonempty(value.reason, `${id} internal public identity reason`);
    const evidence = array(value.evidence, `${id} internal public identity evidence`)
      .map((entry) => nonempty(entry, `${id} internal public identity evidence`));
    canonicalUnique(evidence, (entry) => entry, `${id} internal public identity evidence`);
  } else throw new Error(`${id}: public identity has an unsupported kind`);
  return value;
}

export function parseIdentityForms(value, id) {
  exact(value, ["kind", "callable_spellings", "constant_spellings"], `${id} identity forms`);
  const callable = canonicalSpellings(value.callable_spellings, id, `${id} callable spellings`);
  const constants = canonicalSpellings(value.constant_spellings, id, `${id} constant spellings`);
  const expected = callable.length && constants.length ? "callable_and_constant"
    : callable.length ? "callable" : constants.length ? "constant" : "unobserved";
  if (enumValue(value.kind, ["callable", "constant", "callable_and_constant", "unobserved"], `${id} form kind`) !== expected) {
    throw new Error(`${id}: form kind does not match its exact spelling sets`);
  }
  return value;
}

export function parseImplementationAuthority(value, id) {
  exact(value, ["callable", "constant"], `${id} implementation authority`);
  parseFormImplementationAuthority(value.callable, id, "callable");
  parseFormImplementationAuthority(value.constant, id, "constant");
  return value;
}

function parseFormImplementationAuthority(value, id, form) {
  if (value?.kind === "owned") {
    exact(value, ["kind", "owner_path", "bindings"], `${id} ${form} implementation authority`);
    repositoryPath(value.owner_path, `${id} implementation owner`);
    const bindings = array(value.bindings, `${id} implementation bindings`).map((entry) => {
      if (form === "callable" && entry?.kind === "canonical_binding") {
        exact(entry, ["kind", "function", "variant", "builtin_path", "native_symbol"], `${id} canonical implementation binding`);
        nonempty(entry.variant, `${id} implementation binding variant`);
        const symbol = nonempty(entry.native_symbol, `${id} implementation native symbol`);
        if (symbol !== nativeSymbol(id, entry.variant)) {
          throw new Error(`${id}: implementation native symbol does not encode its exact binding identity`);
        }
      } else if (form === "constant" && entry?.kind === "constant_registration") {
        exact(entry, ["kind", "constant", "builtin_path"], `${id} constant implementation binding`);
        const constant = identity(entry.constant, `${id} implementation constant`).toLowerCase();
        if (constant !== id) throw new Error(`${id}: implementation constant must belong to its identity`);
      } else throw new Error(`${id}: implementation binding has an unsupported kind`);
      if (entry.kind !== "constant_registration") {
        rustIdentifier(entry.function, `${id} implementation binding function`);
      }
      rustModulePath(entry.builtin_path, `${id} implementation builtin path`);
      return entry;
    });
    canonicalUnique(bindings, implementationBindingKey, `${id} implementation bindings`);
    const semanticKeys = bindings.map(implementationRegistrationKey);
    if (new Set(semanticKeys).size !== semanticKeys.length) {
      throw new Error(`${id}: ${form} implementation bindings repeat one semantic registration identity`);
    }
  } else if (value?.kind === "none") {
    exact(value, ["kind", "reason"], `${id} ${form} implementation authority`);
    enumValue(value.reason, ["alias-resolution", `no-${form}-form`], `${id} absent implementation reason`);
  } else throw new Error(`${id}: implementation authority has an unsupported kind`);
}

export function validateIdentityAuthorityGraph(identities) {
  const publicSpellings = new Map();
  const normalizedIdentities = new Set();
  for (const [id, row] of identities) {
    const normalized = identity(id, "identity authority graph id").toLowerCase();
    if (normalizedIdentities.has(normalized)) {
      throw new Error(`identity authority graph ids collide case-insensitively at ${id}`);
    }
    normalizedIdentities.add(normalized);
    for (const spelling of controlledSpellings(row.public_identity)) {
      const folded = spelling.spelling.toLowerCase();
      const prior = publicSpellings.get(folded);
      if (prior && prior !== id) {
        throw new Error(`public spellings collide case-insensitively: ${prior} and ${id}`);
      }
      publicSpellings.set(folded, id);
    }
    validateFormImplementation(id, row);
  }
  for (const [id, row] of identities) {
    if (row.public_identity.kind !== "alias") continue;
    const targetId = row.public_identity.canonical_identity.toLowerCase();
    const target = identities.get(targetId);
    if (!target || target.public_identity.kind !== "primary") {
      throw new Error(`${id}: alias target must be a present primary public identity`);
    }
  }
}

export function primarySpelling(value) {
  return value.kind === "primary" ? value.primary_spelling.spelling
    : value.kind === "alias" ? value.alias_spelling.spelling : null;
}

export function implementationBindings(value) {
  return [value.callable, value.constant]
    .flatMap((authority) => authority.kind === "owned" ? authority.bindings : []);
}

export function implementationOwners(value) {
  return [value.callable, value.constant]
    .filter((authority) => authority.kind === "owned")
    .map((authority) => authority.owner_path);
}

export function observedIdentityForms(sourceRow) {
  const callableSpellings = canonicalValues([
    ...(sourceRow.semantic_authority?.catalog_entries ?? []).map((entry) => entry.identity?.name),
    ...(sourceRow.semantic_authority?.catalog_aliases ?? []).map((entry) => entry.alias?.name),
    ...(sourceRow.semantic_authority?.legacy_functions ?? []).map((entry) => entry.name),
    ...(sourceRow.semantic_authority?.runtime_bindings ?? []).map((entry) => entry.name),
    ...(sourceRow.semantic_authority?.implementation_provenance ?? []).map((entry) => entry.name),
  ]);
  const constantSpellings = canonicalValues([
    ...(sourceRow.semantic_authority?.constants ?? []).map(nameOf),
    ...(sourceRow.semantic_authority?.runtime_constants ?? []).map(nameOf),
  ]);
  return {
    kind: callableSpellings.length && constantSpellings.length ? "callable_and_constant"
      : callableSpellings.length ? "callable"
        : constantSpellings.length ? "constant" : "unobserved",
    callable_spellings: callableSpellings,
    constant_spellings: constantSpellings,
  };
}

function validateFormImplementation(id, row) {
  const callable = row.forms.callable_spellings.length > 0;
  const constant = row.forms.constant_spellings.length > 0;
  if (row.public_identity.kind === "alias") {
    if ([row.implementation.callable, row.implementation.constant]
      .some((authority) => authority.kind !== "none" || authority.reason !== "alias-resolution")) {
      throw new Error(`${id}: alias cannot own an independent implementation`);
    }
    return;
  }
  validateFormPresence(id, "callable", callable, row.implementation.callable);
  validateFormPresence(id, "constant", constant, row.implementation.constant);
}

function validateFormPresence(id, form, present, authority) {
  if (present && authority.kind !== "owned") {
    throw new Error(`${id}: observed ${form} form requires one implementation owner`);
  }
  if (!present && (authority.kind !== "none" || authority.reason !== `no-${form}-form`)) {
    throw new Error(`${id}: absent ${form} form must have an explicit absent authority`);
  }
}

function controlledSpellings(value) {
  if (value.kind === "primary") return [value.primary_spelling];
  if (value.kind === "alias") return [value.alias_spelling];
  return [];
}

function parseSpelling(value, label) {
  exact(value, ["identity", "spelling"], label);
  const id = identity(value.identity, `${label} identity`).toLowerCase();
  const spelling = identity(value.spelling, `${label} spelling`);
  if (spelling.toLowerCase() !== id) throw new Error(`${label} must case-fold to its identity`);
  return { identity: id, spelling };
}

function canonicalSpellings(value, ownerIdentity, label) {
  const entries = array(value, label, { empty: true }).map((entry) => identity(entry, label));
  if (entries.some((entry) => entry.toLowerCase() !== ownerIdentity)) {
    throw new Error(`${label} must case-fold to the identity that owns the form`);
  }
  canonicalUnique(entries, (entry) => entry, label);
  return entries;
}

function canonicalValues(values) {
  return [...new Set(values.filter((value) => typeof value === "string" && value.length > 0))]
    .sort(compareCodePoint);
}

function nameOf(value) { return typeof value === "string" ? value : value?.name; }

function canonicalUnique(values, key, label) {
  const keys = values.map(key);
  if (new Set(keys).size !== keys.length) throw new Error(`${label} must be unique`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error(`${label} must use canonical order`);
  }
}

export function implementationBindingKey(entry) {
  if (entry.kind === "canonical_binding") {
    return `function\0${entry.function}\0${entry.variant}\0${entry.builtin_path}\0${entry.native_symbol}`;
  }
  return `constant\0${entry.constant}\0${entry.builtin_path}`;
}

function implementationRegistrationKey(entry) {
  if (entry.kind === "canonical_binding") return `canonical\0${entry.variant}`;
  return `constant\0${entry.constant}`;
}

function nativeSymbol(name, variant) {
  return `runmat_builtin_binding_v1_${Buffer.from(name).toString("hex")}_${Buffer.from(variant).toString("hex")}`;
}
