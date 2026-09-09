import { sorted } from "./constants.mjs";
import { emptyClassification } from "./records.mjs";

export function emptyDispositionInput() {
  return { schema_version: 1, kind: "runmat-builtin-dispositions", identities: {} };
}

export function buildDispositionSeed(inventory) {
  return {
    schema_version: 1,
    kind: "runmat-builtin-dispositions",
    authority: "review-input-only",
    identities: Object.fromEntries(inventory.identities.map((entry) => [entry.identity, emptyClassification()])),
  };
}

export function validateDispositionInput(input) {
  if (!input || input.schema_version !== 1 || input.kind !== "runmat-builtin-dispositions" || typeof input.identities !== "object" || Array.isArray(input.identities)) {
    throw new Error("Disposition input must use schema_version 1, kind runmat-builtin-dispositions, and an identities object");
  }
  for (const [identity, value] of Object.entries(input.identities)) validateRow(identity, value);
}

export function normalizeDisposition(input) {
  return {
    disposition: input.disposition ?? null,
    canonical: input.canonical?.toLowerCase() ?? null,
    domain: input.domain ?? null,
    family: input.family ?? null,
    reason: input.reason ?? null,
    review: { status: input.review?.status ?? "unreviewed", evidence: sorted(input.review?.evidence ?? []) },
  };
}

export function validateReviewedRelationships(records, diagnostics) {
  for (const item of records.values()) {
    if (item.catalogPaths.size && item.input.disposition && item.input.disposition !== "canonical") {
      diagnostics.push({ severity: "error", code: "disposition-contradicts-catalog", path: sorted(item.catalogPaths)[0], detail: `${item.identity} was reviewed as ${item.input.disposition}` });
    }
    if (item.input.disposition === "alias" && !records.has(item.input.canonical)) {
      diagnostics.push({ severity: "error", code: "dangling-alias", path: "<disposition-input>", detail: `${item.identity} -> ${item.input.canonical}` });
    }
  }
  for (const item of records.values()) {
    if (item.input.disposition !== "alias") continue;
    const seen = new Set([item.identity]);
    let target = item.input.canonical;
    while (target && records.get(target)?.input.disposition === "alias") {
      if (seen.has(target)) {
        diagnostics.push({ severity: "error", code: "alias-cycle", path: "<disposition-input>", detail: [...seen, target].join(" -> ") });
        break;
      }
      seen.add(target);
      target = records.get(target).input.canonical;
    }
  }
}

function validateRow(identity, value) {
  if (!/^[A-Za-z][A-Za-z0-9_.]*$/.test(identity) || !value || typeof value !== "object") throw new Error(`Invalid disposition input for ${identity}`);
  if (!value.review || !["unreviewed", "reviewed"].includes(value.review.status) || !Array.isArray(value.review.evidence)) throw new Error(`${identity}: review must declare status and evidence array`);
  if (value.review.status === "unreviewed") {
    if (value.disposition || value.canonical || value.domain || value.family || value.reason) throw new Error(`${identity}: unreviewed seed rows must not contain classification values`);
    return;
  }
  if (value.review.evidence.length === 0 || value.review.evidence.some((entry) => !String(entry).trim())) throw new Error(`${identity}: reviewed rows require nonempty evidence`);
  if (!["canonical", "alias", "internal"].includes(value.disposition)) throw new Error(`${identity}: disposition must be canonical, alias, or internal`);
  if (value.disposition === "alias" && !value.canonical) throw new Error(`${identity}: aliases require a canonical target`);
  if (value.canonical && !/^[A-Za-z][A-Za-z0-9_.]*$/.test(value.canonical)) throw new Error(`${identity}: canonical target is not a safe identity`);
  for (const field of ["domain", "family"]) {
    if (value[field] && !/^[a-z][a-z0-9_]*(?:\/[a-z][a-z0-9_]*)*$/.test(value[field])) throw new Error(`${identity}: ${field} must be a lowercase path of Rust identifiers`);
  }
  if (value.disposition === "internal" && !String(value.reason ?? "").trim()) throw new Error(`${identity}: internal dispositions require a reviewed reason`);
  if (value.disposition !== "alias" && value.canonical) throw new Error(`${identity}: canonical target is only valid for aliases`);
}
