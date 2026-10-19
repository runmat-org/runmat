import { compareCodePoint } from "../constants.mjs";
import {
  array,
  digest,
  enumValue,
  exact,
  identity,
  nonempty,
  repositoryPath,
  sourceRevision,
  stableId,
} from "../schema.mjs";

export const COHORT_REVIEW_KIND = "runmat-builtin-topology-cohort-review";
export const COHORT_REVIEW_VERSION = 2;
export const COHORT_REVIEW_AUTHORITY = "reviewer-authored-development-input";
export const TOPOLOGY_PROGRAM = "RM-1064/C00-C07";

const COHORTS = Object.freeze(["C01", "C02", "C03", "C04", "C05", "C06", "C07"]);
const TARGET_CLASSIFICATIONS = Object.freeze(["preserved", "normalization", "semantic-correction", "newly-classified"]);
const PACKAGE_PATH = /^[a-z][a-z0-9_]*(?:\/[a-z][a-z0-9_]*)*$/;

export function parseReviewBaseline(value) {
  exact(value, ["revision", "inventory_digest", "component_graph_digest", "control_draft_digest"], "cohort review baseline");
  sourceRevision(value.revision, "cohort review baseline revision");
  digest(value.inventory_digest, "cohort review inventory digest");
  digest(value.component_graph_digest, "cohort review component graph digest");
  digest(value.control_draft_digest, "cohort review control draft digest");
  return value;
}

export function parseCohort(value, label = "cohort") {
  return enumValue(value, COHORTS, label);
}

export function parseIdentityTarget(value, label) {
  exact(value, ["identity", "domain", "family", "classification", "evidence"], label);
  identity(value.identity, `${label} identity`);
  packagePath(value.domain, `${label} domain`, { nested: false });
  packagePath(value.family, `${label} family`);
  parseTargetMetadata({ classification: value.classification, evidence: value.evidence }, label);
  return value;
}

export function parseTargetMetadata(value, label) {
  exact(value, ["classification", "evidence"], label);
  enumValue(value.classification, TARGET_CLASSIFICATIONS, `${label} classification`);
  canonicalStrings(value.evidence, `${label} evidence`);
  return value;
}

export function parseBundleComposition(value, label) {
  exact(value, ["kind", "target_packages", "authored_write_set", "shared_authority_sources", "evidence"], label);
  enumValue(value.kind, ["single-component", "shared-target-package", "mixed-family-component"], `${label} kind`);
  const targetPackages = array(value.target_packages, `${label} target packages`).map((entry, index) => parseTargetPackage(entry, `${label} target package ${index}`));
  canonicalUnique(targetPackages, targetPackageKey, `${label} target packages`);
  const authoredWriteSet = array(value.authored_write_set, `${label} authored write set`, { empty: true }).map((entry, index) => parseAuthoredScope(entry, `${label} authored write set ${index}`));
  canonicalUnique(authoredWriteSet, authoredScopeKey, `${label} authored write set`);
  canonicalRepositoryPaths(value.shared_authority_sources, `${label} shared authority sources`, { empty: true });
  canonicalStrings(value.evidence, `${label} evidence`);
  return value;
}

export function parseTargetPackage(value, label) {
  exact(value, ["domain", "family"], label);
  packagePath(value.domain, `${label} domain`, { nested: false });
  packagePath(value.family, `${label} family`);
  return value;
}

export function parseAuthoredScope(value, label) {
  exact(value, ["kind", "path"], label);
  enumValue(value.kind, ["file", "tree"], `${label} kind`);
  repositoryPath(value.path, `${label} path`);
  return value;
}

export function parseComponentId(value, label = "authority component") {
  const result = stableId(value, label);
  if (!result.startsWith("component-") || result.length === "component-".length) throw new Error(`${label} must start with component-`);
  return result;
}

export function canonicalIdentities(value, label, { empty = false } = {}) {
  const entries = array(value, label, { empty }).map((entry) => identity(entry, label));
  canonicalUnique(entries, (entry) => entry.toLowerCase(), label);
  return entries;
}

export function canonicalComponentIds(value, label, { empty = false } = {}) {
  const entries = array(value, label, { empty }).map((entry) => parseComponentId(entry, label));
  canonicalUnique(entries, (entry) => entry, label);
  return entries;
}

export function canonicalCohorts(value, label) {
  const entries = array(value, label).map((entry) => parseCohort(entry, label));
  canonicalUnique(entries, (entry) => entry, label);
  return entries;
}

export function canonicalStrings(value, label, { empty = false } = {}) {
  const entries = array(value, label, { empty }).map((entry) => nonempty(entry, label));
  canonicalUnique(entries, (entry) => entry, label);
  return entries;
}

export function canonicalRepositoryPaths(value, label, { empty = false } = {}) {
  const entries = array(value, label, { empty }).map((entry) => repositoryPath(entry, label));
  canonicalUnique(entries, (entry) => entry, label);
  return entries;
}

export function canonicalUnique(values, key, label) {
  const keys = values.map(key);
  if (new Set(keys).size !== keys.length) throw new Error(`${label} must be unique`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error(`${label} must use canonical order`);
  return values;
}

export function targetPackageKey(value) {
  return `${value.domain}/${value.family}`;
}

function authoredScopeKey(value) {
  return `${value.kind}:${value.path}`;
}

function packagePath(value, label, { nested = true } = {}) {
  const result = nonempty(value, label);
  if (!PACKAGE_PATH.test(result) || (!nested && result.includes("/"))) throw new Error(`${label} must be a lowercase package path`);
  return result;
}
