import { compareCodePoint } from "../constants.mjs";
import { parseBundleComposition, targetPackageKey } from "./schema.mjs";

export function validateBundleComposition(claim, composition) {
  parseBundleComposition(composition, `${claim.bundle} composition`);
  const targetPackages = [...new Map(claim.identity_targets.map((target) => [
    targetPackageKey(target),
    { domain: target.domain, family: target.family },
  ])).values()].sort((left, right) => compareCodePoint(targetPackageKey(left), targetPackageKey(right)));

  if (!same(composition.target_packages, targetPackages)) {
    throw new Error(`${claim.bundle}: composition target packages must equal the exact identity-target package set`);
  }

  const componentCount = claim.components.length;
  if (componentCount === 1 && targetPackages.length === 1) {
    if (composition.kind !== "single-component") {
      throw new Error(`${claim.bundle}: one-component one-family bundle must use single-component composition`);
    }
    if (composition.shared_authority_sources.length !== 0) {
      throw new Error(`${claim.bundle}: single-component composition cannot claim cross-family shared authority`);
    }
    return composition;
  }
  if (componentCount > 1 && targetPackages.length === 1) {
    if (composition.kind !== "shared-target-package") {
      throw new Error(`${claim.bundle}: independent components sharing one package must use shared-target-package composition`);
    }
    if (composition.authored_write_set.length === 0) {
      throw new Error(`${claim.bundle}: shared-target-package composition requires an explicit authored write set`);
    }
    if (composition.shared_authority_sources.length !== 0) {
      throw new Error(`${claim.bundle}: independent components cannot claim one shared authority source`);
    }
    return composition;
  }
  if (componentCount === 1 && targetPackages.length > 1) {
    if (composition.kind !== "mixed-family-component") {
      throw new Error(`${claim.bundle}: one component crossing package families must use mixed-family-component composition`);
    }
    if (composition.shared_authority_sources.length === 0) {
      throw new Error(`${claim.bundle}: mixed-family component requires exact shared authority source evidence`);
    }
    return composition;
  }
  throw new Error(`${claim.bundle}: multiple independent authority components cannot be combined across target packages`);
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}
