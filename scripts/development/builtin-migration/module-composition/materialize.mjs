import { compareCodePoint } from "../constants.mjs";
import { contentDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { assertActiveLease } from "../lease.mjs";
import { validateReviewedModuleCompositionAuthority } from "./authority.mjs";
import {
  assertEffectiveChildStatesUnchanged, inspectEffectiveChildStates,
} from "./child-state.mjs";
import { deriveModuleCompositionMaterializationState } from "./effective-state.mjs";
import { inspectCanonicalMaterializationTargets } from "./prior-state.mjs";
import {
  inspectMaterializationTargets, materializationChildKey,
} from "./repository-state.mjs";
import { renewCompositionRepositoryLock } from "./transaction-lock.mjs";
import { installCompositionSet, renderCompositionSetTwice, withCompositionTransaction } from "./transaction.mjs";

export function materializeEffectiveModuleComposition({
  repository, control, queueState, queueCheckpoint, lease, clock = Date.now,
}, options = {}) {
  const state = deriveModuleCompositionMaterializationState({
    control, queueState, queueCheckpoint, lease, clock,
  });
  if (state === null) throw new Error("operational module composition requires a reviewed projection");
  validateReviewedModuleCompositionAuthority(control.integrationProducts, state.prior);
  validateReviewedModuleCompositionAuthority(control.integrationProducts, state.effective);
  const active = assertActiveLease(lease, control, clock);
  const productIds = selectedProductIds(control, active.bundle);
  const effectiveById = new Map(state.effective.products
    .map((product) => [product.product_id, product]));
  const priorById = new Map(state.prior.products
    .map((product) => [product.product_id, product]));
  const products = productIds.map((id) => {
    const product = effectiveById.get(id);
    if (!product) throw new Error(`${id}: active integration product is absent from effective composition`);
    return product;
  });
  const priorProducts = productIds.map((id) => {
    const product = priorById.get(id);
    if (!product) throw new Error(`${id}: integration product is absent from prior composition`);
    return product;
  });
  const authorizedPendingChildKeys = deriveAuthorizedPendingChildKeys(
    products, priorById,
  );
  return withCompositionTransaction(repository, (lock, recovery) => {
    const priorPresent = priorProducts.filter((product) => product.state === "present");
    const renderedPriorIds = new Set(state.priorRenderedProductIds);
    const priorRendered = renderCompositionSetTwice(
      priorPresent.filter((product) => renderedPriorIds.has(product.product_id)),
    );
    renewCompositionRepositoryLock(lock);
    const priorDigests = new Map(priorRendered
      .map((entry) => [entry.product.product_id, contentDigest(entry.content)]));
    for (const product of priorPresent) {
      if (priorDigests.has(product.product_id)) continue;
      const reviewed = control.integrationProducts.get(product.product_id);
      if (!reviewed || reviewed.baseline_digest === null) {
        throw new Error(`${product.product_id}: present baseline product lacks its frozen digest`);
      }
      priorDigests.set(product.product_id, reviewed.baseline_digest);
    }
    const proveAuthority = (expectedChildStates = null) => {
      const observed = deriveModuleCompositionMaterializationState({
        control, queueState, queueCheckpoint, lease, clock,
      });
      if (JSON.stringify(observed) !== JSON.stringify(state)) {
        throw new Error("operational module composition authority changed during materialization");
      }
      validateReviewedModuleCompositionAuthority(control.integrationProducts, observed.prior);
      validateReviewedModuleCompositionAuthority(control.integrationProducts, observed.effective);
      const inventory = inspectCanonicalMaterializationTargets(
        lock.repository, priorProducts, priorDigests,
      );
      inspectMaterializationTargets(lock.repository, products, { authorizedPendingChildKeys });
      const childStates = inspectEffectiveChildStates(
        lock.repository, products, { authorizedPendingChildKeys },
      );
      if (expectedChildStates !== null) {
        assertEffectiveChildStatesUnchanged(expectedChildStates, childStates);
      }
      return deepImmutable({ inventory, childStates });
    };
    const initialAuthority = proveAuthority();
    renewCompositionRepositoryLock(lock);
    const { inventory } = initialAuthority;
    options.afterAudit?.({ inventory, lock });
    const affectedIds = new Set(products.filter((product) => {
      const prior = priorById.get(product.product_id);
      return prior.state === "present" || product.state === "present";
    }).map((product) => product.product_id));
    const affected = products.filter((product) => affectedIds.has(product.product_id));
    const renderedPresent = renderCompositionSetTwice(
      affected.filter((product) => product.state === "present"), options.render,
    );
    const renderedById = new Map(renderedPresent
      .map((entry) => [entry.product.product_id, entry]));
    const desired = affected.map((product) => renderedById.get(product.product_id)
      ?? { product, content: null });
    renewCompositionRepositoryLock(lock);
    const before = inventory.filter((entry) => affectedIds.has(entry.product_id));
    const transaction = installCompositionSet(
      lock.repository, desired, before, lock, options.installHooks,
      () => { proveAuthority(initialAuthority.childStates); },
    );
    return { product_ids: productIds, inventory, installed: transaction.installed, transaction: { recovery, cleanup: transaction.cleanup, cleanup_errors: transaction.cleanup_errors } };
  });
}

function deriveAuthorizedPendingChildKeys(products, priorById) {
  const activatedPaths = new Set(products
    .filter((product) => product.state === "present"
      && priorById.get(product.product_id)?.state === "absent")
    .map((product) => product.path));
  const keys = new Set();
  for (const product of products) {
    const priorChildren = new Set((priorById.get(product.product_id)?.children ?? [])
      .map((child) => child.source_path));
    for (const child of product.children) {
      if (activatedPaths.has(child.source_path) && !priorChildren.has(child.source_path)) {
        keys.add(materializationChildKey(product.product_id, child.source_path));
      }
    }
  }
  return keys;
}

function selectedProductIds(control, bundle) {
  const ids = new Set([...control.integrationProducts.values()]
    .filter((product) => product.verification.kind === "rust_module_composition"
      && product.lifecycle.kind === "reviewed-baseline-only")
    .map((product) => product.product_id));
  for (const output of bundle.integration_outputs) {
    const product = control.integrationProducts.get(output.product_id);
    if (!product) throw new Error(`${bundle.id}: active bundle references unknown integration product ${output.product_id}`);
    if (product.verification.kind === "rust_module_composition") ids.add(product.product_id);
  }
  return [...ids].sort(compareCodePoint);
}
