import {
  parseTrustedReviewedModuleCompositionBaseline,
  validateReviewedModuleCompositionBaseline,
} from "./baseline-authority.mjs";
import { inspectCompositionRepository } from "./repository-state.mjs";
import { renewCompositionRepositoryLock } from "./transaction-lock.mjs";
import { installCompositionSet, renderCompositionSetTwice, withCompositionTransaction } from "./transaction.mjs";

export function bootstrapModuleComposition({
  repository, reviewedBaseline, trustedReviewedBaselineDigest, trustedSignerFingerprint,
}, options = {}) {
  parseTrustedReviewedModuleCompositionBaseline(
    reviewedBaseline, trustedReviewedBaselineDigest,
  );
  return withCompositionTransaction(repository, (lock, recovery) => {
    const validateBaseline = () => validateReviewedModuleCompositionBaseline(
      reviewedBaseline, trustedReviewedBaselineDigest, lock.repository,
      trustedSignerFingerprint, {
        ...options.sourceObservation,
        allowCompositionLock: true,
        allowedCompositionArtifacts: [],
      },
    );
    const baseline = validateBaseline();
    renewCompositionRepositoryLock(lock);
    const reviewed = baseline.projection;
    const inventory = inspectCompositionRepository(lock.repository, reviewed.products);
    renewCompositionRepositoryLock(lock);
    options.afterAudit?.({ inventory, lock });
    for (const observed of inventory.products) {
      const expected = reviewed.products.find((product) => product.product_id === observed.product_id);
      if (observed.state !== expected.state) {
        throw new Error(`${observed.product_id}: repository presence differs from the reviewed baseline`);
      }
    }
    const products = reviewed.products.filter((product) => product.state === "present");
    const rendered = renderCompositionSetTwice(products, options.render);
    renewCompositionRepositoryLock(lock);
    const before = inventory.products.filter((entry) => entry.state === "present");
    validateBaseline();
    renewCompositionRepositoryLock(lock);
    const authorityGuard = ({ repository: root, transactionArtifacts }) => {
      validateReviewedModuleCompositionBaseline(
        reviewedBaseline, trustedReviewedBaselineDigest, root, trustedSignerFingerprint,
        {
          ...options.sourceObservation,
          allowCompositionLock: true,
          allowedCompositionArtifacts: transactionArtifacts,
        },
      );
    };
    const transaction = installCompositionSet(
      lock.repository, rendered, before, lock, options.installHooks, authorityGuard,
    );
    return { inventory, installed: transaction.installed, transaction: { recovery, cleanup: transaction.cleanup, cleanup_errors: transaction.cleanup_errors } };
  });
}
