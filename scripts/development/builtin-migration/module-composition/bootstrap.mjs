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
    const present = new Set(inventory.products.filter((entry) => entry.state === "present").map((entry) => entry.product_id));
    const products = reviewed.products.filter((product) => present.has(product.product_id));
    const rendered = renderCompositionSetTwice(products, options.render);
    renewCompositionRepositoryLock(lock);
    const before = inventory.products.filter((entry) => present.has(entry.product_id));
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
