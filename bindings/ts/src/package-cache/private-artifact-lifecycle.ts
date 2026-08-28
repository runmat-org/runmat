type PrivateArtifactInvalidationListener = () => void;

const listeners = new Set<PrivateArtifactInvalidationListener>();
export const PRIVATE_PACKAGE_INVALIDATION_EVENT =
  "runmat:private-package-invalidate";

export function subscribePrivateArtifactInvalidation(
  listener: PrivateArtifactInvalidationListener
): () => void {
  listeners.add(listener);
  const invalidate = (): void => listener();
  globalThis.addEventListener?.(PRIVATE_PACKAGE_INVALIDATION_EVENT, invalidate);
  return () => {
    listeners.delete(listener);
    globalThis.removeEventListener?.(
      PRIVATE_PACKAGE_INVALIDATION_EVENT,
      invalidate
    );
  };
}

/**
 * Drops every decrypted browser package mount owned by this JavaScript realm.
 * Product hosts call this before logout or organization changes. Worker-hosted
 * resolvers are invalidated by disposing or terminating their owning worker.
 */
export function invalidateBrowserPrivatePackageArtifacts(): void {
  for (const listener of [...listeners]) {
    listener();
  }
}
