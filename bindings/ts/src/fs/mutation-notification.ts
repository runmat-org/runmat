/** Post-commit observers cannot turn a committed mutation into a failed operation. */
export function notifyCommittedMutation(
  observer?: () => void,
  onObserverError?: (error: unknown) => void
): void {
  try {
    observer?.();
  } catch (error) {
    if (onObserverError) {
      try {
        onObserverError(error);
        return;
      } catch (sinkError) {
        reportObserverFailure(error, sinkError);
        return;
      }
    }
    reportObserverFailure(error);
  }
}

function reportObserverFailure(error: unknown, sinkError?: unknown): void {
  try {
    console.error("RunMat filesystem observer failed after commit", { error, sinkError });
  } catch {
    // A host may replace console.error. Reporting is best effort; its failure
    // cannot roll back an already committed operation or reject that operation.
  }
}
