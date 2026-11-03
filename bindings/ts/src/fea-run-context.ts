/**
 * Exact payload and source binding for a local owned worker. This record is
 * structured-cloneable delegation of an already admitted root lease.
 * The host owns worker creation, cancellation and result delivery; Rust checks
 * this binding and validates the prepared program before executing it.
 */
import type { FeaRunRequest, FeaRunLease } from "./fea-run-admission.js";
export interface LocalFeaWorkerContext {
  request: FeaRunRequest;
  /** Canonical SHA-256 digest of the exact serialized worker payload. */
  preparedStudyIdentity: string;
  lease: FeaRunLease;
  rootStartedAtUnixMillis: number;
  delegationId: string;
}
