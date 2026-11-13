/**
 * Exact payload and source binding for a local owned worker. This record is
 * structured-cloneable binding to a child grant. It starts with zero capacity;
 * the parent retains the disjoint reservation and grants batches explicitly.
 * The host owns worker creation, cancellation and result delivery; Rust checks
 * this binding and validates the prepared program before executing it.
 */
import type { FeaRunRequest, FeaRunLease } from "./fea-run-admission.js";
export type FeaWorkerWorkClass = "numerical" | "occtPreparation";
export interface LocalFeaWorkerContext {
  request: FeaRunRequest;
  /** Canonical SHA-256 digest of the exact serialized worker payload. */
  preparedStudyIdentity: string;
  lease: FeaRunLease;
  rootStartedAtUnixMillis: number;
  delegationId: string;
  /** Parent-assigned rsua-1 measurement domain; the child cannot change it. */
  workClass: FeaWorkerWorkClass;
}
