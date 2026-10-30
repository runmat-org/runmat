/**
 * Exact payload and source binding for a local owned worker. This record is
 * structured-cloneable context, not a billing lease or authorization token.
 * The host owns worker creation, cancellation and result delivery; Rust checks
 * this binding and validates the prepared program before executing it.
 */
export interface LocalFeaWorkerContext {
  request: {
    runId: string;
    kind: "meshing" | "solve" | "meshAndSolve";
    site: "local";
    allocatedCpuMillicores: 1000;
    estimatedWallSeconds?: number | null;
    studyIdentity: string;
  };
  /** Canonical SHA-256 digest of the exact serialized worker payload. */
  preparedStudyIdentity: string;
}
