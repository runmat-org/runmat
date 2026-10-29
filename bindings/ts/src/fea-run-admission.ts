/** One lease covers the complete local mesh/solve run, including all cases. */
export type FeaRunRequest = {
  runId: string;
  kind: "meshing" | "solve" | "meshAndSolve";
  allocatedCpuMillicores: number;
  estimatedWallSeconds?: number;
  studyIdentity: string;
};

export type FeaRunDenied = {
  reason: "signInRequired" | "allowanceUsed" | "leaseExpired" | "licenseInvalid" | "serverUnreachable";
  refillsAt?: string;
};

export type FeaRunUsage = { cpuCoreMillis: number; wallMillis: number };
export type FeaRunOutcome = "completed" | "cancelled" | "failed";

/**
 * The host validates the Server's signed lease before returning its leaseId as
 * permitId. Heartbeat/settle must journal offline reports for later replay;
 * reporting failures never interrupt an admitted run. Failed runs release their
 * reservation; completed and cancelled runs charge allocated cores × wall time.
 */
export interface FeaRunAdmission {
  admit(request: FeaRunRequest): Promise<{ permitId: string } | { denied: FeaRunDenied }>;
  heartbeat?(permitId: string, usage: FeaRunUsage): Promise<void>;
  settle(permitId: string, outcome: FeaRunOutcome, usage: FeaRunUsage): Promise<void>;
}

/** Context supplied to the owned geometry/numerical host callbacks. Not authorization. */
export interface FeaRunDelegation {
  request: FeaRunRequest;
  permitId: string;
  /** SHA-256 of the exact serialized job, binding authorization to its contents. */
  preparedStudyIdentity: string;
}

/**
 * A standalone worker needs its own host admission. A delegated worker also
 * requires host admission: the adapter must authorize the existing parent lease
 * and return that permitId, without creating a new lease. It must validate the
 * runId, studyIdentity, allocation and exact job digest against the parent job.
 * For delegation the parent alone owns heartbeat/settle and the full wall time.
 */
export interface FeaRunWorkerAdmission {
  admission: FeaRunAdmission;
  delegation?: FeaRunDelegation;
}
