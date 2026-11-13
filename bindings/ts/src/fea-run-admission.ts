import { emptySimulationUsage, parseCompletedSimulationUsage, simulationOperationCount, simulationUsageCategories, simulationUsageTotal, validateSimulationProgress, type CompletedSimulationUsage, type SimulationOperationCount } from "./simulation-usage.js";
import { parseSimulationLedgerError, type SimulationLedgerError } from "./simulation-usage.js";
export { simulationOperationCount } from "./simulation-usage.js";
export type { CompletedSimulationUsage, SimulationOperationCount } from "./simulation-usage.js";
export interface FeaRunRequest {
  runId: string;
  kind: "meshing" | "solve" | "meshAndSolve";
  site: "local";
  allocatedCpuMillicores: number;
  estimatedWallSeconds?: number | null;
  studyIdentity: string;
  measurementVersion: string;
}
export interface FeaRunUsage { completedOps: CompletedSimulationUsage }
export interface FeaRunSettlement { outcome: FeaRunOutcome; usage: FeaRunUsage }
export interface FeaRunWorkerReport { phase: "start" | "finish"; usage: FeaRunUsage }
export interface FeaRunWorkerGrantRequest { requiredOps: SimulationOperationCount; usage: FeaRunUsage }
export interface FeaWorkerAssignment { preparedStudyIdentity: string; workClass: import("./fea-run-context.js").FeaWorkerWorkClass }
export type FeaRunOutcome = "completed" | "cancelled" | "failed";
export interface FeaRunLeaseRenewal { state: "active"; budgetOps: SimulationOperationCount; grantedOps: SimulationOperationCount; expiresAt: string }
export interface FeaRunLease extends FeaRunLeaseRenewal { leaseId: string; heartbeatSeconds: number }
export interface FeaRunDenied {
  reason: "signInRequired" | "allowanceUsed" | "leaseExpired" | "licenseInvalid" | "serverUnreachable" | "allowanceUnpublished" | "budgetUnavailable" | "accountingInvalid";
  refillsAt?: string;
  availableOps?: SimulationOperationCount;
  cause?: SimulationLedgerError;
}
export type FeaRunStopReason = FeaRunDenied | { reason: "cancelled" | "completed" };
/** Issued by the UI worker coordinator only after this generation is gone. */
export interface FeaRunRetirementReceipt {
  runId: string;
  delegationId: string | null;
  workerId: string | null;
  retired: true;
}
/** Implement on the page/UI coordinator, never the busy numerical worker.
 * stopOwnedWork waits for this request's mesh/solve workers to terminate.
 */
export interface FeaRunAdmission {
  start(request: FeaRunRequest): Promise<FeaRunLease>;
  heartbeat(leaseId: string, usage: FeaRunUsage): Promise<FeaRunLeaseRenewal>;
  /** Absolute root grant requested before an additional numerical batch. */
  requestGrant(leaseId: string, requiredOps: SimulationOperationCount, usage: FeaRunUsage): Promise<FeaRunLeaseRenewal>;
  settle(leaseId: string, outcome: FeaRunOutcome, usage: FeaRunUsage): Promise<void>;
  stopOwnedWork(request: FeaRunRequest, reason: FeaRunStopReason, delegationId: string | null): Promise<FeaRunRetirementReceipt>;
}
export class FeaRunAdmissionError extends Error {
  readonly code: string;
  /** Trusted supervisor receipt; preserve this field when proxying an error. */
  workersRetired?: boolean;
  constructor(readonly denied: FeaRunDenied, cause?: unknown) {
    super(denied.reason === "signInRequired" ? "Sign in to RunMat to mesh or solve."
      : denied.reason === "allowanceUsed" ? `You've used your simulation allowance.${denied.refillsAt ? ` It refills on ${denied.refillsAt}.` : ""} See https://runmat.com/pricing`
      : denied.reason === "serverUnreachable" ? "Meshing and solving need a connection to RunMat Server."
      : denied.reason === "allowanceUnpublished" ? "The simulation allowance for this RunMat version has not been published."
      : denied.reason === "budgetUnavailable" ? "The requested simulation spending cap exceeds your available allowance."
      : denied.reason === "accountingInvalid" ? "Simulation usage could not be validated."
      : "Your FEA lease needs refreshing. Sign in to RunMat, then run the study again.", { cause });
    this.name = "FeaRunAdmissionError";
    const originalCode = record(cause)?.["code"];
    this.code = denied.reason === "accountingInvalid" && (originalCode === "RM.FEA.BudgetReached" || originalCode === "RM.FEA.GrantRequired") ? originalCode
      : denied.reason === "accountingInvalid" && typeof denied.cause === "object" && denied.cause !== null && "BudgetExceeded" in denied.cause ? "RM.FEA.BudgetReached"
      : denied.reason === "accountingInvalid" && typeof denied.cause === "object" && denied.cause !== null && "MoreGrantRequired" in denied.cause ? "RM.FEA.GrantRequired"
      : denialCodes[denied.reason];
  }
}
const denialCodes: Readonly<Record<FeaRunDenied["reason"], string>> = {
  signInRequired: "RM.FEA.SignInRequired", allowanceUsed: "RM.FEA.AllowanceUsed",
  leaseExpired: "RM.FEA.LeaseInvalid", licenseInvalid: "RM.FEA.LeaseInvalid",
  serverUnreachable: "RM.FEA.ServerUnreachable", allowanceUnpublished: "RM.FEA.AllowanceUnpublished",
  budgetUnavailable: "RM.FEA.BudgetUnavailable", accountingInvalid: "RM.FEA.AccountingInvalid",
};
function denialReason(value: unknown): value is FeaRunDenied["reason"] {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(denialCodes, value);
}
const record = (value: unknown): Record<string, unknown> | undefined => value !== null && typeof value === "object" ? value as Record<string, unknown> : undefined;
/** Accept typed denials or the server's standard {status, body:{error}} failure. */
export function feaRunAdmissionError(error: unknown): FeaRunAdmissionError {
  if (error instanceof FeaRunAdmissionError) return error;
  const value = record(error), denied = record(value?.["denied"]);
  if (denialReason(denied?.["reason"])) {
    try {
      return new FeaRunAdmissionError({ reason: denied["reason"], ...(typeof denied["refillsAt"] === "string" ? { refillsAt: denied["refillsAt"] } : {}), ...(typeof denied["availableOps"] === "string" ? { availableOps: simulationOperationCount(denied["availableOps"]) } : {}), ...(denied["reason"] === "accountingInvalid" && denied["cause"] !== undefined ? { cause: parseSimulationLedgerError(denied["cause"]) } : {}) }, error);
    } catch { return new FeaRunAdmissionError({ reason: "licenseInvalid" }, error); }
  }
  const status = value?.["status"] ?? value?.["statusCode"], body = record(value?.["body"]) ?? value, detail = record(body?.["error"]);
  if (status === 401) return new FeaRunAdmissionError({ reason: "signInRequired" }, error);
  if (status === 409 && detail?.["code"] === "simulation.allowance_used") {
    const refillsAt = record(detail["details"])?.["refillsAt"];
    return new FeaRunAdmissionError({ reason: "allowanceUsed", ...(typeof refillsAt === "string" ? { refillsAt } : {}) }, error);
  }
  if (status === 409 && detail?.["code"] === "simulation.allowance_unpublished") return new FeaRunAdmissionError({ reason: "allowanceUnpublished" }, error);
  if (status === 409 && detail?.["code"] === "simulation.budget_exceeds_available") {
    const available = record(detail["details"])?.["availableOps"];
    try { return new FeaRunAdmissionError({ reason: "budgetUnavailable", ...(typeof available === "string" ? { availableOps: simulationOperationCount(available) } : {}) }, error); }
    catch { return new FeaRunAdmissionError({ reason: "licenseInvalid" }, error); }
  }
  if (status === 404) return new FeaRunAdmissionError({ reason: "leaseExpired" }, error);
  if (status === 403 || status === 422 || status === 409) return new FeaRunAdmissionError({ reason: "licenseInvalid" }, error);
  return new FeaRunAdmissionError({ reason: "serverUnreachable" }, error);
}
const positiveInteger = (value: number) => Number.isSafeInteger(value) && value > 0;
function validateRenewal(value: FeaRunLeaseRenewal): void {
  if (!value || value.state !== "active" || !Number.isFinite(Date.parse(value.expiresAt)) || Date.parse(value.expiresAt) <= Date.now()) throw new FeaRunAdmissionError({ reason: "leaseExpired" });
  try {
    if (BigInt(simulationOperationCount(value.budgetOps)) === 0n || BigInt(simulationOperationCount(value.grantedOps)) > BigInt(value.budgetOps)) throw new RangeError("invalid grant");
  } catch { throw new FeaRunAdmissionError({ reason: "licenseInvalid" }); }
}
export interface FeaRunLeaseState { lease: FeaRunLease; rootStartedAtUnixMillis: number }
export interface FeaRunWorkerLeaseState extends FeaRunLeaseState { delegationId: string }
export interface FeaRunWorkerAdmission {
  /** Revalidate against the trusted parent's admitted-root cache, not these JSON fields. */
  validateWorker(context: import("./fea-run-context.js").LocalFeaWorkerContext, report: FeaRunWorkerReport): Promise<FeaRunLeaseState>;
  /** Await parent-owned capacity before a numerical batch reserves any work. */
  acquireWorkerGrant(context: import("./fea-run-context.js").LocalFeaWorkerContext, request: FeaRunWorkerGrantRequest): Promise<FeaRunLeaseState>;
}
/** The parent WASM session owns the disjoint ledger reservation. This port must
 * remain on the UI coordinator; the worker receives only the admission proxy.
 */
export interface FeaRunWorkerOwner {
  delegate(context: import("./fea-run-context.js").LocalFeaWorkerContext): Promise<FeaRunLeaseState>;
  start(context: import("./fea-run-context.js").LocalFeaWorkerContext): Promise<void>;
  acquire(context: import("./fea-run-context.js").LocalFeaWorkerContext, requiredOps: SimulationOperationCount): Promise<FeaRunLeaseState>;
  report(context: import("./fea-run-context.js").LocalFeaWorkerContext, usage: FeaRunUsage, finalReport: boolean): Promise<FeaRunUsage>;
  /** Called only after stopOwnedWork proves physical retirement. */
  retire(context: import("./fea-run-context.js").LocalFeaWorkerContext, receipt: FeaRunRetirementReceipt): Promise<void>;
  discardUndispatched(context: import("./fea-run-context.js").LocalFeaWorkerContext, receipt: FeaRunRetirementReceipt): Promise<void>;
}
interface ActiveLease extends FeaRunLeaseState {
  request: FeaRunRequest;
  usage: FeaRunUsage;
  heartbeat?: ReturnType<typeof setTimeout>;
  deadline?: ReturnType<typeof setTimeout>;
  denial?: FeaRunAdmissionError;
  stopped?: Promise<void>;
  settlement?: Promise<void>;
  finalReport?: FeaRunSettlement;
  generation: number;
  worker?: { id: string; payload: string; phase: "delegating" | "available" | "running" | "finishing" | "retiring"; context: import("./fea-run-context.js").LocalFeaWorkerContext; lease: FeaRunLease; completed: CompletedSimulationUsage; retirement?: Promise<void> };
}
/** Create on the UI thread and proxy these methods to a runtime coordinator if
 * needed. This owner must outlive the root; numerical workers receive data only.
 */
export function createFeaRunAdmissionHook(host: FeaRunAdmission, cancelExecution: () => void, onRenew: (runId: string, lease: FeaRunLease) => void) {
  const workerGlobal = (globalThis as { WorkerGlobalScope?: { prototype: object } }).WorkerGlobalScope;
  if (workerGlobal && workerGlobal.prototype.isPrototypeOf(globalThis)) throw new Error("FEA lease supervision must run on the page/UI coordinator, not a busy worker.");
  const active = new Map<string, ActiveLease>();
  let owner: FeaRunWorkerOwner | undefined;
  let starting = false;
  const usage = (run: ActiveLease): FeaRunUsage => structuredClone(run.usage);
  const reportUsage = (run: ActiveLease, report: FeaRunUsage) => {
    try {
      const next = parseCompletedSimulationUsage(report.completedOps);
      validateSimulationProgress(run.usage.completedOps, next, run.lease.grantedOps);
      run.usage = { completedOps: next };
    } catch (error) { throw new FeaRunAdmissionError({ reason: "accountingInvalid" }, error); }
  };
  const clear = (run: ActiveLease) => { clearTimeout(run.heartbeat); clearTimeout(run.deadline); };
  const requireRetirement = (run: ActiveLease, delegationId: string | null, receipt: FeaRunRetirementReceipt): void => {
    if (!receipt || receipt.retired !== true || receipt.runId !== run.request.runId || receipt.delegationId !== delegationId || (receipt.workerId !== null && (typeof receipt.workerId !== "string" || !receipt.workerId || receipt.workerId.length > 256))) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
    if (run.worker && run.worker.id === delegationId && run.worker.phase !== "available" && run.worker.phase !== "retiring" && receipt.workerId === null) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
  };
  const retireReportedWorker = async (run: ActiveLease, worker: NonNullable<ActiveLease["worker"]>, receipt: FeaRunRetirementReceipt): Promise<void> => {
    if (!owner) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
    if (worker.phase === "available" || worker.phase === "delegating" || worker.phase === "retiring") {
      await owner.discardUndispatched(worker.context, receipt);
      return;
    }
    if (worker.phase === "running" && worker.context.workClass === "occtPreparation") {
      // The physical worker is gone. OCCT preparation has no billable rsua-1
      // reference work even if its C++ execution ended before a final reply.
      const zero = emptySimulationUsage();
      const authoritative = await owner.report(worker.context, { completedOps: zero }, true);
      worker.completed = zero;
      reportUsage(run, authoritative);
      worker.phase = "finishing";
    }
    await owner.retire(worker.context, receipt);
  };
  const stop = (run: ActiveLease, error: unknown): Promise<void> => {
    run.denial ??= feaRunAdmissionError(error);
    clear(run);
    if (!run.stopped) {
      // A disposed/coordinator RPC must not prevent physical worker retirement.
      try { cancelExecution(); } catch { /* stopOwnedWork remains authoritative. */ }
      run.stopped = Promise.resolve().then(async () => {
        const worker = run.worker;
        const receipt = await host.stopOwnedWork(run.request, run.denial!.denied, worker?.id ?? null);
        requireRetirement(run, worker?.id ?? null, receipt);
        if (run.worker !== worker) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
        if (worker) {
          await retireReportedWorker(run, worker, receipt);
          delete run.worker;
        }
      }).catch(error => { delete run.stopped; throw error; });
      void run.stopped.catch(() => {});
    }
    return run.stopped;
  };
  const armDeadline = (run: ActiveLease) => {
    clearTimeout(run.deadline);
    const remaining = Date.parse(run.lease.expiresAt) - Date.now();
    run.deadline = setTimeout(() => { void stop(run, new FeaRunAdmissionError({ reason: "leaseExpired" })); }, Math.max(0, Math.min(remaining, 2_147_483_647)));
  };
  const armHeartbeat = (run: ActiveLease) => {
    run.heartbeat = setTimeout(() => {
      void (async () => {
        try {
          const renewal = await host.heartbeat(run.lease.leaseId, usage(run));
          if (run.settlement || run.denial) return;
          validateRenewal(renewal);
          if (renewal.budgetOps !== run.lease.budgetOps || BigInt(renewal.grantedOps) < BigInt(run.lease.grantedOps) || Date.parse(renewal.expiresAt) < Date.parse(run.lease.expiresAt)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
          run.lease = { ...run.lease, ...renewal };
          onRenew(run.request.runId, { ...run.lease });
          armDeadline(run); armHeartbeat(run);
        } catch (error) { await stop(run, error); }
      })().catch(() => {});
    }, run.lease.heartbeatSeconds * 1000);
  };
  const lookup = (request: FeaRunRequest): ActiveLease => {
    const run = active.get(request.runId);
    if (!run || run.request.runId !== request.runId || run.request.kind !== request.kind || run.request.site !== request.site || run.request.allocatedCpuMillicores !== request.allocatedCpuMillicores || run.request.studyIdentity !== request.studyIdentity || run.request.measurementVersion !== request.measurementVersion || (run.request.estimatedWallSeconds ?? null) !== (request.estimatedWallSeconds ?? null)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
    return run;
  };
  const retireFinished = async (run: ActiveLease): Promise<void> => {
    const worker = run.worker;
    if (worker?.phase !== "finishing") return;
    // The parent calls this only after its worker RPC has returned the bytes.
    // Normal retirement must preserve those bytes and the successful root.
    worker.retirement ??= (async () => {
      const receipt = await host.stopOwnedWork(run.request, { reason: "completed" }, worker.id);
      requireRetirement(run, worker.id, receipt);
      if (!owner) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      await owner.retire(worker.context, receipt);
    })().catch(error => { delete worker.retirement; throw error; });
    await worker.retirement;
    if (run.worker === worker) delete run.worker;
  };
  const check = async (request: FeaRunRequest, report?: FeaRunUsage): Promise<FeaRunLeaseState> => {
    const run = lookup(request);
    if (run.settlement) throw new FeaRunAdmissionError({ reason: "leaseExpired" });
    if (report) reportUsage(run, report);
    if (!run.denial && Date.now() >= Date.parse(run.lease.expiresAt)) await stop(run, new FeaRunAdmissionError({ reason: "leaseExpired" }));
    if (run.denial) { await run.stopped; throw run.denial; }
    return { lease: { ...run.lease }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis };
  };
  const retireUndispatched = async (request: FeaRunRequest): Promise<void> => {
    const run = lookup(request), worker = run.worker;
    if (!worker || (worker.phase !== "available" && worker.phase !== "retiring") || !owner) return;
    worker.phase = "retiring";
    const receipt = await host.stopOwnedWork(run.request, { reason: "cancelled" }, worker.id);
    requireRetirement(run, worker.id, receipt);
    await owner.discardUndispatched(worker.context, receipt);
    if (run.worker === worker) delete run.worker;
  };
  return {
    bindOwner(port: FeaRunWorkerOwner): void {
      if (active.size !== 0 || owner) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      owner = port;
    },
    async retryPendingSettlement(runId: string): Promise<void> {
      const run = active.get(runId);
      if (!run?.finalReport || run.settlement) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      return this.settle(run.request, run.finalReport);
    },
    async start(request: FeaRunRequest): Promise<FeaRunLeaseState> {
      if (starting || active.size !== 0) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      starting = true;
      try {
        if (!positiveInteger(request.allocatedCpuMillicores) || request.allocatedCpuMillicores > 4_096_000 || typeof request.measurementVersion !== "string" || !request.measurementVersion || request.measurementVersion.length > 64) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
        const lease = await host.start(request);
        validateRenewal(lease);
        if (typeof lease.leaseId !== "string" || lease.leaseId.length === 0 || lease.leaseId.length > 256 || !positiveInteger(lease.heartbeatSeconds) || lease.heartbeatSeconds > 120) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
        const run: ActiveLease = { request: structuredClone(request), lease: { ...lease }, usage: { completedOps: emptySimulationUsage() }, rootStartedAtUnixMillis: Date.now(), generation: 0 };
        active.set(request.runId, run); armDeadline(run); armHeartbeat(run);
        return { lease: { ...run.lease }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis };
      } catch (error) { throw feaRunAdmissionError(error); }
      finally { starting = false; }
    },
    check,
    async retireFinished(request: FeaRunRequest): Promise<void> {
      await retireFinished(lookup(request));
    },
    retireUndispatched,
    async grant(request: FeaRunRequest, requiredOps: SimulationOperationCount): Promise<FeaRunLeaseState> {
      const run = lookup(request), required = simulationOperationCount(requiredOps);
      await check(request);
      try {
        const renewal = await host.requestGrant(run.lease.leaseId, required, usage(run));
        if (run.denial || run.settlement) throw new FeaRunAdmissionError({ reason: "leaseExpired" });
        validateRenewal(renewal);
        if (renewal.budgetOps !== run.lease.budgetOps || BigInt(renewal.grantedOps) < BigInt(run.lease.grantedOps) || Date.parse(renewal.expiresAt) < Date.parse(run.lease.expiresAt)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
        if (BigInt(renewal.grantedOps) < BigInt(required)) throw new FeaRunAdmissionError({ reason: "accountingInvalid", cause: { MoreGrantRequired: { required, granted: renewal.grantedOps } } });
        run.lease = { ...run.lease, ...renewal };
        onRenew(run.request.runId, { ...run.lease });
        armDeadline(run);
        return { lease: { ...run.lease }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis };
      } catch (error) { throw feaRunAdmissionError(error); }
    },
    async delegate(request: FeaRunRequest, assignment: FeaWorkerAssignment): Promise<FeaRunWorkerLeaseState> {
      const run = lookup(request);
      await retireFinished(run);
      if (!owner) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      if (!assignment || typeof assignment !== "object") throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      const { preparedStudyIdentity, workClass } = assignment;
      if (!/^sha256:[a-f0-9]{64}$/.test(preparedStudyIdentity) || (workClass !== "numerical" && workClass !== "occtPreparation")
        || (run.worker && (run.worker.phase !== "available" || run.worker.payload !== preparedStudyIdentity || run.worker.context.workClass !== workClass))) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      const state = await check(request);
      if (!run.worker) {
        const id = `${run.request.runId}:worker:${++run.generation}:${crypto.randomUUID()}`;
        const context = { request: structuredClone(run.request), preparedStudyIdentity, lease: { ...run.lease, grantedOps: simulationOperationCount("0") }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis, delegationId: id, workClass };
        const worker: NonNullable<ActiveLease["worker"]> = { id, payload: preparedStudyIdentity, phase: "delegating", context, lease: context.lease, completed: emptySimulationUsage() };
        run.worker = worker;
        try {
          const child = await owner.delegate(context);
          if (child.lease.leaseId !== state.lease.leaseId || child.rootStartedAtUnixMillis !== state.rootStartedAtUnixMillis || child.lease.grantedOps !== "0" || child.lease.budgetOps !== state.lease.budgetOps) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
          if (run.worker !== worker || run.denial || run.settlement) throw new FeaRunAdmissionError({ reason: "leaseExpired" });
          worker.lease = child.lease;
          worker.phase = "available";
        } catch (error) {
          worker.phase = "retiring";
          const receipt = await host.stopOwnedWork(run.request, { reason: "cancelled" }, id);
          requireRetirement(run, id, receipt);
          await owner.discardUndispatched(context, receipt);
          if (run.worker === worker) delete run.worker;
          throw error;
        }
      }
      if (!Number.isSafeInteger(run.generation)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      return { lease: { ...run.worker.lease }, rootStartedAtUnixMillis: state.rootStartedAtUnixMillis, delegationId: run.worker.id };
    },
    async acquireWorkerGrant(context: import("./fea-run-context.js").LocalFeaWorkerContext, request: FeaRunWorkerGrantRequest): Promise<FeaRunLeaseState> {
      const run = lookup(context.request), worker = run.worker;
      if (!owner || !worker || worker.phase !== "running" || worker.id !== context.delegationId || worker.payload !== context.preparedStudyIdentity || context.workClass !== "numerical" || worker.context.workClass !== "numerical" || context.lease.leaseId !== run.lease.leaseId || context.rootStartedAtUnixMillis !== run.rootStartedAtUnixMillis) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      await check(context.request);
      const required = simulationOperationCount(request.requiredOps);
      if (BigInt(required) > BigInt(run.lease.budgetOps)) throw new FeaRunAdmissionError({ reason: "accountingInvalid", cause: { BudgetExceeded: { required, budget: run.lease.budgetOps } } });
      let completed: CompletedSimulationUsage;
      try {
        completed = parseCompletedSimulationUsage(request.usage.completedOps);
        validateSimulationProgress(worker.completed, completed, worker.lease.grantedOps);
      } catch (error) { throw new FeaRunAdmissionError({ reason: "accountingInvalid" }, error); }
      const authoritative = await owner.report(worker.context, { completedOps: completed }, false);
      worker.completed = completed;
      reportUsage(run, authoritative);
      if (run.worker !== worker || worker.phase !== "running") throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      const state = await owner.acquire(worker.context, required);
      await check(context.request);
      if (run.worker !== worker || worker.phase !== "running") throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      if (state.lease.leaseId !== run.lease.leaseId || state.rootStartedAtUnixMillis !== run.rootStartedAtUnixMillis || state.lease.budgetOps !== run.lease.budgetOps || state.lease.heartbeatSeconds !== worker.lease.heartbeatSeconds || BigInt(state.lease.grantedOps) < BigInt(required) || BigInt(state.lease.grantedOps) < BigInt(worker.lease.grantedOps) || Date.parse(state.lease.expiresAt) < Date.parse(worker.lease.expiresAt)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      worker.lease = { ...state.lease };
      return { lease: { ...worker.lease }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis };
    },
    async validateWorker(context: import("./fea-run-context.js").LocalFeaWorkerContext, report: FeaRunWorkerReport): Promise<FeaRunLeaseState> {
      const phase = report.phase;
      const run = lookup(context.request);
      const worker = run.worker;
      if (!worker || worker.id !== context.delegationId || worker.payload !== context.preparedStudyIdentity || worker.context.workClass !== context.workClass || context.lease.leaseId !== run.lease.leaseId || context.rootStartedAtUnixMillis !== run.rootStartedAtUnixMillis || (phase !== "start" && phase !== "finish") || worker.phase !== (phase === "start" ? "available" : "running") || !owner) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      // Claim before awaiting so two starts cannot spend the same allocation.
      let completed: CompletedSimulationUsage;
      try { completed = parseCompletedSimulationUsage(report.usage.completedOps); }
      catch (error) { throw new FeaRunAdmissionError({ reason: "accountingInvalid" }, error); }
      if (worker.context.workClass === "occtPreparation" && simulationUsageTotal(completed) !== 0n) throw new FeaRunAdmissionError({ reason: "accountingInvalid" });
      if (phase === "start" && simulationUsageTotal(completed) !== 0n) throw new FeaRunAdmissionError({ reason: "accountingInvalid" });
      if (phase === "finish") {
        try { validateSimulationProgress(worker.completed, completed, worker.lease.grantedOps); }
        catch (error) { throw new FeaRunAdmissionError({ reason: "accountingInvalid" }, error); }
        worker.phase = "finishing";
        let authoritative: FeaRunUsage;
        try { authoritative = await owner.report(worker.context, { completedOps: completed }, true); }
        catch (error) { worker.phase = "running"; throw error; }
        worker.completed = completed;
        reportUsage(run, authoritative);
      } else {
        worker.phase = "running";
        await owner.start(worker.context);
      }
      const state = await check(context.request);
      worker.lease = { ...worker.lease, expiresAt: state.lease.expiresAt };
      return { lease: { ...worker.lease }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis };
    },
    async settle(request: FeaRunRequest, report: FeaRunSettlement): Promise<void> {
      const run = lookup(request);
      const outcome = report.outcome;
      if (outcome !== "completed" && outcome !== "cancelled" && outcome !== "failed") throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      if (run.finalReport) {
        const completed = parseCompletedSimulationUsage(report.usage.completedOps);
        if (outcome !== run.finalReport.outcome || simulationUsageCategories.some(category => completed[category] !== run.finalReport!.usage.completedOps[category])) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
        if (run.settlement) return run.settlement;
      } else {
        reportUsage(run, report.usage);
        run.finalReport = { outcome, usage: usage(run) };
      }
      if (run.worker?.phase === "available" || run.worker?.phase === "retiring") await retireUndispatched(request);
      if (run.worker?.phase === "running" && !run.stopped && outcome === "completed") {
        // A root cannot retire its hold while a claimed worker still runs.
        void stop(run, new FeaRunAdmissionError({ reason: "licenseInvalid" }));
      }
      let settledRemote = false;
      const attempt = (async () => {
        if ((outcome === "cancelled" || (outcome === "failed" && run.worker?.phase === "running")) && !run.stopped) {
          try { cancelExecution(); } catch { /* Still retire the physical worker. */ }
          run.stopped = Promise.resolve().then(async () => {
            const worker = run.worker;
            const receipt = await host.stopOwnedWork(run.request, { reason: "cancelled" }, worker?.id ?? null);
            requireRetirement(run, worker?.id ?? null, receipt);
            if (run.worker !== worker) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
            if (worker) {
              await retireReportedWorker(run, worker, receipt);
              delete run.worker;
            }
          }).catch(error => { delete run.stopped; throw error; });
        }
        if (run.stopped) await run.stopped;
        else await retireFinished(run);
        clear(run);
        await host.settle(run.lease.leaseId, run.denial ? "cancelled" : outcome, run.finalReport!.usage);
        settledRemote = true;
        active.delete(request.runId);
        if (run.denial) throw run.denial;
      })();
      run.settlement = attempt.catch(error => {
        if (active.get(request.runId) === run) delete run.settlement;
        const failure = feaRunAdmissionError(error);
        if (settledRemote) failure.workersRetired = true;
        throw failure;
      });
      return run.settlement;
    },
  };
}
export type FeaRunAdmissionHook = ReturnType<typeof createFeaRunAdmissionHook>;
