export interface FeaRunRequest {
  runId: string;
  kind: "meshing" | "solve" | "meshAndSolve";
  site: "local";
  allocatedCpuMillicores: number;
  estimatedWallSeconds?: number | null;
  studyIdentity: string;
}
export interface FeaRunUsage { cpuCoreMillis: number; wallMillis: number }
export type FeaRunOutcome = "completed" | "cancelled" | "failed";
export interface FeaRunLeaseRenewal { heldCoreMillis: number; expiresAt: string }
export interface FeaRunLease extends FeaRunLeaseRenewal { leaseId: string; heartbeatSeconds: number }
export interface FeaRunDenied {
  reason: "signInRequired" | "allowanceUsed" | "leaseExpired" | "licenseInvalid" | "serverUnreachable";
  refillsAt?: string;
}
export type FeaRunStopReason = FeaRunDenied | { reason: "cancelled" | "completed" };
/** Implement on the page/UI coordinator, never the busy numerical worker.
 * stopOwnedWork waits for this request's mesh/solve workers to terminate.
 */
export interface FeaRunAdmission {
  start(request: FeaRunRequest): Promise<FeaRunLease>;
  heartbeat(leaseId: string, usage: FeaRunUsage): Promise<FeaRunLeaseRenewal>;
  settle(leaseId: string, outcome: FeaRunOutcome, usage: FeaRunUsage): Promise<void>;
  stopOwnedWork(request: FeaRunRequest, reason: FeaRunStopReason): void | Promise<void>;
}
export class FeaRunAdmissionError extends Error {
  readonly code: string;
  /** Trusted supervisor receipt; preserve this field when proxying an error. */
  workersRetired?: boolean;
  constructor(readonly denied: FeaRunDenied) {
    super(denied.reason === "signInRequired" ? "Sign in to RunMat to mesh or solve. FEA uses core-hours."
      : denied.reason === "allowanceUsed" ? `You've used this week's core-hours.${denied.refillsAt ? ` They refill on ${denied.refillsAt}.` : ""} Add a core-hour pack or move to Scale: https://runmat.com/pricing`
      : denied.reason === "serverUnreachable" ? "Meshing and solving need a connection to RunMat Server."
      : "Your FEA lease needs refreshing. Sign in to RunMat, then run the study again.");
    this.name = "FeaRunAdmissionError";
    this.code = denied.reason === "signInRequired" ? "RM.FEA.SignInRequired" : denied.reason === "allowanceUsed" ? "RM.FEA.AllowanceUsed" : denied.reason === "serverUnreachable" ? "RM.FEA.ServerUnreachable" : "RM.FEA.LeaseInvalid";
  }
}
const record = (value: unknown): Record<string, unknown> | undefined => value !== null && typeof value === "object" ? value as Record<string, unknown> : undefined;
/** Accept typed denials or the server's standard {status, body:{error}} failure. */
export function feaRunAdmissionError(error: unknown): FeaRunAdmissionError {
  if (error instanceof FeaRunAdmissionError) return error;
  const value = record(error), denied = record(value?.denied);
  if (typeof denied?.reason === "string" && ["signInRequired", "allowanceUsed", "leaseExpired", "licenseInvalid", "serverUnreachable"].includes(denied.reason)) return new FeaRunAdmissionError(denied as unknown as FeaRunDenied);
  const status = value?.status ?? value?.statusCode, body = record(value?.body) ?? value, detail = record(body?.error);
  if (status === 401) return new FeaRunAdmissionError({ reason: "signInRequired" });
  if (status === 409 && detail?.code === "fea.core_hours_used") {
    const refillsAt = record(detail.details)?.refillsAt;
    return new FeaRunAdmissionError({ reason: "allowanceUsed", ...(typeof refillsAt === "string" ? { refillsAt } : {}) });
  }
  if (status === 404) return new FeaRunAdmissionError({ reason: "leaseExpired" });
  if (status === 403 || status === 422 || status === 409) return new FeaRunAdmissionError({ reason: "licenseInvalid" });
  return new FeaRunAdmissionError({ reason: "serverUnreachable" });
}
const positiveInteger = (value: number) => Number.isSafeInteger(value) && value > 0;
function validateRenewal(value: FeaRunLeaseRenewal): void {
  if (!value || !positiveInteger(value.heldCoreMillis) || !Number.isFinite(Date.parse(value.expiresAt)) || Date.parse(value.expiresAt) <= Date.now()) throw new FeaRunAdmissionError({ reason: "leaseExpired" });
}
export interface FeaRunLeaseState { lease: FeaRunLease; rootStartedAtUnixMillis: number }
export interface FeaRunWorkerLeaseState extends FeaRunLeaseState { delegationId: string }
export interface FeaRunWorkerAdmission {
  /** Revalidate against the trusted parent's admitted-root cache, not these JSON fields. */
  validateWorker(context: import("./fea-run-context.js").LocalFeaWorkerContext, phase: "start" | "finish"): Promise<FeaRunLeaseState>;
}
interface ActiveLease extends FeaRunLeaseState {
  request: FeaRunRequest;
  started: number;
  heartbeat?: ReturnType<typeof setTimeout>;
  deadline?: ReturnType<typeof setTimeout>;
  denial?: FeaRunAdmissionError;
  stopped?: Promise<void>;
  settlement?: Promise<void>;
  generation: number;
  worker?: { id: string; payload: string; phase: "available" | "running" | "finishing"; retirement?: Promise<void> };
}
/** Create on the UI thread and proxy these methods to a runtime coordinator if
 * needed. This owner must outlive the root; numerical workers receive data only.
 */
export function createFeaRunAdmissionHook(host: FeaRunAdmission, cancelExecution: () => void, onRenew: (runId: string, lease: FeaRunLease) => void) {
  const workerGlobal = (globalThis as { WorkerGlobalScope?: { prototype: object } }).WorkerGlobalScope;
  if (workerGlobal && workerGlobal.prototype.isPrototypeOf(globalThis)) throw new Error("FEA lease supervision must run on the page/UI coordinator, not a busy worker.");
  const active = new Map<string, ActiveLease>();
  let starting = false;
  const usage = (run: ActiveLease): FeaRunUsage => {
    const wallMillis = Math.max(0, Math.ceil(performance.now() - run.started));
    const cpuCoreMillis = Math.ceil(wallMillis * run.request.allocatedCpuMillicores / 1000);
    if (!Number.isSafeInteger(cpuCoreMillis)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
    return { wallMillis, cpuCoreMillis };
  };
  const clear = (run: ActiveLease) => { clearTimeout(run.heartbeat); clearTimeout(run.deadline); };
  const stop = (run: ActiveLease, error: unknown): Promise<void> => {
    run.denial ??= feaRunAdmissionError(error);
    clear(run);
    if (!run.stopped) {
      // A disposed/coordinator RPC must not prevent physical worker retirement.
      try { cancelExecution(); } catch { /* stopOwnedWork remains authoritative. */ }
      run.stopped = Promise.resolve().then(() => host.stopOwnedWork(run.request, run.denial!.denied));
      void run.stopped.then(() => { run.worker = undefined; }, () => {});
      void run.stopped.catch(() => {});
    }
    return run.stopped;
  };
  const armDeadline = (run: ActiveLease) => {
    clearTimeout(run.deadline);
    const remaining = Math.min((run.lease.heldCoreMillis - usage(run).cpuCoreMillis) * 1000 / run.request.allocatedCpuMillicores, Date.parse(run.lease.expiresAt) - Date.now());
    run.deadline = setTimeout(() => { void stop(run, new FeaRunAdmissionError({ reason: "leaseExpired" })); }, Math.max(0, Math.min(remaining, 2_147_483_647)));
  };
  const armHeartbeat = (run: ActiveLease) => {
    run.heartbeat = setTimeout(() => {
      void (async () => {
        try {
          const renewal = await host.heartbeat(run.lease.leaseId, usage(run));
          if (run.settlement || run.denial) return;
          validateRenewal(renewal);
          if (renewal.heldCoreMillis < run.lease.heldCoreMillis) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
          run.lease = { ...run.lease, ...renewal };
          onRenew(run.request.runId, { ...run.lease });
          if (usage(run).cpuCoreMillis >= run.lease.heldCoreMillis) throw new FeaRunAdmissionError({ reason: "allowanceUsed" });
          armDeadline(run); armHeartbeat(run);
        } catch (error) { await stop(run, error); }
      })().catch(() => {});
    }, run.lease.heartbeatSeconds * 1000);
  };
  const lookup = (request: FeaRunRequest): ActiveLease => {
    const run = active.get(request.runId);
    if (!run || run.request.runId !== request.runId || run.request.kind !== request.kind || run.request.site !== request.site || run.request.allocatedCpuMillicores !== request.allocatedCpuMillicores || run.request.studyIdentity !== request.studyIdentity || (run.request.estimatedWallSeconds ?? null) !== (request.estimatedWallSeconds ?? null)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
    return run;
  };
  const retireFinished = async (run: ActiveLease): Promise<void> => {
    const worker = run.worker;
    if (worker?.phase !== "finishing") return;
    // The parent calls this only after its worker RPC has returned the bytes.
    // Normal retirement must preserve those bytes and the successful root.
    worker.retirement ??= Promise.resolve().then(() => host.stopOwnedWork(run.request, { reason: "completed" }));
    await worker.retirement;
    if (run.worker === worker) run.worker = undefined;
  };
  const check = async (request: FeaRunRequest): Promise<FeaRunLeaseState> => {
    const run = lookup(request);
    if (run.settlement) throw new FeaRunAdmissionError({ reason: "leaseExpired" });
    if (!run.denial && (usage(run).cpuCoreMillis >= run.lease.heldCoreMillis || Date.now() >= Date.parse(run.lease.expiresAt))) await stop(run, new FeaRunAdmissionError({ reason: "leaseExpired" }));
    if (run.denial) { await run.stopped; throw run.denial; }
    return { lease: { ...run.lease }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis };
  };
  return {
    async start(request: FeaRunRequest): Promise<FeaRunLeaseState> {
      if (starting || active.size !== 0) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      starting = true;
      try {
        if (!positiveInteger(request.allocatedCpuMillicores) || request.allocatedCpuMillicores > 4_096_000) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
        const lease = await host.start(request);
        validateRenewal(lease);
        if (typeof lease.leaseId !== "string" || lease.leaseId.length === 0 || lease.leaseId.length > 256 || !positiveInteger(lease.heartbeatSeconds) || lease.heartbeatSeconds > 120) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
        const run: ActiveLease = { request: structuredClone(request), lease: { ...lease }, started: performance.now(), rootStartedAtUnixMillis: Date.now(), generation: 0 };
        active.set(request.runId, run); armDeadline(run); armHeartbeat(run);
        return { lease: { ...run.lease }, rootStartedAtUnixMillis: run.rootStartedAtUnixMillis };
      } catch (error) { throw feaRunAdmissionError(error); }
      finally { starting = false; }
    },
    check,
    async delegate(request: FeaRunRequest, preparedStudyIdentity: string): Promise<FeaRunWorkerLeaseState> {
      const run = lookup(request);
      await retireFinished(run);
      if (!/^sha256:[a-f0-9]{64}$/.test(preparedStudyIdentity) || (run.worker && (run.worker.phase !== "available" || run.worker.payload !== preparedStudyIdentity))) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      const state = await check(request);
      run.worker ??= { id: `${run.request.runId}:worker:${++run.generation}`, payload: preparedStudyIdentity, phase: "available" };
      if (!Number.isSafeInteger(run.generation)) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      return { ...state, delegationId: run.worker.id };
    },
    async validateWorker(context: import("./fea-run-context.js").LocalFeaWorkerContext, phase: "start" | "finish"): Promise<FeaRunLeaseState> {
      const run = lookup(context.request);
      const worker = run.worker;
      if (!worker || worker.id !== context.delegationId || worker.payload !== context.preparedStudyIdentity || context.lease.leaseId !== run.lease.leaseId || context.rootStartedAtUnixMillis !== run.rootStartedAtUnixMillis || (phase !== "start" && phase !== "finish") || worker.phase !== (phase === "start" ? "available" : "running")) throw new FeaRunAdmissionError({ reason: "licenseInvalid" });
      // Claim before awaiting so two starts cannot spend the same allocation.
      worker.phase = phase === "start" ? "running" : "finishing";
      const state = await check(context.request);
      return state;
    },
    async settle(request: FeaRunRequest, outcome: FeaRunOutcome): Promise<void> {
      const run = lookup(request);
      if (run.worker?.phase === "running" && !run.stopped && outcome === "completed") {
        // A root cannot retire its hold while a claimed worker still runs.
        void stop(run, new FeaRunAdmissionError({ reason: "licenseInvalid" }));
      }
      run.settlement ??= (async () => {
        if ((outcome === "cancelled" || (outcome === "failed" && run.worker?.phase === "running")) && !run.stopped) {
          try { cancelExecution(); } catch { /* Still retire the physical worker. */ }
          run.stopped = Promise.resolve().then(() => host.stopOwnedWork(run.request, { reason: "cancelled" }));
          void run.stopped.then(() => { run.worker = undefined; }, () => {});
        }
        if (run.stopped) await run.stopped;
        else await retireFinished(run);
        clear(run);
        try {
          await host.settle(run.lease.leaseId, run.denial ? "cancelled" : outcome, usage(run));
          if (run.denial) throw run.denial;
        }
        catch (error) {
          const failure = feaRunAdmissionError(error);
          failure.workersRetired = true;
          throw failure;
        }
        finally { active.delete(request.runId); }
      })();
      return run.settlement;
    },
  };
}
export type FeaRunAdmissionHook = ReturnType<typeof createFeaRunAdmissionHook>;
