import { afterEach, describe, expect, it, vi } from "vitest";
import { createFeaRunAdmissionHook, feaRunAdmissionError, type FeaRunAdmission, type FeaRunRequest, type FeaRunLease } from "./fea-run-admission.js";

import { emptySimulationUsage, parseCompletedSimulationUsage, simulationOperationCount as count } from "./simulation-usage.js";
const measured = (solve = "0") => ({ completedOps: parseCompletedSimulationUsage({ ...emptySimulationUsage(), solve }) });
const workerReport = (phase: "start" | "finish") => ({ phase, usage: measured(phase === "finish" ? "23" : "0") });

afterEach(() => vi.useRealTimers());
const request: FeaRunRequest = { runId: `run_${"1".repeat(32)}`, site: "local", kind: "meshAndSolve", allocatedCpuMillicores: 1000, studyIdentity: `sha256:${"2".repeat(64)}`, measurementVersion: "rsua-1" };
const payload = `sha256:${"3".repeat(64)}`;
const numerical = { preparedStudyIdentity: payload, workClass: "numerical" as const };
const occt = { preparedStudyIdentity: payload, workClass: "occtPreparation" as const };
const retirement = (request: FeaRunRequest, delegationId: string | null) => ({ runId: request.runId, delegationId, workerId: delegationId ? `worker:${delegationId}` : null, retired: true as const });
function fixture(grantedOps = 120_000, heartbeatSeconds = 30, lifetimeMillis = 120_000) {
  const lease: FeaRunLease = { leaseId: "fealease-original", state: "active", budgetOps: count("120000"), grantedOps: count(String(grantedOps)), heartbeatSeconds, expiresAt: new Date(Date.now() + lifetimeMillis).toISOString() };
  const host: FeaRunAdmission = { start: vi.fn(async () => lease), heartbeat: vi.fn(async () => lease), requestGrant: vi.fn(async () => lease), settle: vi.fn(async () => {}), stopOwnedWork: vi.fn(async (request, _reason, id) => retirement(request, id)) };
  const cancel = vi.fn(), renew = vi.fn();
  const hook = createFeaRunAdmissionHook(host, cancel, renew);
  const grants = new Map<string, ReturnType<typeof count>>();
  hook.bindOwner({
    async delegate(context) { grants.set(context.delegationId, count("0")); return { lease: context.lease, rootStartedAtUnixMillis: context.rootStartedAtUnixMillis }; },
    async start() {},
    async acquire(context, requiredOps) { grants.set(context.delegationId, requiredOps); return { lease: { ...context.lease, grantedOps: requiredOps }, rootStartedAtUnixMillis: context.rootStartedAtUnixMillis }; },
    async report(context, usage) { if (BigInt(usage.completedOps.solve) > BigInt(grants.get(context.delegationId) ?? "0")) throw new Error("ungranted work"); return usage; },
    async retire(context) { grants.delete(context.delegationId); },
    async discardUndispatched(context) { grants.delete(context.delegationId); },
  });
  return { host, lease, cancel, renew, hook };
}

describe("page-owned FEA lease supervision", () => {
  it("maps exact lease API errors at the boundary", () => {
    expect(feaRunAdmissionError({ status: 401 }).denied.reason).toBe("signInRequired");
    expect(feaRunAdmissionError({ status: 409, body: { error: { code: "simulation.allowance_used", details: { refillsAt: "2099-01-01T00:00:00Z" } } } }).denied).toEqual({ reason: "allowanceUsed", refillsAt: "2099-01-01T00:00:00Z" });
    for (const status of [403, 422]) expect(feaRunAdmissionError({ status }).denied.reason).toBe("licenseInvalid");
    expect(feaRunAdmissionError({ status: 404 }).denied.reason).toBe("leaseExpired");
    for (const error of [{ status: 503 }, new TypeError("network")]) expect(feaRunAdmissionError(error).denied.reason).toBe("serverUnreachable");
    for (const error of [{ status: 401 }, { status: 404 }, { status: 403 }, { status: 422 }, { status: 409 }, { status: 503 }, new TypeError("network")]) {
      const normalized = feaRunAdmissionError(error);
      expect(normalized.cause).toBe(error);
      expect(Object.keys(normalized)).not.toContain("cause");
    }
    for (const [code, cause] of [
      ["RM.FEA.BudgetReached", { BudgetExceeded: { required: "18446744073709551615", budget: "5" } }],
      ["RM.FEA.GrantRequired", { MoreGrantRequired: { required: "9", granted: "5" } }],
    ] as const) {
      const error = { code, denied: { reason: "accountingInvalid", cause } };
      const normalized = feaRunAdmissionError(error);
      expect(normalized.code).toBe(code);
      expect(normalized.denied.cause).toEqual(cause);
      expect(normalized.cause).toBe(error);
    }
    for (const cause of [{ MoreGrantRequired: { required: 9, granted: "5" } }, { BudgetExceeded: { required: "9", budget: "5", extra: "0" } }, "toString"]) {
      expect(feaRunAdmissionError({ denied: { reason: "accountingInvalid", cause } }).denied.reason).toBe("licenseInvalid");
    }
  });

  it("requires the exact admitted cache and registered payload, never the supplied JSON budget", async () => {
    const { hook, host } = fixture();
    const state = await hook.start(request);
    const unregistered = { request, preparedStudyIdentity: payload, workClass: "numerical" as const, ...state, delegationId: "unregistered" };
    await expect(hook.validateWorker(unregistered, workerReport("start"))).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    const delegated = await hook.delegate(request, numerical);
    const context = { request, preparedStudyIdentity: payload, workClass: "numerical" as const, ...delegated };
    expect(await hook.validateWorker({ ...context, lease: { ...state.lease, grantedOps: count("18446744073709551615") } }, workerReport("start"))).toEqual({ lease: delegated.lease, rootStartedAtUnixMillis: state.rootStartedAtUnixMillis });
    await expect(hook.validateWorker(context, workerReport("start"))).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await expect(hook.delegate(request, numerical)).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await expect(hook.validateWorker({ ...context, request: { ...request, allocatedCpuMillicores: 2000 } }, workerReport("finish"))).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await expect(hook.validateWorker({ ...context, preparedStudyIdentity: `sha256:${"4".repeat(64)}` }, workerReport("finish"))).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await hook.acquireWorkerGrant(context, { requiredOps: count("23"), usage: measured() });
    await hook.validateWorker(context, workerReport("finish"));
    await expect(hook.validateWorker(context, workerReport("start"))).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    const next = await hook.delegate(request, numerical);
    expect(next.delegationId).not.toBe(context.delegationId);
    await hook.settle(request, { outcome: "completed", usage: measured("23") });
    expect(host.start).toHaveBeenCalledTimes(1);
    expect(host.settle).toHaveBeenCalledTimes(1);
  });

  it("binds excluded OCCT preparation to its parent assignment and zero billable reference work", async () => {
    const { hook, host } = fixture();
    await hook.start(request);
    const delegated = await hook.delegate(request, occt);
    const context = { request, preparedStudyIdentity: payload, workClass: "occtPreparation" as const, ...delegated };
    await expect(hook.validateWorker({ ...context, workClass: "numerical" }, workerReport("start")))
      .rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await hook.validateWorker(context, { phase: "start", usage: measured() });
    await expect(hook.acquireWorkerGrant(context, { requiredOps: count("1"), usage: measured() }))
      .rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await expect(hook.validateWorker(context, { phase: "finish", usage: measured("1") }))
      .rejects.toMatchObject({ code: "RM.FEA.AccountingInvalid" });
    await hook.validateWorker(context, { phase: "finish", usage: measured() });
    await hook.retireFinished(request);
    await hook.settle(request, { outcome: "completed", usage: measured() });
    expect(host.settle).toHaveBeenCalledWith("fealease-original", "completed", measured());
    const interrupted = fixture();
    await interrupted.hook.start(request);
    const interruptedLease = await interrupted.hook.delegate(request, occt);
    const interruptedContext = { request, preparedStudyIdentity: payload, workClass: "occtPreparation" as const, ...interruptedLease };
    await interrupted.hook.validateWorker(interruptedContext, { phase: "start", usage: measured() });
    await interrupted.hook.settle(request, { outcome: "cancelled", usage: measured() });
    expect(interrupted.host.stopOwnedWork).toHaveBeenCalledWith(request, { reason: "cancelled" }, interruptedContext.delegationId);
    expect(interrupted.host.settle).toHaveBeenCalledWith("fealease-original", "cancelled", measured());
  });

  it("stops a busy owned worker at the hold even when a heartbeat never resolves", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date", "performance"] });
    const { hook, host, cancel } = fixture(3000, 1, 3000);
    vi.mocked(host.heartbeat).mockImplementation(() => new Promise(() => {}));
    let retired!: () => void;
    vi.mocked(host.stopOwnedWork).mockImplementation((request, _reason, id) => new Promise(resolve => { retired = () => resolve(retirement(request, id)); }));
    await hook.start(request);
    await hook.check(request, measured("17"));
    await vi.advanceTimersByTimeAsync(3000);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(host.stopOwnedWork).toHaveBeenCalledTimes(1);
    const settled = hook.settle(request, { outcome: "failed", usage: measured("17") });
    await expect(hook.settle(request, { outcome: "failed", usage: measured("18") })).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await Promise.resolve();
    expect(host.settle).not.toHaveBeenCalled();
    retired();
    await expect(settled).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    expect(host.settle).toHaveBeenCalledWith("fealease-original", "cancelled", measured("17"));
    expect(host.start).toHaveBeenCalledTimes(1);
  });

  it("keeps a finished slot owned until the parent confirms physical retirement", async () => {
    const { hook, host } = fixture();
    const state = await hook.start(request);
    const delegated = await hook.delegate(request, numerical);
    const context = { request, preparedStudyIdentity: payload, workClass: "numerical" as const, ...delegated };
    await hook.validateWorker(context, workerReport("start"));
    await hook.acquireWorkerGrant(context, { requiredOps: count("23"), usage: measured() });
    await hook.validateWorker(context, workerReport("finish"));
    let retired!: () => void;
    vi.mocked(host.stopOwnedWork).mockImplementationOnce((request, _reason, id) => new Promise(resolve => { retired = () => resolve(retirement(request, id)); }));
    let reusable = false;
    const next = hook.delegate(request, numerical).then((value) => { reusable = true; return value; });
    await Promise.resolve();
    await Promise.resolve();
    expect(host.stopOwnedWork).toHaveBeenCalledWith(request, { reason: "completed" }, context.delegationId);
    expect(reusable).toBe(false);
    await expect(hook.validateWorker(context, workerReport("start"))).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    retired();
    expect((await next).delegationId).not.toBe(context.delegationId);
    await hook.settle(request, { outcome: "completed", usage: measured("23") });
    expect(host.settle).toHaveBeenCalledTimes(1);
    expect(state.lease.leaseId).toBe("fealease-original");
  });

  it("does not discard an undispatched child before physical retirement", async () => {
    const { hook, host } = fixture();
    await hook.start(request);
    const delegated = await hook.delegate(request, numerical);
    let retired!: () => void;
    vi.mocked(host.stopOwnedWork).mockImplementation((request, _reason, id) => new Promise(resolve => { retired = () => resolve(retirement(request, id)); }));
    const discarding = hook.retireUndispatched(request);
    await Promise.resolve();
    await expect(hook.validateWorker({ request, preparedStudyIdentity: payload, ...delegated }, workerReport("start"))).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    retired();
    await discarding;
    await hook.settle(request, { outcome: "failed", usage: measured() });
    expect(host.settle).toHaveBeenCalledTimes(1);
  });

  it("retires a finished generation before the root settles", async () => {
    const { hook, host } = fixture();
    await hook.start(request);
    const delegated = await hook.delegate(request, numerical);
    const context = { request, preparedStudyIdentity: payload, workClass: "numerical" as const, ...delegated };
    await hook.validateWorker(context, workerReport("start"));
    await hook.acquireWorkerGrant(context, { requiredOps: count("23"), usage: measured() });
    await hook.validateWorker(context, workerReport("finish"));
    await hook.retireFinished(request);
    expect(host.stopOwnedWork).toHaveBeenCalledWith(request, { reason: "completed" }, context.delegationId);
    await hook.settle(request, { outcome: "completed", usage: measured("23") });
    expect(host.settle).toHaveBeenCalledTimes(1);
  });

  it("refuses a retirement receipt for another worker generation", async () => {
    const { hook, host } = fixture();
    await hook.start(request);
    const delegated = await hook.delegate(request, numerical);
    const context = { request, preparedStudyIdentity: payload, workClass: "numerical" as const, ...delegated };
    await hook.validateWorker(context, workerReport("start"));
    await hook.acquireWorkerGrant(context, { requiredOps: count("23"), usage: measured() });
    await hook.validateWorker(context, workerReport("finish"));
    vi.mocked(host.stopOwnedWork).mockResolvedValueOnce(retirement(request, "another-generation"));
    await expect(hook.retireFinished(request)).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    expect(host.settle).not.toHaveBeenCalled();
    await hook.retireFinished(request);
    await hook.settle(request, { outcome: "completed", usage: measured("23") });
  });

  it("retains an ambiguous settlement for an exact retry", async () => {
    const { hook, host } = fixture();
    await hook.start(request);
    vi.mocked(host.settle).mockRejectedValueOnce(new TypeError("lost acknowledgement"));
    const final = { outcome: "failed" as const, usage: measured() };
    await expect(hook.settle(request, final)).rejects.toMatchObject({ code: "RM.FEA.ServerUnreachable" });
    expect(host.settle).toHaveBeenCalledTimes(1);
    await hook.retryPendingSettlement(request.runId);
    expect(host.settle).toHaveBeenCalledTimes(2);
  });

  it("pushes authentic cumulative renewals to the root control and extends the hard deadline", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date", "performance"] });
    const { hook, host, lease, cancel, renew } = fixture(3000, 1, 3000);
    const renewed = { ...lease, grantedOps: count("9000"), expiresAt: new Date(Date.now() + 9000).toISOString() };
    vi.mocked(host.heartbeat).mockResolvedValue(renewed);
    await hook.start(request);
    await hook.check(request, measured("42"));
    await vi.advanceTimersByTimeAsync(3500);
    expect(renew).toHaveBeenCalledWith(request.runId, renewed);
    expect(cancel).not.toHaveBeenCalled();
    await hook.settle(request, { outcome: "completed", usage: measured("42") });
    expect(host.settle).toHaveBeenCalledWith(lease.leaseId, "completed", measured("42"));
    expect(vi.getTimerCount()).toBe(0);
  });
});
