import { afterEach, describe, expect, it, vi } from "vitest";
import { createFeaRunAdmissionHook, feaRunAdmissionError, type FeaRunAdmission, type FeaRunRequest } from "./fea-run-admission.js";

afterEach(() => vi.useRealTimers());
const request: FeaRunRequest = { runId: `run_${"1".repeat(32)}`, site: "local", kind: "meshAndSolve", allocatedCpuMillicores: 1000, studyIdentity: `sha256:${"2".repeat(64)}` };
const payload = `sha256:${"3".repeat(64)}`;
function fixture(heldCoreMillis = 120_000, heartbeatSeconds = 30) {
  const lease = { leaseId: "fealease-original", heldCoreMillis, heartbeatSeconds, expiresAt: new Date(Date.now() + 120_000).toISOString() };
  const host: FeaRunAdmission = { start: vi.fn(async () => lease), heartbeat: vi.fn(async () => lease), settle: vi.fn(async () => {}), stopOwnedWork: vi.fn(async () => {}) };
  const cancel = vi.fn(), renew = vi.fn();
  return { host, lease, cancel, renew, hook: createFeaRunAdmissionHook(host, cancel, renew) };
}

describe("page-owned FEA lease supervision", () => {
  it("maps exact lease API errors at the boundary", () => {
    expect(feaRunAdmissionError({ status: 401 }).denied.reason).toBe("signInRequired");
    expect(feaRunAdmissionError({ status: 409, body: { error: { code: "fea.core_hours_used", details: { refillsAt: "2099-01-01T00:00:00Z" } } } }).denied).toEqual({ reason: "allowanceUsed", refillsAt: "2099-01-01T00:00:00Z" });
    for (const status of [403, 422]) expect(feaRunAdmissionError({ status }).denied.reason).toBe("licenseInvalid");
    expect(feaRunAdmissionError({ status: 404 }).denied.reason).toBe("leaseExpired");
    for (const error of [{ status: 503 }, new TypeError("network")]) expect(feaRunAdmissionError(error).denied.reason).toBe("serverUnreachable");
  });

  it("requires the exact admitted cache and registered payload, never the supplied JSON budget", async () => {
    const { hook, host } = fixture();
    const state = await hook.start(request);
    const unregistered = { request, preparedStudyIdentity: payload, ...state, delegationId: "unregistered" };
    await expect(hook.validateWorker(unregistered, "start")).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    const delegated = await hook.delegate(request, payload);
    const context = { request, preparedStudyIdentity: payload, ...delegated };
    expect(await hook.validateWorker({ ...context, lease: { ...state.lease, heldCoreMillis: Number.MAX_SAFE_INTEGER } }, "start")).toEqual(state);
    await expect(hook.validateWorker(context, "start")).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await expect(hook.delegate(request, payload)).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await expect(hook.validateWorker({ ...context, request: { ...request, allocatedCpuMillicores: 2000 } }, "finish")).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await expect(hook.validateWorker({ ...context, preparedStudyIdentity: `sha256:${"4".repeat(64)}` }, "finish")).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    await hook.validateWorker(context, "finish");
    await expect(hook.validateWorker(context, "start")).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    const next = await hook.delegate(request, payload);
    expect(next.delegationId).not.toBe(context.delegationId);
    await hook.settle(request, "completed");
    expect(host.start).toHaveBeenCalledTimes(1);
    expect(host.settle).toHaveBeenCalledTimes(1);
  });

  it("stops a busy owned worker at the hold even when a heartbeat never resolves", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date", "performance"] });
    const { hook, host, cancel } = fixture(3000, 1);
    vi.mocked(host.heartbeat).mockImplementation(() => new Promise(() => {}));
    let retired!: () => void;
    vi.mocked(host.stopOwnedWork).mockImplementation(() => new Promise<void>((resolve) => { retired = resolve; }));
    await hook.start(request);
    await vi.advanceTimersByTimeAsync(3000);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(host.stopOwnedWork).toHaveBeenCalledTimes(1);
    const settled = hook.settle(request, "failed");
    await Promise.resolve();
    expect(host.settle).not.toHaveBeenCalled();
    retired();
    await expect(settled).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    expect(host.settle).toHaveBeenCalledWith("fealease-original", "cancelled", { wallMillis: 3000, cpuCoreMillis: 3000 });
    expect(host.start).toHaveBeenCalledTimes(1);
  });

  it("keeps a finished slot owned until the parent confirms physical retirement", async () => {
    const { hook, host } = fixture();
    const state = await hook.start(request);
    const delegated = await hook.delegate(request, payload);
    const context = { request, preparedStudyIdentity: payload, ...delegated };
    await hook.validateWorker(context, "start");
    await hook.validateWorker(context, "finish");
    let retired!: () => void;
    vi.mocked(host.stopOwnedWork).mockImplementation(() => new Promise<void>((resolve) => { retired = resolve; }));
    let reusable = false;
    const next = hook.delegate(request, payload).then((value) => { reusable = true; return value; });
    await Promise.resolve();
    await Promise.resolve();
    expect(host.stopOwnedWork).toHaveBeenCalledWith(request, { reason: "completed" });
    expect(reusable).toBe(false);
    await expect(hook.validateWorker(context, "start")).rejects.toMatchObject({ code: "RM.FEA.LeaseInvalid" });
    retired();
    expect((await next).delegationId).not.toBe(context.delegationId);
    await hook.settle(request, "completed");
    expect(host.settle).toHaveBeenCalledTimes(1);
    expect(state.lease.leaseId).toBe("fealease-original");
  });

  it("pushes authentic cumulative renewals to the root control and extends the hard deadline", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date", "performance"] });
    const { hook, host, lease, cancel, renew } = fixture(3000, 1);
    vi.mocked(host.heartbeat).mockResolvedValue({ ...lease, heldCoreMillis: 9000 });
    await hook.start(request);
    await vi.advanceTimersByTimeAsync(3500);
    expect(renew).toHaveBeenCalledWith(request.runId, { ...lease, heldCoreMillis: 9000 });
    expect(cancel).not.toHaveBeenCalled();
    await hook.settle(request, "completed");
    expect(host.settle).toHaveBeenCalledWith(lease.leaseId, "completed", { wallMillis: 3500, cpuCoreMillis: 3500 });
    expect(vi.getTimerCount()).toBe(0);
  });
});
