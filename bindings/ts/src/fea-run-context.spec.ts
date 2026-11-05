import { afterEach, describe, expect, it, vi } from "vitest";
import { __internals, executePreparedFeaStudy, executeOwnedFeaStudyStep, initRunMat, type LocalFeaWorkerContext } from "./index.js";
import { emptySimulationUsage } from "./simulation-usage.js";
const usage = { completedOps: emptySimulationUsage() };
import { createHermeticFeaAdmission } from "./test/fea-admission.js";

afterEach(() => __internals.setNativeModuleOverride(null));

describe("metered browser FEA boundary", () => {
  it("initializes signed out without calling a lease provider", async () => {
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, initRunMat: async () => ({}) } as never);
    const session = await initRunMat({ enableGpu: false });
    expect(typeof session.setFeaRunAdmission).toBe("function");
  });

  it("denies standalone numerical work without an admitted delegation", async () => {
    const execute = vi.fn(async () => "numerical-result");
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, executePreparedFeaStudy: execute } as never);
    await expect(executePreparedFeaStudy("prepared-json", 1024)).rejects.toMatchObject({ code: "RM.FEA.SignInRequired" });
    expect(execute).not.toHaveBeenCalled();
  });

  it("forwards exact context with the explicit trusted worker admission port", async () => {
    let ownedWorkLive = false;
    const fixture = createHermeticFeaAdmission({ cancelExecution: vi.fn(), renewFeaRunLease: vi.fn(), stopOwnedWork: async () => { ownedWorkLive = false; } });
    const request: LocalFeaWorkerContext["request"] = { runId: "run-parent", kind: "meshAndSolve", site: "local", allocatedCpuMillicores: 1000, studyIdentity: "source-digest", measurementVersion: "rsua-1" };
    await fixture.hook.start(request);
    const context = await fixture.delegate(request, "prepared-json");
    const admission = { validateWorker: fixture.hook.validateWorker, acquireWorkerGrant: fixture.hook.acquireWorkerGrant };
    // This unit test qualifies binding forwarding, not a numerical solver.
    const execute = vi.fn(async (_json, _memory, delegated, authority) => {
      await authority.validateWorker(delegated, { phase: "start", usage });
      ownedWorkLive = true;
      await authority.validateWorker(delegated, { phase: "finish", usage });
      return "numerical-result";
    });
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, executePreparedFeaStudy: execute } as never);
    await expect(executePreparedFeaStudy("prepared-json", 1024, undefined, context)).rejects.toMatchObject({ code: "RM.FEA.SignInRequired" });
    expect(execute).not.toHaveBeenCalled();
    await expect(executePreparedFeaStudy("prepared-json", 1024, undefined, context, admission)).resolves.toBe("numerical-result");
    expect(execute).toHaveBeenCalledWith("prepared-json", 1024, context, admission);
    expect(ownedWorkLive).toBe(true);
    const stepContext = await fixture.delegate(request, "step-json");
    expect(ownedWorkLive).toBe(false);
    const step = vi.fn(async (_json, _memory, delegated, authority) => {
      await authority.validateWorker(delegated, { phase: "start", usage });
      ownedWorkLive = true;
      await authority.validateWorker(delegated, { phase: "finish", usage });
      return "typed-step-result";
    });
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, executeOwnedFeaStudyStep: step } as never);
    await expect(executeOwnedFeaStudyStep("step-json", 1024, undefined, context)).rejects.toMatchObject({ code: "RM.FEA.SignInRequired" });
    expect(step).not.toHaveBeenCalled();
    await expect(executeOwnedFeaStudyStep("step-json", 1024, undefined, stepContext, admission)).resolves.toBe("typed-step-result");
    expect(step).toHaveBeenCalledWith("step-json", 1024, stepContext, admission);
    expect(ownedWorkLive).toBe(true);
    await fixture.hook.settle(request, { outcome: "completed", usage });
    expect(ownedWorkLive).toBe(false);
    expect(fixture.starts).toHaveLength(1);
    expect(fixture.settlements).toHaveLength(1);
    expect(fixture.retirements).toHaveLength(2);
  });

  it("reports missing numerical assets without a sign-in denial", async () => {
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {} } as never);
    await expect(executePreparedFeaStudy("json", 1024)).rejects.toThrow("prepared FEA execution support");
  });
});
