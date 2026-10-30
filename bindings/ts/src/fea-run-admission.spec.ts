import { afterEach, describe, expect, it, vi } from "vitest";
import { __internals, executePreparedFeaStudy, initRunMat, type FeaRunAdmission, type FeaRunWorkerAdmission } from "./index.js";

afterEach(() => __internals.setNativeModuleOverride(null));

function admission(): FeaRunAdmission {
  return { admit: vi.fn(async () => ({ permitId: "lease-parent" })), settle: vi.fn(async () => {}) };
}

describe("browser FEA admission boundary", () => {
  it("installs the exact asynchronous host adapter on the session", async () => {
    const install = vi.fn();
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, initRunMat: async () => ({ setFeaRunAdmission: install }) } as never);
    const session = await initRunMat({ enableGpu: false });
    const adapter = admission();
    session.setFeaRunAdmission(adapter);
    expect(install).toHaveBeenCalledTimes(1);
    expect(install).toHaveBeenCalledWith(adapter);
    expect(adapter.admit).not.toHaveBeenCalled();
  });

  it("does not silently ignore admission on an older runtime", async () => {
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, initRunMat: async () => ({}) } as never);
    const session = await initRunMat({ enableGpu: false });
    expect(() => session.setFeaRunAdmission(admission())).toThrow("configure FEA admission");
  });

  it("forwards worker authorization and exact parent identity without acquiring another lease", async () => {
    const execute = vi.fn(async () => "numerical-result");
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, executePreparedFeaStudy: execute, feaRunAdmissionSupported: () => true } as never);
    const authorization: FeaRunWorkerAdmission = {
      admission: admission(),
      delegation: {
        request: { runId: "run-parent", kind: "meshAndSolve", allocatedCpuMillicores: 1000, studyIdentity: "source-digest" },
        permitId: "lease-parent", preparedStudyIdentity: "exact-job-digest",
      },
    };
    await expect(executePreparedFeaStudy("prepared-json", 1024, undefined, authorization)).resolves.toBe("numerical-result");
    expect(execute).toHaveBeenCalledTimes(1);
    expect(execute).toHaveBeenCalledWith("prepared-json", 1024, authorization);
    expect(authorization.admission.admit).not.toHaveBeenCalled();
  });

  it("fails closed without worker admission, including against older numerical assets", async () => {
    const execute = vi.fn(() => "must-not-run");
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, executePreparedFeaStudy: execute } as never);
    await expect(executePreparedFeaStudy("json", 1024)).rejects.toMatchObject({ identifier: "RM.FEA.SignInRequired" });
    await expect(executePreparedFeaStudy("json", 1024, undefined, { admission: admission() })).rejects.toThrow("FEA admission support");
    expect(execute).not.toHaveBeenCalled();
  });
});
