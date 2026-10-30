import { afterEach, describe, expect, it, vi } from "vitest";
import { __internals, executePreparedFeaStudy, initRunMat, type LocalFeaWorkerContext } from "./index.js";

afterEach(() => __internals.setNativeModuleOverride(null));

describe("unmetered browser FEA boundary", () => {
  it("initializes signed out without a billing installation API", async () => {
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, initRunMat: async () => ({}) } as never);
    const session = await initRunMat({ enableGpu: false });
    expect("setFeaRunAdmission" in session).toBe(false);
  });

  it("executes standalone numerical work without an account or adapter", async () => {
    const execute = vi.fn(async () => "numerical-result");
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, executePreparedFeaStudy: execute } as never);
    await expect(executePreparedFeaStudy("prepared-json", 1024)).resolves.toBe("numerical-result");
    expect(execute).toHaveBeenCalledWith("prepared-json", 1024, undefined);
  });

  it("forwards exact local identity and payload context without callback delegation", async () => {
    const execute = vi.fn(async () => "numerical-result");
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {}, executePreparedFeaStudy: execute } as never);
    const context: LocalFeaWorkerContext = {
      request: { runId: "run-parent", kind: "meshAndSolve", site: "local", allocatedCpuMillicores: 1000, studyIdentity: "source-digest" },
      preparedStudyIdentity: "exact-job-digest",
    };
    await expect(executePreparedFeaStudy("prepared-json", 1024, undefined, context)).resolves.toBe("numerical-result");
    expect(execute).toHaveBeenCalledWith("prepared-json", 1024, context);
  });

  it("reports missing numerical assets without a sign-in denial", async () => {
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: async () => {} } as never);
    await expect(executePreparedFeaStudy("json", 1024)).rejects.toThrow("prepared FEA execution support");
  });
});
