import { afterEach, describe, expect, it, vi } from "vitest";
import { __internals, workerProgramInventory, createWorkerProgramInvocation,
  sameProgramAssignment, isRetiredProgramTerminal,
  type RunMatProgramExecutionAssignment, type RunMatProgramTaskTerminal } from "./index.js";

const assignment: RunMatProgramExecutionAssignment = {
  scope_id: "scope", pool_id: "pool", task_id: "task", attempt_id: "attempt", worker_id: "worker",
  backend: "browser_workers", resources: { accelerator_leases: [{ epoch: 1n, device: "worker-device" }] }
};
afterEach(() => __internals.setNativeModuleOverride(null));
describe("selected program Runtime boundary", () => {
  it("compares every fenced identity and lease without lossy JSON conversion", () => {
    expect(sameProgramAssignment(assignment, structuredClone(assignment))).toBe(true);
    for (const key of ["scope_id", "pool_id", "task_id", "attempt_id", "worker_id", "backend"] as const) {
      expect(sameProgramAssignment(assignment, { ...assignment, [key]: "foreign" })).toBe(false);
    }
    expect(sameProgramAssignment(assignment, { ...assignment, resources: { accelerator_leases: [{ epoch: 2n, device: "worker-device" }] } })).toBe(false);
    const terminal: RunMatProgramTaskTerminal = { schema_version: 1, assignment, response: { outcome: "success" }, retirement: { state: "retired" } };
    expect(isRetiredProgramTerminal(assignment, terminal)).toBe(true);
    expect(isRetiredProgramTerminal(assignment, { ...terminal, assignment: { ...assignment, attempt_id: "foreign" } })).toBe(false);
    expect(isRetiredProgramTerminal(assignment, { ...terminal, retirement: { state: "unverified", diagnostic: "cleanup failed" } })).toBe(false);
  });
  it("returns the actual worker inventory and original invocation controller", async () => {
    const inventory = { schema_version: 1, host: {}, accelerators: [] };
    const invocation = { assignment, run: vi.fn(), cancel: vi.fn(), free: vi.fn() };
    const create = vi.fn(() => invocation);
    __internals.setNativeModuleOverride({ default: vi.fn(), initRunMat: vi.fn(), workerProgramInventory: () => inventory, createWorkerProgramInvocation: create } as never);
    expect(await workerProgramInventory()).toBe(inventory);
    const request = { assignment };
    expect(await createWorkerProgramInvocation(request)).toBe(invocation);
    expect(create).toHaveBeenCalledWith(request);
  });
  it("reports a missing native contract instead of fabricating inventory", async () => {
    __internals.setNativeModuleOverride({ default: vi.fn(), initRunMat: vi.fn() } as never);
    await expect(workerProgramInventory()).rejects.toThrow("Rebuild the SDK and WASM together");
    await expect(createWorkerProgramInvocation({})).rejects.toThrow("Rebuild the SDK and WASM together");
  });
});
