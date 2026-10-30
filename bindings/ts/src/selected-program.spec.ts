import { afterEach, describe, expect, it, vi } from "vitest";
import { __internals, workerProgramInventory, createWorkerProgramInvocation,
  sameProgramAssignment, isRetiredProgramTerminal,
  type RunMatProgramExecutionAssignment, type RunMatProgramTaskTerminal } from "./index.js";

const assignment: RunMatProgramExecutionAssignment = {
  scope_id: "scope", pool_id: "pool", task_id: "task", attempt_id: "attempt", worker_id: "worker",
  backend: "browser_workers", resources: { accelerator_leases: [{
    device_id: "worker-device", allocation_domain: "worker-domain", class: "gpu",
    provider: { id: "test", version: "1", abi_fingerprint: "fixture" },
    max_allocation_bytes: 16n, features: [], inventory_epoch: 1n,
    fencing_token: 1n, lease_id: "lease",
  }] }
};
afterEach(() => __internals.setNativeModuleOverride(null));
describe("selected program Runtime boundary", () => {
  it("compares every fenced identity and lease without lossy JSON conversion", () => {
    expect(sameProgramAssignment(assignment, structuredClone(assignment))).toBe(true);
    for (const key of ["scope_id", "pool_id", "task_id", "attempt_id", "worker_id", "backend"] as const) {
      expect(sameProgramAssignment(assignment, { ...assignment, [key]: "foreign" })).toBe(false);
    }
    expect(sameProgramAssignment(assignment, { ...assignment, resources: { accelerator_leases: [{
      ...assignment.resources.accelerator_leases[0], inventory_epoch: 2n,
    }] } })).toBe(false);
    const terminal: RunMatProgramTaskTerminal = { schema_version: 1, assignment, response: { outcome: "success", outputs: [] }, retirement: { state: "retired" } };
    expect(isRetiredProgramTerminal(assignment, terminal)).toBe(true);
    expect(isRetiredProgramTerminal(assignment, { ...terminal, assignment: { ...assignment, attempt_id: "foreign" } })).toBe(false);
    expect(isRetiredProgramTerminal(assignment, { ...terminal, retirement: { state: "unverified", diagnostic: "cleanup failed" } })).toBe(false);
  });
  it("returns the actual worker inventory and original invocation controller", async () => {
    const inventory = { schema_version: 1, host: {}, accelerators: [] };
    const invocation = { assignment, run: vi.fn(), cancel: vi.fn(), free: vi.fn() };
    const create = vi.fn(() => invocation);
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: vi.fn(), initRunMat: vi.fn(), workerProgramInventory: () => inventory, createWorkerProgramInvocation: create } as never);
    expect(await workerProgramInventory()).toBe(inventory);
    const request = { assignment };
    expect(await createWorkerProgramInvocation(request)).toBe(invocation);
    expect(create).toHaveBeenCalledWith(request);
  });
  it("reports a missing native contract instead of fabricating inventory", async () => {
    __internals.setNativeModuleOverride({ stringElementWireSchemaVersion: () => 1, default: vi.fn(), initRunMat: vi.fn() } as never);
    await expect(workerProgramInventory()).rejects.toThrow("The loaded RunMat module has no qualified worker factory inventory.");
    await expect(createWorkerProgramInvocation({})).rejects.toThrow("The loaded RunMat module does not support controlled worker programs.");
  });
});
