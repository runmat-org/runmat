import { describe, expect, it } from "vitest";
import {
  sameProgramAssignment, isRetiredProgramTerminal,
  type RunMatProgramExecutionAssignment,
} from "./program-terminal.js";

describe("selected program wire fences", () => {
  it("compares the complete assignment and never accepts an unverified terminal", () => {
    const assignment: RunMatProgramExecutionAssignment = {
      scope_id: "scope", pool_id: "pool", task_id: "task", attempt_id: "attempt",
      worker_id: "worker", backend: "browser_workers", resources: { accelerator_leases: [] },
    };
    expect(sameProgramAssignment(assignment, structuredClone(assignment))).toBe(true);
    for (const key of ["scope_id", "pool_id", "task_id", "attempt_id", "worker_id"] as const) {
      expect(sameProgramAssignment(assignment, { ...assignment, [key]: "other" })).toBe(false);
    }
    expect(sameProgramAssignment(assignment, { ...assignment, extra: true })).toBe(false);
    expect(isRetiredProgramTerminal(assignment, {
      schema_version: 1, assignment, retirement: { state: "retired" },
    })).toBe(true);
    expect(isRetiredProgramTerminal(assignment, {
      schema_version: 1, assignment,
      retirement: { state: "unverified", diagnostic: "live owner" },
    })).toBe(false);
  });
});
