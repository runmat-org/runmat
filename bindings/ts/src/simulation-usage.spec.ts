import { describe, expect, it } from "vitest";
import { addSimulationUsage, emptySimulationUsage, parseCompletedSimulationUsage, simulationOperationCount as count, simulationUsageTotal, validateSimulationProgress } from "./simulation-usage.js";

describe("exact simulation work at the SDK boundary", () => {
  it("preserves the full run range and rejects numeric, malformed and missing measurements", () => {
    const large = parseCompletedSimulationUsage({ ...emptySimulationUsage(), solve: "18446744073709551615" });
    expect(simulationUsageTotal(large)).toBe(18_446_744_073_709_551_615n);
    for (const value of ["", "01", "-1", "1.0", "18446744073709551616", "9".repeat(1000)]) expect(() => count(value)).toThrow();
    expect(() => parseCompletedSimulationUsage({ solve: "0" })).toThrow();
    expect(() => parseCompletedSimulationUsage(Object.assign(Object.create(emptySimulationUsage()), { a: "0", b: "0", c: "0", d: "0", e: "0" }))).toThrow();
    expect(() => parseCompletedSimulationUsage({ ...emptySimulationUsage(), solve: 1 })).toThrow();
    expect(() => parseCompletedSimulationUsage({ ...large, assembly: "1" })).toThrow();
  });
  it("conserves disjoint work and requires category-monotonic snapshots within the grant", () => {
    const first = parseCompletedSimulationUsage({ ...emptySimulationUsage(), solve: "5" });
    const second = parseCompletedSimulationUsage({ ...emptySimulationUsage(), assembly: "3" });
    const combined = addSimulationUsage(first, second);
    expect(simulationUsageTotal(combined)).toBe(8n);
    expect(() => validateSimulationProgress(first, combined, count("8"))).not.toThrow();
    expect(() => validateSimulationProgress(first, combined, count("7"))).toThrow();
    expect(() => validateSimulationProgress(first, parseCompletedSimulationUsage({ ...second, solve: "4", meshing: "10" }), count("100"))).toThrow();
  });
});
