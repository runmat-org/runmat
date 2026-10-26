import { describe, expect, it } from "vitest";
import { physicalLengthUnit, resolvedSourceLengthUnit, sourceLengthUnitLabel } from "./physical.js";
import type { LengthUnitEvidence } from "./physical.js";

describe("authored source unit overrides", () => {
  it("uses the authored conversion while retaining the original CAD evidence", () => {
    const evidence: LengthUnitEvidence = {
      status: "authored_override",
      declaration: { context: "study", source_symbol: "mm", unit: physicalLengthUnit("millimeter") },
      declared: [{ context: "STEP", source_symbol: "m", unit: physicalLengthUnit("meter") }],
      source: "study source units",
    };
    expect(resolvedSourceLengthUnit(evidence)?.scale).toBe(0.001);
    expect(sourceLengthUnitLabel(evidence)).toBe("mm (override)");
    expect(evidence.declared[0].unit.scale).toBe(1);
  });
});
