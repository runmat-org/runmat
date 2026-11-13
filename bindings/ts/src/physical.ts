/** Shared physical unit wire projections. The numeric catalog is generated from Rust. */
import type { LengthUnit, LengthUnitEvidence } from "./generated/fea-semantic-contracts.js";
import { PHYSICAL_LENGTH_UNITS } from "./generated/physical-unit-catalog.js";
export { PHYSICAL_LENGTH_UNITS };
export type { LengthUnit, LengthUnitEvidence };

export function physicalLengthUnit(name: keyof typeof PHYSICAL_LENGTH_UNITS): LengthUnit {
  const unit = PHYSICAL_LENGTH_UNITS[name];
  return { dimension: { ...unit.dimension }, scale: unit.scale, offset: unit.offset };
}
export function isPhysicalLengthUnit(value: unknown): value is LengthUnit {
  if (!value || typeof value !== "object") return false;
  const unit = value as Record<string, unknown>;
  const dimension = unit["dimension"];
  if (!dimension || typeof dimension !== "object" || typeof unit["scale"] !== "number" || !Number.isFinite(unit["scale"]) || unit["scale"] <= 0 || unit["offset"] !== 0 || (unit["angle_power"] !== undefined && unit["angle_power"] !== 0)) return false;
  const expected = PHYSICAL_LENGTH_UNITS.meter.dimension;
  return Object.entries(expected).every(([key, exponent]) => (dimension as Record<string, unknown>)[key] === exponent);
}
export function physicalLengthUnitName(unit: LengthUnit): string {
  for (const [name, candidate] of Object.entries(PHYSICAL_LENGTH_UNITS)) {
    if (candidate.scale === unit.scale && candidate.offset === unit.offset) return name.replaceAll("_", " ");
  }
  return `${unit.scale} m per coordinate unit`;
}
export function resolvedSourceLengthUnit(evidence: LengthUnitEvidence): LengthUnit | null {
  switch (evidence.status) {
    case "declared": {
      const first = evidence.contexts[0]?.unit;
      return first && evidence.contexts.every(context => context.unit.scale === first.scale && context.unit.offset === first.offset) ? first : null;
    }
    case "explicit_fallback": case "format_convention": case "authored_override": return evidence.declaration.unit;
    case "missing": case "unsupported": case "conflicting": return null;
  }
}
export function sourceLengthUnitLabel(evidence: LengthUnitEvidence): string {
  switch (evidence.status) {
    case "declared": return resolvedSourceLengthUnit(evidence) ? evidence.contexts[0]!.source_symbol : "mixed declared length units";
    case "explicit_fallback": return `${evidence.declaration.source_symbol} (specified)`;
    case "authored_override": return `${evidence.declaration.source_symbol} (override)`;
    case "format_convention": return `${evidence.declaration.source_symbol} (${evidence.format})`;
    case "missing": return "physical scale not established";
    case "unsupported": return "unsupported length-unit declaration";
    case "conflicting": return "conflicting length-unit declarations";
  }
}
export function explicitSourceLengthUnit(unit: LengthUnit, source: string): LengthUnitEvidence {
  if (!isPhysicalLengthUnit(unit) || !source.trim()) throw new Error("Explicit source units require a valid length conversion and attribution");
  return { status: "explicit_fallback", declaration: { context: "unitless-source", source_symbol: physicalLengthUnitName(unit), unit }, source };
}
