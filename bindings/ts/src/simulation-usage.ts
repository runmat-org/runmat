/** Exact operation-count wire values. Time and resource telemetry are separate. */
declare const operationCount: unique symbol;
export type SimulationOperationCount = string & { readonly [operationCount]: true };
const maximum = 18_446_744_073_709_551_615n;
export const simulationUsageCategories = ["meshing", "assembly", "factorization", "solve", "other"] as const;
export type SimulationUsageCategory = typeof simulationUsageCategories[number];
export type CompletedSimulationUsage = Readonly<Record<SimulationUsageCategory, SimulationOperationCount>>;

const ledgerFailures = {
  CountOverflow: true, InvalidGrant: true, InvalidReservation: true,
  NonMonotonicUsage: true, OutstandingWork: true, Closed: true,
  InvalidState: true, GrantUnavailable: true,
} as const;
export type SimulationLedgerError = keyof typeof ledgerFailures
  | { BudgetExceeded: { required: SimulationOperationCount; budget: SimulationOperationCount } }
  | { MoreGrantRequired: { required: SimulationOperationCount; granted: SimulationOperationCount } };

/** Decode the shared Rust ledger's externally tagged error at the wire boundary. */
export function parseSimulationLedgerError(value: unknown): SimulationLedgerError {
  if (typeof value === "string" && Object.prototype.hasOwnProperty.call(ledgerFailures, value)) return value as keyof typeof ledgerFailures;
  if (value === null || typeof value !== "object" || Object.keys(value).length !== 1) throw new TypeError("Invalid simulation ledger error");
  const input = value as Record<string, unknown>;
  const kind = Object.prototype.hasOwnProperty.call(input, "BudgetExceeded") ? "BudgetExceeded"
    : Object.prototype.hasOwnProperty.call(input, "MoreGrantRequired") ? "MoreGrantRequired" : undefined;
  const detail = kind && input[kind];
  if (!kind || detail === null || typeof detail !== "object" || Object.keys(detail).length !== 2) throw new TypeError("Invalid simulation ledger error");
  const fields = detail as Record<string, unknown>;
  const limit = kind === "BudgetExceeded" ? "budget" : "granted";
  if (!Object.prototype.hasOwnProperty.call(fields, "required") || !Object.prototype.hasOwnProperty.call(fields, limit) || typeof fields["required"] !== "string" || typeof fields[limit] !== "string") throw new TypeError("Invalid simulation ledger error counts");
  const required = simulationOperationCount(fields["required"]);
  const bound = simulationOperationCount(fields[limit]);
  return kind === "BudgetExceeded" ? { BudgetExceeded: { required, budget: bound } }
    : { MoreGrantRequired: { required, granted: bound } };
}

export function simulationOperationCount(value: string): SimulationOperationCount {
  if (typeof value !== "string" || value.length > 20 || !/^(0|[1-9][0-9]*)$/.test(value) || BigInt(value) > maximum) {
    throw new RangeError("Simulation operation counts must be canonical unsigned 64-bit decimal strings.");
  }
  return value as SimulationOperationCount;
}

export function emptySimulationUsage(): CompletedSimulationUsage {
  const zero = simulationOperationCount("0");
  return { meshing: zero, assembly: zero, factorization: zero, solve: zero, other: zero };
}

/** Every category is required. An absent measurement is not a measured zero. */
export function parseCompletedSimulationUsage(value: unknown): CompletedSimulationUsage {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).length !== simulationUsageCategories.length) {
    throw new TypeError("Simulation usage requires every completed-work category.");
  }
  const input = value as Record<string, unknown>;
  if (!simulationUsageCategories.every(category => Object.prototype.hasOwnProperty.call(input, category))) {
    throw new TypeError("Simulation usage categories must be explicit fields.");
  }
  const parse = (category: SimulationUsageCategory) => {
    const count = input[category];
    if (typeof count !== "string") throw new TypeError(`Simulation ${category} usage must be an exact decimal string.`);
    return simulationOperationCount(count);
  };
  const result = { meshing: parse("meshing"), assembly: parse("assembly"), factorization: parse("factorization"), solve: parse("solve"), other: parse("other") };
  simulationUsageTotal(result);
  return result;
}

export function simulationUsageTotal(value: CompletedSimulationUsage): bigint {
  const total = simulationUsageCategories.reduce((sum, category) => sum + BigInt(simulationOperationCount(value[category])), 0n);
  if (total > maximum) throw new RangeError("Completed simulation work exceeds the run's operation-count range.");
  return total;
}

export function validateSimulationProgress(previous: CompletedSimulationUsage, current: CompletedSimulationUsage, grant: SimulationOperationCount): void {
  if (simulationUsageCategories.some(category => BigInt(current[category]) < BigInt(previous[category]))) {
    throw new RangeError("Completed simulation usage cannot decrease in any category.");
  }
  if (simulationUsageTotal(current) > BigInt(simulationOperationCount(grant))) {
    throw new RangeError("Completed simulation work exceeds its admitted grant.");
  }
}

/** Merge disjoint owners, never repeated snapshots of the same owner. */
export function addSimulationUsage(left: CompletedSimulationUsage, right: CompletedSimulationUsage): CompletedSimulationUsage {
  const add = (category: SimulationUsageCategory) => simulationOperationCount((BigInt(left[category]) + BigInt(right[category])).toString());
  const result = { meshing: add("meshing"), assembly: add("assembly"), factorization: add("factorization"), solve: add("solve"), other: add("other") };
  simulationUsageTotal(result);
  return result;
}
