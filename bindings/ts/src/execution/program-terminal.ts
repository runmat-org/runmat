/** Canonical selected-program wire contracts from runmat-execution-artifact.
 * Scheduler attempt IDs and local TaskHandle generations are distinct.
 */
export interface RunMatProgramExecutionAssignment {
  scope_id: string;
  pool_id: string;
  task_id: string;
  attempt_id: string;
  worker_id: string;
  backend: "serial" | "local_processes" | "browser_workers" | "remote";
  resources: {
    accelerator_leases: RunMatProgramAcceleratorLease[];
  };
}

export interface RunMatProgramAcceleratorLease {
  device_id: string;
  allocation_domain: string;
  class: string;
  provider: { id: string; version: string; abi_fingerprint: string };
  max_allocation_bytes: bigint;
  features: (
    | "compute"
    | "unified_memory"
    | "peer_to_peer"
    | "confidential_compute"
    | { custom: string }
  )[];
  inventory_epoch: bigint;
  fencing_token: bigint;
  lease_id: string;
}

export interface RunMatWorkerAcceleratorDevice {
  id: string;
  allocation_domain: string;
  class: string;
  provider: RunMatProgramAcceleratorLease["provider"];
  max_allocation_bytes: bigint;
  features: RunMatProgramAcceleratorLease["features"];
  inventory_epoch: bigint;
}

export type RunMatWorkerHostCapability =
  | "HostRuntime" | "Filesystem" | "Network" | "UserInterface" | "Accelerator"
  | "NativeCode" | "ForeignRuntime" | "ParallelRuntime" | "DistributedRuntime";

export interface RunMatProgramWorkerInventory {
  schema_version: 1;
  host: {
    schema_version: 1;
    semantic_schema: number;
    compiler_schema: number;
    runtime_fingerprint: string;
    catalog_fingerprint: string;
    compatibility_modes: ("runmat" | "matlab" | "strict")[];
    target: { kind: "browser_wasm" };
    capabilities: RunMatWorkerHostCapability[];
    adapters: {
      adapter: string;
      version: number;
      capabilities: RunMatWorkerHostCapability[];
      foreign_capabilities: ("invoke" | "read" | "write" | "callback" | "transfer" | "serialize" | "zero_copy")[];
      execution_stack: "Any" | "Process";
      supports_wasm: boolean;
      supports_host_bridge: boolean;
    }[];
    process_stack_available: boolean;
    host_bridge_available: boolean;
    trust_tier: "customer-trusted" | "hosted-ordinary" | "attested-confidential";
  };
  accelerators: RunMatWorkerAcceleratorDevice[];
}

export type RunMatProgramCancellationReason =
  | "user"
  | "parent_scope"
  | "dependency_failed"
  | "deadline"
  | "quota"
  | "shutdown";

export interface RunMatProgramTaskCancellation {
  schema_version: 1;
  assignment: RunMatProgramExecutionAssignment;
  reason: RunMatProgramCancellationReason;
}

export type RunMatProgramTaskRetirement =
  | { state: "retired" }
  | {
      state: "unverified";
      diagnostic: string;
      cleanup_failure?: RunMatProgramRuntimeFailure;
    };

export interface RunMatProgramRuntimeFailure {
  message: string;
  identifier: string | null | undefined;
  span: { offset: bigint; length: bigint } | null | undefined;
  builtin: string | null | undefined;
  task_id: string | null | undefined;
  call_frames: {
    function: string;
    source_id: number | null | undefined;
    span: { offset: bigint; length: bigint } | null | undefined;
    detached_diagnostic?: {
      display_name: string;
      line: bigint | null | undefined;
      column: bigint | null | undefined;
    };
  }[];
  call_frames_elided: number;
  call_stack: string[];
  phase: string | null | undefined;
}

export type RunMatProgramExecutionResponse =
  | { outcome: "success"; outputs: unknown[] }
  | { outcome: "graph_success"; graph: unknown }
  | { outcome: "externalized_success"; outputs: unknown[]; result_objects: unknown[] }
  | { outcome: "spmd_success"; outputs: unknown[] }
  | { outcome: "failure"; message: string }
  | { outcome: "runtime_failure"; failure: RunMatProgramRuntimeFailure };

export interface RunMatProgramTaskTerminal {
  schema_version: 1;
  assignment: RunMatProgramExecutionAssignment;
  response?: RunMatProgramExecutionResponse;
  retirement: RunMatProgramTaskRetirement;
}

export interface RunMatWorkerProgramInvocation {
  readonly assignment: RunMatProgramExecutionAssignment;
  run(): Promise<RunMatProgramTaskTerminal>;
  /** Request delivery only. Await run() for the actual retirement terminal. */
  cancel(request: RunMatProgramTaskCancellation): void;
  free(): void;
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Exact bounded metadata equality. Rust performs canonical validation too.
 * No JSON stringification, number coercion, or attempt/generation inference.
 */
export function sameProgramAssignment(
  expected: RunMatProgramExecutionAssignment,
  actual: unknown
): actual is RunMatProgramExecutionAssignment {
  if (!record(actual) || Object.keys(actual).length !== 7) return false;
  for (const key of [
    "scope_id", "pool_id", "task_id", "attempt_id", "worker_id", "backend"
  ] as const) {
    if (actual[key] !== expected[key]) return false;
  }
  if (!record(actual.resources) || Object.keys(actual.resources).length !== 1) return false;
  const leases = actual.resources.accelerator_leases;
  if (!Array.isArray(leases) || leases.length !== expected.resources.accelerator_leases.length) {
    return false;
  }
  return expected.resources.accelerator_leases.every((lease, index) => {
    const candidate: unknown = leases[index];
    if (!record(candidate) || Object.keys(candidate).length !== 9) return false;
    for (const key of [
      "device_id", "allocation_domain", "class", "max_allocation_bytes",
      "inventory_epoch", "fencing_token", "lease_id"
    ] as const) {
      if (candidate[key] !== lease[key]) return false;
    }
    const provider = candidate.provider;
    if (!record(provider) || Object.keys(provider).length !== 3
      || provider.id !== lease.provider.id
      || provider.version !== lease.provider.version
      || provider.abi_fingerprint !== lease.provider.abi_fingerprint) return false;
    const features = candidate.features;
    if (!Array.isArray(features) || features.length !== lease.features.length) return false;
    return lease.features.every((feature, featureIndex) => {
      const value: unknown = features[featureIndex];
      return typeof feature === "string"
        ? value === feature
        : record(value) && Object.keys(value).length === 1 && value.custom === feature.custom;
    });
  });
}

export function isRetiredProgramTerminal(
  expected: RunMatProgramExecutionAssignment,
  terminal: RunMatProgramTaskTerminal
): boolean {
  return record(terminal)
    && Object.keys(terminal).every((key) => [
      "schema_version", "assignment", "response", "retirement"
    ].includes(key))
    && (terminal.response === undefined || record(terminal.response))
    && terminal.schema_version === 1
    && sameProgramAssignment(expected, terminal.assignment)
    && record(terminal.retirement)
    && Object.keys(terminal.retirement).length === 1
    && terminal.retirement.state === "retired";
}
