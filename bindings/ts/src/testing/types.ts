export type TestIsolation = "auto" | "worker" | "session" | "none";
export type TestReportFormat = "human" | "json" | "junit" | "tap";

export interface FrozenTestSubmission {
  plan: Record<string, unknown>;
  snapshot: Record<string, unknown>;
}

export interface BrowserTestRunOptions {
  isolation?: TestIsolation;
  jobs?: number;
  timeoutMs?: number;
  cancellationGraceMs?: number;
  maxAttempts?: number;
  shardIndex?: number;
  shardCount?: number;
  reports?: TestReportFormat[];
}

export interface BrowserTestRunInput extends FrozenTestSubmission {
  options?: BrowserTestRunOptions;
}

export interface RenderedTestReport {
  name: string;
  mediaType: string;
  bytes: Uint8Array;
}

export interface BrowserTestRunOutput {
  result: {
    run_id: string;
    state: {
      disposition: string;
      failed: boolean;
      incomplete: boolean;
    };
    tests: unknown[];
  };
  events: unknown[];
  reports: RenderedTestReport[];
  infrastructureFailures: number;
  isolation: Exclude<TestIsolation, "auto">;
}

export interface RunMatTestNative {
  runTests(
    input: BrowserTestRunInput,
    backend: BrowserWorkerBackendPort
  ): Promise<BrowserTestRunOutput>;
}

export interface RunMatTestSession {
  installProjectHandoff?(handoff: unknown): unknown;
  executeTestAttempt(input: {
    plan: Record<string, unknown>;
    snapshot: Record<string, unknown>;
    testId: string;
    attempt: number;
  }): Promise<WorkerExecution>;
  cancelExecution(): void;
  dispose?(): void;
}

export interface WorkerExecution {
  result: unknown;
  events: unknown[];
}

export interface BrowserWorkerBackendPort {
  capabilities(): {
    isolation: Array<"worker" | "session" | "none">;
    maxWorkers: number;
  };
  spawn(input: WorkerSpawnInput): Promise<BrowserWorkerHandle>;
  execute(
    handle: BrowserWorkerHandle,
    input: WorkerExecutionInput
  ): Promise<WorkerExecution>;
  cancel(
    handle: BrowserWorkerHandle,
    input: WorkerCancellationInput
  ): Promise<WorkerExecution | null>;
  terminate(handle: BrowserWorkerHandle): Promise<void>;
  shutdown(handle: BrowserWorkerHandle): Promise<void>;
  isCancelled(): boolean;
  cancellationReason(): string | undefined;
  waitForCancellation(): Promise<string>;
}

export interface WorkerSpawnInput extends FrozenTestSubmission {
  isolation: Exclude<TestIsolation, "auto">;
}

export interface WorkerExecutionInput {
  testId: string;
  attempt: number;
  deadlineMs?: number;
}

export interface WorkerCancellationInput {
  runId: string;
  reason: string;
  graceDeadlineMs: number;
}

export interface BrowserWorkerHandle {
  id: string;
}
