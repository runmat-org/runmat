export type TestIsolation = "auto" | "worker" | "session" | "none";
export type TestReportFormat = "human" | "json" | "junit" | "tap";
export type CoverageReportFormat = "json" | "lcov" | "cobertura" | "html";

export interface FrozenTestSubmission {
  plan: Record<string, unknown>;
  snapshot: Record<string, unknown>;
}

export interface SavedRunSource {
  owner_identity: string;
  relative_path: string;
  content: string;
}

export interface UnsavedRunBuffer extends SavedRunSource {}

export interface BrowserTestPrepareRequest {
  manifestPath: string;
  projectRevision: unknown;
  targets?: string[];
  names?: string[];
  tags?: string[];
  excludedTags?: string[];
  unsavedBuffers?: UnsavedRunBuffer[];
}

export interface BrowserPreparedTestRun {
  snapshot: Record<string, unknown>;
  discovery: Record<string, unknown>;
  plan: Record<string, unknown>;
  filesystemSnapshot?: BrowserTestFilesystemEntry[];
  projectHandoff?: unknown;
}

export interface BrowserTestFilesystemEntry {
  path: string;
  bytes: Uint8Array;
}

export interface ProjectTestLayout {
  sourceRoots: string[];
  testRoots: string[];
  testPaths: string[];
  testConfigDigest: string;
}

export interface RunMatTestPreparationNative {
  projectTestLayout(input: {
    manifestPath: string;
    manifestContent: string;
  }): ProjectTestLayout | Promise<ProjectTestLayout>;
  freezeTestSnapshot(input: {
    graphDigest: string;
    baseSourceDigest: string;
    testConfigDigest: string;
    savedSources: SavedRunSource[];
    unsavedBuffers: UnsavedRunBuffer[];
  }): Record<string, unknown> | Promise<Record<string, unknown>>;
}

export interface RunMatTestPreparationSession {
  prepareTests(
    snapshot: Record<string, unknown>,
    selector: Record<string, unknown>
  ): BrowserPreparedTestRun | Promise<BrowserPreparedTestRun>;
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
  coverage?: {
    enabled?: boolean;
    formats?: CoverageReportFormat[];
    roots?: string[];
    exclude?: string[];
    includeGenerated?: boolean;
    includeVendor?: boolean;
  };
}

export interface BrowserTestRunInput extends FrozenTestSubmission {
  projectHandoff?: unknown;
  filesystemSnapshot?: BrowserTestFilesystemEntry[];
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
  events: BrowserTestEvent[];
  reports: RenderedTestReport[];
  infrastructureFailures: number;
  pluginFailures: number;
  isolation: Exclude<TestIsolation, "auto">;
  coverage: CoverageAggregate;
}

export type BrowserTestEvent = Record<string, unknown>;

export interface RunMatTestNative {
  runTests(
    input: BrowserTestRunInput,
    backend: BrowserWorkerBackendPort
  ): Promise<BrowserTestRunOutput>;
  runTestsWithEvents?(
    input: BrowserTestRunInput,
    backend: BrowserWorkerBackendPort,
    observer: (event: BrowserTestEvent) => void
  ): Promise<BrowserTestRunOutput>;
}

export interface RunMatTestSession extends RunMatTestPreparationSession {
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
  coverage?: CoverageFragment[];
}

export interface CoverageFragment {
  program_revision: string;
  plan_revision: string;
  sites: CoverageSite[];
  counts: Record<string, number>;
}

export interface CoverageAggregate {
  program_revision: string | null;
  sites: CoverageSite[];
  counts: Record<string, number>;
}

export interface CoverageSite {
  id: string;
  counter_key: string;
  metric: "function" | "statement" | "decision" | "condition" | "mcdc_condition";
  owner_identity: string;
  relative_path: string;
  semantic_path: string;
  source_id: number;
  start_byte: number;
  end_byte: number;
  start_line: number;
  start_column: number;
  end_line: number;
  end_column: number;
  instrumented: boolean;
  unsupported_reason: string | null;
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
  projectHandoff?: unknown;
  filesystemSnapshot?: BrowserTestFilesystemEntry[];
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
