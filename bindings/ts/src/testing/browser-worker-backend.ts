import type {
  BrowserWorkerBackendPort,
  BrowserWorkerHandle,
  RunMatTestSession,
  WorkerCancellationInput,
  WorkerExecution,
  WorkerExecutionInput,
  WorkerSpawnInput
} from "./types.js";

export interface BrowserWorkerBackendOptions {
  workerFactory?: () => Worker;
  sessionFactory: (
    input: WorkerSpawnInput
  ) => Promise<RunMatTestSession> | RunMatTestSession;
  projectHandoff?: unknown;
  filesystemSnapshot?: import("./types.js").BrowserTestFilesystemEntry[];
  maxWorkers?: number;
  signal?: AbortSignal;
  /** Narrows capacity when the caller explicitly selects shared-session isolation. */
  requestedIsolation?: "auto" | "worker" | "session" | "none";
}

type InternalHandle =
  | { id: string; kind: "worker"; client: WorkerClient }
  | {
      id: string;
      kind: "session" | "none";
      session: RunMatTestSession;
      submission: WorkerSpawnInput;
      active?: Promise<WorkerExecution>;
    };

export class BrowserWorkerBackend implements BrowserWorkerBackendPort {
  private readonly handles = new Map<string, InternalHandle>();
  private readonly cancellation: Promise<string>;
  private resolveCancellation!: (reason: string) => void;
  private nextId = 1;
  private sharedSession?: RunMatTestSession;

  constructor(private readonly options: BrowserWorkerBackendOptions) {
    this.cancellation = new Promise((resolve) => {
      this.resolveCancellation = resolve;
    });
    if (options.signal) {
      const cancel = () =>
        this.resolveCancellation(options.signal?.reason?.toString() ?? "browser run cancelled");
      if (options.signal.aborted) {
        cancel();
      } else {
        options.signal.addEventListener("abort", cancel, { once: true });
      }
    }
  }

  capabilities(): {
    isolation: Array<"worker" | "session" | "none">;
    maxWorkers: number;
  } {
    return {
      isolation: this.options.workerFactory
        ? ["worker", "session", "none"]
        : ["session", "none"],
      maxWorkers:
        this.options.requestedIsolation === "none"
          ? 1
          : Math.max(1, this.options.maxWorkers ?? navigatorConcurrency())
    };
  }

  async spawn(input: WorkerSpawnInput): Promise<BrowserWorkerHandle> {
    const id = `browser-test-${this.nextId++}`;
    const submission: WorkerSpawnInput = {
      ...input,
      ...(this.options.projectHandoff === undefined
        ? {}
        : { projectHandoff: this.options.projectHandoff }),
      ...(this.options.filesystemSnapshot === undefined
        ? {}
        : { filesystemSnapshot: this.options.filesystemSnapshot })
    };
    let handle: InternalHandle;
    if (input.isolation === "worker") {
      if (!this.options.workerFactory) {
        throw new Error("dedicated Web Worker isolation is unavailable");
      }
      const client = new WorkerClient(this.options.workerFactory());
      await client.request("install", submission);
      handle = { id, kind: "worker", client };
    } else {
      const session =
        input.isolation === "none"
          ? await this.noneSession(submission)
          : await this.options.sessionFactory(submission);
      if (this.options.projectHandoff !== undefined) {
        session.installProjectHandoff?.(this.options.projectHandoff);
      }
      handle = {
        id,
        kind: input.isolation,
        session,
        submission
      };
    }
    this.handles.set(id, handle);
    return handle;
  }

  async execute(
    external: BrowserWorkerHandle,
    input: WorkerExecutionInput
  ): Promise<WorkerExecution> {
    const handle = this.handle(external);
    if (handle.kind === "worker") {
      return (await handle.client.request("execute", input)) as WorkerExecution;
    }
    const active = handle.session.executeTestAttempt({
      plan: handle.submission.plan,
      snapshot: handle.submission.snapshot,
      testId: input.testId,
      attempt: input.attempt
    });
    handle.active = active;
    try {
      return await active;
    } finally {
      handle.active = undefined;
    }
  }

  async cancel(
    external: BrowserWorkerHandle,
    input: WorkerCancellationInput
  ): Promise<WorkerExecution | null> {
    const handle = this.handle(external);
    if (handle.kind === "worker") {
      return (await handle.client.request("cancel", input)) as WorkerExecution | null;
    }
    handle.session.cancelExecution();
    return handle.active ? await handle.active : null;
  }

  async terminate(external: BrowserWorkerHandle): Promise<void> {
    const handle = this.handle(external);
    this.handles.delete(handle.id);
    if (handle.kind === "worker") {
      handle.client.terminate(new Error("test worker was hard-terminated"));
    } else if (handle.kind === "session") {
      handle.session.dispose?.();
    }
  }

  async shutdown(external: BrowserWorkerHandle): Promise<void> {
    const handle = this.handle(external);
    this.handles.delete(handle.id);
    if (handle.kind === "worker") {
      await handle.client.request("shutdown");
      handle.client.terminate();
    } else if (handle.kind === "session") {
      handle.session.dispose?.();
    }
  }

  isCancelled(): boolean {
    return this.options.signal?.aborted ?? false;
  }

  cancellationReason(): string | undefined {
    return this.options.signal?.aborted
      ? this.options.signal.reason?.toString() ?? "browser run cancelled"
      : undefined;
  }

  waitForCancellation(): Promise<string> {
    return this.cancellation;
  }

  async dispose(): Promise<void> {
    await Promise.all(
      [...this.handles.values()].map((handle) => this.terminate(handle))
    );
    this.sharedSession?.dispose?.();
    this.sharedSession = undefined;
  }

  private handle(external: BrowserWorkerHandle): InternalHandle {
    const handle = this.handles.get(external.id);
    if (!handle) {
      throw new Error(`unknown browser test worker '${external.id}'`);
    }
    return handle;
  }

  private async noneSession(input: WorkerSpawnInput): Promise<RunMatTestSession> {
    this.sharedSession ??= await this.options.sessionFactory(input);
    return this.sharedSession;
  }
}

class WorkerClient {
  private readonly pending = new Map<
    number,
    { resolve(value: unknown): void; reject(error: Error): void }
  >();
  private nextId = 1;
  private closed = false;

  constructor(private readonly worker: Worker) {
    worker.addEventListener("message", (event: MessageEvent) => {
      const response = event.data as {
        id: number;
        ok: boolean;
        value?: unknown;
        error?: string;
      };
      const pending = this.pending.get(response.id);
      if (!pending) return;
      this.pending.delete(response.id);
      if (response.ok) {
        pending.resolve(response.value);
      } else {
        pending.reject(new Error(response.error ?? "test worker rejected request"));
      }
    });
    worker.addEventListener("error", (event) => {
      this.terminate(new Error(event.message || "test worker crashed"));
    });
    worker.addEventListener("messageerror", () => {
      this.terminate(new Error("test worker produced an invalid message"));
    });
  }

  request(type: string, payload?: unknown): Promise<unknown> {
    if (this.closed) {
      return Promise.reject(new Error("test worker is closed"));
    }
    const id = this.nextId++;
    const promise = new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.worker.postMessage({ id, type, payload });
    return promise;
  }

  terminate(reason = new Error("test worker terminated")): void {
    if (this.closed) return;
    this.closed = true;
    this.worker.terminate();
    for (const pending of this.pending.values()) {
      pending.reject(reason);
    }
    this.pending.clear();
  }
}

function navigatorConcurrency(): number {
  return typeof navigator === "undefined"
    ? 1
    : Math.max(1, navigator.hardwareConcurrency || 1);
}
