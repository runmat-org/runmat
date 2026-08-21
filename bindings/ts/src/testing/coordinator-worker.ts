import {
  BrowserTestRunner,
  type BrowserTestRunnerOptions
} from "./runner.js";
import type {
  BrowserTestEvent,
  BrowserTestRunInput,
  BrowserTestRunOutput
} from "./types.js";

interface CoordinatorRequest {
  id: number;
  type: "run" | "cancel" | "dispose";
  input?: BrowserTestRunInput;
  reason?: string;
}

interface CoordinatorResultResponse {
  id: number;
  type: "result";
  ok: boolean;
  output?: BrowserTestRunOutput;
  error?: string;
}

interface CoordinatorEventResponse {
  id: number;
  type: "event";
  event: BrowserTestEvent;
}

type CoordinatorResponse = CoordinatorResultResponse | CoordinatorEventResponse;

export interface TestCoordinatorScope {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<CoordinatorRequest>) => void
  ): void;
  postMessage(message: CoordinatorResponse): void;
}

export type TestCoordinatorHostOptions = Omit<
  BrowserTestRunnerOptions,
  "signal"
>;

/**
 * Hosts the portable Rust coordinator in a dedicated coordinator worker. The
 * host creates child test workers but contains no scheduling policy.
 */
export function installRunMatTestCoordinatorHost(
  scope: TestCoordinatorScope,
  options: TestCoordinatorHostOptions
): void {
  let active:
    | { id: number; cancellation: AbortController; promise: Promise<BrowserTestRunOutput> }
    | undefined;

  scope.addEventListener("message", (event) => {
    const request = event.data;
    if (request.type === "cancel") {
      active?.cancellation.abort(request.reason ?? "browser run cancelled");
      return;
    }
    if (request.type === "dispose") {
      active?.cancellation.abort("coordinator disposed");
      active = undefined;
      return;
    }
    if (!request.input) {
      respond(scope, request.id, Promise.reject(new Error("run input is required")));
      return;
    }
    if (active) {
      respond(
        scope,
        request.id,
        Promise.reject(new Error("coordinator already has an active run"))
      );
      return;
    }
    const cancellation = new AbortController();
    const runner = new BrowserTestRunner({
      ...options,
      signal: cancellation.signal,
      onEvent: (testEvent) => {
        scope.postMessage({
          id: request.id,
          type: "event",
          event: testEvent
        });
      }
    });
    const promise = runner.run(request.input);
    active = { id: request.id, cancellation, promise };
    respond(scope, request.id, promise).finally(() => {
      if (active?.id === request.id) active = undefined;
    });
  });
}

export class BrowserTestCoordinatorClient {
  private nextId = 1;
  private active:
    | {
        id: number;
        resolve(output: BrowserTestRunOutput): void;
        reject(error: Error): void;
        onEvent?: (event: BrowserTestEvent) => void;
      }
    | undefined;

  constructor(private readonly worker: Worker) {
    worker.addEventListener(
      "message",
      (event: MessageEvent<CoordinatorResponse>) => {
        const response = event.data;
        if (!this.active || this.active.id !== response.id) return;
        if (response.type === "event") {
          this.active.onEvent?.(response.event);
          return;
        }
        const active = this.active;
        this.active = undefined;
        if (response.ok && response.output) {
          active.resolve(normalizeReportBytes(response.output));
        } else {
          active.reject(new Error(response.error ?? "test coordinator failed"));
        }
      }
    );
    worker.addEventListener("error", (event) => {
      this.rejectActive(new Error(event.message || "test coordinator worker crashed"));
    });
    worker.addEventListener("messageerror", () => {
      this.rejectActive(new Error("test coordinator returned an invalid message"));
    });
  }

  run(
    input: BrowserTestRunInput,
    signal?: AbortSignal,
    onEvent?: (event: BrowserTestEvent) => void
  ): Promise<BrowserTestRunOutput> {
    if (this.active) {
      return Promise.reject(new Error("test coordinator already has an active run"));
    }
    const id = this.nextId++;
    const promise = new Promise<BrowserTestRunOutput>((resolve, reject) => {
      this.active = { id, resolve, reject, onEvent };
    });
    if (signal) {
      const cancel = () => {
        this.worker.postMessage({
          id,
          type: "cancel",
          reason: signal.reason?.toString()
        } satisfies CoordinatorRequest);
      };
      if (signal.aborted) cancel();
      else signal.addEventListener("abort", cancel, { once: true });
    }
    this.worker.postMessage({ id, type: "run", input } satisfies CoordinatorRequest);
    return promise;
  }

  dispose(): void {
    this.worker.postMessage({
      id: this.active?.id ?? 0,
      type: "dispose"
    } satisfies CoordinatorRequest);
    this.worker.terminate();
    this.rejectActive(new Error("test coordinator disposed"));
  }

  private rejectActive(error: Error): void {
    this.active?.reject(error);
    this.active = undefined;
  }
}

async function respond(
  scope: TestCoordinatorScope,
  id: number,
  promise: Promise<BrowserTestRunOutput>
): Promise<void> {
  try {
    scope.postMessage({ id, type: "result", ok: true, output: await promise });
  } catch (error) {
    scope.postMessage({
      id,
      type: "result",
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
}

function normalizeReportBytes(output: BrowserTestRunOutput): BrowserTestRunOutput {
  for (const report of output.reports) {
    if (!(report.bytes instanceof Uint8Array)) {
      report.bytes = new Uint8Array(report.bytes);
    }
  }
  return output;
}
