import type {
  FrozenTestSubmission,
  RunMatTestSession,
  WorkerExecution
} from "./types.js";

interface HostRequest {
  id: number;
  type: "install" | "execute" | "cancel" | "shutdown";
  payload?: unknown;
}

interface HostResponse {
  id: number;
  ok: boolean;
  value?: unknown;
  error?: string;
}

export interface TestWorkerHostOptions {
  createSession(): Promise<RunMatTestSession> | RunMatTestSession;
  projectHandoff?: unknown;
}

export interface TestWorkerScope {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<HostRequest>) => void
  ): void;
  postMessage(message: HostResponse): void;
}

/**
 * Installs the deliberately small child-worker protocol. This code owns
 * transport and session lifetime only; Rust owns scheduling and semantics.
 */
export function installRunMatTestWorkerHost(
  scope: TestWorkerScope,
  options: TestWorkerHostOptions
): void {
  let session: RunMatTestSession | undefined;
  let submission: FrozenTestSubmission | undefined;
  let active: Promise<WorkerExecution> | undefined;

  scope.addEventListener("message", (event: MessageEvent<HostRequest>) => {
    const request = event.data;
    void respond(request.id, async () => {
      switch (request.type) {
        case "install": {
          session?.dispose?.();
          session = await options.createSession();
          if (options.projectHandoff !== undefined) {
            session.installProjectHandoff?.(options.projectHandoff);
          }
          submission = deepFreeze(
            structuredClone(request.payload as FrozenTestSubmission)
          );
          return null;
        }
        case "execute": {
          if (!session || !submission) {
            throw new Error("test worker has no installed frozen submission");
          }
          const input = request.payload as {
            testId: string;
            attempt: number;
          };
          active = session.executeTestAttempt({
            ...submission,
            testId: input.testId,
            attempt: input.attempt
          });
          try {
            return await active;
          } finally {
            active = undefined;
          }
        }
        case "cancel": {
          session?.cancelExecution();
          return active ? await active : null;
        }
        case "shutdown": {
          session?.dispose?.();
          session = undefined;
          submission = undefined;
          return null;
        }
      }
    });
  });

  async function respond(id: number, operation: () => Promise<unknown>): Promise<void> {
    try {
      const value = await operation();
      scope.postMessage({ id, ok: true, value } satisfies HostResponse);
    } catch (error) {
      scope.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      } satisfies HostResponse);
    }
  }
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const nested of Object.values(value as Record<string, unknown>)) {
      deepFreeze(nested);
    }
  }
  return value;
}
