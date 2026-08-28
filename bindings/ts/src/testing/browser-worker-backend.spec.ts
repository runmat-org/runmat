import { describe, expect, it, vi } from "vitest";

import { BrowserWorkerBackend } from "./browser-worker-backend.js";
import { BrowserTestRunner } from "./runner.js";
import type {
  BrowserTestEvent,
  BrowserTestRunInput,
  BrowserTestRunOutput,
  BrowserWorkerBackendPort,
  RunMatTestSession,
  WorkerExecution
} from "./types.js";

function controlledSession(): {
  session: RunMatTestSession;
  cancelled: ReturnType<typeof vi.fn>;
  disposed: ReturnType<typeof vi.fn>;
} {
  let complete: ((value: WorkerExecution) => void) | undefined;
  const cancelled = vi.fn(() => {
    complete?.({ result: { state: "cancelled" }, events: [] });
  });
  const disposed = vi.fn();
  return {
    cancelled,
    disposed,
    session: {
      async executeTestAttempt() {
        return new Promise((resolve) => {
          complete = resolve;
        });
      },
      cancelExecution: cancelled,
      dispose: disposed
    }
  };
}

describe("browser test worker backend", () => {
  it("reports strongest available isolation without claiming process support", () => {
    const backend = new BrowserWorkerBackend({
      sessionFactory: () => controlledSession().session,
      maxWorkers: 3
    });
    expect(backend.capabilities()).toEqual({
      isolation: ["session", "none"],
      maxWorkers: 3
    });
  });

  it("serializes explicit none isolation at the advertised host boundary", () => {
    const backend = new BrowserWorkerBackend({
      sessionFactory: () => controlledSession().session,
      maxWorkers: 8,
      requestedIsolation: "none"
    });
    expect(backend.capabilities()).toEqual({
      isolation: ["session", "none"],
      maxWorkers: 1
    });
  });

  it("cooperatively cancels a session attempt and returns its teardown result", async () => {
    const controlled = controlledSession();
    const backend = new BrowserWorkerBackend({
      sessionFactory: () => controlled.session
    });
    const handle = await backend.spawn({
      isolation: "session",
      plan: {},
      snapshot: {}
    });
    const execution = backend.execute(handle, {
      testId: "test",
      attempt: 1
    });
    const cancelled = await backend.cancel(handle, {
      runId: "run",
      reason: "stop",
      graceDeadlineMs: Date.now() + 100
    });

    expect(cancelled).toEqual({
      result: { state: "cancelled" },
      events: []
    });
    await expect(execution).resolves.toEqual(cancelled);
    expect(controlled.cancelled).toHaveBeenCalledOnce();
    await backend.terminate(handle);
    expect(controlled.disposed).toHaveBeenCalledOnce();
  });

  it("keeps scheduling in native Rust and persists returned reports", async () => {
    const output: BrowserTestRunOutput = {
      result: {
        run_id: "run",
        state: { disposition: "passed", failed: false, incomplete: false },
        tests: []
      },
      events: [],
      reports: [
        {
          name: "results.json",
          mediaType: "application/json",
          bytes: new Uint8Array([1, 2, 3])
        }
      ],
      infrastructureFailures: 0,
      pluginFailures: 0,
      coverage: { program_revision: null, sites: [], counts: {} },
      isolation: "session"
    };
    const put = vi.fn(async () => {});
    const runTests = vi.fn(async (_input, backend) => {
      expect(backend.capabilities().isolation).toContain("session");
      return output;
    });
    const runner = new BrowserTestRunner({
      native: { runTests },
      sessionFactory: () => controlledSession().session,
      artifactStore: { put }
    });

    await expect(runner.run({ plan: {}, snapshot: {} })).resolves.toBe(output);
    expect(runTests).toHaveBeenCalledOnce();
    expect(put).toHaveBeenCalledWith("run", output.reports[0]);
  });

  it("streams canonical native events before returning the completed run", async () => {
    const events: Array<Record<string, unknown>> = [];
    const output: BrowserTestRunOutput = {
      result: {
        run_id: "run",
        state: { disposition: "passed", failed: false, incomplete: false },
        tests: []
      },
      events: [{ sequence: 1 }],
      reports: [],
      infrastructureFailures: 0,
      pluginFailures: 0,
      isolation: "worker",
      coverage: { program_revision: null, sites: [], counts: {} }
    };
    const runTestsWithEvents = vi.fn(
      async (
        _input: BrowserTestRunInput,
        _backend: BrowserWorkerBackendPort,
        observer: (event: BrowserTestEvent) => void
      ) => {
        observer(output.events[0]);
        return output;
      }
    );
    const runner = new BrowserTestRunner({
      native: {
        runTests: vi.fn(),
        runTestsWithEvents
      },
      sessionFactory: () => controlledSession().session,
      onEvent: (event) => events.push(event)
    });

    await expect(runner.run({ plan: {}, snapshot: {} })).resolves.toBe(output);
    expect(events).toEqual(output.events);
    expect(runTestsWithEvents).toHaveBeenCalledOnce();
  });
});
