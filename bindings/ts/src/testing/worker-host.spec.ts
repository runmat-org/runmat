import { expect, it, vi } from "vitest";

import { installRunMatTestWorkerHost } from "./worker-host.js";
import type { TestWorkerScope } from "./worker-host.js";

it("awaits project installation and sends only attempt fields to the executor", async () => {
  let listener: Parameters<TestWorkerScope["addEventListener"]>[1];
  let finishInstall!: () => void;
  const installation = new Promise<void>((resolve) => { finishInstall = resolve; });
  const responses: unknown[] = [];
  let responseReady!: () => void;
  const response = () => new Promise<void>((resolve) => { responseReady = resolve; });
  const execute = vi.fn(async () => ({ result: {}, events: [] }));
  const install = vi.fn(() => installation);
  const dispose = vi.fn();
  installRunMatTestWorkerHost({
    addEventListener: (_type, callback) => { listener = callback; },
    postMessage: (value) => { responses.push(value); responseReady(); }
  }, {
    createSession: () => ({
      prepareTests: vi.fn(), executeTestAttempt: execute,
      installProjectHandoff: install, cancelExecution: vi.fn(), dispose
    })
  });
  const plan = { name: "plan" }, snapshot = { name: "snapshot" };
  const installed = response();
  listener!({ data: { id: 1, type: "install", payload: {
    plan, snapshot, isolation: "worker", projectHandoff: { graph: "frozen" },
    filesystemSnapshot: [{ path: "/test.m", bytes: new Uint8Array([1]) }]
  } } } as MessageEvent);
  await vi.waitFor(() => expect(install).toHaveBeenCalledOnce());
  expect(responses).toEqual([]);
  finishInstall();
  await installed;
  expect(responses).toEqual([{ id: 1, ok: true, value: null }]);
  const completed = response();
  listener!({ data: { id: 2, type: "execute", payload: { testId: "test", attempt: 1 } } } as MessageEvent);
  await completed;
  expect(execute).toHaveBeenCalledWith({ plan, snapshot, testId: "test", attempt: 1 });
  install.mockRejectedValueOnce(new Error("project installation rejected"));
  const rejected = response();
  listener!({ data: { id: 3, type: "install", payload: {
    plan, snapshot, isolation: "worker", projectHandoff: { graph: "rejected" }
  } } } as MessageEvent);
  await rejected;
  expect(responses.at(-1)).toEqual({ id: 3, ok: false, error: "project installation rejected" });
  expect(dispose).toHaveBeenCalledTimes(2);
  const staleAttempt = response();
  listener!({ data: { id: 4, type: "execute", payload: { testId: "test", attempt: 1 } } } as MessageEvent);
  await staleAttempt;
  expect(responses.at(-1)).toEqual({ id: 4, ok: false, error: "test worker has no installed frozen submission" });
  expect(execute).toHaveBeenCalledOnce();
});
