import { Worker } from "node:worker_threads";

const WAIT_MS = 5_000;
const WORKER = new URL("./transaction-lock-fence-worker.mjs", import.meta.url);

export function acquireProcessFence(token, options = {}) {
  const state = sharedState();
  const worker = createWorker(options, { mode: "serve", token, state: state.buffer });
  let acquired = false;
  try {
    waitForChange(state, 0, "composition process fence acquisition timed out", options.waitMs);
    if (Atomics.load(state, 0) !== 1) throw new Error("composition process fence acquisition failed");
    const port = Atomics.load(state, 1);
    if (!Number.isInteger(port) || port < 1 || port > 65_535) {
      throw new Error("composition process fence returned an invalid port");
    }
    acquired = true;
    return { worker, state, port, token, waitMs: options.waitMs };
  } finally {
    if (!acquired) worker.terminate();
  }
}

export function processFenceIsActive(port, token, options = {}) {
  const state = sharedState();
  const waitMs = options.waitMs ?? WAIT_MS;
  const worker = createWorker(options, { mode: "probe", port, token, timeoutMs: waitMs, state: state.buffer });
  try {
    waitForChange(state, 0, "composition process fence probe timed out", waitMs);
    const result = Atomics.load(state, 0);
    if (result < 0) throw new Error("composition process fence probe failed");
    return result === 1;
  } finally { worker.terminate(); }
}

export function assertProcessFence(fence) {
  if (!fence || Atomics.load(fence.state, 0) !== 1) {
    throw new Error("module composition process fence is not active");
  }
  if (!processFenceIsActive(fence.port, fence.token)) {
    throw new Error("module composition process fence does not answer with its owner token");
  }
}

export function releaseProcessFence(fence) {
  try {
    if (!fence || Atomics.load(fence.state, 0) !== 1) {
      throw new Error("module composition process fence is not active");
    }
    fence.worker.postMessage("close");
    waitForChange(fence.state, 1, "composition process fence release timed out", fence.waitMs);
    if (Atomics.load(fence.state, 0) !== 2) throw new Error("composition process fence release failed");
  } finally { fence?.worker?.terminate(); }
}

function sharedState() { return new Int32Array(new SharedArrayBuffer(2 * Int32Array.BYTES_PER_ELEMENT)); }
function waitForChange(state, expected, message, waitMs = WAIT_MS) { if (Atomics.wait(state, 0, expected, waitMs) === "timed-out") throw new Error(message); }
function createWorker(options, workerData) { return options.createWorker ? options.createWorker(workerData) : new Worker(WORKER, { workerData }); }
