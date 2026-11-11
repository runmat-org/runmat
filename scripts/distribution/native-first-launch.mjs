import { spawnSync } from "node:child_process";

// Cold OS loading and Rosetta translation precede the bounded program cases.
// Keep this launch visible in the acceptance record and fail on its own deadline.
export const FIRST_LAUNCH_TIMEOUT_MS = 180_000;

export function nativeFirstLaunch(binary, { cwd, environment, spawn = spawnSync, now = () => performance.now() }) {
  const started = now();
  const execution = spawn(binary, ["--version"], { cwd, env: environment,
    encoding: "utf8", timeout: FIRST_LAUNCH_TIMEOUT_MS, maxBuffer: 1024 * 1024 });
  return { name: "first-launch", timeoutMs: FIRST_LAUNCH_TIMEOUT_MS,
    elapsedMs: now() - started, exitCode: execution.status, signal: execution.signal,
    error: execution.error?.message, stdout: execution.stdout, stderr: execution.stderr };
}
