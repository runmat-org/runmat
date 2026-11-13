import assert from "node:assert/strict";
import test from "node:test";
import { FIRST_LAUNCH_TIMEOUT_MS, nativeFirstLaunch } from "./native-first-launch.mjs";

test("cold launch has a separate bounded deadline and conserves success and process failures", () => {
  const environment = { NO_GUI: "1" };
  for (const execution of [
    { status: 0, signal: null, stdout: "runmat 0.7.0\n", stderr: "" },
    { status: null, signal: "SIGTERM", error: new Error("ETIMEDOUT"), stdout: "", stderr: "" },
  ]) {
    const ticks = [100, 43_100];
    const result = nativeFirstLaunch("/product/runmat", { cwd: "/probe", environment,
      now: () => ticks.shift(), spawn: (binary, args, options) => {
        assert.equal(binary, "/product/runmat");
        assert.deepEqual(args, ["--version"]);
        assert.equal(options.env, environment);
        assert.equal(options.cwd, "/probe");
        assert.equal(options.timeout, FIRST_LAUNCH_TIMEOUT_MS);
        return execution;
      } });
    assert.equal(result.elapsedMs, 43_000);
    assert.equal(result.exitCode, execution.status);
    assert.equal(result.signal, execution.signal);
    assert.equal(result.error, execution.error?.message);
    assert.equal(result.stdout, execution.stdout);
    assert.equal(result.timeoutMs, 180_000);
  }
});
