#!/usr/bin/env node
// Exercise a real light CLI and its exact pack. This does not build or fetch
// either input, and uses an isolated cache and synthetic source throughout.
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, copyFileSync, createReadStream, existsSync, lstatSync, mkdirSync, mkdtempSync,
  openSync, readdirSync, rmSync, writeFileSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";

assert.equal(process.argv.length, 5, "Expected light CLI, matching pack ZIP and new receipt path");
const [binary, pack, output] = process.argv.slice(2).map(value => resolve(value));
assert(lstatSync(binary).isFile() && lstatSync(pack).isFile(), "Inputs must be regular files");
assert(!existsSync(output), "Acceptance receipt already exists");
const directory = mkdtempSync(join(tmpdir(), "runmat-aot-acceptance-"));
const cache = join(directory, "cache");
const results = [];
let status = "failed";
let failure = null;
const identity = async filename => {
  const hash = createHash("sha256");
  for await (const bytes of createReadStream(filename)) hash.update(bytes);
  return { sha256: `sha256:${hash.digest("hex")}`, bytes: lstatSync(filename).size };
};
let inputs = null;

try {
  inputs = { binary: await identity(binary), pack: await identity(pack) };
  const configuration = join(directory, "runtime.toml");
  writeFileSync(configuration, '[runtime.telemetry]\nenabled = false\n[runtime.logging.otel]\nenabled = false\n[runtime.plotting]\nmode = "headless"\n[runtime.accelerate]\nenabled = false\n');
  const cliConfig = join(directory, "cli");
  mkdirSync(cliConfig);
  writeFileSync(join(cliConfig, "remote.json"), JSON.stringify({ server_url: "http://127.0.0.1:0", credential_store: "file" }));
  const environment = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !key.startsWith("RUNMAT_") && !key.startsWith("OTEL_") && !key.startsWith("WGPU_") && key !== "RUST_LOG"));
  Object.assign(environment, { RUNMAT_CONFIG: configuration, RUNMAT_CLI_CONFIG_DIR: cliConfig,
    RUNMAT_AOT_CACHE_DIR: cache, RUNMAT_PACKAGE_CACHE_DIR: join(directory, "packages"),
    RUNMAT_EXECUTION_STATE_DIR: join(directory, "execution"), RUNMAT_SERVER_URL: "http://127.0.0.1:0",
    RUNMAT_TELEMETRY_ENABLED: "false", RUNMAT_OTEL_ENABLED: "false", NO_GUI: "1", NO_COLOR: "1" });
  const userSettings = join(directory, "user-settings.json");
  writeFileSync(userSettings, "{}\n");
  environment.RUNMAT_USER_SETTINGS = userSettings;
  if (process.platform === "linux") {
    const query = spawnSync("pkg-config", ["--libs-only-L", "hdf5"], { encoding: "utf8", timeout: 10000 });
    assert.equal(query.status, 0, query.stderr);
    const paths = query.stdout.trim().split(/\s+/).filter(flag => flag.startsWith("-L")).map(flag => flag.slice(2));
    assert(paths.length && paths.every(path => existsSync(path)), "HDF5 development linker directories are required");
    environment.LIBRARY_PATH = [...paths, environment.LIBRARY_PATH].filter(Boolean).join(delimiter);
  }
  const options = { cwd: directory, env: environment, encoding: "utf8", timeout: 180000, maxBuffer: 1024 * 1024 };
  const run = (name, args, command = binary) => {
    const started = performance.now();
    const result = spawnSync(command, args, options);
    results.push({ name, exitCode: result.status, signal: result.signal, elapsedMs: performance.now() - started,
      error: result.error?.message, stdout: result.stdout, stderr: result.stderr });
    assert(!result.error && !result.signal, `Process failure in ${name}`);
    return result;
  };
  const source = join(directory, "plain.m");
  const program = join(directory, process.platform === "win32" ? "plain.exe" : "plain");
  writeFileSync(source, "disp(2 + 3);\n");
  const ordinary = run("ordinary-script-without-pack", ["--offline", source]);
  assert.equal(ordinary.status, 0, ordinary.stderr);
  assert.equal(ordinary.stdout.trim(), "5");
  assert(!existsSync(cache), "Ordinary execution touched the compile-time cache");
  const miss = run("offline-cache-miss", ["--offline", "compile", source, "-o", program]);
  assert.notEqual(miss.status, 0);
  assert.match(miss.stderr, /not cached/);
  const unattended = run("unattended-missing-pack", ["compile", source, "-o", program]);
  assert.notEqual(unattended.status, 0);
  assert.match(unattended.stderr, /--install-aot-runtime/);
  assert(!existsSync(cache) && !existsSync(program), "Refusal changed the installation or produced an executable");

  mkdirSync(cache);
  for (const digit of ["1", "2", "3"]) writeFileSync(join(cache, `${digit.repeat(64)}.zip`), "obsolete cache fixture");
  const install = name => new Promise((resolveResult, reject) => {
    const started = performance.now();
    const child = spawn(binary, ["--offline", "aot", "install", "--from", pack],
      { cwd: directory, env: environment, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.on("data", bytes => { stdout = (stdout + bytes).slice(-65536); });
    child.stderr.on("data", bytes => { stderr = (stderr + bytes).slice(-65536); });
    const timeout = setTimeout(() => child.kill(), 180000);
    child.on("error", error => { clearTimeout(timeout); reject(error); });
    child.on("close", (exitCode, signal) => {
      clearTimeout(timeout);
      results.push({ name, exitCode, signal, elapsedMs: performance.now() - started, stdout, stderr });
      resolveResult({ exitCode, signal, stderr });
    });
  });
  for (const installed of await Promise.all([install("concurrent-install-1"), install("concurrent-install-2")])) {
    assert.equal(installed.signal, null);
    assert.equal(installed.exitCode, 0, installed.stderr);
  }
  const cachedPack = join(cache, `${inputs.pack.sha256.slice(7)}.zip`);
  assert.deepEqual(await identity(cachedPack), inputs.pack);
  assert.equal(readdirSync(cache).filter(name => /^[a-f0-9]{64}\.zip$/.test(name)).length, 2);
  assert(!readdirSync(cache).some(name => name.endsWith(".tmp")), "Staging files leaked");
  const compiled = run("cached-offline-compile", ["--offline", "compile", source, "-o", program]);
  assert.equal(compiled.status, 0, compiled.stderr);
  const executed = run("standalone-execution", [], program);
  assert.equal(executed.status, 0, executed.stderr);
  assert.equal(executed.stdout.trim(), "5");

  const corrupt = join(directory, "corrupt.zip");
  copyFileSync(pack, corrupt);
  const file = openSync(corrupt, "r+");
  try { writeSync(file, Buffer.from([0]), 0, 1, 0); } finally { closeSync(file); }
  const refused = run("corrupt-pack-refused-with-valid-cache", ["--offline", "aot", "install", "--from", corrupt]);
  assert.notEqual(refused.status, 0);
  assert.deepEqual(await identity(cachedPack), inputs.pack);
  assert.deepEqual(await identity(binary), inputs.binary);
  assert.deepEqual(await identity(pack), inputs.pack);
  status = "passed";
} catch (error) {
  failure = error.message;
  throw error;
} finally {
  try {
    writeFileSync(output, JSON.stringify({ schema: "runmat.aot-pack.acceptance.v1", status, failure, inputs, results }, null, 2) + "\n", { flag: "wx" });
  } finally { rmSync(directory, { recursive: true, force: true }); }
}
