#!/usr/bin/env node
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { resolveArtifactPath, validateArtifactManifest } from "../runtime/builtin-example-verifier/artifacts.mjs";
import { runNativeEmbeddedAotProbe } from "../runtime/builtin-example-verifier/product-probe.mjs";

const manifestPath = resolve(process.argv[2]);
const output = resolve(process.argv[3]);
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
validateArtifactManifest(manifest, { verifyFiles: true, manifestPath });
assert.equal(manifest.product, "native-cli");
const binary = resolveArtifactPath(manifest, manifestPath, "runmat-binary");
const directory = mkdtempSync(join(tmpdir(), "runmat-native-acceptance-"));
const results = [];
let status = "failed";
try {
  const configuration = join(directory, "runmat.toml");
  writeFileSync(configuration, '[runtime.telemetry]\nenabled = false\n[runtime.logging.otel]\nenabled = false\n[runtime.accelerate]\nenabled = false\nprovider = "inprocess"\n[runtime.plotting]\nmode = "headless"\n');
  const cliConfiguration = join(directory, "cli-config");
  mkdirSync(cliConfiguration);
  writeFileSync(join(cliConfiguration, "remote.json"), JSON.stringify({
    server_url: "http://127.0.0.1:0", credential_store: "file",
  }), { mode: 0o600 });
  const userSettings = join(directory, "user-settings.json");
  writeFileSync(userSettings, "{}\n");
  const clean = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !key.startsWith("RUNMAT_") && !key.startsWith("OTEL_") && !key.startsWith("WGPU_")
      && key !== "RUST_LOG"));
  const environment = { ...clean, RUNMAT_CONFIG: configuration,
    RUNMAT_CLI_CONFIG_DIR: cliConfiguration, RUNMAT_USER_SETTINGS: userSettings,
    RUNMAT_SERVER_URL: "http://127.0.0.1:0", RUNMAT_TELEMETRY_ENABLED: "false",
    RUNMAT_OTEL_ENABLED: "false", RUNMAT_PACKAGE_CACHE_DIR: join(directory, "packages"),
    RUNMAT_EXECUTION_STATE_DIR: join(directory, "execution"), NO_GUI: "1", NO_COLOR: "1" };
  // Match the documented Debian development-library setup used by the native
  // compiler witness. Its unversioned HDF5 linker name lives outside /usr/lib.
  if (process.platform === "linux") {
    const query = spawnSync("pkg-config", ["--libs-only-L", "hdf5"], { encoding: "utf8", timeout: 10000 });
    assert.equal(query.status, 0, query.stderr);
    const paths = query.stdout.trim().split(/\s+/).filter(flag => flag.startsWith("-L")).map(flag => flag.slice(2));
    assert(paths.length && paths.every(path => existsSync(path)), "HDF5 development linker directories are required");
    environment.LIBRARY_PATH = [...paths, environment.LIBRARY_PATH].filter(Boolean).join(delimiter);
  }
  const cases = [
    ["scalar-division", 'x=[1 2 4]; assert(isequal(x/2,[0.5 1 2])); disp("accepted");', 0],
    ["callback", 'f=@(x) x.*x; assert(isequal(arrayfun(f,[1 2 3]),[1 4 9])); disp("accepted");', 0],
    ["original-error", "error('RunMat:Acceptance:Expected','original error');", 1],
  ];
  for (const [name, source, failure] of cases) {
    const file = join(directory, `${name}.m`);
    writeFileSync(file, source);
    const execution = spawnSync(binary, [file], { cwd: directory, env: environment,
      encoding: "utf8", timeout: 45000, maxBuffer: 1024 * 1024 });
    results.push({ name, exitCode: execution.status, signal: execution.signal,
      error: execution.error?.message, stdout: execution.stdout, stderr: execution.stderr });
    assert(!execution.error && !execution.signal, JSON.stringify(results.at(-1)));
    if (failure) {
      assert.notEqual(execution.status, 0, JSON.stringify(results.at(-1)));
      assert.match(execution.stdout + execution.stderr, /original error/);
    } else {
      assert.equal(execution.status, 0, JSON.stringify(results.at(-1)));
      assert.match(execution.stdout, /accepted/);
    }
  }
  const probe = runNativeEmbeddedAotProbe({ artifactManifest: manifest,
    artifactManifestPath: manifestPath, sourceRevision: manifest.sourceRevision,
    environment });
  results.push(probe);
  assert.equal(probe.result.status, "passed", JSON.stringify(probe));
  validateArtifactManifest(manifest, { verifyFiles: true, manifestPath });
  status = "passed";
  console.log("Three staged CLI cases and native-image compile/execute passed.");
} finally {
  writeFileSync(output, JSON.stringify({ status, artifactManifestDigest: manifest.artifactManifestDigest, results }, null, 2) + "\n");
  rmSync(directory, { recursive: true, force: true });
}
