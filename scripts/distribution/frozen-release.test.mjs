import assert from "node:assert/strict";
import test from "node:test";
import { buildArtifactManifest } from "../runtime/builtin-example-verifier/artifacts.mjs";
import { PLATFORMS, latestJobs, validateFrozenProducts, validateRetirement, validateAcceptance } from "./frozen-release.mjs";

const source = "a".repeat(40);
const options = { source, tag: "v0.7.1" };
const run = () => ({ head_sha: source, head_repository: { full_name: "runmat-org/runmat" },
  head_branch: options.tag, path: ".github/workflows/release.yml", event: "push", status: "completed", conclusion: "cancelled" });
const step = name => ({ name, status: "completed", conclusion: "success" });
const jobs = () => [
  ...["validate-release-state", "Verify qualified CI checkpoint and publication contracts",
    "Verify meshing catalogs and scientific release evidence", "Publish wasm bindings to npm / build-and-publish"]
    .map(name => ({ name, status: "completed", conclusion: "success" })),
  ...PLATFORMS.map((platform, id) => {
    const system = platform.startsWith("windows") ? "Windows" : "Unix";
    return { id, name: `Build ${platform}`, run_attempt: 1, status: "completed", conclusion: "cancelled", steps: [
      step(`Build release CLI with embedded AOT runtime (${system})`),
      ...(platform.startsWith("macos") ? [step("Sign and notarize release artifact (macOS)")] : []),
      step(`Stage and accept distributed product (${system})`), step(`Package as archive (${system})`),
      step("Publish release artifacts"), { name: "Upload rolling native checkpoint", status: "completed", conclusion: "cancelled" },
    ] };
  }),
];

test("merges partial reruns without losing earlier accepted product jobs", () => {
  const original = { name: "Build linux-x86_64", run_attempt: 1, id: 1 };
  const failedPublish = { name: "Publish crates to crates.io", run_attempt: 1, id: 2 };
  const resumedPublish = { ...failedPublish, run_attempt: 2, id: 3 };
  assert.deepEqual(latestJobs([original, failedPublish, resumedPublish]), [original, resumedPublish]);
  assert.throws(() => latestJobs([original, original]));
  assert.throws(() => latestJobs([{ name: "missing attempt" }]));
});

test("selects exact platform and attempt artifacts after accepted products", () => {
  assert.deepEqual(validateFrozenProducts(run(), jobs(), options).map(product => product.artifact),
    PLATFORMS.map(platform => `runmat-v0.7.1-${platform}-1`));
});

test("uses the actual execution attempt for GitHub's carried-forward completed jobs", () => {
  const originals = jobs().map(job => ({ ...job, run_attempt: 1,
    started_at: "2026-11-13T10:37:51Z", completed_at: "2026-11-13T11:44:36Z" }));
  const carried = originals.map(job => ({ ...structuredClone(job), id: (job.id ?? 100) + 200, run_attempt: 2 }));
  assert.deepEqual(latestJobs([...originals, ...carried]).slice(4), originals.slice(4));
  assert.deepEqual(validateFrozenProducts(run(), latestJobs([...originals, ...carried]), options)
    .map(product => product.artifact), PLATFORMS.map(platform => `runmat-v0.7.1-${platform}-1`));
});

test("keeps a genuinely rerun execution and refuses to infer origin from incomplete metadata", () => {
  const original = { ...jobs()[4], started_at: "2026-11-13T10:37:51Z", completed_at: "2026-11-13T11:44:36Z" };
  const rerun = { ...structuredClone(original), id: 200, run_attempt: 2, started_at: "2026-11-13T18:46:00Z" };
  assert.deepEqual(latestJobs([original, rerun]), [rerun]);
  const changedSteps = { ...structuredClone(original), id: 201, run_attempt: 2 };
  changedSteps.steps[0].completed_at = "2026-11-13T18:47:00Z";
  assert.deepEqual(latestJobs([original, changedSteps]), [changedSteps]);
  const missingTimes = { ...structuredClone(original), id: 202, run_attempt: 2 };
  delete missingTimes.started_at;
  assert.deepEqual(latestJobs([original, missingTimes]), [missingTimes]);
  const active = { ...structuredClone(original), id: 203, run_attempt: 2, status: "in_progress" };
  assert.deepEqual(latestJobs([original, active]), [active]);
});
test("permits retirement only after every product is safely uploaded", () => {
  const active = { ...run(), status: "in_progress" };
  assert.throws(() => validateFrozenProducts(active, jobs(), options));
  assert.equal(validateFrozenProducts(active, jobs(), { ...options, completed: false }).length, 4);
});
test("retirement never interrupts a live registry publisher or unrelated job", () => {
  const active = { ...run(), status: "in_progress" };
  assert.equal(validateRetirement(active, jobs(), options).length, 4);
  for (const name of ["Publish crates to crates.io", "Another test job"]) {
    assert.throws(() => validateRetirement(active, [...jobs(), { name, status: "in_progress" }], options));
  }
});
test("rejects source, workflow, repository, tag and event mismatches", () => {
  for (const delta of [{ head_sha: "b".repeat(40) }, { head_repository: { full_name: "other/repo" } },
    { head_branch: "main" }, { path: ".github/workflows/ci.yml" }, { event: "workflow_dispatch" }]) {
    assert.throws(() => validateFrozenProducts({ ...run(), ...delta }, jobs(), options));
  }
});
test("rejects missing, failed and duplicate source gates", () => {
  assert.throws(() => validateFrozenProducts(run(), jobs().slice(1), options));
  const failed = jobs(); failed[0].conclusion = "failure";
  assert.throws(() => validateFrozenProducts(run(), failed, options));
  assert.throws(() => validateFrozenProducts(run(), [...jobs(), jobs()[0]], options));
});
test("rejects missing native jobs, attempts, acceptance, packaging and uploads", () => {
  assert.throws(() => validateFrozenProducts(run(), jobs().slice(0, -1), options));
  for (const conclusion of ["failure", "skipped", "cancelled", null]) {
    for (const name of ["Stage and accept distributed product (Unix)", "Package as archive (Unix)", "Publish release artifacts"]) {
      const input = jobs(); input[4].steps.find(step => step.name === name).conclusion = conclusion;
      assert.throws(() => validateFrozenProducts(run(), input, options));
    }
  }
  const missingAttempt = jobs(); delete missingAttempt[4].run_attempt;
  assert.throws(() => validateFrozenProducts(run(), missingAttempt, options));
});
test("never accepts a failed, cancelled or unfinished compiler", () => {
  for (const delta of [{ conclusion: "failure" }, { conclusion: "cancelled" }, { status: "in_progress", conclusion: null }]) {
    const input = jobs(); Object.assign(input[4].steps[0], delta);
    assert.throws(() => validateFrozenProducts(run(), input, options));
  }
});
test("requires macOS signing and refuses new post-product execution", () => {
  const unsigned = jobs(); unsigned[6].steps.find(step => step.name === "Sign and notarize release artifact (macOS)").conclusion = "skipped";
  assert.throws(() => validateFrozenProducts(run(), unsigned, options));
  const unexpected = jobs(); unexpected[4].steps.push(step("Run another compiler"));
  assert.throws(() => validateFrozenProducts(run(), unexpected, options));
});

function evidence() {
  const manifest = buildArtifactManifest({ product: "native-cli", artifactProfile: "embedded-aot", sourceRevision: source,
    layout: { kind: "recursive-tree", root: "product" }, entrypoints: [{ role: "runmat-binary", path: "runmat" }],
    files: [{ path: "runmat", size: 0, sha256: `sha256:${"0".repeat(64)}` }] });
  const receipt = { status: "passed", artifactManifestDigest: manifest.artifactManifestDigest, processDiagnostics: [], results: [
    ...[["first-launch", 0], ["scalar-division", 0], ["callback", 0], ["original-error", 1]]
      .map(([name, exitCode]) => ({ name, exitCode, signal: null })),
    { sourceRevision: source, artifactManifestDigest: manifest.artifactManifestDigest,
      probeKind: "native-embedded-aot-compile-and-execute-v1", result: { kind: "compile-and-execute", status: "passed", failureKind: "none", prepareExitCode: 0, executeExitCode: 0 } },
  ] };
  return { manifest, receipt };
}
test("requires all five manifest-bound native acceptance checks", () => {
  const { manifest, receipt } = evidence();
  validateAcceptance(manifest, receipt, source);
  for (const change of [r => r.results.pop(), r => { r.results[3].exitCode = 0; },
    r => { r.results[4].result.status = "failed"; }, r => { r.artifactManifestDigest = `sha256:${"f".repeat(64)}`; },
    r => { r.results[4].sourceRevision = "b".repeat(40); }, r => { r.processDiagnostics.push("timeout"); }]) {
    const altered = structuredClone(receipt); change(altered);
    assert.throws(() => validateAcceptance(manifest, altered, source));
  }
});
