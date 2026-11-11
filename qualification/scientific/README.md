# Scientific qualification evidence

Run the long test groups listed in `policy.json` during release qualification.
Ordinary CI runs the shorter numerical and integration tests through
`scripts/development/run-workspace-test-group.py` and checks the committed
scientific results with:

```sh
python3 scripts/qualification/scientific.py verify
```

Each required group needs a passing receipt that matches its source inputs and
test policy. CI rejects missing or outdated receipts. Extended transient-rod
qualification remains deferred under RM-1297; its existing failures are recorded
in that ticket.

The 0.7.0 release also defers the governed `meshing-release` cohort under the
owner's November 4 approval. Its 12 commands have no qualifying run or passing
receipt for this release. Shorter meshing tests remain in ordinary CI. Authored
CLI physics and driver recovery still require their scientific receipts.

The full induction-gear harmonic witness runs manually. Its current model
meshes for about 65 minutes before refusing a predicted 628,552,420,949-byte
numerical allocation. The three-frequency solve remains unqualified. Bounded
Maxwell regressions run in ordinary CI. This routing addition leaves the
commands, input identities and measured results of the existing measured cohorts
unchanged; their receipts retain that evidence under the revised policy hash.

The full-airliner drawing memory check also runs offline and requires a passing
receipt. It compiles the 3,200-occurrence assembly sheet and three staged views
through the native browser-host adapter, retaining the test's Rust-heap ceiling.
Ordinary CI runs the bounded drawing and assembly regressions. Actual browser
acceptance separately qualifies the distributed CAD host.

## Run a cohort

Print the command for a test group:

```sh
python3 scripts/qualification/scientific.py command --cohort authored-cli-physics
python3 scripts/qualification/scientific.py command --cohort driver-recovery-physics
```

Run the printed command through the machine's resource gate, using its existing
Cargo cache. Physics and board-timing cohorts use the release profile; the
airliner heap check uses the test profile. Record the actual profile in the
receipt. Keep `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0` and
`CARGO_INCREMENTAL=0` for development/test builds so they reuse the same cache.

The meshing recipe is `scripts/qualification/meshing-release.sh`. Supply a new
absolute `RUNMAT_SCIENTIFIC_OUTPUT` directory and `RUNMAT_QUALIFY_NATIVE_TARGET`.
It runs the governed corpus and the 15-sample stable performance/resource checks
with their original assertions. Results include one log per command and an exit
status table.

Its 12 commands reuse the three driver-replacement cases measured by the
separate required `driver-recovery-physics` cohort. All 18 recovery cases still
need their receipt; the meshing recipe does not repeat those three solves.

Use the workspace runner's `--include-extended` option to execute all test groups,
including the deferred refinement tests.

## Commit the measured evidence

Store one compact receipt per required cohort under `receipts/<cohort>.json`.
Include:

- the original source revision and canonical scientific input identity;
- target, toolchain and actual command argument vector;
- named cases, pass/failure/ignore/unselected counts and exit disposition;
- SHA-256 identities and retained locations of the executed artifacts and logs;
- the exact qualification-policy hash.

Commit the tested source, then use `scientific.py fingerprint` to calculate its
input identity. The command hashes Git's mode/type/blob/path entries for the
declared source roots and rejects uncommitted changes. Added or removed files,
mode changes and changed source or dependencies produce a different identity.
The policy hash identifies the test selection.

Before committing a receipt, review its logs, artifact hashes and results.
Keep the raw logs and binaries in the qualification archive and record their
locations. Commit the receipt and small review records. Keep credentials private.
The release workflow attests the receipt files; their original execution details
remain in the receipts.

## Reuse existing results

Reuse a receipt while its source inputs and test policy match. For unrelated
source changes, add a conservation review here explaining why the existing
results still apply. Keep the original measured revision and artifact hashes.

List each changed input's before/after Git entry and explain its effect on the
test group. Include the results of checks covering those changes. The verifier
uses these entries to reconstruct the measured source inventory and checks the
review and supporting result hashes. It rejects missing edits or changed records.
This comparison also supports a coordinated history rewrite while preserving
the original execution record.

Review changes to solvers, meshers, numerical dependencies, execution code,
fixtures and expected results before reusing their measurements. Rerun the
affected workloads when their scientific behavior changes. Support any claim
of equivalent behavior with the source review and corresponding checks.
Run artifact and loader smoke tests against the final packaged binaries.
