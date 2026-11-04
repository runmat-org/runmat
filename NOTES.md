# q-snapshot-5 notes (base integration/0.7.1 = f81788a4a8, T1)

Scope: website P1 items 4–6 (+ item 8 notes): structural snapshot capture profiles, structural
harmonic execution, pinned release binary for the six cantilever example captures.

## Baseline on T1 (release CLI runmat-hero-a5906bba1a, contains T1)
Receipts: runmat-build-mac:~/runs/q-snapshot-5/out/base/<study>/ (snapshot.receipt.json, verify.json).
- direct harmonic: solves (78 s, 1.2 GB). The "frequency observation source catalog is missing"
  failure is already gone on 0.7.1. verify-snapshot FAILED: "snapshot has no saved render topology"
  (structural frequency records saved no render tiles).
- modal_harmonic (runs/cantilever-modal-harmonic, modal 8 modes → modal_harmonic): reproduces
  "the second authored step must be an unresolved modal transient" (4 s). Cause: native placement
  sent every modal→X study that was not RS/random to the modal-transient coordinator; the runtime
  had no completed-result variant for modal→frequency ("ordinary modal-frequency typed publication
  has no completed study result variant").
- random vibration: snapshot drew `structural.random_response.variance.displacement` (m^2) as
  "Displacement magnitude" on a shape deformed by the variance. Cause: RMS and variance fields had
  identical layouts; the statistic was only inferred from field-id strings in the runtime.
- response spectrum: drew the envelope as a plain deformed "Displacement magnitude".
- buckling: modes captured, titled "Mode n relative amplitude" (no load factor).
- transient: the snapshot picked `transient.displacement.120` by tie order. Transient displacement,
  strain, stress and von Mises snapshots carried ordinal `Single` (velocity carried `Step(i)`).
- plastic (Tet4 100 mm): final step captured (134 s, 1.0 GB).

## Changes (see commits)
1. Typed field statistic at the producer: `runmat_analysis_core::FieldStatistic`
   (ComponentMagnitudeEnvelope, RootMeanSquare, Variance) on `AnalysisFieldLayout.statistic`
   (serde-optional; absent = a state). Random-response and response-spectrum producers set it;
   the runtime descriptor's `statistic` projects it from the layout. Removed the field-id string
   inference (`infer_field_statistic`) and the runtime-only `AnalysisFieldStatistic` enum.
2. Transient snapshots carry `Step(index)` ordinals at the producer (move_snapshot_field).
3. Structural frequency records save render tiles of the mesh they were solved on (same as standard
   runs), so verify-snapshot and figures replay a harmonic sweep from the saved record.
4. modal → modal_harmonic runs end to end: native placement routes ModalFrequency consumers to
   the existing modal-frequency coordinator; new runtime result
   `StructuralModalFrequency(StructuralStudyModalFrequencyNumericalResult)`, typed evidence,
   publication options (ModalReduced), intent completion (modal run + basis state + frequency run
   admitted against the basis via new `admit_modal_structural_frequency_publication`).
   The remote/exact cluster path does not produce this result kind yet (it never routed
   modal→frequency before either).
5. Capture profiles (snapshot/views/profiles.rs), all from typed descriptors:
   - transient: every time-stepped quantity at the step of peak nodal displacement magnitude,
     titled with the retained time (read through the transient index selection);
   - nonlinear: final step of the load path, titled "load factor X" from the per-step LoadFactor
     globals; displacement + von Mises;
   - buckling: mode shapes titled "Buckling mode n … load factor λ" from the per-mode LoadFactor list;
   - random vibration: `rms-displacement` (RootMeanSquare), undeformed; variance never drawn;
   - response spectrum: `peak-displacement` (ComponentMagnitudeEnvelope), undeformed;
   - harmonic: amplitude at the frequency of largest response (existing), titled with frequency.

## Item 8 (other families) template
The profile is family-neutral: a run's views come from (quantity, representation, statistic,
location, ordinal, sample). Per family, what is still needed:
- thermal transient / acoustic transient / fluid transient: the peak-instant rule keys on nodal
  displacement; other families show their latest step. Peak-instant for them needs a typed
  "primary response" per run (e.g. temperature, acoustic pressure, velocity) — choose by quantity
  priority in DISPLAYS rather than displacement only.
- EM / electrothermal / acoustics harmonic: frequency-peak choice already works per quantity.
- Any family's statistical outputs: set `FieldStatistic` at the producer, add a Display row.

## Handover state (2026-11-04, hand-back at context cap)
Branch q-snapshot-5 (worktree /Users/nallana/.claude/jobs/b8482b6b/tmp/q-snapshot-5-wt):
- 577df8da85 typed FieldStatistic on the layout; transient snapshots carry Step ordinals
- 6216c93a57 structural frequency records save render tiles (verify-snapshot for harmonic)
- c6478b011e modal -> modal_harmonic end to end; snapshot capture profiles + 7 profile tests
- 3fe0e42ec0 merge integration/0.7.1 (T1.1 = 7507f51f46)
NONE of this has compiled yet (the earlier build attempts died: disk < 100 GB, then the restart).
One earlier compile found a missing re-export (fixed before commit c6478b011e).

Running on runmat-build-mac (detached with nohup, survives a main-Mac restart):
- PID 34361 ~/runs/q-snapshot-5/build.sh -> gate build: `cargo test --no-run --message-format short
  -p runmat -p runmat-analysis-core -p runmat-analysis-fea -p runmat-runtime
  -p runmat-execution-runner-native --lib`, src ~/src/q-snapshot-5 (synced at 3fe0e42ec0),
  target ~/targets/q-snapshot-5, log ~/runs/q-snapshot-5/test1.log (ends with "EXIT <code>").
  Read errors with: tr "\r" "\n" < test1.log | grep -a -E "^[^ ].*(error|warning)|^EXIT"

Next steps for the successor:
1. Fix compile errors, if any (likely spots: modal_frequency_completion.rs, storage loaders,
   profiles.rs StoredValues::state_time uses page.data.descriptor.sample).
2. Run tests (detached, via build.sh with `test --offline ...`):
   - runmat (CLI) lib: `snapshot::views::profiles` (13 tests incl. 7 new);
   - runmat-runtime lib: `statistical_field_descriptors_preserve_statistic_units_and_locations`,
     storage frequency tests, `analysis::` frequency/typed publication tests;
   - runmat-analysis-fea lib: transient recovery + statistical_fields tests;
   - runmat-analysis-core lib: field tests.
   Watch for tests asserting transient snapshot ordinals == Single (changed to Step).
3. Release: `$J/bin/fea-release <worktree> snap5-<hash>` (J=/Users/nallana/.claude/jobs/b8482b6b/tmp).
4. Captures: `~/runs/q-snapshot-5/capture.sh snap5-<hash> final harmonic modal-harmonic
   random-vibration response-spectrum buckling transient plastic` (detached with nohup; it waits for
   the RunMat server at 127.0.0.1:8080 to answer for 3 min and retries ServerUnreachable 3x;
   studies in ~/runs/q-snapshot-5/studies/cantilever-*/, output snapshot-final/meta.json +
   out/final/<study>/{snapshot,verify}.{log,json}, summary out/final/summary.txt).
   Expected views: harmonic `displacement · <f> Hz`; random `rms-displacement` (m, undeformed);
   RS `peak-displacement`; buckling `mode-1/2 … load factor 410.xx`; transient displacement/
   stress/velocity at the peak step `t = … ms`; plastic displacement + stress `load factor 1.000`.
5. Copy scenes to the main Mac and run
   `node /Users/nallana/.claude/jobs/b8482b6b/tmp/work/q-snapshot-4/viewer/check-scenes.mjs <scene.json...>`
   (fetches the renderer from :1001 read-only). Scene format is unchanged by this branch.
6. Post /tmp/sister-comms/fea-pinned-build-<hash>.txt: tag, commit, `runmat --version`,
   `shasum -a 256 ~/fea-cli/runmat-snap5-<hash>`, six snapshot paths + verify results.
Baseline receipts (T1 CLI runmat-hero-a5906bba1a): ~/runs/q-snapshot-5/out/base*/.

## q-snapshot-6 (2026-11-04)
- Build of 82d1a8439e compiled clean (test1.log EXIT 0).
- Tests at 82d1a8439e (logs runmat-build-mac:~/runs/q-snapshot-5/t-*.log):
  - runmat-analysis-core --lib: 152 passed, 0 failed.
  - runmat-analysis-fea --lib (QUAL skip list): 2095 passed, 0 failed, 13 ignored.
  - runmat-runtime --test analysis: 7 passed.
  - runmat (CLI) --lib snapshot: 20 passed.
  - runmat-runtime --lib: 17824 passed, 33 failed. 29 are builtins (runtime-sister area); 4 are analysis::tests:
    native_sparse_fea_cancellation_is_terminal_and_releases_resources (pre-existing per q-thermal-field NOTES),
    analysis_run_linear_static_with_thermo_mechanical_coupling_reports_fields and
    analysis_run_transient_returns_native_transient_result (scalar component_count Some(1) vs None; contracts.rs:1314
    from_entry, not touched here), analysis_run_modal_returns_native_modal_result (orthogonality Vector vs Tensor).
  - runmat-execution-runner-native --lib needs --features occt-native (without it, dead-code lint on a
    test helper, pre-existing). With it: 459 passed, 4 failed (remote driver_replacement x3, fea_proxy reservation;
    remote paths, not in this diff).
- Release snap5-82d1a8439e: six cantilever studies captured + verified (out/final/). modal-harmonic failed at
  publication; fixed in ae15407493 (defaults_for harmonic ModalReduced) and 8f04be4280 (intent ModalFrequency arm).
- 8f04be4280 runtime --lib analysis:: : 428 passed, 4 failed (same 4 as above); new assertion in
  solver_defaults_admit_only_implemented_exact_procedures passes.
- PINNED BUILD: snap5-8f04be4280 (runmat-build-mac ~/fea-cli, sha256 7b9a654f…578c0f, `runmat 0.6.2`).
  Six studies recaptured + verified with it: studies/cantilever-<s>/snapshot-pinned/, receipts out/pinned/<s>/.
  16/16 scenes import with the :1001 renderer 6c140ff499d2. Posted sister-comms/fea-pinned-build-8f04be4280.txt.
- OPEN: modal -> modal_harmonic now solves and publishes, but it is rejected by the strict gate in
  runtime frequency_result/admission.rs: the physical balance error is 0.0923 (8-mode truncation) against the
  1e-8 tolerance, while the reduced solver residual is about 5e-17. The same gate applies to prestressed
  modal-reduced sweeps. A decision is needed on the modal-reduced criterion: gate the reduced residual and
  report balance as a truncation measure, or add residual (static-correction) vectors to the basis
  (residual_vector_directions exists in the runtime options but not in the .fea language).
  Receipt: out/final3/modal-harmonic/, run manifest in studies/cantilever-modal-harmonic/.artifacts.

## q-snapshot-7 (2026-11-04, branch q-snapshot-7 from integration/0.7.1 = eb3dacbd25)
Work dirs on runmat-build-mac: src ~/src/q-snapshot-5, target ~/targets/q-snapshot-5 (pruned to the
two newest hashes per workspace crate: 109 GB -> 47 GB), runs ~/runs/q-snapshot-7/.
1. Modal-reduced frequency acceptance:
   - runtime frequency_result/admission.rs: `classify_structural_frequency_quality` — Direct is
     gated on solver residual and full-order balance (unchanged 1e-8); ModalReduced on the
     reduced-space residual only. Full-space balance is published, never gated:
     `ModalTruncationBalance` (fea receipt accessor `modal_truncation_balance()`) and
     `AnalysisResultsSummary.modal_truncation_balance_error` / `_frequency_hz`
     (`max_frequency_balance_error` is None for modal-reduced).
   - Language: `structural modal|prestressed_modal { residual_vectors = [[0,0,1], ...] }`
     (inertia residual vectors, existed in runtime) and modal-reduced frequency steps
     `static_correction = applied_loads | none` (new: K^-1 f at the plan's reference frequency,
     mass-orthogonalized against modes + residual vectors). Core: ProcedureSettings::
     StructuralModalBasis / StructuralModalFrequency. Grammar group "modal static corrections";
     LSP hover; ledger capability structural.modal-reduced.residual-vectors; procedures.md section.
   - ModalReducedSystem::project_enriched (modes + basis residual vectors + static corrections)
     is used by the frequency path only; transient/RS/random keep project().
2. analysis::tests failures (exist at 7507f51f46: typed layouts came in a38ce761ae, in T1.1):
   - component_count scalar Some(1): tests held the stale truth (producer from_entry/from_field
     derive it from the typed value shape) -> tests updated.
   - orthogonality kind: producer defect (Tensor{[n]} per mode classified Vector when n=3) ->
     modal.m_orthogonality is one Global Tensor{[n,n]}; test expects component_count n*n.
   - cancellation: producer defect, coercivity symbolic analysis mapped SolverError::Cancelled to
     SymbolicOrdering -> FEA_INVALID_REAL_SYSTEM; now CoercivityError::Cancelled.
3. Snapshot profiles: PRIMARY_RESPONSES picks a transient's instant per family (displacement,
   acoustic pressure, fluid velocity at peak; temperature at the final state); new real
   "sound-pressure" display for acoustic transients; 6 family tests.
