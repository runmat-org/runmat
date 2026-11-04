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
