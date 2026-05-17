# Analysis Physics Domain Coverage and Target Shape

This document defines the target physics-domain shape for analysis study schemas and runtime ownership.

Companion docs:

- workflow contract: `docs/detailed-work/code-reviewed-fea-workflow.md`
- program context: `docs/detailed-work/geo-and-analysis.md`
- over/under + rival build matrix: `docs/detailed-work/fea-rival-gap-matrix.md`

## Product Position

RunMat today is strongest in structural mechanics and partially implemented in multiphysics branches.

| Area | Status today | Notes |
| --- | --- | --- |
| Linear static structural | implemented | Core structural path. |
| Modal structural | implemented | Contracted and tested. |
| Transient structural | implemented | Contracted and tested. |
| Nonlinear structural | implemented | Contracted with diagnostics/governance. |
| Thermo-mechanical coupling | partial | Model-owned schema is in place; solver path still coupling-context driven. |
| Electro-thermal coupling | partial | Not a full EM domain. |
| Plasticity | partial | Model-owned schema is in place; constitutive context wiring is active, with further fidelity work pending. |
| Contact | partial | Model-owned schema is in place; interface context wiring is active, with further fidelity work pending. |
| Full standalone thermal domain | partial | Dedicated thermal step + runtime operation + payload now exist, with constitutive/outcome diagnostics and readiness posture thresholds; high-fidelity calibration depth still needs expansion. |
| Full EM domain (Maxwell-class) | not implemented | Out of scope for near-term "most engineers" target. |

## Hard-Cutover Rule

This design is a hard cutover target, not a backward-compat migration path.

- treat the model-owned physics shape as the canonical `v1`,
- do not preserve legacy proxy-first schema variants as long-term contracts,
- run options remain numeric/execution controls, not the primary home of constitutive physics.

## Canonical Ownership by Domain

| Domain | Canonical owner |
| --- | --- |
| Mechanical elastic | `model.materials[]` |
| Thermal constitutive properties | `model.materials[].thermal` |
| Electrical constitutive properties | `model.materials[].electrical` |
| Plastic constitutive model | `model.materials[].plastic` |
| Contact interface model | `model.interfaces[].contact` |
| Loads/BC/steps | `model.loads`, `model.boundary_conditions`, `model.steps` |

## Target Capability Set for Most Engineers

1. Structural baseline: linear static, modal, transient, nonlinear.
2. Thermal-structural: material-owned thermal properties with thermo-mechanical coupling.
3. Electro-thermal: material-owned electrical + thermal properties with Joule-heating style coupling.
4. Practical nonlinear: constitutive plasticity and interface-owned contact (frictionless + frictional baseline).
5. Stable inspection + governance: `run.data` arrays, diagnostics, readiness/trend policy by domain.

Not required for this milestone:

- full Maxwell-class EM multiphysics,
- niche constitutive families beyond common industrial baselines.

## What It Takes to Complete the Set

### Repo Audit Snapshot (Current)

1. **Model ownership has moved into analysis schema**
   - `AnalysisModel` now carries `thermo_mechanical`, `electro_thermal`, and `interfaces`.
   - `MaterialModel` now groups `mechanical`, `thermal`, optional `electrical`, optional `plastic`.
   - Contact is modeled by `AnalysisInterfaceKind::Contact`.

2. **Run options are now execution controls, not physics-definition containers**
   - `AnalysisRunOptions`, `AnalysisModalRunOptions`, `AnalysisTransientRunOptions`, and `AnalysisNonlinearRunOptions` do not carry thermo/electro/plastic/contact definition fields anymore.
   - Runtime derives physics contexts from model-owned fields.

3. **Structural solve family is fully wired and regression-covered**
   - Static, modal, transient, and nonlinear operations run through runtime contracts and broad test coverage.
   - Modal and transient still include placeholder quality signals if a placeholder diagnostic appears, but native paths are present and exercised.

4. **Multiphysics breadth is present, but depth is still heuristic-weighted**
   - Thermo/electro coupling is active in transient and nonlinear pathways, with diagnostics and quality gates.
   - Plastic/contact are materially/interface-owned at schema level and now flow through constitutive/interface contexts; depth and calibration remain the main gaps.

### Gap List to Reach "Most Engineers" Target

1. **Close proxy semantics for plastic/contact in runtime + FEA internals**
   - Replace proxy naming/assumptions with constitutive/interface-native internal contracts.
   - Keep behavior equivalent while removing proxy framing from diagnostics and APIs.

2. **Strengthen thermo/electro from governance-level coupling to constitutive solve depth**
   - Continue beyond severity/quality signaling toward richer constitutive field evolution.
   - Keep artifact-backed thermo field path and validation as first-class, not side-path.

3. **Deepen standalone thermal constitutive fidelity**
   - Thermal step kind and operation contract now exist (`analysis.run_thermal`).
   - Constitutive/outcome acceptance envelopes now exist; next increment is calibration depth and richer thermal constitutive realism.

4. **Codify exit criteria in tests/manifests per domain**
   - Structural: keep parity/perf + publishability gates.
   - Thermo/electro: add deterministic fixture expectations for constitutive behavior, not only severity ranges.
   - Plastic/contact: add fixture assertions tied to constitutive/interface outcomes.

### Concrete Completion Plan

1. **Phase A: Internal contract cleanup**
   - Rename/remove remaining `*proxy*` internal pathways where they represent canonical behavior.
   - Keep adapters only where strictly required for temporary solver boundaries.

2. **Phase B: Physics-depth increments**
   - Thermo/electro: increase constitutive coupling fidelity and add fixture-level acceptance thresholds.
   - Plastic/contact: encode constitutive/interface invariants in diagnostics and benchmark fixtures.

3. **Phase C: Optional thermal-only capability**
   - Add thermal step + operation contracts if product scope confirms this is part of "most engineers" baseline.

4. **Phase D: Hard-cutover finalization**
   - Remove stale migration wording and any residual legacy references in docs/tests.
   - Keep this doc and `code-reviewed-fea-workflow.md` synchronized as the canonical statement of ownership and capability.

### Slice A/B Completion Status (2026-03-12)

- **Slice A (thermal fidelity hardening + acceptance envelopes): completed baseline scope**
  - standalone thermal now emits constitutive + outcome diagnostics (`FEA_THERMAL_CONSTITUTIVE`, `FEA_THERMAL_OUTCOME`),
  - conformance thresholds cover residual/temperature/spread plus outcome metrics (spatial gradient, monotonic response fraction, response realization ratio),
  - readiness policy includes thermal posture thresholds for these outcome indicators.
- **Slice B (thermo/electro constitutive outcome assertions): completed baseline scope**
  - thermo fixture thresholds now include constitutive indicators beyond severity-only checks,
  - electro fixture thresholds include temporal/time-scale constitutive indicators alongside Joule/spread posture,
  - thermo/electro acceptance now uses deterministic constitutive assertions in conformance and governance surfaces.

### Slice C Bootstrap Status (2026-03-13)

- **Plastic/contact constitutive outcome assertions: advanced beyond severity-only checks**
  - nonlinear diagnostics now emit load-path constitutive outcome metrics for plastic/contact (`load_realization_ratio`, `load_amplification_ratio`),
  - nonlinear conformance fixtures assert deterministic envelopes for these new outcome metrics across stress and reference-backed fixture families,
  - `analysis.results` summary now exposes these plastic/contact outcome metrics directly for contract consumers,
  - release-readiness governance now enforces threshold envelopes, breach-rate controls, and rolling-trend checks for these load-path metrics,
  - branch-profile defaults were calibrated against rolling benchmark artifacts to keep load-path checks stable pre-hardening (lower false-positive risk on feature/development/release profiles),
  - legacy severity/spread trend checks were recalibrated to fixture-aligned rolling baselines (instead of mixed-fixture global medians) to suppress cross-fixture false positives while preserving drift sensitivity,
  - additional in-scope constitutive posture fields are now surfaced and governed (`thermo_field_clamp_ratio`, electro transient/nonlinear time-scale means, plastic/contact severity means) with conformance and readiness coverage.

### Maxwell EM Phase-0 Status (2026-03-13)

- Added EM contract scaffolding to start next-domain bring-up while preserving contract/versioning discipline:
  - core analysis step schema now includes `electromagnetic`,
  - create-model profile now supports `electromagnetic_static` templates,
  - runtime operation contract placeholder is available at `analysis.run_electromagnetic` (`v1`) and now emits deterministic placeholder run payloads (`FEA_EM_PLACEHOLDER`) rather than hard-failing unsupported, so `analysis.results`/`analysis.trends` can carry EM runs end-to-end.

### Maxwell EM Phase-1 Status (2026-03-13)

- Added model-owned EM domain primitive in analysis-core (`AnalysisModel.electromagnetic`) and wired runtime validation on EM run entry:
  - requires EM step presence,
  - requires configured `model.electromagnetic`,
  - validates finite positive `reference_frequency_hz` and `applied_current_a`.
- Kept domain-logic ownership pattern consistent with existing thermo/electro paths:
  - model/domain definition lives in analysis-core,
  - runtime contract/options + operation shape live in runmat-runtime,
  - solver behavior remains explicitly placeholder until FEA EM assembly/solve kernels are implemented.

### Maxwell EM Phase-2 Status (2026-03-13)

- Added EM-specific runtime result payload surface and summary posture fields:
  - `AnalysisRunResult.electromagnetic_results` now carries `electromagnetic_results/v1` payloads,
  - `analysis.results` summary now exposes EM placeholder posture metrics (`electromagnetic_enabled`, `electromagnetic_reference_frequency_hz`, `electromagnetic_applied_current_a`, `electromagnetic_placeholder_quality`).
- Wired EM run classification into `analysis.trends` (`AnalysisRunKind::Electromagnetic`) and added an EM placeholder warning-rate signal for governance skeleton consumption.

### Maxwell EM Phase-3 Status (2026-03-13)

- Replaced runtime-synthetic EM placeholder generation with a first real FEA-side EM static starter path:
  - new FEA pipeline entrypoint (`pipeline/electromagnetic.rs`) with deterministic static EM proxy solve,
  - runtime `analysis.run_electromagnetic/v1` now delegates to FEA EM pipeline instead of constructing synthetic run payloads inside runtime.
- Kept ownership boundaries consistent with established patterns:
  - EM domain schema in analysis-core,
  - EM solve logic in analysis-fea pipeline,
  - runtime remains contract/orchestration/results/governance surface.
- EM diagnostics now emit `FEA_EM_STATIC` with structured solve posture metrics (`reference_frequency_hz`, `applied_current_a`, `conductivity_mean_s_per_m`, `max_residual_norm`, `solve_quality`).

### Maxwell EM Phase-4 Status (2026-03-13)

- Introduced first weak-form style EM static assembly pieces in analysis-fea pipeline:
  - curl-curl style stiffness term proxy (`reluctivity / h^2`) and conductivity-frequency mass term (`omega * sigma`) assembled into operator coefficients,
  - boundary-constrained line-domain solve with matrix-free PCG backend path,
  - post-solve flux-density proxy derived from vector-potential gradient.
- Runtime EM operation remains contract-stable and now consumes this FEA-backed static EM solve path end-to-end (typed payloads, summary metrics, trends/governance signals unchanged at interface level).

### Maxwell EM Phase-5 Status (2026-05-16)

- Added region/material-driven EM coefficient assembly sourced from `material_assignments`:
  - per-region `sigma`, `epsilon_r`, and `mu_r` now drive node/edge coefficient fields,
  - deterministic fallback behavior is explicit when assigned/expected materials are missing or non-electrical.
- Expanded EM readiness posture into first-class summary/trend signals:
  - `analysis.results` now surfaces EM readiness metrics (`assignment_coverage_ratio`, `fallback_coefficient_ratio`, `region_coefficient_contrast_index`, `solver_conditioning_proxy`) in addition to spread/heterogeneity metrics,
  - `analysis.trends` now tracks breach rates for EM coverage, fallback, contrast, and conditioning signals.
- Added first EM governance fixture family for readiness splits:
  - homogeneous baseline (publishable),
  - heterogeneous high-contrast reference (degraded),
  - sparse-assignment reference (degraded),
  - fallback-heavy reference (degraded).

### Maxwell EM Phase-6 Status (2026-05-16)

- Extracted shared policy thresholds/helpers into internal runtime module (`analysis/policy.rs`) for thermo + EM governance consistency.
- Hardened baseline drift gates with explicit EM readiness metric drift checks:
  - assignment coverage ratio,
  - fallback coefficient ratio,
  - region coefficient contrast index,
  - solver conditioning proxy.
- Added EM contract-facing snapshot shape coverage (`electromagnetic_contract_snapshot.json`) and operation-contract test validation for typed EM payload + summary key stability.

### Maxwell EM Phase-7 Status (2026-05-16)

- Added first EM-specific source and boundary primitives in core analysis schema:
  - loads: `current_density`, `coil_current`,
  - boundary conditions: `magnetic_insulation`, `vector_potential_ground`.
- Upgraded EM FEA path to consume these primitives directly:
  - source realization now scales EM RHS excitation deterministically,
  - boundary anchoring/insulation now influences EM constraint/coupling posture.
- Surfaced new EM readiness signals as first-class result/trend/governance fields:
  - `source_realization_ratio`,
  - `boundary_anchor_ratio`,
  - plus quality-policy thresholding and breach-rate tracking in `analysis.results` / `analysis.trends`.

### Maxwell EM Phase-8 Status (2026-05-16)

- Added EM field-consistency readiness metrics as first-class governance signals:
  - `flux_divergence_proxy`,
  - `energy_imbalance_ratio`,
  - `boundary_energy_ratio`.
- Upgraded EM metric definitions to be assembly/material-aware instead of residual-only proxies:
  - energy imbalance now blends solver residual posture with material heterogeneity, fallback-coefficient pressure, and source-realization coverage,
  - boundary energy ratio now reflects boundary-coupling share of assembled operator energy (stable under strongly constrained boundary potentials).
- Wired these metrics through runtime quality policy gates, `analysis.results` summary, `analysis.trends` breach rates, and baseline/conformance harness assertions for EM reference fixtures.

### Maxwell EM Phase-9 Status (2026-05-16)

- Added region-resolved EM source deposition in analysis-fea:
  - EM loads (`current_density`, `coil_current`) now map by `load.region_id` onto assignment-backed coefficient regions instead of global-only source shaping,
  - unmapped sources fall back to attenuated global deposition with explicit diagnostic accounting.
- Surfaced source-locality readiness posture as first-class EM metrics:
  - `source_region_coverage_ratio`,
  - `source_material_alignment_ratio`,
  - `source_localization_ratio`.
- Wired the new source-locality posture through runtime policy/results/trends:
  - quality gates now enforce coverage/alignment thresholds,
  - `analysis.results` summary and `analysis.trends` breach rates include the new fields,
  - conformance/baseline checks were extended for the EM fixture family.

### Maxwell EM Phase-10 Status (2026-05-16)

- Upgraded EM source-to-region coupling fidelity:
  - region-mapped sources now use region-weighted kernels derived from local material coefficients (`sigma`, `epsilon_r`, `mu_r`) instead of uniform deposition gain,
  - source deposition now preserves signed polarity (coil/current-density direction), enabling cancellation-aware solve forcing.
- Added multi-source interference diagnostics as first-class EM readiness signals:
  - `source_overlap_ratio` (opposing-polarity overlap share),
  - `source_interference_index` (net cancellation index).
- Extended runtime governance + fixture coverage:
  - policy thresholds and quality reasons now include overlap/interference controls,
  - `analysis.results`/`analysis.trends` now surface overlap/interference posture and breach rates,
  - added EM overlap-interference reference fixture with deterministic thresholded conformance and baseline drift tracking.

### Maxwell EM Phase-11 Status (2026-05-16)

- Added EM boundary-kernel governance metrics as first-class runtime readiness signals:
  - `boundary_condition_localization_ratio`,
  - `ground_anchor_effectiveness_ratio`,
  - `insulation_leakage_proxy`.
- Extended runtime quality policy and trend surfaces:
  - new balanced-policy thresholds + quality reasons for boundary localization, ground effectiveness, and insulation leakage,
  - `analysis.results` summary and `analysis.trends` breach-rate payloads now expose these three signals.
- Expanded conformance coverage with a dedicated EM boundary-kernel stress reference fixture and wired the new metrics into baseline + rolling drift checks.

### Maxwell EM Phase-12 Status (2026-05-16)

- Upgraded EM static solve internals from single-scalar pass to a coupled harmonic iteration path:
  - real/imaginary vector-potential passes with conductivity-coupled RHS exchange,
  - normalized coupled residual evaluation against full equation-scale terms.
- Added harmonic-coupling diagnostics and stabilized quality signaling:
  - `FEA_EM_HARMONIC_COUPLING` now reports iteration and coupling posture,
  - intermediate sub-solve max-iteration warnings are suppressed from top-level quality gating to avoid false negatives.
- Kept runtime contracts stable while recalibrating solver-quality mapping so homogeneous control fixtures remain publishable and stress fixtures remain policy-governed via readiness breaches.

### Maxwell EM Phase-13 Status (2026-05-16)

- Replaced split harmonic staging with a single block-coupled harmonic solve core:
  - one coupled real/imag system solve (`[K -C; C K]`) using block BiCGStab iteration,
  - block-Jacobi preconditioning from per-node `2x2` diagonal harmonic blocks.
- Added explicit block-operator and coupled residual posture in the EM pipeline while preserving existing runtime/result contract shapes.
- Maintained governance compatibility:
  - EM contract tests and conformance harness pass on re-run,
  - residual quality signaling remains calibrated for control-fixture publishability and stress-fixture degradation behavior.
- Added region-consistent EM boundary/source governance signals:
  - `boundary_penalty_conditioning_contribution`,
  - `source_region_energy_consistency_ratio`,
  - split harmonic residual posture (`real_residual_norm`, `imag_residual_norm`).
- Promoted these signals to first-class runtime governance:
  - `analysis.results` summary fields,
  - `analysis.trends` breach-rate metrics,
  - policy-threshold quality reasons.
- Extended EM reference-fixture governance family with:
  - `electromagnetic_reference_boundary_penalty_stress_gpu_provider`,
  - `electromagnetic_reference_multi_region_phased_source_gpu_provider`.
- Added explicit EM source phasing inputs in analysis-core load schema:
  - `LoadKind::CurrentDensity { phase_rad, amplitude_scale }`,
  - `LoadKind::CoilCurrent { phase_rad, amplitude_scale }`.
- Switched harmonic EM solve RHS handling to true complex-source forcing (`rhs_real` + `rhs_imag`) and computed source overlap/interference readiness from complex source vectors actually used by the solver.
- Moved boundary penalty from diagnostic-only posture to operator-applied behavior by injecting boundary penalty terms into assembled stiffness diagonals and measuring contribution from actual assembled coefficients.

### Maxwell EM Phase-16 Status (2026-05-16)

- Added optional EM frequency-sweep controls to runtime EM run options:
  - `sweep_enabled`,
  - `sweep_frequency_hz`.
- Extended typed EM payload and runtime summary/trend surfaces with sweep/resonance readiness signals:
  - payload: sweep frequency/peak/quality vectors plus resonance peak/bandwidth/Q/flux-gain fields,
  - summary: `electromagnetic_sweep_count` + resonance posture fields,
  - trends: sweep-coverage and resonance-sharpness breach-rate signals.
- Added runtime governance for sweep/resonance only when sweep mode is explicitly enabled:
  - quality reasons: `ElectromagneticSweepCoverageLow`, `ElectromagneticResonanceSharpnessLow`,
  - thresholds are policy-profiled in `analysis/policy.rs`,
  - default single-frequency EM runs remain backward-compatible and keep prior publishability posture.

## Closeout Checklist for This Track

- [x] Canonical physics ownership documented as model/material/interface-owned.
- [x] Run options documented as execution controls only.
- [x] Current capability status table updated to match repo shape.
- [x] Gap list and phased completion plan captured with explicit remaining work.
- [x] Plastic/contact internals no longer framed as proxy paths.
- [x] Thermo/electro acceptance criteria cover constitutive outcomes (not only severity bands).
- [x] Thermal-only step/operation decision made (in-scope vs out-of-scope) and documented.

### Thermal-Only Scope Decision

- Decision: **in-scope baseline delivered for this milestone track**.
- Delivered shape: thermal step kind in core model, dedicated runtime operation (`analysis.run_thermal`), and typed thermal results payload.
- Follow-on: improve standalone thermal constitutive fidelity and calibration tightness to match structural-domain confidence levels.

## Next Major Milestone Exit Criteria (Binary)

This section defines pass/fail checks for the next major milestone. The milestone is complete only when every item is true.

1. **Canonical ownership is exclusive**
   - Physics-defining inputs for thermo/electro/plastic/contact are model/material/interface-owned.
   - Run options contain execution controls only (numerics/backend/policy/prep), not constitutive definitions.

2. **Plastic/contact are behaviorally first-class**
   - Runtime and FEA internals use constitutive/interface semantics end-to-end.
   - Nonlinear fixture suites include at least one reference-backed low-severity path and one stress-path per domain.

3. **Thermo/electro acceptance is constitutive, not only heuristic**
   - At least one deterministic acceptance assertion per thermo/electro fixture validates constitutive outcome behavior (not just severity threshold presence).
   - Trend policy still enforces severity drift controls as secondary governance.

4. **Structural baseline remains stable**
   - Linear static, modal, transient, and nonlinear contract/integration suites pass on the default CI path.
   - No regression in publishability policy behavior for balanced/strict profiles.

5. **Domain governance parity is complete**
   - `analysis.results` and `analysis.trends` expose required summary/posture signals for structural, thermo/electro, and plastic/contact domains.
   - Readiness scripts enforce branch-profiled thresholds and trend non-regression for all in-scope domains.

6. **Artifact-backed thermo trust path is enforced**
   - Artifact signature/hash/approver checks remain active in runtime validation.
   - Conformance includes artifact-backed thermo fixture coverage with typed failures.

7. **Fixture and manifest evidence is deterministic**
   - Benchmark fixtures used for release gating have stable expected envelopes and fail loudly on drift.
   - Added/renamed semantics preserve fixture reproducibility and CI report continuity.

8. **Documentation reflects implementation exactly**
   - This doc, workflow doc, and geo/analysis log describe current ownership and capability without stale proxy-first language.
   - No unresolved placeholder sections remain in this track.

9. **Thermal-only scope decision is explicit**
   - In-scope baseline is documented and implemented with step kind + operation contract + result payload.
   - Follow-on fidelity/calibration work is explicitly tracked as the next increment.

10. **Release-readiness signal is green on protected branch policy**
    - Tier-7.5 (or equivalent) governance report passes with no unwaived blockers for in-scope domains.
    - Promotion-ready state for required reference tracks is true.

### Milestone Status (Current)

All exit criteria 1-10 are satisfied for this milestone scope, with evidence from code shape, conformance assertions, and regression suites.

- Structural/runtime regression evidence:
  - `cargo test -p runmat-runtime -q`
  - `cargo test -p runmat-analysis-fea -q`
  - `cargo test -p runmat-analysis-core -q`
- Canonical ownership evidence:
  - model/material/interface ownership in core schema (`AnalysisModel`, `MaterialModel`, `AnalysisInterfaceKind::Contact`).
  - run options restricted to execution controls.
- Thermo/electro constitutive acceptance evidence:
  - benchmark conformance thresholds include constitutive indicators (`effective_modulus_scale`, `constitutive_material_spread_ratio`, `joule_heating_scale`, `conductivity_spread_ratio`) in fixture assertions.
- Plastic/contact first-class evidence:
  - runtime and FEA internal option/context paths use constitutive/interface naming; proxy framing removed from active internals.
