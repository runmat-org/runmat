# Multi-Physics Parity Roadmap

## Purpose

Define the development plan from the current Tier-7.5 state to a stellar programmatic multi-physics FEA platform that can credibly rival ANSYS/COMSOL for code-first workflows.

This roadmap is internal engineering guidance. Customer-facing capability messaging should remain in `docs/analysis/prep-aware-solves.md` and `docs/geometry/prep-for-analysis.md`.

## Current Baseline

### Strengths already in place

- Programmatic runtime/contract surfaces for geometry prep, analysis model creation, solves, results, and trends.
- Deterministic prep-aware assembly pipeline with topology, region, element, connectivity, graph, and solver-shaping tiers.
- Calibration/acceptance governance with drift detection, recommendation artifacts, evidence validation/promotion, and CI/release policy integration.
- Strong reproducibility posture (artifact lineage, diagnostics, replay checks, release-readiness gates).

### Major remaining gap to ANSYS/COMSOL parity

- Physics/modeling breadth and depth (element formulations, coupled PDE sets, robust nonlinear/contact/material behavior, advanced solvers, V&V corpus, and HPC scaling).

## Strategy

Do not pursue "platform 100% first" as a serial gate. Use co-development tracks:

- Track A (Physics thin slices): deliver end-to-end credible multiphysics slices with benchmark parity targets.
- Track B (Platform hardening): evolve contracts/artifacts/tooling only where Track A pressure proves reuse needs.

Planned allocation:

- Phase 1-2: 60% physics slices / 40% platform.
- Phase 3-4: 50% / 50%.
- Phase 5+: 40% physics slice expansion / 60% solver/perf/ops scaling.

## Program Principles

- Determinism first: same inputs must yield stable artifacts/diagnostics/fingerprints.
- Additive contracts: avoid breaking operation payloads; evolve with optional fields and schema versioning.
- Benchmark-driven acceptance: no major feature considered complete without reference benchmark thresholds.
- Programmatic UX over UI parity: optimize agent-authored code/config workflows instead of replicating GUI interaction models.
- Governed promotion: calibration/evidence updates must pass explicit promotion gates.

## Target Architecture End-State

### Authoring layer

- RunMat code/config DSL that can express multiphysics domains, couplings, materials, BC/load schedules, studies, and parametric sweeps.
- Agent-assisted generation/validation of models and studies.

### Solve layer

- Native element/PDE assembly for each physics family.
- Coupled solve orchestration (monolithic and partitioned strategies as appropriate).
- Adaptive controls (time stepping, continuation, nonlinear damping, contact stabilization, error indicators).

### Verification/governance layer

- Reference benchmark suite by physics and coupling class.
- Structured acceptance metrics and trend/drift governance.
- Release readiness policy profiles tied to branch/release risk posture.

## Phased Plan

## Phase 1: Credible Single Coupled Slice (Thermo-Mechanical)

### Objectives

- Deliver industrially credible thermo-mechanical coupling in programmatic workflow.
- Prove that current platform abstractions hold under real coupled PDE pressure.

### Scope

- Heat conduction + linear thermoelastic coupling.
- Temperature-dependent material properties (first-order support).
- Coupled study definitions (steady and transient basic).
- Post-processing fields: temperature, displacement, stress, thermal strain contributions.

### Deliverables

- New coupled operation profile(s) and study schema additions.
- Element assembly path for thermal and mechanical operators with coupling blocks.
- Coupled diagnostics (assembly/solver/convergence/coupling residual).
- Reference benchmarks: canonical beam/plate thermal gradient and constrained expansion cases.

### Exit criteria

- Benchmarks meet predefined tolerance envelopes vs reference solutions.
- Deterministic replay for diagnostics/artifacts across repeated runs.
- CI includes coupled benchmark and readiness checks.

## Phase 2: Solver Robustness and Nonlinear Foundation

### Objectives

- Raise solver reliability for difficult coupled conditions.
- Prepare for contact/plasticity-level complexity.

### Scope

- Robust nonlinear controls: line search, adaptive damping, continuation ramps.
- Improved preconditioner strategy selection for coupled block systems.
- Time-step adaptivity and stiffness-aware controls.

### Deliverables

- Coupled solver policy profiles (fast/balanced/robust).
- Structured solver-performance metrics in results/trends (iterations, residual history, fallback reasons).
- New stress tests for convergence edge cases.

### Exit criteria

- Convergence success rate and bounded runtime variance across nonlinear benchmark set.
- No unexplained drift in acceptance metrics over rolling windows.

## Phase 3: Second Coupled Family (Electro-Thermal or Fluid-Thermal)

### Objectives

- Validate portability of abstractions across a distinct physics family.
- Expose abstraction gaps before broad expansion.

### Scope

- Preferred path: electro-thermal Joule heating coupling (lower initial turbulence complexity).
- Alternative path: incompressible fluid + thermal transport (if CFD stack foundation is prioritized).

### Deliverables

- New coupled PDE assembly blocks and coupling terms.
- Family-specific benchmark suite with trend governance.
- Shared coupling APIs generalized only where duplicate logic is proven.

### Exit criteria

- Two coupled families operate under common study contracts without feature forks.
- Regression suite and readiness policies cover both families.

## Phase 4: Materials and Contact Depth

### Objectives

- Close major realism gap for structural and coupled analyses.

### Scope

- Elastic-plastic constitutive support (start with isotropic hardening).
- Basic contact mechanics (frictionless contact, gap/penalty/augmented options).
- Material model extensibility hooks for future creep/viscoelastic models.

### Deliverables

- Nonlinear material/contact diagnostics and stabilization controls.
- Contact/material benchmark pack.

### Exit criteria

- Stable convergence on representative contact/plastic scenarios.
- Verified stress/strain response against reference data for selected models.

## Phase 5: Performance and Scale

### Objectives

- Move from correctness-first to production-scale throughput.

### Scope

- Sparse matrix/data layout optimization.
- Multi-thread/distributed decomposition roadmap.
- GPU kernels for broader operator components, not just selected paths.
- Memory and latency profiling in CI trend baselines.

### Deliverables

- Performance SLO dashboard metrics (size/runtime/iterations/memory).
- Scale benchmarks (problem sizes and mesh complexity classes).

### Exit criteria

- Demonstrated scaling curve improvements on target hardware classes.
- Release readiness includes performance regression gates for key workloads.

## Phase 6: Productization for Programmatic UX

### Objectives

- Make model/study authoring and result analysis efficient for agent-authored and human-authored code/config.

### Scope

- Rich model/study schema documentation and validation errors.
- Parameter studies, sweep orchestration, reusable templates.
- Result extraction APIs for automated downstream workflows.

### Deliverables

- Canonical example library (by physics/coupling class).
- Agent runbook patterns for generating and validating studies.

### Exit criteria

- New use cases can be built from templates without ad-hoc internal changes.
- Programmatic workflow is consistently faster than equivalent GUI-driven setup for target scenarios.

## Quality and Governance Plan

## Test layers

- Unit: formulation, assembly blocks, profile selection, artifact validation.
- Integration: operation contract + end-to-end solve workflows.
- Conformance: deterministic replay and benchmark envelope checks.
- Trend: rolling drift/performance checks with recommendation pressure tracking.

## Release policy

- Branch-specific governance profiles remain in force and are tightened as benchmark confidence grows.
- Promotion gates required for evidence/profile updates.
- No bypass allowed for required recommendation/evidence artifacts in protected/release workflows.

## Benchmark and V&V Expansion

- Maintain a benchmark matrix by physics family, coupling type, linearity class, and mesh complexity.
- For each benchmark define:
  - reference source,
  - expected scalar/vector envelopes,
  - performance target bands,
  - acceptance drift policies.

## Risks and Mitigations

- Over-generalized abstractions too early.
  - Mitigation: only generalize after at least two physics families require same pattern.
- Solver instability on coupled nonlinear/contact cases.
  - Mitigation: robust profile defaults + staged benchmark hardening before broad rollout.
- Governance fatigue from overly strict policies during active development.
  - Mitigation: branch-aware policy profiles and progressive tightening.
- Performance regressions hidden behind correctness checks.
  - Mitigation: performance SLOs as first-class readiness inputs.

## Near-Term Next 3 Chunks

1. Phase 1 kickstart: thermo-mechanical coupled study contract + minimal coupled assembly path + first benchmark fixture.
2. Coupled solver diagnostics and convergence policy profile for thermo-mechanical benchmarks.
3. Benchmark/trend/readiness expansion for thermo-mechanical acceptance and drift envelopes.

## Success Milestones

- M1: Credible thermo-mechanical slice with deterministic governance and benchmark pass.
- M2: Second coupled family integrated with shared contracts and stable trends.
- M3: Nonlinear material/contact scenarios passing bounded reliability/performance gates.
- M4: Scale/perf readiness on target hardware with regression-protected release policy.
- M5: Programmatic multiphysics workflows considered production-ready for prioritized domains.
