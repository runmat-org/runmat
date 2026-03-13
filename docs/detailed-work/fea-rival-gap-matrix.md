# FEA Rival Gap Matrix (Physics Over/Under)

## Purpose

Capture the exact current-vs-missing physics shape and the concrete build requirements needed to credibly position RunMat as a stellar programmatic multi-physics FEA platform for code-first workflows.

This is an engineering planning document. It is not customer-facing capability messaging.

Companion docs:

- physics ownership and current scope: `docs/detailed-work/analysis-physics-domain-coverage-and-migration.md`
- phased roadmap context: `docs/detailed-work/multi-physics-parity-roadmap.md`
- program log: `docs/detailed-work/geo-and-analysis.md`

## Current Physics Over/Under

| Domain | Current status | Concrete evidence in repo | Missing to claim strong industrial parity |
| --- | --- | --- | --- |
| Linear static structural | implemented | Runtime op contract `analysis.run_linear_static` and operation contract tests in `crates/runmat-runtime/src/analysis/mod.rs` and `crates/runmat-runtime/tests/operation_contracts.rs`. | Broader reference matrix across geometry/order/load classes and tighter external calibration envelopes. |
| Modal structural | implemented | Runtime op contract `analysis.run_modal` and contract/integration coverage in runtime tests. | Deeper modal quality evidence and larger benchmark corpus. |
| Transient structural | implemented | Runtime op contract `analysis.run_transient`; runtime contracts and harness coverage active. | More difficult stiffness/time-scale cases, expanded adaptivity confidence envelope. |
| Nonlinear structural | implemented | Runtime op `analysis.run_nonlinear` with line-search/backtrack/rebuild diagnostics and governance integration. | Greater constitutive/formulation depth and larger reference-backed nonlinear matrix. |
| Standalone thermal | implemented baseline | Runtime op `analysis.run_thermal`; constitutive/outcome diagnostics and governance posture checks are active. | Higher-fidelity constitutive calibration depth and broader thermal benchmark classes. |
| Thermo-mechanical coupling | partial (functional baseline) | Model-owned schema + active coupling path + fixture/governance assertions are present. | Deeper coupled-field solve fidelity and wider reference-backed coupled scenarios. |
| Electro-thermal coupling | partial (functional baseline) | Material-owned electrical+thermal schema with Joule-style coupling and conformance/governance checks. | This is not full EM; requires Maxwell-class PDE family for full EM parity claims. |
| Plasticity | partial (advancing) | Model-owned plastic contexts, nonlinear diagnostics, and load-path outcome metrics (`load_realization_ratio`, `load_amplification_ratio`) in conformance/governance. | More constitutive law families and stronger external-reference coverage depth. |
| Contact | partial (advancing) | Interface-owned contact contexts, nonlinear diagnostics, and load-path outcome metrics with threshold/trend gating. | Broader contact formulations and richer scenario matrix (robustness + realism). |
| Full EM domain (Maxwell-class) | not implemented | Explicitly documented as not implemented in physics-coverage doc. No EM runtime operations in current runtime analysis op surface. | New EM contract family, schema, assembly/solve stack, diagnostics, and benchmark/governance track. |
| Fluids / CFD (Navier-Stokes class) | not implemented | No CFD runtime operations or fluid schema in current runtime analysis op surface. | New CFD contract family, fluid schema, solver stack, diagnostics, and benchmark/governance track. |
| Fluid-thermal (CHT) | not implemented | Present only as future roadmap option. | Coupled fluid+thermal contracts, transfer operators, CHT benchmark/governance tracks. |
| FSI | not implemented | No FSI operations/schemas in current runtime contracts. | New coupled FSI contracts, interface transfer model, partitioned/monolithic coupling strategies, and benchmark/governance tracks. |
| Acoustics | not implemented | No acoustic operation/scheme/solver family in runtime contracts. | Acoustic contracts, schema, solver diagnostics, benchmarks, governance. |

## Domain Build Matrix (What Must Exist)

| Domain family | Contracts/API required | Schema required | Solver/assembly required | Diagnostics required | Benchmark/validation required | Governance gates required |
| --- | --- | --- | --- | --- | --- | --- |
| Structural core (linear/modal/transient/nonlinear) | Maintain current `analysis.run_*` contracts and add explicit strategy/policy profile controls where needed. | Keep model/material/interface ownership; extend only for advanced study controls. | Continue robustness improvements (conditioning/preconditioning/adaptivity/nonlinear strategy portfolio). | Rich convergence + stability + quality signal coverage per run family. | Expand reference packs by fixture family and complexity tier. | Fixture-aligned trend gates, non-regression policy, backend parity confidence signals. |
| Thermal + thermo-mechanical | Preserve `analysis.run_thermal` and coupling contracts; keep additive contract evolution only. | Extend thermal constitutive/BC schemas where needed for richer industrial classes. | Improve constitutive realism and coupled solve fidelity. | Add conservation/consistency diagnostics in addition to existing outcome metrics. | Add wider thermal/coupled benchmark matrix with external references. | Branch-profile thresholds + trend ratchets per fixture family. |
| Electro-thermal (current scope) | Keep Joule-coupling APIs explicit and versioned. | Extend electrical model support for stronger temperature dependence and material classes. | Improve stiffness handling and coupling stability; maintain deterministic behavior. | Expand constitutive consistency diagnostics across time scales. | Add wider electro-thermal reference-backed scenarios. | Separate breach/trend controls for Joule, spread, constitutive drift. |
| Plastic/contact depth | Keep constitutive/interface semantics canonical in contracts/results. | Expand plastic/contact schema for additional law/formulation families. | Broaden constitutive and contact formulations; improve robustness on hard scenarios. | Keep load-path outcomes and add deeper constitutive/interface invariants. | Larger reference-backed nonlinear matrix (stress + low-severity reference paths). | Promotion-ready evidence policy, blocker budgets, ratcheted trend controls. |
| Full EM (Maxwell-class) | Introduce dedicated EM operations (static/harmonic first, transient later) under versioned analysis contracts. | Add EM material/source/BC model namespaces (permittivity/permeability/conductivity, currents/fields). | Build EM PDE assembly and solve stack with stable strategy controls. | Add EM-specific quality/constraint diagnostics (field consistency/divergence/energy norms). | Stand up canonical EM benchmark corpus with external references. | Add EM readiness profile with branch-specific threshold/trend policy. |
| CFD/fluid family | Introduce fluid operations (steady/transient first) under versioned contracts. | Add fluid material/BC/turbulence namespaces. | Build fluid solver stack with stable pressure-velocity strategy and transport handling. | Add conservation/stability diagnostics (mass, boundedness, CFL-style controls). | Build CFD benchmark corpus (laminar-first, then expanded complexity). | Add CFD readiness profile and trend ratchet policy. |
| Coupled families beyond current scope (CHT/FSI/acoustics) | Define explicit coupled operation contracts and dependency semantics. | Add cross-domain interface/coupling schema objects. | Implement deterministic transfer operators and coupling strategies. | Add interface-consistency and coupled stability diagnostics. | Build dedicated coupled benchmark tracks with external references. | Add per-family promotion readiness criteria and blocker policies. |

## Program-Level Gaps to Rival-Class Credibility

1. Missing entire major physics families (Maxwell EM, CFD, FSI, acoustics).
2. Existing in-scope coupled/nonlinear domains are baseline-capable but not yet broad industrial-depth.
3. Reference-backed benchmark breadth is not yet large enough for broad parity claims across domains.
4. Governance framework is strong, but additional domain-specific gates are required as new physics families land.

## Recommended Domain Expansion Order

1. Deepen current in-scope physics first (thermal + thermo/electro + plastic/contact realism and reference coverage).
2. Add first missing major family: Maxwell-class EM.
3. Add second missing major family: CFD/fluid core.
4. Add coupled expansions: CHT then FSI.
5. Keep benchmark/governance gates mandatory before each domain is treated as production-credible.

## Exit Criteria for "Credible Rival" Messaging (Engineering)

Treat a domain as parity-credible only when all are true:

1. Versioned runtime operation contracts exist and are stable.
2. Canonical model/material/interface ownership is explicit and documented.
3. Solver/assembly paths are deterministic and robust on representative hard cases.
4. Domain-native diagnostics are present and surfaced through results/trends.
5. Benchmark corpus includes external-reference checks with deterministic acceptance envelopes.
6. Release-readiness governance has branch-profiled thresholds, breach rates, and trend non-regression controls.

## Maxwell EM Bring-Up Status (Phase 0)

Initial EM contract scaffolding is now in place:

1. Analysis schema now includes an EM step kind (`electromagnetic`) and create-model profile template (`electromagnetic_static`).
2. Runtime operation contract placeholders exist for EM execution:
   - `analysis.run_electromagnetic`
   - `analysis.run_electromagnetic/v1`
3. Current EM execution path is intentionally a contract placeholder and returns deterministic unsupported errors (`ANALYSIS_RUN_ELECTROMAGNETIC_UNSUPPORTED`) after step-shape validation, until solver kernels are implemented.

This keeps contract/versioning discipline in place while solver/assembly implementation remains explicitly pending.

## Maxwell EM Bring-Up Status (Phase 1)

1. EM model schema now has a model-owned domain primitive (`electromagnetic`) in analysis-core, matching existing domain ownership patterns.
2. EM runtime operation now validates domain configuration and emits deterministic placeholder run payloads (`FEA_EM_PLACEHOLDER`) rather than failing unsupported, enabling end-to-end result/trend pipeline integration.
3. EM solver fidelity is still pending; placeholder mode is explicit and non-publishable by default (`RunStatus::Degraded`).

## In-Scope Deepening Closure (Thermal + Thermo/Electro + Plastic/Contact)

The current in-scope deepening track (items 1-6) is closed at baseline with the following concrete outcomes:

1. Constitutive-depth diagnostics were strengthened for in-scope domains using deterministic fields already emitted by transient/nonlinear thermal and coupling paths (including thermo field-clamp and electro time-scale indicators).
2. Conformance thresholds were expanded to assert those additional constitutive indicators in thermo/electro nonlinear path fixtures.
3. `analysis.results` summary now surfaces added posture signals used for in-scope governance:
   - `thermo_field_clamp_ratio`,
   - `electro_transient_time_scale_mean`, `electro_nonlinear_time_scale_mean`,
   - `plastic_nonlinear_severity_mean`, `contact_nonlinear_severity_mean`.
4. Runtime conformance report records and contract snapshots were updated to include the new summary fields.
5. Release-readiness governance now includes threshold/breach/trend controls for new in-scope realism metrics:
   - thermo field-clamp posture (`THERMO_FIELD_CLAMP_*`),
   - electro time-scale posture (`ELECTRO_TIME_SCALE_*`).
6. Documentation and work-log entries were synchronized with implementation and test evidence.
