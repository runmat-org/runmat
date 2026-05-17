# Analysis Worklog

Append-only concise log for significant simulation-system changes.

## 2026-05-17

- Reorganized `docs/detailed-work` around canonical high-signal docs anchored by `GOAL.md`.
- Created canonical docs: `STATUS.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `GOVERNANCE.md`, `EM_TRACK.md`, `WORKLOG.md`, `README.md`.
- Moved prior long-form planning/history docs into `ARCHIVE/legacy-2026-05-17/` for read-only context.
- Folded prep-aware trust flow, failure taxonomy, prep lifecycle knobs, and prep-health observability into canonical architecture/governance docs.
- Archived `docs/analysis/prep-aware-solves.md` and `docs/geometry/prep-for-analysis.md` into `ARCHIVE/legacy-2026-05-17/`.
- Renamed `docs/detailed-work` to `docs/simulation`.
- Added additive EM material frequency-response schema and FEA `sigma(omega)` interpolation, plus EM conformance thresholds for frequency-response metrics; verified with `cargo test -p runmat-analysis-core`, `cargo test -p runmat-analysis-fea`, and `cargo test -p runmat-runtime`.
- Extended M6 external-reference benchmark generation to support threshold-assertion-backed metrics and added EM `sigma(omega)` comparator entries; verified with Python governance tests plus benchmark generation+validation scripts.
- Added EM dispersive-loss coupling from frequency-response points (`dispersive_loss_scale`) and promoted dispersive diagnostics/threshold assertions plus external-reference baseline entries; verified with FEA/runtime test suites and external-reference benchmark generation+validation.
- Tightened thermo/thermal conformance thresholds across gradient/ramp/shock and standalone thermal fixtures to narrow permissive drift windows; verified with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib`.
- Tightened nonlinear plastic/contact proxy and reference threshold bands (severity and load realization/amplification) to lock constitutive benchmark behavior; verified with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib -- --test-threads=1`.
- Tightened electro-thermal benign/pathological and nonlinear mixed-load threshold bands (Joule coupling, conductivity spread, severity, temporal variation, and time-scale metrics); verified with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib -- --test-threads=1`.
- Tightened nonlinear assembly/stress/softening convergence threshold bands (line-search backtracks, increment/residual norms, spike/stall counts, and tangent rebuilds); verified with `cargo test -p runmat-runtime --test analysis` and `cargo test -p runmat-runtime --lib -- --test-threads=1`.
- Isolated thermo-field artifact unit tests to per-test temporary roots via `RUNMAT_THERMO_FIELD_ARTIFACT_ROOT` to prevent shared fixture directory deletion during test cleanup; verified with `cargo test -p runmat-runtime --lib -- --test-threads=1`.
- Added additive CFD schema kickoff (`AnalysisStepKind::Cfd`, model-owned CFD domain contracts) and `analysis.create_model` steady/transient CFD templates with runtime tests; verified with `cargo test -p runmat-analysis-core`, `cargo test -p runmat-runtime --lib -- --test-threads=1`, and `cargo test -p runmat-runtime --test analysis`.
- Added first CFD runtime execution baseline via `analysis.run_cfd/v1` and `AnalysisCfdRunOptions`, including model/domain + option validation, prep-context resolution, and `FEA_CFD_FLOW` diagnostics with runtime tests; verified with `cargo test -p runmat-runtime --lib -- --test-threads=1` and `cargo test -p runmat-runtime --test analysis`.
- Added CFD governance baseline by introducing conformance harness fixtures (`cfd_steady_cpu`, `cfd_steady_gpu_provider`, `cfd_steady_gpu_fallback`) with `FEA_CFD_FLOW` threshold assertions and by classifying CFD in runtime trends (`AnalysisRunKind::Cfd`); verified with `cargo test -p runmat-runtime --lib -- --test-threads=1` and `cargo test -p runmat-runtime --test analysis`.
- Added CHT contract kickoff via new additive `analysis.create_model` profile `cht_coupled` (seeded CFD + thermal steps and baseline coupled domains) with runtime profile contract tests; verified with `cargo test -p runmat-runtime --lib -- --test-threads=1` and `cargo test -p runmat-runtime --test analysis`.
- Added first CHT runtime execution baseline via `analysis.run_cht/v1` and `AnalysisChtRunOptions`, including coupled model/domain + option validation, typed transient+thermal payload output, and coupled diagnostics (`FEA_CHT_COUPLING`, `FEA_CFD_FLOW`) with runtime tests; verified with `cargo test -p runmat-runtime --lib -- --test-threads=1` and `cargo test -p runmat-runtime --test analysis`.
- Added CHT governance baseline by introducing conformance harness fixtures (`cht_coupled_cpu`, `cht_coupled_gpu_provider`, `cht_coupled_gpu_fallback`) with coupled threshold assertions (`FEA_CHT_COUPLING`, `FEA_CFD_FLOW`) and by classifying CHT in runtime trends (`AnalysisRunKind::Cht`); verified with `cargo test -p runmat-runtime --lib -- --test-threads=1` and `cargo test -p runmat-runtime --test analysis`.
