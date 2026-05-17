# Analysis Governance and Readiness

Last updated: 2026-05-17

## Governance Model

A slice is not complete when code compiles; it is complete when:

1. benchmark/conformance thresholds pass,
2. trend and drift checks pass,
3. protected-branch readiness policy passes,
4. evidence artifacts are valid and present.

## Required Gate Families

- Contract conformance gates.
- Domain benchmark acceptance gates.
- Trend/drift non-regression gates.
- External-reference comparator gates (M6 track).
- Artifact schema/presence gates.

## Branch Posture

- Protected branches (`main`, `release/*`) enforce strict gating.
- Development/feature branches may use wider thresholds but still emit full signals.

## Domain Completion Governance (Parity-Credible)

A domain is parity-credible only when all are true:

1. stable versioned contracts,
2. canonical schema ownership,
3. deterministic + robust solver behavior on representative hard cases,
4. domain-native diagnostics in results/trends,
5. reference benchmark suite with acceptance envelopes,
6. enforced branch-profiled trend/drift gates.

## External Reference (M6)

Required direction:

- maintain machine-checkable external reference artifact(s),
- compare observed vs reference metrics with explicit pass/fail envelopes,
- enforce metric pass behavior on protected branches.

## Evidence Artifacts

Required artifact classes:

- run results/diagnostics/trends,
- readiness verdict,
- benchmark report,
- external-reference comparator report,
- optional calibration/recommendation artifacts where domain policy requires them.
