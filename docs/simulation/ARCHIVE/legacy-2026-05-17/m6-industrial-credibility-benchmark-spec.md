# M6 Industrial Credibility Benchmark Spec

## Goal

Deliver one reference-grade, code-first multiphysics workflow that is externally anchored,
reproducible, and release-gated.

Initial target slice:

- Primary external baseline: NAFEMS-style elastoplastic verification case.
- Secondary cross-check baseline: CalculiX run of same setup.
- Runtime path under test: thermo/contact/plastic nonlinear governance stack.

## Scope (First Pass)

- One canonical scenario ID:
  - `m6_elastoplastic_contact_bracket_v1`
- One canonical reference artifact path:
  - `target/runmat-analysis-artifacts/external_reference_benchmark.json`
- One placeholder validator path:
  - `scripts/analysis/governance/validate_external_reference_benchmark.py`
- One comparator/generator path:
  - `scripts/analysis/governance/generate_external_reference_benchmark.py`

## External Reference Strategy

### Tier 1 (Credibility)

- Published benchmark targets from open engineering references (NAFEMS/literature) for:
  - load-displacement response points,
  - stress hotspot probe,
  - plastic strain probe.

### Tier 2 (Operational Reproducibility)

- CalculiX reproduction of the same case to produce machine-checkable reference artifacts.

## Reference Artifact Schema (v1 draft)

Top-level fields:

- `schema_version`: `external-reference-benchmark/v1`
- `scenario_id`: string
- `reference_source`: object
  - `primary`: string
  - `secondary`: string
- `generated_at`: RFC3339 UTC timestamp
- `metrics`: list of metric objects

Metric object fields:

- `name`: string
- `observed`: number
- `reference`: number
- `tolerance_abs`: number (optional)
- `tolerance_rel`: number (optional)
- `pass`: bool

## Pass/Fail Rules (First Pass)

- Validator behavior:
  - enforcing mode fails when artifact missing/schema invalid.
  - optional `require_pass` mode fails when any metric is out-of-band.
- CI policy:
  - always enforce artifact/schema validation.
  - enforce all metric passes on protected branches.

## Planned Expansion (Second Pass)

- Add richer comparator metrics (load-displacement key points, stress/plastic probes) beyond severity proxies.
- Add trend drift checks on selected external-reference metrics.
