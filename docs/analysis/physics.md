---
title: "Physics Models"
category: "Analysis & Simulation"
section: "13.4"
last_updated: "June 9, 2026"
---

# Physics Models

Choose a physics family based on the engineering question you want to answer.

For each supported family, RunMat defines:

- model schema in `runmat-analysis-core`,
- a runtime operation contract,
- an execution path in `runmat-analysis-fea` or an equivalent implementation,
- result payloads and diagnostics,
- quality gates and status boundaries.

## Supported Families

| Family | Typical question | Run operation |
| --- | --- | --- |
| Linear static structural | How does the structure respond to static loads? | `analysis.run_linear_static/v1` |
| Modal structural | What are the natural modes and frequencies? | `analysis.run_modal/v1` |
| Acoustic harmonic | What acoustic response is expected at a driving frequency? | `analysis.run_acoustic/v1` |
| Thermal standalone | How does heat move through the model? | `analysis.run_thermal/v1` |
| Structural transient | How does the structure evolve over time? | `analysis.run_transient/v1` |
| Nonlinear structural | How do nonlinear material, load-path, plastic, or contact effects behave? | `analysis.run_nonlinear/v1` |
| Thermo-mechanical | How does temperature affect structural behavior? | Structural run families with thermo-mechanical domain state. |
| Electro-thermal | How does electrical behavior drive heating? | Coupled thermal/nonlinear paths with electro-thermal domain state. |
| Electromagnetic | How do sources, materials, frequencies, and boundaries affect electromagnetic fields? | `analysis.run_electromagnetic/v1` |
| CFD | How does the fluid state evolve? | `analysis.run_cfd/v1` |
| CHT | How do fluid and thermal fields interact? | `analysis.run_cht/v1` |
| FSI | How do fluid and structural states interact? | `analysis.run_fsi/v1` |

## Choosing A Family

Start from the engineering question, then check the model requirements:

| Question | You need |
| --- | --- |
| Is this a structural response problem? | Structural materials, loads, constraints, and static/modal/transient/nonlinear steps. |
| Is time important? | Transient options or a coupled family with time-profile data. |
| Is heat the target or a coupling input? | Thermal material state and thermal or coupled domain data. |
| Are contacts, plasticity, or large load-path effects important? | Nonlinear profile, nonlinear options, and relevant interface/material state. |
| Are fluid fields part of the question? | CFD domain data and CFD-compatible model setup. |
| Are source frequency, material electrical properties, or field proxies important? | Electromagnetic domain data, electrical material properties, and compatible sources and boundaries. |

## Result Payloads

All run families return the common `AnalysisRunResult` shape. Domain families can add typed payloads:

| Payload family | Used by |
| --- | --- |
| `ModalResultsData` | Modal and acoustic paths. |
| `ThermalResultsData` | Thermal and coupled thermal paths. |
| `TransientResultsData` | Transient, CFD, CHT, and FSI paths where applicable. |
| `NonlinearResultsData` | Nonlinear structural and coupled nonlinear paths. |
| `ElectromagneticResultsData` | Electromagnetic paths. |

Diagnostics and quality reasons explain whether the result is publishable, degraded, or rejected under the selected quality policy.

## Backends And Provenance

Run operations accept a requested `ComputeBackend`. Requested backend alone does not define the actual execution path. Provenance records:

- requested backend,
- selected solver backend,
- precision mode,
- deterministic mode,
- solver method,
- preconditioner,
- fallback events.

GPU requests can fall back. That fallback is expected to be visible in provenance and diagnostics.

## Current Boundaries

- Structural, thermal, nonlinear, coupled, electromagnetic, acoustic, CFD, CHT, and FSI paths are represented in the operation surface.
- Fidelity is not uniform across domains. Some families have deeper solver coverage; others are baseline paths.
- Production readiness is evaluated per domain and fixture family. A run operation is only one part of that decision.
- Larger industrial workloads, richer CAD semantics, and deeper domain-specific solver fidelity are tracked in [Current Status](/docs/runtime/analysis/status).
