---
title: "Analysis Models"
category: "Analysis & Simulation"
section: "13.3"
last_updated: "June 9, 2026"
---

# Models

An analysis model turns geometry into solver-ready input.

The model connects:

- geometry identity and revision,
- units and reference frame,
- materials and material assignments,
- loads and boundary conditions,
- interfaces such as contact,
- physics domains,
- ordered analysis steps.

Geometry identifies where things are. The analysis model defines the materials, loads, constraints, domains, and steps to run there.

## Create A Model

Use `analysis.create_model/v1` with an `AnalysisCreateModelIntentSpec`. The profile selects the starting scaffold:

| Profile | Intended solve family |
| --- | --- |
| `linear_static_structural` | Linear static structural solve. |
| `modal_structural` | Modal structural solve. |
| `acoustic_harmonic` | Acoustic harmonic solve. |
| `thermal_standalone` | Thermal solve. |
| `transient_structural` | Structural transient solve. |
| `nonlinear_structural` | Nonlinear structural solve. |
| `thermo_mechanical_coupled` | Structural solve with thermo-mechanical domain context. |
| `electromagnetic_static` | Electromagnetic solve. |
| `cfd_steady_state`, `cfd_transient` | CFD solve. |
| `cht_coupled` | Coupled CFD and thermal solve. |
| `fsi_coupled` | Coupled structural transient and CFD solve. |

The generated model is a starting point. Hosts can refine materials, assignments, loads, constraints, domains, and options before validation and execution.

## Attach Physics To Geometry

Most failed or low-quality runs come from incomplete model setup:

| Model concern | Why it matters |
| --- | --- |
| Materials | Solvers need material properties for the selected physics. |
| Material assignments | Materials need to cover the regions that matter. |
| Loads | The model needs explicit forcing or source terms. |
| Boundary conditions | The model needs enough constraints or anchors to be well posed. |
| Domains | Coupled or specialized physics need domain state. |
| Steps | The run operation must match a step the model actually contains. |

Geometry inspection and prep make setup more reliable by exposing region and topology data before the model is built.

## Validate Before Solving

Use `analysis.validate/v1` before a run. Validation checks model-level requirements such as:

- required geometry identity and revision,
- supported units and frame data,
- material and assignment coherence,
- load and boundary-condition compatibility,
- required domain state for the selected physics,
- required analysis steps.

Run operations also perform operation-specific validation. For example, a domain run rejects a model that does not contain the matching step or required domain data.

## Prep-Aware Models

`analysis.create_model/v1` can consume prep context. Run operations can also receive a `prep_artifact_id`.

Use prep-aware modeling when:

- region mappings matter,
- model setup needs deterministic topology data,
- run acceptance should depend on prep freshness,
- later reviewers need to reproduce the geometry data used by a run.

For prep lifecycle details, see [Geometry](/docs/runtime/analysis/geometry).

## Current Boundaries

- Model profiles are scaffolds. They do not replace domain-specific material, load, and constraint choices.
- Generated defaults are useful for tests and early integration, but meaningful engineering runs need intentional model data.
- Validation catches schema and compatibility problems. The engineering fit of the physics model still depends on the materials, loads, constraints, domains, and assumptions chosen by the caller.
