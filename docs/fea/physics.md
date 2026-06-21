---
title: "Physics Families"
category: "FEA"
section: "13.4"
last_updated: "June 21, 2026"
---

# Physics Families

Choose the physics family from the engineering question. The selected model profile determines required model data, selects the solver, validates run options, and shapes diagnostics, result payloads, and current support boundaries.

## Families

| Family | Use when you need to know... | Selected solver |
| --- | --- | --- |
| Linear static structural | Static displacement and stress response. | `linear_static` |
| Modal structural | Natural modes and modal frequencies. | `modal` |
| Acoustic harmonic | Acoustic pressure and particle-velocity response from a damped Helmholtz domain graph. | `acoustic` |
| Thermal standalone | Temperature response and thermal quality signals. | `thermal` |
| Structural transient | Structural response over time. | `transient` |
| Nonlinear structural | Nonlinear, plasticity, contact, or increment-controlled response. | `nonlinear` |
| Thermo-mechanical | Structural response with thermal coupling context. | `transient` |
| Electro-thermal | Heating behavior from electrical coupling context. | Coupled thermal, transient, or nonlinear paths with electro-thermal domain data. |
| Electromagnetic | Electromagnetic source, boundary, material, frequency, and sweep behavior. | `electromagnetic` |
| CFD | Fluid-domain baseline behavior and diagnostics. | `cfd` |
| CHT | Coupled fluid and thermal baseline behavior. | `cht` |
| FSI | Coupled fluid and structural baseline behavior. | `fsi` |

## Choosing A Family

| If the question is about... | Check these model inputs |
| --- | --- |
| Static structural response | Mechanical material properties, loads, fixed or prescribed boundary conditions, static step. |
| Natural frequencies | Mechanical material properties, constraints, modal step, requested mode count. |
| Thermal response | Thermal material properties, thermal step, thermal domain or coupled thermal context. |
| Time response | Transient step, time profile, time-step controls, stability diagnostics. |
| Plasticity or contact | Nonlinear step, plastic material data, contact interfaces, nonlinear options. |
| Electromagnetic behavior | Electrical material properties, electromagnetic domain, sources, boundaries, harmonic or sweep options. |
| Fluid behavior | CFD domain properties, CFD step, inlet or flow controls, coupled thermal or structural data when applicable. |

## Run Options

The `.fea` `run.options` block is typed by the solver selected from `model.profile`.

Common option themes include:

- deterministic mode,
- precision mode,
- quality policy,
- prep artifact or prep context,
- preconditioner mode where applicable.

Families can add controls such as modal mode count, transient time stepping, nonlinear increment policy, or electromagnetic harmonic and sweep settings.

## Result Payloads

All families return common run fields: `run_id`, fields, diagnostics, quality gates, quality reasons, publishability, and provenance. Some families add domain payloads:

| Payload | Used by |
| --- | --- |
| `modal_results` | Modal paths. |
| `thermal_results` | Thermal and coupled thermal paths. |
| `transient_results` | Transient, CFD, CHT, and FSI paths where applicable. |
| `nonlinear_results` | Nonlinear structural paths. |
| `electromagnetic_results` | Electromagnetic paths. |

## Support Is Per Family

A family can be implemented and tested without being production-grade for every engineering use. Current support varies by family:

- dedicated solver-backed paths exist for several structural, thermal, nonlinear, transient, modal, acoustic, and electromagnetic workflows,
- CFD, CHT, and FSI currently have baseline paths with typed diagnostics and payloads,
- V&V maturity depends on known-answer tests, convergence checks, independent references, and governed thresholds.

Use [Current Status](/docs/fea/status) before relying on a family for a workflow, and use [Verification & Validation](/docs/fea/validation) for the evidence standard.
