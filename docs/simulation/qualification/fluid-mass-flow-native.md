# Signed canonical mass-flow sources

Status: native source and cold-continuation witnesses pass. This source vertical does not promote
CFD, close the original structural/J2 science failures, or supply arbitrary spatial-error evidence.

## Physical contract

`MassFlow` is a signed total mass input in kg/s: positive enters the selected fluid boundary.
Boundary-balance reports retain the outward domain sign. On exactly resolved original exterior
faces, the analysis owner realizes uniform normal speed `-rate / (density * total_area)` and checks
the actual integrated mass against the authored projection tolerance. This is an integral source,
not a velocity intensity or a hydraulic-diameter/profile correlation.

The source requires scalar authored mass-rate signals, a fixed global frame, no direction,
`PreserveResultant`, and exact boundary-region resolution. Overlapping face laws, moment claims,
wrong quantities, alternate authorities and invalid budgets reject. Pressure-reference authority
can supply a gauge on those faces because it supplies no additional face law.

Static and physical-time samples use the shared signal owner. Portable steady/transient admission
and canonical preparation share typed input capabilities. The existing task pool, sparse solver,
checkpoint, receipt, accuracy and publication owners carry this source; no private execution route
or source sampler was added. Resource revision 89 reserves selection, per-face realization and shared ILU work
before binding/factorization. Whole-history and block profiles share the potential boundary-authority
inventory, including signed source IDs across the complete schedule. Requested storage is not a measured RSS claim.

The original 100-iteration/restart-8 native linear controls exposed diagonal-preconditioner
stagnation at relative defect 1.94e-4 against 1e-10. Shared fixed-support ILU(0) resolves the same
unseeded accelerating reference without changing iteration caps or tolerances. Its actual factor
storage/work is checked against the admitted count envelope. No fill, pivot modification, fallback
or separate CFD solver was introduced. Numerical-input identity v7 distinguishes this method;
linear failures now retain iteration, residual and breakdown evidence.

## Independent evidence and execution boundary

The shared validation fixture prescribes analytic uniform translation and zero pressure fluctuation,
with a separately retained 1e12-Pa reference. Tests cover positive, negative and zero speed, different
densities, rotated geometry and partitioned port sums. A nonuniform triangle-area fixture has an
independently known 2-by-3 cross section: total area six, with unchanged speed when total rate is
scaled by six. Invalid quantity/frame/direction/distribution/overlap and incompatible closed mass
balance are explicit rejection cases.

The accelerating reference has `u(t)=0.001+0.01*t` and `grad(p')=-rho*[0.01,0,0]`. Native witnesses
exercise original fields/state, ordered receipts, cold publication and split immutable continuation
after terminated producer sessions. They also keep mandatory missing spatial evidence indeterminate
and assert that publication is blocked without a `markIndeterminate` policy. The original Fourier velocity/flow/shear convergence gate remains
byte-identical after sharing lifecycle scaffolding.

The exact candidate passes 48 canonical CFD tests, 7 shared ILU tests, 6 uniqueness-provider
regressions, 22 portable fluid tests, 175 resource tests and 205 execution-bridge tests. Strict
FEA/bridge library and test lint passes; native package library/test lint also passes with
`--no-deps` after removing a redundant witness-fixture conversion. Native signed steady cases and the accelerating full-versus-
split trajectory pass, including three terminated producer sessions and cold original-field recovery.
The original three-mesh Fourier matrix and original transient continuation witness also pass under
the shared ILU method and unchanged acceptance limits. Dependency-inclusive native lint currently
fails in the excluded renderer at `geometry_scene.rs:1188`; this is not a whole-release lint pass.
The exact candidate also retains the two excluded HEAD renderer module-size debts.

Run from the repository:

```sh
cargo test -p runmat-analysis-fea --lib physics::cfd::canonical
cargo test -p runmat-execution-runner-native --test analysis_process_conformance -- --fluid-massflow-only
cargo test -p runmat-execution-runner-native --test analysis_process_conformance -- --fluid-fourier-only
```

The canonical serial CPU scope is Tet4 laminar flow. Authored transient
whole-study roots, symmetry/periodic/slip operators, remote/distributed/GPU, CHT/FSI, measured resource
qualification and release evidence remain open. A retained source hash is provenance, not an
independent mathematical accuracy certificate.
