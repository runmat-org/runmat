# Physics guides

RunMat studies use one geometry, one solver mesh, a physical model, and an analysis graph. The
physics block inside each `analysis` declaration states the equation family and procedure; it does
not silently supply materials, references, boundary conditions, damping, or initial state.

Choose the guide that matches the question you need to answer. Together these pages document every
physics family exposed by the language, every currently named procedure, the actual admitted
numerical realization (or its absence), required model data, fields/state, validation checks, and
common failure modes.

| Study | Guide | Typed `.fea` contract | Native `.fea` execution |
| --- | --- | --- | --- |
| Stress, deformation, modes, vibration, or buckling | [Structural analysis](./structural) | Available | Available |
| Steady or transient temperature | [Heat transfer](./thermal) | Available | Not yet connected end to end |
| Electric or magnetic fields | [Electromagnetics](./electromagnetic) | Available | Not yet connected end to end |
| Pressure modes or sound propagation | [Acoustics](./acoustic) | Available | Not yet connected end to end |
| Incompressible flow | [Fluid flow](./fluid) | Available | Not yet connected end to end |
| Temperature-dependent deformation | [Thermomechanical analysis](./thermomechanical) | Available | Not yet connected end to end |
| Joule heating and temperature | [Electrothermal analysis](./electrothermal) | Available | Not yet connected end to end |
| Flow with heat transfer | [Conjugate heat transfer](./conjugate-heat-transfer) | Available | Not yet connected end to end |
| Two-way fluid and structural motion | [Fluid–structure interaction](./fluid-structure-interaction) | Available | Not yet connected end to end |

“Typed contract available” means the parser and checker understand the family, its quantities,
conditions, independent domain, and state types. It does not mean that the public executor can run
that family today. `runmat check` validates source; `runmat run` is the execution boundary.
See [Current status](../validation/status.md) for the qualification matrix.

All guides use the same workflow: inspect or construct geometry, name stable regions, assign
materials to solid/cell regions, declare conditions on regions with compatible topology, define
one or more analyses, then request typed outputs. Placement and worker policy are separate from
physics, so moving a qualified study to a cluster cannot change its equations.

## Which procedures can I declare in a `.fea` file?

| Family | Procedures understood by the checker | Currently registered low-level realization |
| --- | --- | --- |
| Structural | static, preload, modal, transient, modal transient, harmonic, modal harmonic, prestressed modal, buckling, prestressed harmonic, modal response spectrum, random response, nonlinear | Multiple structural realizations; public typed executor is structural-only |
| Thermal | steady, transient | Tet4/Tet10 Galerkin conduction; steady linear and backward-Euler transient |
| Electromagnetic | static, transient, harmonic, nonlinear | Reduced edge-graph curl–curl harmonic only |
| Acoustic | static, modal, transient, harmonic, nonlinear | Reduced Helmholtz graph |
| Fluid | steady, transient, harmonic, nonlinear | Reduced channel steady/transient only |
| Thermomechanical | steady, transient, nonlinear | One-pass prescribed-temperature structural response |
| Electrothermal | steady, transient, nonlinear | None |
| Conjugate heat transfer | steady, transient, nonlinear | Reduced partitioned flow/thermal transfer |
| Fluid–structure interaction | steady, transient, nonlinear | Reduced partitioned relaxation |

The checker accepts the generic `static`/`steady`, `transient`, `harmonic`, and `nonlinear` shapes
where the family contract exposes them. A registered low-level realization is still not a public
typed execution path. Admission requires an exact family, procedure, topology, source projection,
boundary transfer, coupling, and backend match.

## How does RunMat keep physics and execution separate?

A procedure compiles into typed physical state and operators. An execution policy selects eligible
placement, workers, accelerator requirements, resilience, and publication; it cannot change the
governing equation or substitute a reduced formulation. Result evidence records the chosen spatial,
constitutive, temporal, recovery, source, topology, boundary-transfer, and coupling implementations.
If no registered realization satisfies the study's `formulation` requirement, execution fails closed.

## Shared references

- [Write and run an FEA study](../get-started.md)
- [FEA language](../language/)
- [Geometry and region selection](../geometry)
- [Meshing](../meshing)
- [Constraints](../constraints/)
- [Loads and excitations](../loads/)
- [Verification and validation](../validation)
