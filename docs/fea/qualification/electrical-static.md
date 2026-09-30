# Canonical static electrical qualification

The CPU library entry `physics::electrical::solve_canonical_static` solves distinct
electrostatic and steady DC conduction equations on canonical Tet4/Tet10 topology.
This is not the legacy reduced EM edge graph, native authored dispatch, portable
workload/restart, production resource admission or a release-qualified envelope.

## Form and independent references

The scalar weak form is `Kij=integral a grad(Ni).grad(Nj) dV`, with
`a=epsilon0*epsilon_r` for electrostatics and `a=sigma` for conduction. The shared
[FEniCS Poisson documentation](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_poisson.html)
documents the corresponding gradient weak form. Vacuum permittivity uses the
[NIST 2022 CODATA value](https://www.physics.nist.gov/cgi-bin/cuu/Value?eqep0=).
The physical references below are derived analytically, not from another RunMat solve.

For a unit slab with potentials 0 and V, `phi=V*x`, `E=(-V,0,0)`,
capacitance `C=epsilon*A/L`, conductance `G=sigma*A/L`, energy `C*V^2/2`,
and Joule power `G*V^2`. A mesh-aligned two-layer slab has
`a_eff=1/(L1/a1+L2/a2)` for unit area, continuous potential and constant
normal constitutive flux. With equal half-thicknesses and `a2=4*a1`, the
interface potential is `0.8 V`, and `a_eff=1.6*a1`.

The manufactured harmonic potential `phi=x^2-y^2` requires no invented volume
source. Tet10 should reproduce its potential/gradient exactly on straight meshes;
Tet4 field errors should converge with refinement. On the unit cube,
`integral |E|^2 dV=8/3`, giving energy `4*epsilon/3` or power `8*sigma/3`.
The test retains the 2/4/8-cell refinement sequence for both orders.

## Numerical and ownership boundaries

Each element consumes the shared compiled material owner and physical isoparametric
quadrature. No extrema-selected terminal, edge conductance, heuristic current scaling
or product of averaged fields substitutes for the forms. E is volume-averaged only
after integration; energy/Joule density averages the squared quadrature field.
Weak nodal outward D/J flux is `-K*phi`; electrode charge has the opposite sign.
Terminal flux uses exact resolved nodes, with overlapping terminal ownership rejected.

The shared CPU sparse solve uses a normalized operator and component-wise potential
increments: tiny SI epsilon must not let identity Dirichlet rows hide free-node
residuals, and arbitrary gauge offsets must not produce spurious zero-field energy.
The physical operator remains unchanged for recovery. Shared scalar connected-component
nullspace and prepared numerical coercivity evidence are mandatory; resource-limit
rejection is not convergence or a physics result.

Initially supported inputs are prescribed real potential/reference constraints;
other exterior faces have natural zero flux. Static/harmonic intent is a closed
typed domain contract with no dummy frequency. Unsupported source actions, explicit
natural/Robin boundaries, floating/equipotential/integral-current/periodic constraints,
dispersive/temperature-dependent laws, imported state and coupling fail closed.

## Verification status and remaining gates

Implementation and focused evidence are in progress; no production promotion.
Initial tests pass homogeneous, layered, gauge and invalid cases. The largest
quadratic refinement currently hits the shared sparse coercivity work limit before
PCG; it remains in the suite. A bounded exact minimum-degree ordering experiment
also failed this case and was discarded. No limit was raised, proof bypassed or
case shrunk. Core model library tests pass 98/98; semantic typegen fixtures pass.
Broad FEA regression (explicitly excluding the known failing refinement) passed
1216 tests and failed one static-intent diagnostic-text assertion; wording corrected,
focused rerun pending. Runtime compilation and strict lint remain pending.
This is not a full numerical gate pass.

Typed voltage/current/charge/current-density source realization, charged Poisson,
broader electrical constraints, portable workload/result and cold field identities,
authored native dispatch, scheduler/resources/accuracy, process/restart and release
qualification remain real Stage7 work. Magnetostatics, Maxwell and electrothermal
coupling are not supplied by this initial scalar envelope.
