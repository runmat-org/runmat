# Canonical static electrical qualification

This document records the initial source-free milestones and their historical
boundaries. The retained authored/native chain is described in
[electrical-authored-static.md](electrical-authored-static.md); the current volume
charge extension and its explicit remaining gates are described in
[electrical-volume-charge.md](electrical-volume-charge.md). Earlier statements below
that native dispatch/publication/resources were absent refer to those initial slices,
not the current repository. Stage 7 qualification and the largest Tet10 proof gate remain open.

The CPU library entry `physics::electrical::solve_canonical_static` solves distinct
electrostatic and steady DC conduction equations on canonical Tet4/Tet10 topology.
This is not the legacy reduced EM edge graph, native authored dispatch, restart,
whole-study resource admission or a release-qualified envelope. A numerical-only portable
workload/result contract now binds that kernel to its explicit equation, producer,
source and canonical topology; it does not open scheduler or authored execution.

## Current admitted proof and original refinement gate (2026-10-07)

The unchanged electrical suite now passes36/36, including the original finest Tet10
`harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence` gate.
Original element contributions are captured during one quadrature pass and verified
against the actual constrained CSR using [anchored local energies](anchored-local-energy.md).
Shared preparation owns the process-local proof; ordinary PCG, constraints, recovery,
fields and physical balances are unchanged. Source/checker demand fits the original
proof limits, and source storage is additive in the pre-assembly task plan (revision79).

Shared algebra71/71, resource170/170, bridge13/13 and native6/6 pass. Strict FEA,
bridge and native library/test lint pass. Actual electrothermal workers and cold
publication pass both orders; exact source and unrelated-change preservation are verified. Cold retained fields remain consistency/provenance evidence,
not a decoded proof or independent accuracy certificate.
Statements below about the original largest-Tet10 failure describe earlier checkpoints;
that particular failure is superseded by the current unchanged-reference pass.

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

The retained numerical response distinguishes absolute L2 residual from relative residual.
The latter uses the actual canonical Dirichlet-lifted RHS, not voltage magnitude or the
unlifted load vector; zero RHS/zero residual is explicitly zero. Shared system preparation
and electrical residual normalization consume the same RHS owner, without a second matrix
preparation or coercivity proof. Prescribed-row cold checks allow gauge-add roundoff, not
an error allowance scaled by the solver tolerance and arbitrary absolute gauge.

Initially supported inputs are prescribed real potential/reference constraints;
other exterior faces have natural zero flux. Static/harmonic intent is a closed
typed domain contract with no dummy frequency. Unsupported source actions, explicit
natural/Robin boundaries, floating/equipotential/integral-current/periodic constraints,
dispersive/temperature-dependent laws, imported state and coupling fail closed.

## Verification status and remaining gates

Implementation and focused evidence are in progress; no stage promotion.
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
broader electrical constraints, governed field/state publication,
authored native dispatch, scheduler/resources/accuracy, process/restart and release
qualification remain real Stage7 work. Magnetostatics, Maxwell and electrothermal
coupling are not supplied by this initial scalar envelope.

## Portable numerical contract boundary

`ElectricalStaticWorkload` uses the same input validator as the numerical kernel.
It admits only fresh singleton static/direct numerical work, with no imported state,
observation or activation request silently ignored. Complete static-electrical output
declarations are retained in workload identity, as required by model validation; this
numerical entry returns physical fields but does not execute semantic publication.

Exact registry identity distinguishes both equations and both element orders. Its scalar
Galerkin gradient, epsilon/sigma law, static temporal solve and D/energy versus J/Joule
recovery cannot borrow the harmonic reduced edge graph.

`ElectricalStaticSolverResult` retains SI arrays behind an immutable, validated contract.
Cold source-bound import checks workload/producer/mesh/projection/order/formulation identity,
field cardinality and finiteness, equation-specific inventory, numerical closure bounds,
terminal ownership and weak nodal flux accounting. It also checks prescribed potentials
against source data and integrated retained densities against canonical element volumes.
Those consistency checks do not reconstruct fields, assemble an operator or solve again.
Bare decoding cannot authenticate the external source; consumers must use source-bound import.

This bounded contract is not scheduler allocation, numerical receipt/shared accuracy
evaluation, persisted semantic state, authored process path or restart qualification.
The previously failed largest Tet10 refinement remains a failed gate, unchanged.

2026-09-30 bounded verification: electrical matrix runs 14 tests, with 13 passing and
the unchanged largest Tet10 proof-resource gate failing (pivot2265, updates43863736 above
limit43840305). Final portable contract tests pass5/5, including all four equation/order
combinations, exact cold arrays/identities, malformed imports, voltage scales1e-6/1e6 and
constant1e12 gauge substitution. Shared admission and algebra regressions each pass40/40;
FEA library/tests strict Clippy and scoped formatting pass. Architecture still reports four
unrelated runtime module-size debts. No resource, process/restart or qualification pass is inferred.

## Requested-heap admission and transport boundary

`build_electrical_static_resource_plan` binds both scalar equations and orders to the
same canonical topology and shared solver-work envelope. It reserves physical/normalized/
constrained matrix lifetimes, property/projection preparation, proof and vector work,
field recovery, canonical result construction and shared transport closure. Checked raw
node incidence establishes the proof support without numerical matrix assembly or physical
quadrature allocation. The support-preparation ceiling is checked before its allocation;
the final requested heap must also fit the authored accuracy ceiling. Resource-estimation
model revision42 distinguishes this admission contract.

`execute_with_resource_plan` rederives the complete plan before numerical assembly and
passes its exact policy to the shared scoped solver. A correctly hashed but altered policy
is not authority. The bridge prepares ordinary shared workload/mesh objects and validates
the imported plan plus exact input closures and chunk policy. Retained result fields are
source-bound on cold import; they are not calculated again.

This is a local numerical requested-heap contract, not measured peak RSS, scheduler
allocation, native authored execution, a numerical receipt or semantic publication.
Those gates remain closed. The pre-existing largest Tet10 physics/proof gate is unchanged;
neither resource ceilings nor independent accuracy requirements are relaxed.

2026-09-30 retention evidence: the three resource checks pass, including both equations/
orders with free interior nodes, exact source-bound admission, valid-hash policy substitution,
payload bounds, malformed facts/overflow, one-byte-below total-memory rejection and support
budget rejection before invalid geometry. The bridge matrix verifies two chunk policies,
canonical input substitution rejection and identical cold retained fields; the full bridge
suite passes103/103. Strict FEA/bridge all-target Clippy passes without suppression, scoped
format and whitespace pass. Architecture still reports only the four unrelated runtime debts.
No measured-residency, scheduler, publication, restart or qualification claim follows.
