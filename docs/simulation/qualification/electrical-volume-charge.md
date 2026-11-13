# Electrostatic volume charge — bounded qualification evidence

This extends the retained canonical scalar electrical chain with typed real volume
charge density. It does not qualify Maxwell, magnetostatics, conduction sources,
coupling, imported-state sequencing, restart, GPU or distributed operators.
Qualification remains open; the largest existing Tet10 proof-resource gate remains
executable and failed, with its limits unchanged.

## Equation and independent reference

For constant positive epsilon, `-div(epsilon grad(phi)) = rho` and `E=-grad(phi)`.
The Galerkin source is `b_i = integral rho N_i dV`, using the shared canonical volume
quadrature and scalar projection evidence. On the unit prism with equal terminal
potential g, choose `rho = s*2*epsilon`, `s=+1` or `-1`. Independently,
`phi=g+s*x*(1-x)`, `E_x=s*(2*x-1)`, outward terminal flux is `s*epsilon` at each end,
total charge `s*2*epsilon`, energy `epsilon/6`, source work `epsilon/3`, terminal
work zero. These formulas are not calculated from RunMat's stiffness or solution.

Tet10 straight-element nodal potential and averaged field reproduce this quadratic
solution, including gauges0,1000 and1e12 through retained component departures.
Tet4 cells2/4/8 independently measure potential L2, field L2 and integrated energy:
potential errors .04564355/.01141089/.00285272, field .28867513/.14433757/.07216878,
energy relative errors .25/.0625/.015625. Predefined gates require potential order>1.7,
field order>.8, energy reduction<.35, finest errors<.005/.1/.025 respectively.

## Source and retained-result authority

`electric_charge_density` accepts scalar C/m^3 on exact solid regions or whole model,
fixed frame, intensity per volume. Signed superposition is physical charge, not a
reinterpretation of total charge. Real static sampling is required by this formulation;
shared complex signal realization does not imply a complex static solve. Unmatched
regions, surface targets, vector/directional/follower loads and conduction charge fail
closed. Each action must belong to a declared load case.

Declared charge, including zero or inactive charge, has a distinct
registry identity from source-free electrostatics. No new source-private quadrature
owner or live callback is introduced. Weak outward flux is `b-K*delta_phi`; conservation
compares summed terminal flux to integrated charge using physical cancellation scale.
Energy compares independently integrated `integral epsilon |E|^2` with volume-source
plus terminal work. Absolute potential gauge is not the numerical field authority.

Cold import retains fields; it never reconstructs E, assembles stiffness or solves
again. Source-bound validation rederives only source projection and checks retained
charge/evidence, departure/gauge, terminal ownership/work, and field integrals. Result,
receipt and native publication identities authenticate the retained arrays. Forged
source, work, departure or projection evidence is rejected.

## Execution and gate boundary

Resource-estimation revision51 adds explicit source/projection/departure reservations
and canonical result evidence bounds. Both orders with signed and zero charge execute
under their exact plan; source-free plan substitution and one-byte-under total-memory
ceilings reject. This is an analytical requested-heap envelope, not measured peak RSS.

Core117/117, all language142/142, full bridge142/142 and focused charge physics/cold
tests10/10 pass. That checkpoint had33 broader electrical passes and a largest Tet10
proof-resource failure. The 2026-10-07 anchored local-energy integration supersedes that
failure: the unchanged suite now passes36/36, including the original refinement gate. The native authored OCCT Tet10 signed/gauged prism test passes
to check actual region resolution, worker solve, four persisted full/partial fields,
receipt/state/root closure and independent analytic data after cold import: both signed
producer processes,686 nodes/347 elements, test138.22s. Strict core/FEA/language/bridge/
native library/test lint passes without suppressions. The ledger records final evidence.
No whole-stage, release, restart or family qualification follows from this slice.
