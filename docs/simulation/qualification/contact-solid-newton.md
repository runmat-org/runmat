# Coupled solid/contact Newton core

The existing canonical solid Newton procedure can compose an explicitly prepared
contact operator with its original material basis. It uses the same accepted
material-point and contact-point histories for every Newton, line-search and
cutback trial. Only a converged increment advances both histories. Accepted state
retains the complete original contact configuration alongside original displacement
and constitutive state; projected contact fields are not restart authority.

The body basis retains original node identity, coordinates and translation rows
once. Coupling checks these against contact's canonical nodal source and row map.
Physical scope composes the original body/equality/source scope with contact's
physical operator identity. Pure execution caps do not change that physical scope.
The uncoupled body's numerical scope and arithmetic remain unchanged.

Body/contact residuals and tangents use shared potential CSR union. Contact tangents
remain GENERAL. Shared affine reduction scatters through planned reduced support,
then the admitted shared GENERAL solver verifies source-independent nonsingularity
and the original physical residual. Initial and cold accepted contact states also
verify the reduced tangent at zero RHS: equilibrium residual alone cannot certify
an unanchored body. These are local discrete operator checks, not global nonlinear
uniqueness or stability evidence. No SPD relabel, private solver or pivot repair.

Explicit additional execution payload, geometry-trial and GENERAL-solver limits
are checked before trial geometry/physical assembly. The finite trial envelope
includes Newton/backtracking/cutback and state-validation evaluations. Requested
additional payload includes original configurations, contact search/trials, merged
and reduced matrices/support, and current-position scratch. Material baseline,
borrowed geometry/model/basis/summary, equality/preparation construction, source
realization, retained outputs/transport, allocator overhead and measured RSS remain
separate obligations. This is not whole governed task admission. Resource revision
100 accounts for body nodal metadata and the enlarged original-state carrier.

The large Newton module is split into preparation, original state validation,
advance, trial equilibrium, contact composition and linear correction. Numeric
constraint reduction has one scatter implementation for discovered/planned support;
its old module-size ceiling is reduced with the extraction.

## Independent evidence and limits

Tet4/Tet10 references solve affine deformable-secondary displacement, with primary
and third bodies fixed by explicit equality reduction. For H=.2m, V=H^3/6,
E=1200Pa, nu=0, the body stiffnesses are 40N/m normal and 20N/m tangential.
Reference area .02m2 and penalty 10000Pa/m yield 200N/m. Original gap .0001m
and loads .016N normal/.01N tangential exercise opening, active penalty and Coulomb
slip. Closed scalar equilibrium/return equations supply independent displacement,
all original point strains/stresses, gap and discrete accumulated friction work.
The original GENERAL slip tangent has reduced columns [20,0] and [60,240];
independent actions and two finite differences verify its nonsymmetric coupling.

Cold scope/missing history/coordinate/work corruption, budget substitution,
rollback, typed cancellation, original-state immutability and observed trial heap
versus requested material-plus-contact payload are checked. This heap evidence is
not RSS. The separate unanchored reference admits its larger actual original CSR
inventory before seeking an initial-state uniqueness verdict.

These are constrained deformable numerical-core references, not authored public
execution or unconstrained Hertz/sliding/mesh/load/penalty convergence qualification.
The governed route still rejects contact. Whole admission, equality/contact overlap
and initial-state transitions, full checkpoint/cursor/native producer-death contact
continuation, mandatory accuracy/publication, feature transfer and within-face
uniqueness remain open. J2 continuation verification on the uncoupled route does
not establish coupled contact continuation. All original science/release gates
remain unchanged and open.

## Accepted contact energy carrier

The original accepted solid carrier now retains the converged contact diagnostic
snapshot and separate cumulative physical friction, tangential algorithmic-loss,
and separation-release totals. Only the converged trial advances these records;
rejected trials, cutbacks and equilibrium replays do not accumulate physical work.
These are the discrete penalty/Coulomb update's work terms, not exact continuous
path work. Algorithmic loss and separation release are not classified as friction
heat. The original configuration remains the point-history authority.

Cold revalidation independently integrates original point-owned friction work using
the prepared original reference quadrature weights and rejects a mismatched total.
Historical algorithmic-loss and separation-release totals cannot be recovered from
the terminal point configuration; decoding and shape validation do not prove their
past trajectory. Governed checkpoint lineage and native contact continuation remain
required before those records can authorize a restart. The public contact route
remains disabled, and whole contact task admission and scientific qualification
remain open. Resource revision101 charges the enlarged accepted-state carrier.

The energy extension passes416focused core tests and strict FEA/bridge lint.
Its uncoupled native J2 continuation passes13parent tests with2ignored helper
entrypoints; native/runtime compilation and strict native lint also pass. This is
not contact continuation or contact qualification.

Prepared contact can now participate in the original numerical load controller;
see [controller scope](contact-load-controller.md). Public contact remains gated.
