# Structural transient fixed-basis execution

This is an execution-work reduction, not a new physical formulation or a physics-gate promotion.
Focused complete-summary/authority/work-count tests pass 6/6 and physical transient regressions
pass 12/12, including prescribed motion, work and cold block continuation. Broad FEA regressions
pass 1266/1266 with seven explicitly filtered draft/proof gates, not unrestricted green.
The original rod-continuum accuracy failures remain open; no numerical formula or threshold
was changed. Strict lint and final retention evidence are recorded separately in the ledger.

## Admitted boundary

The prepared path borrows the initial, freshly assembled and admitted canonical-solid summary,
the immutable active model, the authored procedure step, and the canonical mesh. It supports the
existing uncoupled linear solid forcing and direct/affine constraint channels. Prescribed-motion
lifting, inertial/damping/elastic support coupling, and work accounting remain with their existing
transient owners. No arbitrary operator is classified as constant by a caller assertion.

Synthetic, explicit/mixed, connector, and thermomechanical/electrothermal models use the original
full-assembly path. Neither formulation admission nor the existing projection policy is widened.
The prepared view is local to one run/block; it is not a serialized cache, checkpoint contract,
cross-workload memoization scheme, or field-artifact initial-condition implementation.

## Authority and retained work

One additional property-map compilation at preparation authenticates geometry/topology, material
ownership, and constitutive identity against the admitted initial summary. The temporary map is
dropped. The step must be authored by the borrowed model. The view retains only references, so a
sample cannot substitute a different model, mesh, material, procedure, or source inventory.

Every sample still realizes the authored signals with the exact time and limit approach, validates
structural inputs, compiles/validates the canonical constraint plan, and invokes the same solid
load and constraint lowering as full assembly. The sampled constrained mask, constraint matrix,
and affine reduction must equal the initial authority. Source inventory, load-plan identity,
projection evidence, resultant-moment metadata, and physical/generalized forcing are refreshed.
Equal numerical values at different times still require distinct sampled provenance.

Sampling does not repeat solid property compilation, element recovery construction, or K/M/C
element/operator assembly. Existing `with_load` constructs the returned operator and preserves
its validated nullspace evidence. The transient solver's unchanged-system checks remain active.

## Residency

No extra full operator, mesh, profile, property inventory, or node-space inventory is retained by
the prepared view. The output summary is constructed explicitly with one `with_load` operator
clone, rather than cloning a complete summary and replacing its operator (which would introduce
a transient additional operator allocation). Sampled constraint objects are moved into the output
after comparison. Load/projection/constraint temporaries use the existing authoritative lowering.

The existing resource estimators are unchanged: portable block demand already reserves four
operator copies and 128 full-order work vectors. This change does not justify lowering those
reservations or claiming whole-process/RSS bounds. Preparation's temporary property compilation
uses the same class of metadata already required during full sample assembly; it adds no retained
copy or resource-ceiling exception.

## Verification chain

Focused tests in `assembly/transient_basis/tests.rs` compare the entire returned `AssemblySummary`
against full assembly, not only the numerical right-hand side:

- Constant force, moment, traction, pressure, and body-force sources.
- Time-varying and discontinuous forcing, including left/right limit identity.
- Affine constraints and the prescribed-motion constraint channel.
- Foreign mesh/material/step, changed constraints, and unmapped sources.
- Explicit/mixed, synthetic, coupled, and connector fallback boundaries.

A test-only thread-local hook counts actual solid-operator assembly entry. Tests require one
initial assembly and zero further operator assemblies during preparation or sampling. Pointer
checks confirm the summary and mesh remain borrowed. These are work-count assertions, not timing
or memory measurements. Existing transient physical, support-motion, block/recovery, resource,
and strict-lint tests must also pass before retaining the change. No accuracy tolerance, refinement
gate, integration formula, or checkpoint authority is weakened to make this evidence pass.

## Effective-solve input ownership

Ordinary, affine and multiplier linear solves now share a borrowed numerical input
view containing only the operator, DOF count and resolved constraint references.
Existing public assembly-summary entrypoints construct this view and use the same
solver, preparation, admission, backend and diagnostic paths.

Initial-acceleration, Newmark and normalized equilibrium-correction systems own
only their required numerical matrices and homogeneous constraints. They no longer
clone the complete physical assembly summary and overwrite its operator. Original
material/recovery/source inventories remain with the source assembly, which still
owns physical balance and original output recovery. Matrix and vector workspace
remains admitted under the existing bounds; no resource reservation or cap is reduced.

The refactor passes the existing 25 transient and 88 constraint tests, including
restartable segments, physical oscillator convergence, affine and multiplier
channels, curved normal support and reaction recovery. It changes ownership, not
formulas, tolerances, constraint enforcement, accepted history or publication identity.
The preceding isolated full numerical run finished with 1,663 passing and 14 failing tests; it predates this
ownership refactor. Original rod and other scientific failures remain open.
