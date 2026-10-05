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

## Selected transient block workspace and original continuum limits

Resource model revision 70 distinguishes direct sparse and reduced modal block
algorithms. Direct Newmark blocks no longer reserve 64 dense matrices at the full
physical dimension: that allocation belongs to the modal algorithm and is absent
from the direct solver. Direct blocks instead reserve shared bounded sparse
coercivity construction for both the full topology pattern and its homogeneous
constraint reduction. These proofs execute sequentially, so the larger envelope
is added to the existing vector workspace. Modal dense workspace is unchanged.
The submitter and native worker rederive this same plan before physical block
construction; authored memory ceilings are unchanged.

A both-order, three-mesh witness constructs the actual initial-mass and Newmark
operators and their process-local coercivity certificates. Every reported proof
storage count fits the preassembly solve workspace. On the finest Tet10 case the
workspace is 241,932,096 bytes and the certificate's logical storage count is
8,773,248 bytes. These are requested and certificate-derived counts, not measured
RSS, and do not qualify every scale or backend. The existing resource, preassembly,
physical motion and native continuation suites remain mandatory.

The original sine-rod initial-state discriminator samples exactly the original
nodal displacement and derives acceleration from discrete equilibrium. Initial
acceleration is byte-identical for two future time partitions. Discrete balance
residuals are at most 5.8e-11, while the independent continuum acceleration fails
the same discrete balance. Initial normalized acceleration errors on the original
Tet4 meshes are 0.636/0.387/0.212 and on Tet10 are 0.311/0.116/0.050. This isolates a
spatial initial-state discrepancy before time integration. It does not remove that
sample, project an eigenmode, prescribe a different initial field, relax a history
bound or replace any of the three original failing continuum gates. Full-history
accuracy and independent temporal convergence remain open.

## Immutable initial displacement and velocity fields

The current draft accepts explicitly closed `InitialFieldBinding` inputs for direct
and modal transient workloads. Workload/template schema 2 requires the complete
binding inventory; no missing source becomes an implicit zero state. Shared field
admission validates exact source content, selected quantity, global frame, canonical
node inventory and topology before physical assembly. Initial-state realization reads
original nodal components by canonical index without interpolation or numerical
recovery. Uniform and signal inputs retain their original realization evidence;
nonuniform fields carry their immutable reference rather than invented uniform bits.

Canonical workload materialization includes the complete original field closure in
scope/block inputs. Source residency is checked against the unchanged authored
ceiling; source identity and cold continuation bind the selected content. These are
analytical resource reservations, not measured peak memory.

The current authored-field draft adds bounded lookup through shared analysis storage.
Direct and modal transient preparation resolve active immutable references before
portable admission. Original transient displacement/velocity catalogs attach to the
same atomic publication root; static displacement catalogs describe an existing
retained host field, and never trigger another solve or recovery for omitted output.
The both-order authored cold producer-to-consumer witness passes with exact original
initial displacement and velocity. Source metadata authenticates provenance and
content; it is not a numerical proof. Independent producer-process termination,
modal field-input physics and measured memory remain separate open gates. Original
rod continuum failures remain mandatory and unchanged.

### Shared whole-study source planning

`plan_structural_study_with_fields` accepts the closed source union for a compiled
study. The shared original-field owner rejects missing, duplicate and unselected
bindings; structural planning rejects procedures that cannot consume a selected
field. Direct transient workloads and modal transient templates retain their exact
active original displacement/velocity bindings. A thermal prestress prefix delegates
its selected original temperature to the existing thermal planner. No planner
performs source lookup, interpolation or a producer solve.

Before copying a selected numerical payload into a workload, the shared codec
profiles its borrowed bindings against the consuming step's unchanged source
residency ceiling. The existing owned-workload validation uses the same reservation.
This is source admission, not a measured whole-study peak-memory certificate.

The both-order witness routes actual solved U/V through planning, cold workload
decoding and the first dynamic block. The original nonuniform temperature witness
also checks cold thermal-preload-modal planning. These library checks do not close
standalone `.fea` source transport or independent producer-process termination;
those entrypoint and lifecycle gates remain open.
