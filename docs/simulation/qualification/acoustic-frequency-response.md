# Canonical acoustic frequency response — bounded execution evidence

`AcousticFrequencyWorkload` derives the existing shared `FrequencyResponsePlan`
from one authored canonical harmonic step. It retains the authored source and
reference frequency; it cannot add frequencies, observations, lifecycle activations
or unrelated steps. Each plan entry lowers to the existing source/mesh-bound
`AcousticHarmonicWorkload`, admitted by the existing acoustic task and resource planner.
No reduced graph, new solver, checkpoint implementation or second solve is introduced.

`AcousticFrequencyResult::assemble` is pure fan-in. It validates every completed
entry against the exact workload and canonical mesh, rejects missing/duplicate/
foreign entries, and retains all complex fields and numerical closure evidence in
shared-plan order. Completion order and transport chunking do not change its
scientific identity. Decoding validates the canonical plan/source/field inventory;
`validate_for` additionally binds it to the caller's canonical mesh and authored
workload. This collection does **not** publish the last frequency or the complete
sweep as one `AcousticComplex` lifecycle state.

The FEA tests solve matched plane-wave ducts at 0.1/0.2/0.3 Hz, with `rho=3`,
`c=2`, unit inlet pressure and impedance `rho*c`. Independent references are
`p=exp(-ikx)`, `k=2*pi*f/c`, mean axial intensity `1/(2*rho*c)=1/12 W/m²`
and energy density `1/(2*rho*c²)=1/24 J/m³`. The four-cell Tet4/two-cell Tet10
cases retain the existing bounded pressure RMS, intensity and energy absolute
error gates of 0.01. These are fixed-mesh execution witnesses, not a new spatial
convergence certificate or radiation-boundary qualification.

The native test prepares three separately resource-admitted tasks in reverse
completion order, executes the real native host kernel through serial analysis,
closes/reopens the filesystem object store before importing every accepted entry,
and assembles/reopens the complete response. Both element orders and chunk sizes
1024/8192 must give the same scientific response identities. Its single-element
fixture is an execution witness, not an independent continuum reference.

Whole-study authored dispatch and observation catalogs, assembled artifact
publication/resource admission, actual failed-process restart/replacement,
remote task scheduling, GPU/cooperative distribution, measured resource/accuracy
release gates and full Stage 8 qualification remain open. Existing
failed local-radiation convergence gates are unchanged and remain failed.

## Accepted numerical receipt and shared accuracy

Each canonical harmonic task now emits three exact outputs: accepted fields,
an `AcousticNumericalReceiptArtifact`, and publication profiles. The receipt binds
the exact numerical workload, retained result, producer, mesh, frequency and
accepted solver/power targets. It does not reconstruct lost fields or repeat the
solve. A separate bounded 64 KiB receipt transport envelope is included in shared
resource model revision 41.

Shared accuracy admission requires the original workload, canonical mesh and
retained result, and authenticates the exact authored acceptance plan. The receipt
supports only global algebraic residual and equation-based power conservation.
Mandatory spatial, field or frequency convergence evidence remains unavailable;
these metrics are not an independent accuracy certificate. Missing, duplicate or
substituted source objects and rehashed foreign requirements are rejected.

This closes an accuracy/publication prerequisite, not whole-study authored
dispatch, observations, process restart or Stage 8 qualification.

Receipt verification on 2026-09-30: canonical receipt tests pass 2/2, acoustic
resource tests 4/4 and the full bridge library 92/92. Native acoustic tests pass
11/11, including Tet4/Tet10 ×1024/8192-byte cold filesystem solve→receipt→shared
accuracy and the retained frequency/publication paths. These are native serial
execution/cold-import witnesses, not failed-process restart evidence.

Verification on 2026-09-30: portable acoustic tests pass 7/7; native acoustic tests
pass 10/10, including the sweep matrix. FEA all-target strict Clippy passes;
native all-target strict Clippy with `--no-deps` passes. The joint dependency-inclusive
strict run fails on the unrelated plotting `field_reassign_with_default` lint;
architecture checks retain four unrelated runtime module-size debts. These failures
are recorded, not suppressed or declared green. Full release verification is open.
