# Anchored local-energy coercivity verification

Status: **verified admitted scalar electrical execution; reviewed for retention**.
The original electrical suite passes 36/36, including the finest Tet10 manufactured
field/refinement gate. Both equations retain the original quadrature, assembly arithmetic,
meshes, tolerance and proof limits. This does not qualify the whole formulation,
resolve the modal gate, or measure resource usage.

## What is checked

The shared algebra checker receives the actual prescribed-row CSR operator and
streams the original scalar element matrices. For a local matrix, it uses
`Q = K[1:,1:]` and differences `d_i = u_i - u_0`. A candidate unit-lower LDL
factorization has positive finite pivots. An outward bound on its product residual,
together with an outward majorant of the triangular inverse, proves a positive
lower bound for `dᵀ Q d`. This detects additional local null or negative modes.

The local reconstruction is `Cᵀ Q C`, where `C u = d`. Its difference from the
actual assembled CSR is enclosed entry by entry. Missing CSR entries are checked
as zero; cancellation entries are not dropped from the comparison. Both the row
and column sums bound the global error by `sqrt(||E||₁ ||E||∞)`.

A multi-source BFS starts at prescribed nodes. For each free node, Cauchy–Schwarz
along its path gives `u_i² ≤ distance_i * sum(path differences²)`. Thus the minimum
verified local bound divided by the sum of distances is a lower bound for the free
operator's energy. Prescribed identity rows supply their own unit lower bound.
The global error bound is subtracted. A nonpositive or uncertain remainder is
indeterminate, not a singularity diagnosis. Uncovered components reject.

Cached local bounds are process-local and bounded. Candidate closeness selects only
a possible prior bound. Reuse requires subtracting an outward norm bound for every
coefficient of the new local quotient matrix. There is no drop tolerance, approximate
hash acceptance or rank assumption. Failed block submission poisons the stream, so
partial mutations cannot later issue evidence. Certificate fields are private;
serialized claims cannot construct a trusted proof.

For context on element-energy comparison and floating-point effects, see the
[original support-preconditioner research](https://arxiv.org/abs/cs/0407022) and
[Kahan's discussion of finite-element roundoff](https://digicoll.lib.berkeley.edu/record/136543/files/ERL-m-593.pdf).
The checker and its bounds require their own validation; those references do not
certify this implementation.

## Scope and evidence

The original electrical matrix uses both equations, Tet4/Tet10, and cells2/4/8.
The first passing diagnostic verifies all12 operators. Finest Tet10 has a positive
lower bound near5e-9 and an assembly-error bound near7e-15. Its verification demand
is about10million numerical scalar operations under the original approximately
43million proof-work cap. Final conservative checker storage bounds are about12MB on the finest mesh;
they are requested bounds, not measured RSS. The integrated source adds about 2.8MB
on the finest mesh. Source and checker together fit the unchanged proof envelope.
Shared algebra71/71, resource170/170, bridge13/13 and native6/6 pass.
Strict FEA, bridge and native library/test lint passes without suppressions.
Actual electrothermal workers and cold publication pass for both element orders.

Independent small-matrix tests check the bound against an exact eigenvalue, damaged
operators, extra local modes, asymmetric values, missing anchors, nonfinite input,
work rejection, complete coefficient perturbations and poisoned partial streams.

## Execution and cold evidence boundary

Electrical assembly captures original element contributions during its single existing
quadrature pass, without changing global accumulation. A flat, ordered source inventory
reserves all node indices, dense local values and block offsets before allocation;
incomplete capture rejects. The whole requested-heap plan reserves source residency
explicitly before projection and physical assembly. Resource model revision79 binds
this contract. The actual prepared CSR's unchanged work/storage envelope also bounds
the source and checker together. Normalization occurs once on the captured values.
Candidate comparison uses the complete quotient Q. The checker reconstructs only
entries remaining in the constrained free block, with identity rows checked separately;
it never drops a discrepancy in that prepared operator. This eliminates redundant
coefficient work and the second dense local shadow block while preserving all bounds.

The existing shared linear preparation owner consumes the source to verify its single
constrained matrix and binds the opaque proof to that operator and current RHS/source
nullspace evidence. It frees source values after the proof. Changed operator values
invalidate reuse; failed replacement cannot leave an old proof available. Coercivity
identity revision2 distinguishes sparse-LDL pivot evidence from a verified local-energy
lower bound. Local proof scratch is reserved as bytes; no global sparse factor is retained.
Other formulations continue using the existing general sparse-LDL check.

Cancellation is polled through the shared progress token at source/block/row/graph
boundaries, and produces no completed certificate. Public electrical results and numerical
receipts retain original fields and physical closure. Their cold validation checks source,
arrays, integrals and provenance; it neither reconstructs a mathematical certificate from
hashes nor reruns missing fields. A new numerical execution from cold workload/mesh inputs
captures and verifies the original operator again in process. Certificates have no decoder.

Hashes bind evidence after mathematical verification. Scalar constant-null bounds do not
establish the rigid-body coercivity required by structural modal admission. Full Stage7,
measured resources, all supported execution envelopes and release requirements remain open.
