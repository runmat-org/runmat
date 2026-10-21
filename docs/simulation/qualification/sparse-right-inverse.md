# Source-independent sparse uniqueness certificate

The shared algebra `RightInverseCheck` is a sufficient nonsingularity check for
any nonempty square real CSR operator, including the real block representation
of a complex Helmholtz operator. Authored acoustic planning now requests its
shared candidate provider; this does not close independent accuracy, restart,
measured-resource or release qualification, nor Stage 8.

For a candidate matrix B, let E=I−AB. If ||E||∞<1, the convergent Neumann series
gives (AB)⁻¹=Σ Eᵏ. Thus AB and the square A are invertible. The candidate need not
be an exact factor inverse: its supplied finite FP64 columns define B exactly.
An existing shared ILU may produce those columns, but invertible ILU pivots alone
do not prove the original operator invertible.

Every product and subtraction is enclosed by adjacent representable FP64 bounds;
absolute row sums accumulate upward. Overflow or a bound of at least one yields
`Indeterminate`, never a singularity diagnosis. No source, load magnitude, GMRES
residual, passive power balance, artificial damping or dimensional floor enters
the check. This is uniqueness, not a claim of good physical conditioning or
adequate spatial accuracy.

Columns arrive once in exact order. The checker retains O(n) row bounds and hashes
the complete candidate column inventory; it never retains a dense inverse. The
certificate binds the original shared numerical matrix identity, exact candidate
identity, dimension and outward defect bound. Certificate fields are private and
serialize-only. Cold verification must repeat the numerical check against the
original matrix and candidate; decoding a claimed certificate is not authority.

`RightInverseWorkEnvelope` uses checked count arithmetic and is admitted before
identity construction or row allocation. It covers checker work and workspace,
including conservative shared identity encoding, but excludes the borrowed source
matrix and candidate-column production/factorization. A governed consumer must
add those lifetimes/work, admit the complete task, and retain/reverify its original
operator/factor provenance. Requested storage is not measured RSS.

Independent tests use a literal indefinite inverse and an analytic null vector,
scale changes by 1e±120, shared ILU as a candidate, inadequate candidates for an
invertible matrix, overflow/cancellation-sensitive entries, exact inventory and
pre-allocation admission, and cold operator revalidation. They are shared algebra
evidence, not an acoustic cavity resonance or whole native execution witness.

## Shared candidate provider and actual acoustic discriminator

`solve::uniqueness::certify_sparse_uniqueness` constructs candidate columns with
the existing shared fixed-support right ILU and restarted GMRES. It does not use
a physical source or change physical solve tolerances. Its explicit candidate
policy bounds each column's Krylov work. Factor failure or an unsuccessful column
is `Indeterminate`; no failed factorization is mislabeled singular. The completed
candidate must still pass the independent outward check above. Cancellation never
produces a certificate.

The shared count-only/actual-operator envelope covers all candidate columns,
factor setup, repeated operator/triangular application, Krylov dots/updates, proof,
coexisting factor/vector/identity storage. It excludes the original borrowed CSR.
Admission occurs before factorization. Acoustic task planning adds the complete
component demand to coexisting original operators, physical factor/workspaces,
result/receipt/profile construction and transport reservations. Native workers
rederive the whole plan from authoritative workload/mesh before execution. This
is requested-heap/work admission, not measured resource usage or wall-time prediction.

On original canonical matched ducts (Tet4/Tet10, cells1/2/4), ILU-only candidates
have outward defects2.30–5.84 and cannot prove uniqueness. A test candidate policy
uses tolerance1/(16n), restart64 and at most128 iterations per original column;
these are diagnostic construction limits, not a replacement physical target.
The corrected candidates pass all six original cases at outward defects
0.00235–0.01127, using83–41183 actual total iterations. The finest Tet10 case
has1458 real-block DOFs and35652 retained entries; outward defect0.00234678.
The component's requested additional storage is8742928 bytes, excluding original
CSR; worst-case work reservation750029617920 scalar units is conservative, not
a measured operation count or wall-time forecast. Full task lifetime/usage remains
unverified. The actual full matrix was checked, not sampled columns or a dense
small-case substitute.

For a unit canonical Tet4 in a rigid lossless fluid, consistent simplex integrals
give the independent antisymmetric mode[0,1,−1,0] at omega²=20c²=80. Its actual
operator action is near roundoff; the old homogeneous solve accepts zero residual.
Neither candidate method establishes uniqueness there. Floating representation
may move an exact mathematical pole, so the diagnostic stays indeterminate rather
than claiming exact floating-matrix rank from a small null-vector residual.

## Governed acoustic diagnostic gate and cold semantics

`AcousticStudyPlanningPolicy` requires explicit candidate controls, independent
of physical residual/power targets. The bounded default is candidate tolerance1e-6,
restart64 and128 iterations per column. It is only a construction policy: acceptance
still requires the outward defect below1. The direct library can
explicitly request no diagnostic; missing controls are not silently defaulted when
decoding its numerical policy. These unrequested results establish physical closure
only and acquire no uniqueness or qualification.

The original affine-reduced real CSR is certified once before physical-RHS acceptance
and reused by every power-refinement correction. Any indeterminate outcome rejects
the requested solve with original operator digest, reason, completed columns and
diagnostic iterations. Cancellation stays cancellation; no failed factor is labeled
singular. A fully prescribed pressure has an explicit zero-free-unknown diagnostic,
not an absent record. Success never substitutes for original residual/power gates.

Results retain operator/candidate identities, dimension, outward bound, proof controls
and diagnostic iterations; the exact numerical receipt retains the same claim.
Canonical result identity and publication source digests include these bytes. Cold
imports validate required evidence shape/control consistency and exact original
workload/mesh/result/receipt provenance without assembly or physical-field re-solving.
This is **producer-claim lineage validation**, not independent mathematical replay:
candidate columns were discarded, and hashes cannot re-prove the inequality.
`SparseUniquenessRecord` is deliberately distinct from serialize-only
`RightInverseCertificate`; there is no conversion from decoded claim to trusted proof.
A well-shaped fabricated digest changes result identity and fails against the retained
original receipt. Independent cold algebra revalidation still requires an explicitly
admitted retained witness or candidate replay. General conditioning, continuum field
accuracy, source-independent cavity/resonance envelopes, measured usage and release
qualification remain open.
