# Source-independent sparse uniqueness certificate

The shared algebra `RightInverseCheck` is a sufficient nonsingularity check for
any nonempty square real CSR operator, including the real block representation
of a complex Helmholtz operator. It is not yet wired into authored acoustic
accuracy/restart or release admission and does not close Stage 8.

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
