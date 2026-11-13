# Governed GENERAL sparse solver resources

Resource model revision104 adds `SolveResourcePolicy::GeneralSparse`. Its immutable
validated controls retain the complete shared `GeneralSolvePolicy`, including
candidate, proof and physical-RHS controls. Planning uses the same
`GeneralSolveEnvelope::for_counts` as numerical execution. Dimension, nonzero upper
bound, maximum row width, admitted component workspace/work and maximum solve count
are bound into canonical resource identity. Decoded counts are claims: validation
rederives demand, and execution compares the original controls and actual CSR.

The complete factor, candidate/proof and physical solve component is charged once
as solver workspace. No SPD preconditioner or factor estimate is added to this
GENERAL component. Supported placement is serial CPU with monolithic operator;
accelerator claims and distributed operator placement reject. The source CSR/RHS,
body/equality preparation, state, outputs, transport and allocator overhead remain
separate obligations. Requested workspace is not measured process RSS.

Prepared contact Newton derives this demand from exact reduced symbolic support
before materializing combined/reduced CSR columns. Every invocation rechecks its
actual reduced operator and original controls before invoking the shared solver.
Its demand has maximum solves1 per invocation; the existing contact controller
counts cumulative linear solves separately. This does not yet constitute whole-task
scalar-work admission or native contact execution. The all-fixed case performs no
GENERAL solve. Each executed solve still constructs a fresh original-operator
right-inverse certificate, including zero physical RHS. Resource admission cannot
turn an indeterminate certificate into a successful result.

Independent tests cover a closed nonsymmetric linear reference, singular zero RHS,
changed controls with unchanged workspace, actual support exceeding a valid planned
row-width or nonzero bound, missing/nonfinite controls, storage/work shortage and
cumulative-work overflow. Shared resource-plan tests cover canonical roundtrip,
identity changes, exact additive workspace, unchanged memory-ceiling rejection and
unsupported placement. Final candidate passes683focused tests, including183resource
and7GENERAL provider tests; strict shared/FEA/bridge lint passes. Native compilation
and lint pass on unchanged production source; the final source correction affected
only the negative support-bound test fixture.

Whole contact task admission, typed public interface controls, immutable native
producer-death continuation, independent mandatory contact accuracy and semantic
publication remain open. Local tangent certification is not nonlinear uniqueness,
stability or contact qualification. All original scientific failures
and release acceptance requirements remain unchanged.
