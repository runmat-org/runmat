//! Shared assembly scratch reservation for structural resource plans.

// Sparse rows and contribution entries have separate resident-byte accounting. This
// envelope covers the bounded element-output batch of parallel element assembly
// (`algebra::element_assembly`, at most 96 MiB and never more than the per-element share) plus
// element-local quadrature matrices. Multiworker planners multiply the fixed reservation by
// worker count.
pub(super) const ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
pub(super) const FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER: u64 = 128 * 1024;
