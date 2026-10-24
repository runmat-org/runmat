//! Successful accelerator evidence sizing through its authoritative derived serializer.

use runmat_accelerate_api::{SparseExecutionBinding, SparseOperationTimings};
use runmat_canonical_codec::{
    profile_serialized_shape, CanonicalCodecError, CanonicalLimits, SerializedCanonicalProfile,
};
use runmat_execution::identity::AttemptId;

use super::*;

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 32);

impl AcceleratorSolveEvidence {
    /// Profiles one completed record, not a history or transient duplicate copies.
    /// The shared profiler reserves the widest scalar representation independent of value.
    /// All identities have fixed serialized lengths; the only optional successful field is
    /// profiled present. Completed records cannot contain provider-loss text or failure states.
    /// No dummy arrays or evidence vectors are constructed. Typed ID serialization does allocate
    /// bounded temporary strings, so this traversal is deliberately not called allocation-free.
    pub(crate) fn completed_profile() -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
        profile_serialized_shape(&completed_shape(), PROFILE_LIMITS)
    }

    #[cfg(test)]
    pub(crate) fn completed_evidence_fixture() -> Self {
        completed_shape()
    }
}

fn completed_shape() -> AcceleratorSolveEvidence {
    let identity = Digest::sha256([]);
    AcceleratorSolveEvidence::Completed {
        schema_version: ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION,
        operator_identity: identity,
        right_hand_side_identity: identity,
        solution_identity: identity,
        provider: SparseOperationEvidence {
            schema_version: SPARSE_EVIDENCE_SCHEMA_VERSION,
            binding: SparseExecutionBinding {
                attempt_id: AttemptId::from_bytes([0; 16]),
                operator_identity: identity,
                provider_device_id: 0,
                provider_capability_schema_version: 1,
                provider_capability_revision: 1,
                pattern_id: 0,
                matrix_id: 0,
                preconditioner_id: Some(0),
            },
            timings: SparseOperationTimings::default(),
            resident_pattern_bytes: 0,
            resident_matrix_bytes: 0,
            resident_preconditioner_bytes: 0,
            resident_solution_bytes: 1,
            host_upload_bytes: 0,
            host_download_bytes: 0,
            setup_dispatches: 0,
            spmv_dispatches: 0,
            preconditioner_dispatches: 0,
            vector_dispatches: 0,
            host_synchronizations: 0,
            device_readbacks: 0,
            peak_resident_bytes: 1,
            scratch_bytes: 0,
            cancellation_checks: 0,
            cancellation_observed: false,
            provider_loss: None,
            used_host_fallback: false,
        },
        provider_reported_final_residual_norm: 0.0,
        terminal_solution_readbacks: 1,
        independently_computed_residual_norm: 0.0,
        independently_computed_relative_residual: 0.0,
        required_relative_tolerance: 1.0,
    }
}

#[cfg(test)]
mod tests;
