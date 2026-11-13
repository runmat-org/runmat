use super::*;

#[test]
fn representative_is_valid_completed_shape_and_uses_authoritative_serializer() {
    let evidence = completed_shape();
    evidence.validate().unwrap();
    assert_eq!(
        AcceleratorSolveEvidence::completed_profile().unwrap(),
        runmat_canonical_codec::profile_serialized_shape(&evidence, PROFILE_LIMITS).unwrap(),
    );
    let AcceleratorSolveEvidence::Completed { provider, .. } = evidence else {
        panic!("projection must represent completed evidence");
    };
    assert!(provider.binding.preconditioner_id.is_some());
    assert!(provider.provider_loss.is_none());
    assert_eq!(provider.binding.attempt_id.to_string().len(), 40);
    assert_eq!(provider.binding.operator_identity.to_string().len(), 71);
}

#[test]
fn completed_bound_covers_optional_preconditioner_and_extreme_scalars() {
    let bound = AcceleratorSolveEvidence::completed_profile().unwrap();
    for preconditioner in [None, Some(0), Some(u64::MAX)] {
        let mut evidence = completed_shape();
        let AcceleratorSolveEvidence::Completed {
            operator_identity,
            right_hand_side_identity,
            solution_identity,
            provider,
            provider_reported_final_residual_norm,
            independently_computed_residual_norm,
            independently_computed_relative_residual,
            required_relative_tolerance,
            ..
        } = &mut evidence
        else {
            unreachable!()
        };
        let identity = Digest::sha256(b"different operator and fixed-width identifiers");
        *operator_identity = identity;
        *right_hand_side_identity = identity;
        *solution_identity = identity;
        provider.binding.operator_identity = identity;
        provider.binding.attempt_id = AttemptId::from_bytes([u8::MAX; 16]);
        provider.binding.provider_device_id = u32::MAX;
        provider.binding.provider_capability_schema_version = u16::MAX;
        provider.binding.provider_capability_revision = u64::MAX;
        provider.binding.pattern_id = u64::MAX;
        provider.binding.matrix_id = u64::MAX;
        provider.binding.preconditioner_id = preconditioner;
        provider.timings = SparseOperationTimings {
            preparation_ns: u64::MAX,
            execution_ns: u64::MAX,
            terminal_synchronization_ns: u64::MAX,
        };
        provider.resident_solution_bytes = u64::MAX;
        provider.peak_resident_bytes = u64::MAX;
        provider.host_upload_bytes = u64::MAX;
        provider.host_download_bytes = u64::MAX;
        provider.setup_dispatches = u64::MAX;
        provider.spmv_dispatches = u64::MAX;
        provider.preconditioner_dispatches = u64::MAX;
        provider.vector_dispatches = u64::MAX;
        provider.host_synchronizations = u32::MAX;
        provider.device_readbacks = u32::MAX;
        provider.cancellation_checks = u64::MAX;
        *provider_reported_final_residual_norm = f64::MAX;
        *independently_computed_residual_norm = f64::MAX;
        *independently_computed_relative_residual = f64::MAX;
        *required_relative_tolerance = f64::MIN_POSITIVE;
        evidence.validate().unwrap();
        let actual = runmat_canonical_codec::profile_serialized(&evidence, PROFILE_LIMITS).unwrap();
        assert!(actual.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
        assert!(
            actual.materialization_resident_bytes_estimate
                <= bound.materialization_resident_bytes_estimate
        );
    }
}
