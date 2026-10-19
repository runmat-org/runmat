use super::*;

#[test]
fn complete_original_fluid_state_has_exact_profile_and_cold_source() {
    let (workload, mesh) = crate::execution::fluid_test_workload();
    let result = workload.execute(&mesh).unwrap();
    let requests = fluid_state_publication_requests(&workload, &mesh, &result).unwrap();
    assert!(!requests.is_empty());
    for request in requests {
        let profile = request
            .encoding_profile(SemanticStatePublicationProfileSource::FluidPrimaryFields {
                workload: &workload,
                mesh: &mesh,
                result: &result,
            })
            .unwrap();
        let state =
            CanonicalSemanticState::from_fluid_publication(&request, &workload, &mesh, &result)
                .unwrap();
        assert_eq!(profile, profile_serialized(&state, STATE_LIMITS).unwrap());
        let bytes = state.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= profile.encoded_bytes_upper_bound);
        let cold = CanonicalSemanticState::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(cold, state);
        assert_eq!(
            cold.payload.fluid().unwrap().original_result().unwrap(),
            result
        );
        assert_eq!(
            cold.payload.fluid().unwrap().result_bytes,
            result.canonical_bytes().unwrap()
        );
    }
}

#[test]
fn fluid_state_rejects_original_source_and_declaration_substitution() {
    let (workload, mesh) = crate::execution::fluid_test_workload();
    let result = workload.execute(&mesh).unwrap();
    let request = fluid_state_publication_requests(&workload, &mesh, &result)
        .unwrap()
        .remove(0);
    let state = CanonicalSemanticState::from_fluid_publication(&request, &workload, &mesh, &result)
        .unwrap();
    let mut payload = state.payload.fluid().unwrap().clone();
    payload.result_bytes[0] ^= 1;
    assert!(payload.original_result().is_err());
    let mut substituted = state.payload.fluid().unwrap().clone();
    substituted.workload.policy.momentum_relaxation = 0.75;
    assert!(substituted.original_result().is_err());
    assert!(state
        .payload
        .fluid()
        .unwrap()
        .validate_for(
            Digest::sha256(b"foreign-layout"),
            request.source_result,
            request.producer
        )
        .is_err());
    let mut foreign = request.clone();
    foreign.declaration.state_id.0.push_str("-foreign");
    assert!(
        CanonicalSemanticState::from_fluid_publication(&foreign, &workload, &mesh, &result)
            .is_err()
    );
    let mut capped = workload.clone();
    capped.policy.maximum_resident_bytes = 1;
    assert!(
        CanonicalSemanticState::from_fluid_publication(&request, &capped, &mesh, &result).is_err()
    );
}
