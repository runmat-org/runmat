use super::*;
use crate::execution::{
    structural_state_publication_requests, SequenceIdentity, SolverResultIdentity,
    StatePublicationProfileEntry, StatePublicationProfiles, StepIdentity,
};
use crate::fixtures::{fixture_model, FixtureId};
use runmat_canonical_codec::SerializedCanonicalProfile;

#[test]
fn authored_static_catalog_profile_covers_actual_metadata_without_numerical_buffers() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let step = &model.sequence.steps[0];
    let topology = Digest::sha256(b"static topology");
    let producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&model.sequence).unwrap(),
        step,
    )
    .unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap();
    let requests = structural_state_publication_requests(
        &model,
        step,
        &[
            SemanticStatePublicationSource::StructuralPrimaryFields,
            SemanticStatePublicationSource::StructuralLinearization,
        ],
        topology,
        result,
    )
    .unwrap();
    assert!(!requests.is_empty());
    let catalog = StatePublicationProfiles::new(
        producer,
        result,
        topology,
        Digest::sha256(b"payload"),
        requests
            .into_iter()
            .map(|request| {
                StatePublicationProfileEntry::new(
                    request,
                    SerializedCanonicalProfile {
                        encoded_bytes_upper_bound: 1024,
                        materialization_resident_bytes_estimate: 100_000,
                    },
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let facts = structural_state_profiles_output_resource_facts(&model, step, topology).unwrap();
    assert!(
        catalog.canonical_bytes().unwrap().len() as u64
            <= facts.encoded_payload_bytes_upper_bound.0
    );
    assert_eq!(
        facts.transport_closure_bytes_upper_bound.0,
        facts.encoded_payload_bytes_upper_bound.0 + ANALYSIS_MANIFEST_MAX_ENCODED_BYTES
    );
    assert_eq!(
        facts,
        projected_canonical_output_resource_facts(facts.encoded_payload_bytes_upper_bound.0)
            .unwrap()
    );
}

#[test]
fn canonical_output_projection_rejects_zero_and_overflow() {
    assert!(projected_canonical_output_resource_facts(0).is_err());
    assert!(projected_canonical_output_resource_facts(u64::MAX).is_err());
}
