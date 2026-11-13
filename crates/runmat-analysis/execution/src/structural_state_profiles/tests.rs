use super::*;
use crate::prepare_analysis_objects;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, SemanticStatePublicationRequest,
};
use runmat_canonical_codec::SerializedCanonicalProfile;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};

fn catalog(encoded: u64, payload: Digest) -> StatePublicationProfiles {
    let step = ProcedureStep::structural_static("producer");
    let sequence = SequenceIdentity::new(Digest::sha256(b"sequence")).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &step).unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap();
    let topology = Digest::sha256(b"topology");
    let request = SemanticStatePublicationRequest::new(
        Source::StructuralPrimaryFields,
        producer,
        step.state_outputs[0].clone(),
        topology,
        result,
    )
    .unwrap();
    StatePublicationProfiles::new(
        producer,
        result,
        topology,
        payload,
        vec![StatePublicationProfileEntry::new(
            request,
            SerializedCanonicalProfile {
                encoded_bytes_upper_bound: encoded,
                materialization_resident_bytes_estimate: 100_000,
            },
        )
        .unwrap()],
    )
    .unwrap()
}

#[test]
fn imported_catalog_requires_exact_independent_projection_even_with_valid_rehashed_wire() {
    let source = Digest::sha256(b"source");
    let expected = catalog(1024, source);
    for actual in [
        expected.clone(),
        catalog(1023, source),
        catalog(1024, Digest::sha256(b"other")),
    ] {
        let bytes = actual.canonical_bytes().unwrap();
        assert_eq!(
            StatePublicationProfiles::from_canonical_bytes(&bytes).unwrap(),
            actual
        );
        let objects = prepare_analysis_objects(
            AnalysisNumericalObjectKind::StatePublicationProfiles,
            actual.identity(),
            &bytes,
            64 * 1024,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        assert_eq!(
            validate_profiles(&objects, expected.clone()).is_ok(),
            actual == expected
        );
    }
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        Digest::sha256(b"wrong-logical-id"),
        &expected.canonical_bytes().unwrap(),
        64 * 1024,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert!(validate_profiles(&objects, expected).is_err());
}
