use super::*;
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
use runmat_analysis_core::{ProcedureStep, StateId};

fn primary_request() -> SemanticStatePublicationRequest {
    let step = ProcedureStep::structural_static("profile");
    let sequence = SequenceIdentity::new(Digest::sha256(b"profile-sequence")).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &step).unwrap();
    SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralPrimaryFields,
        producer,
        step.state_outputs[0].clone(),
        Digest::sha256(b"topology"),
        SolverResultIdentity::from_step_result(producer, Digest::sha256(b"result")).unwrap(),
    )
    .unwrap()
}

#[test]
fn primary_profile_matches_selected_state_and_tracks_declaration_length() {
    let fields = vec![
        AnalysisField::host_f64(
            "unpublished",
            AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global),
            vec![256],
            vec![4.0; 256],
        ),
        AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            AnalysisFieldLayout::scalar(ObservationQuantity::Displacement, FieldLocation::Node),
            vec![3],
            vec![0.0, -1.25, 1.0e100],
        ),
    ];
    let mut request = primary_request();
    let mut previous = None;
    for length in [1, 24, 256] {
        request.declaration.state_id = StateId("s".repeat(length));
        let projected = request
            .encoding_profile(
                SemanticStatePublicationProfileSource::StructuralPrimaryFields(&fields),
            )
            .unwrap();
        let actual = CanonicalSemanticState::from_publication(&request, &fields).unwrap();
        assert_eq!(
            projected,
            profile_serialized(&actual, STATE_LIMITS).unwrap()
        );
        assert!(
            actual.canonical_bytes().unwrap().len() as u64 <= projected.encoded_bytes_upper_bound
        );
        if let Some(previous) = previous {
            assert!(projected.encoded_bytes_upper_bound > previous);
        }
        previous = Some(projected.encoded_bytes_upper_bound);
        let selected_only = request
            .encoding_profile(
                SemanticStatePublicationProfileSource::StructuralPrimaryFields(&fields[1..]),
            )
            .unwrap();
        assert_eq!(projected, selected_only);
    }
}

#[test]
fn primary_profile_rejects_missing_duplicate_and_mismatched_sources() {
    let mut request = primary_request();
    assert!(request
        .encoding_profile(SemanticStatePublicationProfileSource::StructuralPrimaryFields(&[]))
        .is_err());
    let field = AnalysisField::host_f64(
        crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        AnalysisFieldLayout::scalar(ObservationQuantity::Displacement, FieldLocation::Node),
        vec![1],
        vec![0.0],
    );
    let fields = [field.clone(), field];
    assert!(request
        .encoding_profile(SemanticStatePublicationProfileSource::StructuralPrimaryFields(&fields))
        .is_err());
    let unrelated_duplicates = [
        fields[0].clone(),
        AnalysisField::host_f64(
            "other",
            AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global),
            vec![1],
            vec![1.0],
        ),
        AnalysisField::host_f64(
            "other",
            AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global),
            vec![1],
            vec![2.0],
        ),
    ];
    assert!(request
        .encoding_profile(
            SemanticStatePublicationProfileSource::StructuralPrimaryFields(&unrelated_duplicates)
        )
        .is_err());
    request.source = SemanticStatePublicationSource::ModalBasis;
    request.declaration.kind = runmat_analysis_core::StateKind::ModalBasis;
    request.validate().unwrap();
    assert!(request
        .encoding_profile(
            SemanticStatePublicationProfileSource::StructuralPrimaryFields(&fields[..1])
        )
        .is_err());
}

#[test]
fn primary_profile_rejects_invalid_storage_and_canonical_metadata_limits() {
    let mut request = primary_request();
    for field in [
        AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            AnalysisFieldLayout::scalar(ObservationQuantity::Displacement, FieldLocation::Node),
            vec![2],
            vec![1.0],
        ),
        AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            AnalysisFieldLayout::scalar(ObservationQuantity::Displacement, FieldLocation::Node),
            vec![0],
            vec![],
        ),
        AnalysisField::host_f64(
            crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            AnalysisFieldLayout::scalar(ObservationQuantity::Displacement, FieldLocation::Node),
            vec![1],
            vec![f64::NAN],
        ),
    ] {
        assert!(request
            .encoding_profile(
                SemanticStatePublicationProfileSource::StructuralPrimaryFields(
                    std::slice::from_ref(&field)
                )
            )
            .is_err());
        assert!(CanonicalSemanticState::from_publication(&request, &[field]).is_err());
    }
    let field = AnalysisField::host_f64(
        crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        AnalysisFieldLayout::scalar(ObservationQuantity::Displacement, FieldLocation::Node),
        vec![1],
        vec![0.0],
    );
    request.declaration.state_id = StateId("s".repeat(STATE_LIMITS.maximum_string_bytes + 1));
    assert!(request
        .encoding_profile(SemanticStatePublicationProfileSource::StructuralPrimaryFields(&[field]))
        .is_err());
}
