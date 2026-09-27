use super::*;
use crate::execution::{
    CanonicalSemanticStateInput, CanonicalSemanticStatePayload, SemanticStatePublicationRequest,
    SemanticStatePublicationSource,
};
use crate::pipeline::structural_linearization::StructuralLinearizationArtifact;

#[test]
fn thermal_prestress_roundtrip_and_publication_bind_exact_structural_result() {
    let (mut model, mesh) = model_and_mesh(ElementOrder::Tet10, true);
    let declaration = StateOutput {
        state_id: StateId("thermal_prestress".into()),
        kind: StateKind::GeometricStiffness,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    };
    model.sequence.steps[1]
        .state_outputs
        .push(declaration.clone());
    let source = solve_temperature(&model, &mesh);
    let result =
        run_linear_static_from_thermal_state(&model, ComputeBackend::Cpu, options(&mesh), &source)
            .unwrap();
    let retained = result.linearization.thermal_source().unwrap();
    assert_eq!(retained.transition(), &result.thermal_transition);
    assert_eq!(retained.structural_result(), result.solver_result_identity);
    let portable = serde_json::to_value(&result.linearization).unwrap();
    let decoded: StructuralLinearizationArtifact =
        serde_json::from_value(portable.clone()).unwrap();
    assert_eq!(decoded, result.linearization);
    for mutation in [
        "absent",
        "null",
        "transition",
        "result",
        "topology",
        "schema",
    ] {
        let mut changed = portable.clone();
        match mutation {
            "absent" => {
                changed.as_object_mut().unwrap().remove("thermal_source");
            }
            "null" => changed["thermal_source"] = serde_json::Value::Null,
            "transition" => {
                changed["thermal_source"]["transition"]["method"] =
                    serde_json::json!("initial_guess_only")
            }
            "result" => {
                changed["thermal_source"]["structural_result"] =
                    serde_json::to_value(source.source_result).unwrap()
            }
            "topology" => {
                changed["topology_digest"] =
                    serde_json::to_value(Digest::sha256("foreign topology")).unwrap()
            }
            _ => changed["schema_version"] = serde_json::json!(1),
        }
        let error = serde_json::from_value::<StructuralLinearizationArtifact>(changed).unwrap_err();
        if matches!(mutation, "absent" | "null") {
            // Context is rejected before the artifact hash comparison, so rehashing
            // cannot turn a solved thermal artifact into an unbound mechanical one.
            assert!(error
                .to_string()
                .contains("thermal source and coupling differ"));
        }
    }
    let mut authored_guess = model.sequence.transitions[0].clone();
    authored_guess.method = StateTransitionMethod::InitialGuessOnly;
    let guess = CanonicalStateTransition::new(crate::execution::CanonicalStateTransitionInput {
        sequence: result.thermal_transition.sequence,
        producer: source.producer,
        consumer: result.thermal_transition.consumer,
        authored: &authored_guess,
        source: &source,
        target_topology_digest: result.linearization.topology_digest(),
    })
    .unwrap();
    guess.validate().unwrap();
    let mut changed = portable.clone();
    changed["thermal_source"]["transition"] = serde_json::to_value(guess).unwrap();
    assert!(
        serde_json::from_value::<StructuralLinearizationArtifact>(changed)
            .unwrap_err()
            .to_string()
            .contains("thermal transition and structural result differ")
    );
    let request = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralLinearization,
        result.thermal_transition.consumer,
        declaration.clone(),
        result.linearization.topology_digest(),
        result.solver_result_identity,
    )
    .unwrap();
    let state =
        CanonicalSemanticState::from_linearization_publication(&request, decoded.clone()).unwrap();
    assert_eq!(
        CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap()).unwrap(),
        state
    );
    let mut wrong = request.clone();
    wrong.source_result = source.source_result;
    assert!(
        CanonicalSemanticState::from_linearization_publication(&wrong, decoded.clone()).is_err()
    );
    wrong = request.clone();
    wrong.producer = source.producer;
    assert!(
        CanonicalSemanticState::from_linearization_publication(&wrong, decoded.clone()).is_err()
    );
    // Direct construction derives a fresh state hash, so rejection is contextual, not a stale hash.
    assert!(CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer: request.producer,
        declaration,
        topology_digest: request.topology_digest,
        source_result: source.source_result,
        payload: CanonicalSemanticStatePayload::StructuralLinearization {
            linearization: Box::new(decoded)
        },
    })
    .is_err());
}
