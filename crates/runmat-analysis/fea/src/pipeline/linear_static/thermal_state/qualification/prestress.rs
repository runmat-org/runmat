use super::*;
use crate::execution::{
    CanonicalSemanticStateInput, CanonicalSemanticStatePayload, SemanticStatePublicationRequest,
    SemanticStatePublicationSource,
};
use crate::pipeline::structural_linearization::StructuralLinearizationArtifact;
use runmat_execution::Digest;

#[test]
fn portable_thermal_result_separates_physical_source_from_transport_commitment() {
    use crate::execution::{
        AnalysisWorkloadIdentity, StructuralLinearNumericalReceiptArtifact,
        StructuralLinearStateInput, StructuralLinearWorkload, StructuralLinearWorkloadResult,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh) = model_and_mesh(order, true);
        let source = solve_temperature(&model, &mesh);
        let solved = run_linear_static_from_thermal_state(
            &model,
            ComputeBackend::Cpu,
            options(&mesh),
            &source,
        )
        .unwrap();
        let physical_identity = solved.solver_result_identity;
        let constraints = solved
            .run
            .structural_constraint_result
            .as_deref()
            .unwrap()
            .clone();
        assert!(
            StructuralLinearWorkloadResult::new(solved.run.clone(), constraints.clone()).is_err()
        );
        let result = StructuralLinearWorkloadResult::with_linearization(
            solved.run,
            constraints,
            solved.linearization,
        )
        .unwrap();
        let retained = result
            .linearization
            .as_ref()
            .unwrap()
            .thermal_source()
            .unwrap();
        retained.validate_model(&model).unwrap();
        let workload = StructuralLinearWorkload {
            schema_version: 2,
            model: model.clone(),
            step_id: retained.transition().consumer_step_id.clone(),
            backend: ComputeBackend::Cpu,
            preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
            algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
            observation_execution: crate::results::observation::ObservationExecutionPolicy::serial(
            ),
            require_solver_mesh_for_solid: true,
            state_input: Some(StructuralLinearStateInput::ThermalPrestress(
                retained.transition().clone(),
            )),
        };
        result.validate_thermal_workload(&workload).unwrap();
        result
            .validate_thermal_source(retained.transition().consumer, physical_identity)
            .unwrap();
        assert!(result
            .validate_thermal_source(retained.transition().producer, physical_identity)
            .is_err());
        let foreign_result = crate::execution::SolverResultIdentity::from_step_result(
            retained.transition().consumer,
            Digest::sha256("foreign physical result"),
        )
        .unwrap();
        assert!(result
            .validate_thermal_source(retained.transition().consumer, foreign_result)
            .is_err());
        let mut foreign_model = model.clone();
        foreign_model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k = runmat_types::physical::Quantity::new(
            foreign_model.materials[0]
                .thermal
                .as_mut()
                .unwrap()
                .expansion_coefficient_per_k
                .canonical_value()
                * (2.0),
        );
        assert!(retained.validate_model(&foreign_model).is_err());
        let mut foreign_workload = workload.clone();
        foreign_workload.model = foreign_model;
        foreign_workload.validate().unwrap();
        assert!(result.validate_thermal_workload(&foreign_workload).is_err());
        let mut stripped_workload = workload.clone();
        stripped_workload.state_input = None;
        assert!(result
            .validate_thermal_workload(&stripped_workload)
            .is_err());
        let first = AnalysisWorkloadIdentity::for_test("thermal first placement").unwrap();
        let second = AnalysisWorkloadIdentity::for_test("thermal second placement").unwrap();
        let receipt = StructuralLinearNumericalReceiptArtifact::new(first, &result).unwrap();
        let other = StructuralLinearNumericalReceiptArtifact::new(second, &result).unwrap();
        assert_eq!(receipt.source_result(), physical_identity);
        assert_eq!(other.source_result(), physical_identity);
        assert_eq!(result.result_identity(first).unwrap(), physical_identity);
        assert_ne!(receipt.identity(), other.identity());
        receipt
            .validate_against(physical_identity, &result)
            .unwrap();
        let decoded = StructuralLinearWorkloadResult::from_canonical_bytes(
            &result.canonical_bytes().unwrap(),
        )
        .unwrap();
        let portable_receipt = StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
            &receipt.canonical_bytes().unwrap(),
        )
        .unwrap();
        portable_receipt
            .validate_against(physical_identity, &decoded)
            .unwrap();
        let mut changed = receipt.clone();
        changed.source_workload = second;
        assert!(changed.validate().is_err());
        changed = receipt.clone();
        changed.source_payload_digest = Digest::sha256("foreign transport payload");
        assert!(changed.validate().is_err());
        let mut changed = result.clone();
        let values = changed
            .run
            .fields
            .iter_mut()
            .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap();
        let AnalysisFieldValues::HostF64(values) = &mut values.values else {
            unreachable!()
        };
        values[0] += 1.0;
        assert!(changed.validate().is_err());
        assert!(receipt
            .validate_against(physical_identity, &changed)
            .is_err());
        let mut stripped = result.clone();
        stripped.linearization = None;
        assert!(stripped.validate().is_err());
    }
}

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
