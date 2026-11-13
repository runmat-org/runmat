use super::*;
use crate::execution::{
    CanonicalSemanticState, FieldRecoveryRequest, SemanticStatePublicationProfileSource,
};
use crate::results::field::{
    canonical_field_partitions, reconstruct_recovered_field, RecoveredFieldArtifact,
    RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};
use runmat_analysis_core::{StateCompleteness, StateId, StateKind, StateOutput, StatePersistence};

fn declared(equation: StaticElectricalEquation) -> ElectricalStaticWorkload {
    let mut workload = workload(equation);
    workload.model.sequence.steps[0].state_outputs = vec![StateOutput {
        state_id: StateId("voltage_state".into()),
        kind: StateKind::ElectromagneticStatic,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }];
    workload
}

#[test]
fn electrical_native_fields_state_and_borrowed_profiles_roundtrip_all_equations_orders() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let workload = declared(equation);
            let mesh = mesh(order, 2);
            let result = workload.execute(&mesh).unwrap();
            let result = ElectricalStaticSolverResult::from_canonical_bytes_for(
                &result.canonical_bytes().unwrap(),
                &workload,
                &mesh,
            )
            .unwrap();
            let request = result.native_field_request(&workload, &mesh).unwrap();
            let request =
                FieldRecoveryRequest::from_canonical_bytes(&request.canonical_bytes().unwrap())
                    .unwrap();
            let artifact = result
                .recover_native_fields(&workload, &mesh, &request)
                .unwrap();
            let cold =
                RecoveredFieldArtifact::from_canonical_bytes(&artifact.canonical_bytes().unwrap())
                    .unwrap();
            assert_eq!(artifact, cold);
            assert_eq!(cold.fields().len(), 4);
            assert_eq!(cold.entity_counts.volume_integration_points, 0);
            for field in cold.fields() {
                assert_eq!(
                    field.definition().numeric_representation(),
                    RecoveredFieldNumericRepresentation::Real
                );
                assert_eq!(field.sample(), RecoveredFieldSample::Final);
                let restored = reconstruct_recovered_field(
                    field.identity(),
                    field.definition().clone(),
                    field.sample(),
                    canonical_field_partitions(field).unwrap(),
                    field.storage().shape[0],
                )
                .unwrap();
                assert_eq!(&restored, field);
            }
            let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
            for (potential, node) in cold.fields()[0]
                .storage()
                .as_host_f64()
                .unwrap()
                .iter()
                .zip(&projection.nodes)
            {
                assert!((potential - 2.0 * node.coordinates_m.canonical_value()[0]).abs() < 1e-8);
            }
            for field in cold.fields()[1]
                .storage()
                .as_host_f64()
                .unwrap()
                .chunks_exact(3)
            {
                assert!(
                    (field[0] + 2.0).abs() < 1e-8 && field[1].abs() < 1e-8 && field[2].abs() < 1e-8
                );
            }
            let coefficient = match equation {
                StaticElectricalEquation::Electrostatics => {
                    4.0 * crate::physics::electrical::VACUUM_PERMITTIVITY_F_PER_M.canonical_value()
                }
                StaticElectricalEquation::SteadyConduction => 3.0,
            };
            for field in cold.fields()[2]
                .storage()
                .as_host_f64()
                .unwrap()
                .chunks_exact(3)
            {
                assert!((field[0] + 2.0 * coefficient).abs() < 1e-8 * coefficient);
                assert!(field[1].abs() < 1e-8 * coefficient && field[2].abs() < 1e-8 * coefficient);
            }
            let density = match equation {
                StaticElectricalEquation::Electrostatics => 2.0 * coefficient,
                StaticElectricalEquation::SteadyConduction => 4.0 * coefficient,
            };
            for value in cold.fields()[3].storage().as_host_f64().unwrap() {
                assert!((value - density).abs() < 1e-8 * density);
            }
            let state =
                CanonicalSemanticState::from_electrical_static_result(&workload, &mesh, &result)
                    .unwrap();
            let cold_state =
                CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                    .unwrap();
            assert_eq!(cold_state, state);
            let payload = cold_state.payload.electrical_static().unwrap();
            assert_eq!((payload.equation, payload.element_order), (equation, order));
            assert_eq!(payload.fields, artifact);
            let requests = result.state_publication_requests().unwrap();
            assert_eq!(requests.len(), 1);
            let borrowed = requests[0]
                .encoding_profile(
                    SemanticStatePublicationProfileSource::StaticElectricalNativeFields(&result),
                )
                .unwrap();
            let actual = runmat_canonical_codec::profile_serialized(
                &state,
                CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
            )
            .unwrap();
            assert_eq!(borrowed, actual);
            let source_digest = Digest::sha256(result.canonical_bytes().unwrap());
            let profiles = result.state_publication_profiles(source_digest).unwrap();
            let cold_profiles = crate::execution::StatePublicationProfiles::from_canonical_bytes(
                &profiles.canonical_bytes().unwrap(),
            )
            .unwrap();
            assert_eq!(profiles, cold_profiles);
            let bounded = workload
                .state_publication_profiles_encoding_profile(Digest::from_bytes(
                    *mesh.canonical_digest.bytes(),
                ))
                .unwrap();
            assert!(
                profiles.canonical_bytes().unwrap().len() as u64
                    <= bounded.encoded_bytes_upper_bound
            );
            assert!(result
                .state_publication_profiles(Digest::sha256("wrong-source"))
                .is_err());
        }
    }
}

#[test]
fn electrical_receipts_bind_actual_closure_and_reject_foreign_evidence() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let workload = declared(equation);
            let mesh = mesh(order, 1);
            let result = workload.execute(&mesh).unwrap();
            let receipt =
                ElectricalNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
            let bytes = receipt.canonical_bytes().unwrap();
            assert!(bytes.len() as u64 <= ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES);
            let cold = ElectricalNumericalReceiptArtifact::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(cold, receipt);
            cold.validate_against(&workload, &result, &mesh).unwrap();
            assert_eq!(
                cold.receipt().linear_relative_residual,
                result.response().linear_relative_residual
            );
            assert_eq!(
                cold.receipt().relative_flux_imbalance,
                result.response().relative_flux_imbalance
            );
            assert_eq!(
                cold.receipt().relative_energy_identity_error,
                result.response().relative_energy_identity_error
            );
            let mut foreign = workload.clone();
            foreign.relative_residual_target *= 2.0;
            assert!(cold.validate_against(&foreign, &result, &mesh).is_err());
            for key in [
                "linear_relative_residual",
                "relative_flux_imbalance",
                "relative_energy_identity_error",
                "total_constitutive_integral",
                "nodes",
            ] {
                let mut altered = serde_json::to_value(&cold).unwrap();
                altered["receipt"][key] = serde_json::json!(123);
                assert!(
                    serde_json::from_value::<ElectricalNumericalReceiptArtifact>(altered).is_err()
                );
            }
        }
    }
}

#[test]
fn electrical_publication_rejects_forged_declaration_and_foreign_field_request() {
    let workload = declared(StaticElectricalEquation::SteadyConduction);
    let mesh = mesh(ElementOrder::Tet10, 1);
    let result = workload.execute(&mesh).unwrap();
    let original = result.state_publication_requests().unwrap().remove(0);
    let mut altered = original.clone();
    altered.declaration.state_id = StateId("forged".into());
    assert!(result.validate_publication_request(&altered).is_err());
    let mut wire = serde_json::to_value(&result).unwrap();
    wire["publication_declaration"]["state_id"] = serde_json::json!("forged");
    let forged = serde_json::from_value::<ElectricalStaticSolverResult>(wire).unwrap();
    assert!(forged.validate_for(&workload, &mesh).is_err());
    let mut request = result.native_field_request_from_result().unwrap();
    request.plan.requirements[0].quantity = runmat_analysis_core::ObservationQuantity::Temperature;
    assert!(result.recover_native_fields_from_result(&request).is_err());
    // Keep the producer structurally valid: static authored steps require a complete
    // typed state. A foreign declaration must not authorize publishing this result
    // under the original request, even though its fields and mesh are compatible.
    let mut foreign_declaration = workload.clone();
    foreign_declaration.model.sequence.steps[0].state_outputs[0].state_id =
        StateId("foreign_voltage_state".into());
    let numerical = foreign_declaration.execute(&mesh).unwrap();
    let foreign_request = numerical.state_publication_requests().unwrap().remove(0);
    assert_ne!(foreign_request.declaration, original.declaration);
    numerical
        .validate_publication_request(&foreign_request)
        .unwrap();
    assert!(
        CanonicalSemanticState::from_electrical_static_publication(&original, &numerical).is_err()
    );
    let state =
        CanonicalSemanticState::from_electrical_static_result(&workload, &mesh, &result).unwrap();
    let mut payload = state.payload.electrical_static().unwrap().clone();
    payload.equation = StaticElectricalEquation::Electrostatics;
    assert!(payload
        .validate_for(state.topology_digest, state.source_result, state.producer)
        .is_err());
}
