use super::*;
use crate::execution::{CanonicalSemanticState, ElectricalNumericalReceiptArtifact};
use crate::physics::electrical::tests::charge_density::charged_model;
use crate::physics::electrical::VACUUM_PERMITTIVITY_F_PER_M;
use runmat_analysis_core::{StateCompleteness, StateId, StateKind, StateOutput, StatePersistence};

fn charged(charge: f64, gauge: f64) -> ElectricalStaticWorkload {
    let mut workload = workload(StaticElectricalEquation::Electrostatics);
    workload.model = charged_model(charge, gauge);
    workload.model.sequence.steps[0].state_outputs = vec![StateOutput {
        state_id: StateId("charged-state".into()),
        kind: StateKind::ElectromagneticStatic,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }];
    workload
}

#[test]
fn charged_results_receipts_and_retained_native_fields_have_cold_source_bound_closure() {
    let epsilon = 4.0 * VACUUM_PERMITTIVITY_F_PER_M.canonical_value();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        for sign in [-1.0, 0.0, 1.0] {
            let input = charged(sign * 2.0 * epsilon, 1000.0);
            let input =
                ElectricalStaticWorkload::from_canonical_bytes(&input.canonical_bytes().unwrap())
                    .unwrap();
            let result = input.execute(&mesh).unwrap();
            let bytes = result.canonical_bytes().unwrap();
            let cold =
                ElectricalStaticSolverResult::from_canonical_bytes_for(&bytes, &input, &mesh)
                    .unwrap();
            assert_eq!(result, cold);
            assert!(
                cold.response().charge_sources.is_some(),
                "zero declared source retains charged identity"
            );
            let receipt = ElectricalNumericalReceiptArtifact::new(&input, &cold, &mesh).unwrap();
            let receipt = ElectricalNumericalReceiptArtifact::from_canonical_bytes(
                &receipt.canonical_bytes().unwrap(),
            )
            .unwrap();
            receipt.validate_against(&input, &cold, &mesh).unwrap();
            assert!(
                (receipt.receipt().net_outward_flux.canonical_value() - sign * 2.0 * epsilon).abs()
                    < epsilon * 1e-9
            );
            let request = cold.native_field_request(&input, &mesh).unwrap();
            let fields = cold.recover_native_fields(&input, &mesh, &request).unwrap();
            let state = CanonicalSemanticState::from_electrical_static_result(&input, &mesh, &cold)
                .unwrap();
            let state =
                CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap())
                    .unwrap();
            assert_eq!(state.payload.electrical_static().unwrap().fields, fields);
            assert_eq!(fields.fields().len(), 4);
            let mut changed = input.clone();
            changed.model.excitations.clear();
            changed.model.excitation_cases.clear();
            assert!(cold.validate_for(&changed, &mesh).is_err());
        }
    }
}

#[test]
fn charged_cold_import_rejects_source_work_charge_and_gauge_increment_substitution() {
    let input = charged(8.0 * VACUUM_PERMITTIVITY_F_PER_M.canonical_value(), 0.0);
    let mesh = mesh(ElementOrder::Tet10, 1);
    let result = input.execute(&mesh).unwrap();
    let baseline = serde_json::to_value(&result).unwrap();
    for (field, replacement) in [
        ("volume_source_work", serde_json::json!(0.0)),
        ("terminal_work", serde_json::json!(1.0)),
        ("charge_sources", serde_json::json!(null)),
        (
            "potential_increment_v",
            serde_json::json!(vec![0.0; mesh.topology.nodes.len()]),
        ),
    ] {
        let mut value = baseline.clone();
        value["response"][field] = replacement;
        let cold = serde_json::from_value::<ElectricalStaticSolverResult>(value);
        assert!(
            cold.is_err() || cold.unwrap().validate_for(&input, &mesh).is_err(),
            "accepted {field}"
        );
    }
    let mut value = baseline;
    value["response"]["charge_sources"]["projection_evidence"][0]["excitation_id"] =
        serde_json::json!("foreign-source");
    let cold: ElectricalStaticSolverResult = serde_json::from_value(value).unwrap();
    assert!(cold.validate_for(&input, &mesh).is_err());
}
