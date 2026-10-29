use super::*;
use crate::physics::electrical::tests::{mesh, model, prescribe};
use runmat_analysis_core::{RegionTarget, RestartPolicy};
use runmat_meshing_core::ElementOrder;

mod charge_density;
mod publication;

fn workload(equation: StaticElectricalEquation) -> ElectricalStaticWorkload {
    let mut model = model(equation);
    prescribe(
        &mut model,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    prescribe(
        &mut model,
        "right".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        2.0,
    );
    ElectricalStaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-11,
        element_temperature_k: None,
        operating_time_s: None,
    }
}

// witness: FEA-EM-QUALIFICATION-001/accuracy @ electrostatic/tet4|tet10/public_study,
//   electrical_conduction/tet4|tet10/public_study;
//   boundary: In-process both scalar equations and orders retain original slab fields and canonical
//   source-bound cold identity; changed controls/equation/mesh reject. No producer death, remote or
//   durable execution.
#[test]
fn equation_order_matrix_retains_exact_fields_and_cold_source_bound_identity() {
    let mut identities = Vec::new();
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let workload = workload(equation);
            let mesh = mesh(order, 2);
            let result = workload.execute(&mesh).unwrap();
            let encoded = result.canonical_bytes().unwrap();
            let identity = result.source_result_identity().unwrap();
            let reopened_workload = ElectricalStaticWorkload::from_canonical_bytes(
                &workload.canonical_bytes().unwrap(),
            )
            .unwrap();
            let cold = ElectricalStaticSolverResult::from_canonical_bytes_for(
                &encoded,
                &reopened_workload,
                &mesh,
            )
            .unwrap();
            assert_eq!(cold, result);
            assert_eq!(cold.canonical_bytes().unwrap(), encoded);
            assert_eq!(cold.source_result_identity().unwrap(), identity);
            assert_eq!(cold.source_workload(), workload.identity().unwrap());
            assert_eq!(cold.producer(), workload.producer_identity().unwrap());
            assert_eq!(cold.response().equation, equation);
            assert!(!identities.contains(&identity));
            identities.push(identity);
            let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
            for (potential, node) in cold
                .response()
                .potential_v
                .canonical_values()
                .iter()
                .zip(&projection.nodes)
            {
                assert!((potential - 2.0 * node.coordinates_m.canonical_value()[0]).abs() < 1e-8);
            }
            for field in cold
                .response()
                .element_electric_field_v_per_m
                .canonical_values()
            {
                assert!((field[0] + 2.0).abs() < 1e-8);
                assert!(field[1].abs() < 1e-8 && field[2].abs() < 1e-8);
            }
            let mut changed = workload.clone();
            changed.relative_residual_target *= 2.0;
            assert!(cold.validate_for(&changed, &mesh).is_err());
            changed = workload.clone();
            changed.model.electromagnetic = Some(ElectromagneticDomain::Static {
                equation: match equation {
                    StaticElectricalEquation::Electrostatics => {
                        StaticElectricalEquation::SteadyConduction
                    }
                    StaticElectricalEquation::SteadyConduction => {
                        StaticElectricalEquation::Electrostatics
                    }
                },
            });
            assert!(cold.validate_for(&changed, &mesh).is_err());
            assert!(cold
                .validate_for(&workload, &super::tests::mesh(order, 1))
                .is_err());
        }
    }
}

#[test]
fn decoding_rejects_malformed_fields_equation_formulation_and_terminal_accounting() {
    let workload = workload(StaticElectricalEquation::Electrostatics);
    let mesh = mesh(ElementOrder::Tet4, 1);
    let result = workload.execute(&mesh).unwrap();
    let baseline = serde_json::to_value(&result).unwrap();
    let reject = |value: serde_json::Value| {
        assert!(serde_json::from_value::<ElectricalStaticSolverResult>(value).is_err());
    };
    let mut value = baseline.clone();
    value["response"]["potential_v"] = serde_json::json!([]);
    reject(value);
    let mut value = baseline.clone();
    value["response"]["ordered_node_ids"][1] = value["response"]["ordered_node_ids"][0].clone();
    reject(value);
    let mut value = baseline.clone();
    value["response"]["equation"] = serde_json::json!(StaticElectricalEquation::SteadyConduction);
    reject(value);
    let mut value = baseline.clone();
    value["response"]["terminals"][0]["outward_flux"] = serde_json::json!(99.0);
    reject(value);
    let mut value = baseline.clone();
    value["response"]["relative_flux_imbalance"] = serde_json::json!(0.1);
    reject(value);
    let mut value = baseline.clone();
    value["response"]["constitutive"]["electrostatics"]["total_energy_j"] = serde_json::json!(-1.0);
    reject(value);
    let mut value = baseline.clone();
    value["response"]["constitutive"]["electrostatics"]["invented"] = serde_json::json!(1);
    reject(value);
    let mut value = baseline.clone();
    value["relative_residual_target"] = serde_json::json!(0);
    reject(value);
    let mut value = baseline;
    value["element_order"] = serde_json::json!(ElementOrder::Tet10);
    reject(value);
}

#[test]
fn source_bound_import_rejects_lineage_and_terminal_substitution_without_rerunning_the_solve() {
    let workload = workload(StaticElectricalEquation::SteadyConduction);
    let mesh = mesh(ElementOrder::Tet4, 1);
    let result = workload.execute(&mesh).unwrap();
    let mut value = serde_json::to_value(&result).unwrap();
    value["source_workload"] = serde_json::to_value(
        AnalysisWorkloadIdentity::derive("different-electrical-source", &Digest::sha256([]))
            .unwrap(),
    )
    .unwrap();
    let decoded: ElectricalStaticSolverResult = serde_json::from_value(value).unwrap();
    assert!(decoded.validate_for(&workload, &mesh).is_err());
    let mut value = serde_json::to_value(&result).unwrap();
    value["response"]["terminals"][0]["constraint_id"] = serde_json::json!("unowned-terminal");
    let decoded: ElectricalStaticSolverResult = serde_json::from_value(value).unwrap();
    assert!(decoded.validate_for(&workload, &mesh).is_err());
    let mut value = serde_json::to_value(&result).unwrap();
    let node = result.response().terminals[1].ordered_node_ids[0];
    let index = result
        .response()
        .ordered_node_ids
        .iter()
        .position(|id| *id == node)
        .unwrap();
    value["response"]["potential_v"][index] = serde_json::json!(3.0);
    let decoded: ElectricalStaticSolverResult = serde_json::from_value(value).unwrap();
    assert!(decoded.validate_for(&workload, &mesh).is_err());
    let mut value = serde_json::to_value(&result).unwrap();
    value["response"]["constitutive"]["steady_conduction"]["total_joule_power_w"] =
        serde_json::json!(24.0);
    // Retained source/terminal work now rejects the forged total at the bare
    // decoding boundary, before source-bound integral validation.
    assert!(serde_json::from_value::<ElectricalStaticSolverResult>(value).is_err());
    let mut value = serde_json::to_value(&result).unwrap();
    let nodes = &result.response().terminals[0].ordered_node_ids;
    for id in nodes {
        let index = result
            .response()
            .ordered_node_ids
            .iter()
            .position(|node| node == id)
            .unwrap();
        value["response"]["nodal_outward_flux"]["value_si"][index] = serde_json::json!(f64::MAX);
    }
    assert!(serde_json::from_value::<ElectricalStaticSolverResult>(value).is_err());
}

#[test]
fn unsupported_lifecycle_and_production_never_become_portable_numerical_intent() {
    let baseline = workload(StaticElectricalEquation::Electrostatics);
    for target in [0.0, 1.0, f64::NAN, f64::INFINITY, -1.0] {
        let mut value = baseline.clone();
        value.relative_residual_target = target;
        assert!(value.validate().is_err());
        assert!(value.canonical_bytes().is_err());
    }
    let mut value = baseline.clone();
    value.model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "not-admitted".into(),
        required_state_digest: "not-admitted".into(),
    };
    assert!(value.validate().is_err());
    let mut value = baseline.clone();
    value.model.sequence.steps[0].state_outputs[0].kind = runmat_analysis_core::StateKind::Thermal;
    assert!(value.validate().is_err());
    let mut value = baseline.clone();
    value
        .model
        .sequence
        .steps
        .push(runmat_analysis_core::ProcedureStep::electromagnetic_static(
            "ignored",
        ));
    assert!(value.validate().is_err());
}

#[test]
fn residual_scale_and_large_zero_field_gauge_do_not_change_acceptance_authority() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = mesh(order, 2);
            for (left, right) in [(0.0, 1e-6), (0.0, 1e6), (1e12, 1e12)] {
                let mut source = workload(equation);
                source.model.constraints.clear();
                prescribe(
                    &mut source.model,
                    "left".into(),
                    RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_left",
                    )),
                    left,
                );
                prescribe(
                    &mut source.model,
                    "right".into(),
                    RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_right",
                    )),
                    right,
                );
                let result = source.execute(&mesh).unwrap();
                assert!(
                    result.response().linear_relative_residual <= source.relative_residual_target
                );
                let bytes = result.canonical_bytes().unwrap();
                assert_eq!(
                    ElectricalStaticSolverResult::from_canonical_bytes_for(&bytes, &source, &mesh)
                        .unwrap(),
                    result
                );
                if left == right {
                    assert_eq!(result.response().linear_absolute_residual, 0.0);
                    assert_eq!(result.response().linear_relative_residual, 0.0);
                    let id = result.response().terminals[0].ordered_node_ids[0];
                    let index = result
                        .response()
                        .ordered_node_ids
                        .iter()
                        .position(|node| *node == id)
                        .unwrap();
                    let mut changed = serde_json::to_value(&result).unwrap();
                    changed["response"]["potential_v"][index] = serde_json::json!(left + 1.0);
                    let changed: ElectricalStaticSolverResult =
                        serde_json::from_value(changed).unwrap();
                    assert!(changed.validate_for(&source, &mesh).is_err());
                }
            }
        }
    }
}
