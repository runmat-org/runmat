use super::*;
use crate::physics::magnetic::test_support::authored_model;
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};
use runmat_analysis_core::{
    ElectricCurrentDensity, ElectromagneticDomain, PhysicalSignal, SignalRepresentation,
    SignalValue, StaticElectricalEquation,
};
use runmat_meshing_core::ElementOrder;

fn workload(current: f64) -> MagnetostaticWorkload {
    MagnetostaticWorkload {
        model: authored_model(current),
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    }
}

// witness: FEA-EM-QUALIFICATION-001/accuracy @ magnetostatic/tet4|tet10/public_study;
//   boundary: In-process affine both-order magnetic source-bound cold bytes and sign/zero-current
//   inventory/work; no native process, restart, partition or measured RSS evidence.
#[test]
fn magnetic_portable_source_bound_cold_result_preserves_both_orders_and_signs() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = match order {
            ElementOrder::Tet4 => slab_mesh(2),
            ElementOrder::Tet10 => quadratic_slab_mesh(2),
        };
        for current in [-2.0, 2.0, 0.0] {
            let w = workload(current);
            let transported =
                MagnetostaticWorkload::from_canonical_bytes(&w.canonical_bytes().unwrap()).unwrap();
            assert_eq!(transported, w);
            let solved = transported.execute(&mesh).unwrap();
            let bytes = solved.canonical_bytes().unwrap();
            let identity = solved.source_result_identity().unwrap();
            let expected = Digest::sha256(&bytes);
            drop(solved);
            drop(transported);
            // Cold consumer receives retained bytes and authenticated source, not a live solver.
            let cold =
                MagnetostaticSolverResult::from_canonical_bytes_for(&bytes, expected, &w, &mesh)
                    .unwrap();
            assert_eq!(cold.source_result_identity().unwrap(), identity);
            assert_eq!(
                cold.response().samples.len(),
                mesh.topology.volume_elements.len() * 15
            );
            assert!(cold.response().relative_energy_work_error < 1.6e-9);
            if current == 0.0 {
                assert_eq!(cold.response().magnetic_energy_j, 0.0);
            } else {
                assert!(cold.response().magnetic_energy_j > 0.0);
            }
        }
    }
}

#[test]
fn magnetic_cold_import_rejects_lineage_inventory_fields_and_integral_substitution() {
    let w = workload(2.0);
    let mesh = slab_mesh(2);
    let r = w.execute(&mesh).unwrap();
    let bytes = r.canonical_bytes().unwrap();
    let digest = Digest::sha256(&bytes);
    for foreign in [workload(-2.0), workload(1.0)] {
        assert!(MagnetostaticSolverResult::from_canonical_bytes_for(
            &bytes, digest, &foreign, &mesh
        )
        .is_err());
    }
    let mut material = w.clone();
    material.model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .relative_permeability = 2.0;
    assert!(r.validate_for(&material, &mesh).is_err());
    assert!(r.validate_for(&w, &slab_mesh(1)).is_err());
    for mutation in 0..8 {
        let mut value = serde_json::to_value(&r).unwrap();
        let response = &mut value["response"];
        match mutation {
            0 => {
                response["edge_circulations_wb"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            1 => {
                response["samples"].as_array_mut().unwrap().pop();
            }
            2 => {
                response["samples"][0]["volume_weight_m3"] = serde_json::json!(-1.0);
            }
            3 => {
                response["magnetic_energy_j"] = serde_json::json!(123.0);
            }
            4 => {
                response["relative_physical_residual"] = serde_json::json!(1.0);
            }
            5 => {
                response["edge_node_identities"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 1);
            }
            6 => {
                response["samples"][0]["flux_density_t"][0] = serde_json::json!(123.0);
            }
            _ => {
                value["relative_residual_target"] = serde_json::json!(0.0);
            }
        }
        let decoded = serde_json::from_value::<MagnetostaticSolverResult>(value);
        assert!(
            decoded.is_err() || decoded.unwrap().validate_for(&w, &mesh).is_err(),
            "accepted mutation{mutation}"
        );
    }
    let mut value = serde_json::to_value(&r).unwrap();
    value["response"]["edge_circulations_wb"][0] = serde_json::json!(1.0);
    let forged: MagnetostaticSolverResult = serde_json::from_value(value).unwrap();
    let altered = forged.canonical_bytes().unwrap();
    assert!(
        MagnetostaticSolverResult::from_canonical_bytes_for(&altered, digest, &w, &mesh).is_err()
    );
}

#[test]
fn magnetic_workload_rejects_unsupported_policy_and_resource_ceiling_before_projection() {
    let base = workload(2.0);
    let mut electrical = base.clone();
    electrical.model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::Electrostatics,
    });
    assert!(electrical.canonical_bytes().is_err());
    let mut invalid = base.clone();
    invalid.relative_residual_target = f64::NAN;
    assert!(invalid.validate().is_err());
    let mut too_small = base.clone();
    too_small.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 1;
    let error = too_small.execute(&slab_mesh(2)).unwrap_err().to_string();
    assert!(error.contains("resource preflight"), "{error}");
    let mut missing = base.clone();
    missing.model.constraints.pop();
    assert!(missing.execute(&slab_mesh(2)).is_err());
    let mut nonglobal = base;
    let PhysicalSignal::ElectricCurrentDensity(signal) = &mut nonglobal.model.signals[0] else {
        unreachable!()
    };
    signal.frame = None;
    signal.representation = SignalRepresentation::Constant(SignalValue::Vector3(
        [0.0, 0.0, 2.0].map(ElectricCurrentDensity::new),
    ));
    assert!(nonglobal.execute(&slab_mesh(2)).is_err());
}
