use super::*;

use crate::connectors::{CanonicalConnectorEntry, CanonicalConnectorResponse};

fn assert_relative(actual: f64, expected: f64, relative_tolerance: f64) {
    let scale = actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE);
    assert!(
        (actual - expected).abs() <= relative_tolerance * scale,
        "actual={actual:.16e} expected={expected:.16e} relative_tolerance={relative_tolerance:.3e}"
    );
}

// witness: FEA-STRUCTURAL-CONSTRAINT-001/connectors @ linear_structural/beam/numerical_kernel;
//   coverage: partial;
//   boundary: An explicit axial connector matches the parallel-spring reference.
#[test]
fn explicit_axial_connector_matches_the_parallel_spring_reference() {
    const APPLIED_FORCE_N: f64 = 1_000.0;
    const CONNECTOR_STIFFNESS_N_PER_M: f64 = 4_000.0;
    const BEAM_YOUNGS_MODULUS_PA: f64 = 200.0e9;
    const BEAM_AREA_M2: f64 = 2.0e-4;
    const BEAM_LENGTH_M: f64 = 1.0;

    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "tip_force_x".into(),
            region_id: "node:2".into(),
            kind: FixtureExcitationKind::Force([APPLIED_FORCE_N, 0.0, 0.0]),
        }],
    );
    model.connectors = vec![runmat_analysis_core::AnalysisConnector {
        connector_id: "parallel_axial_spring".into(),
        kind: runmat_analysis_core::ConnectorKind::AxialSpring {
            primary: runmat_analysis_core::RegionTarget::point("node:1"),
            secondary: runmat_analysis_core::RegionTarget::point("node:2"),
            stiffness: runmat_analysis_core::Stiffness::new(CONNECTOR_STIFFNESS_N_PER_M),
        },
    }];

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect("parallel beam and connector reference must solve");
    let expected_displacement = APPLIED_FORCE_N
        / (BEAM_YOUNGS_MODULUS_PA * BEAM_AREA_M2 / BEAM_LENGTH_M + CONNECTOR_STIFFNESS_N_PER_M);
    let expected_connector_force = CONNECTOR_STIFFNESS_N_PER_M * expected_displacement;
    let expected_connector_energy =
        0.5 * CONNECTOR_STIFFNESS_N_PER_M * expected_displacement * expected_displacement;
    let tip_displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT)[3];
    assert_relative(tip_displacement, expected_displacement, 1.0e-9);

    let connector_result = result
        .structural_connector_result
        .as_ref()
        .expect("structural static solve must publish connector evidence");
    let CanonicalConnectorResponse::AxialSpring {
        extension_meters,
        axial_force_newtons,
        resultant_force_newtons,
        stored_energy_joules,
        ..
    } = &connector_result.responses[0]
    else {
        panic!("expected the authored axial connector response");
    };
    assert_relative(*extension_meters, expected_displacement, 1.0e-9);
    assert_relative(*axial_force_newtons, expected_connector_force, 1.0e-9);
    assert_relative(resultant_force_newtons[0], expected_connector_force, 1.0e-9);
    assert_eq!(resultant_force_newtons[1..], [0.0, 0.0]);
    assert_relative(*stored_energy_joules, expected_connector_energy, 1.0e-9);
}

#[test]
fn mixed_solid_beam_ground_spring_recovers_from_the_shared_tip_dof() {
    const SPRING_STIFFNESS_N_PER_M: f64 = 2_000.0;

    let mut mesh = single_tetrahedron_solver_mesh();
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
    mesh.topology.nodes[0]
        .provenance
        .push(PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        ));
    mesh.topology.nodes[0].provenance.sort();
    add_face_region(&mut mesh, 1, "solid_load");
    reseal_solver_mesh(&mut mesh);
    let model = mixed_solid_beam_model(attached_coordinates);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            solver_mesh_artifact_path: Some("mixed-connector-reference.rmmesh".into()),
            ..baseline_linear_static_options()
        },
    )
    .expect("mixed solid-beam connector reference must solve");

    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    let tip = [displacement[12], displacement[13], displacement[14]];
    let expected_force = tip.map(|value| -SPRING_STIFFNESS_N_PER_M * value);
    let expected_energy =
        0.5 * SPRING_STIFFNESS_N_PER_M * tip.iter().map(|value| value * value).sum::<f64>();
    let connector_result = result
        .structural_connector_result
        .as_ref()
        .expect("mixed solve must publish connector evidence");
    let CanonicalConnectorResponse::IsotropicGroundSpring {
        displacement_meters,
        resultant_force_newtons,
        stored_energy_joules,
        ..
    } = &connector_result.responses[0]
    else {
        panic!("expected the mixed ground-spring response");
    };
    for component in 0..3 {
        assert_relative(displacement_meters[component], tip[component], 1.0e-12);
        assert_relative(
            resultant_force_newtons[component],
            expected_force[component],
            1.0e-12,
        );
    }
    assert_relative(*stored_energy_joules, expected_energy, 1.0e-12);
}

#[test]
fn solid_face_foundation_matches_an_independent_quadrature_sum() {
    const FOUNDATION_MODULUS_N_PER_M3: f64 = 5.0e8;

    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.connectors = vec![runmat_analysis_core::AnalysisConnector {
        connector_id: "tip_foundation".into(),
        kind: runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
            target: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("tip"),
            ),
            modulus: runmat_analysis_core::FoundationModulus::new(FOUNDATION_MODULUS_N_PER_M3),
        },
    }];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            solver_mesh_artifact_path: Some("solid-foundation-reference.rmmesh".into()),
            ..baseline_linear_static_options()
        },
    )
    .expect("solid face foundation reference must solve");
    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    let connector_result = result
        .structural_connector_result
        .as_ref()
        .expect("solid solve must publish connector evidence");
    let CanonicalConnectorEntry::SurfaceNormalFoundation { quadrature, .. } =
        &connector_result.plan.entries[0]
    else {
        panic!("expected the authored face-foundation plan");
    };
    let mut expected_force = [0.0; 3];
    let mut expected_energy = 0.0;
    for point in quadrature {
        let mut point_displacement = [0.0; 3];
        for (node, shape) in point.nodes.iter().zip(&point.shape_values) {
            for component in 0..3 {
                point_displacement[component] +=
                    shape * displacement[node.node_index * 3 + component];
            }
        }
        let normal_displacement = point
            .unit_normal
            .iter()
            .zip(point_displacement)
            .map(|(normal, value)| normal * value)
            .sum::<f64>();
        let force_scale = -FOUNDATION_MODULUS_N_PER_M3 * normal_displacement * point.area_weight_m2;
        for (component, force) in expected_force.iter_mut().enumerate() {
            *force += force_scale * point.unit_normal[component];
        }
        expected_energy += 0.5
            * FOUNDATION_MODULUS_N_PER_M3
            * normal_displacement
            * normal_displacement
            * point.area_weight_m2;
    }
    let CanonicalConnectorResponse::SurfaceNormalFoundation {
        resultant_force_newtons,
        stored_energy_joules,
        ..
    } = &connector_result.responses[0]
    else {
        panic!("expected the authored face-foundation response");
    };
    for component in 0..3 {
        assert_relative(
            resultant_force_newtons[component],
            expected_force[component],
            1.0e-12,
        );
    }
    assert_relative(*stored_energy_joules, expected_energy, 1.0e-12);
    assert!(*stored_energy_joules > 0.0);
}
