//! Independent structural recovery references.
use super::*;

pub(super) fn solved_heterogeneous_tet4_stress_matches_an_independent_constitutive_reference() {
    let model = fixture_model(FixtureId::MultiMaterialAssembly);
    let mesh = crate::tests::multi_region_tetrahedron_solver_mesh();
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("the heterogeneous reference must solve");
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let displacement = result
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let point_stress = result
        .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
        .unwrap();
    let point_values = point_stress.as_host_f64().unwrap();
    assert_eq!(
        point_stress.shape,
        vec![mesh.topology.volume_elements.len() * 4, 6]
    );

    for (element_index, element) in mesh.topology.volume_elements.iter().enumerate() {
        let indices = element.node_ids[..4]
            .iter()
            .map(|node_id| {
                projection
                    .nodes
                    .iter()
                    .position(|node| node.node_id == *node_id)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let coordinates = indices
            .iter()
            .map(|index| projection.nodes[*index].coordinates_m.canonical_value())
            .collect::<Vec<_>>();
        let origin = coordinates[0];
        assert_eq!(coordinates[1], [origin[0] + 1.0, origin[1], origin[2]]);
        assert_eq!(coordinates[2], [origin[0], origin[1] + 1.0, origin[2]]);
        assert_eq!(coordinates[3], [origin[0], origin[1], origin[2] + 1.0]);
        let value = |node: usize, component: usize| displacement[indices[node] * 3 + component];
        let strain = [
            value(1, 0) - value(0, 0),
            value(2, 1) - value(0, 1),
            value(3, 2) - value(0, 2),
            value(3, 1) - value(0, 1) + value(2, 2) - value(0, 2),
            value(3, 0) - value(0, 0) + value(1, 2) - value(0, 2),
            value(2, 0) - value(0, 0) + value(1, 1) - value(0, 1),
        ];
        let assignment = model
            .material_assignments
            .iter()
            .find(|assignment| {
                assignment.selected_region_id()
                    == element.region_id.analysis_selection_id().ok().as_deref()
            })
            .unwrap();
        let material = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.assigned_material_id)
            .unwrap();
        let MaterialElasticityModel::IsotropicLinear(mechanical) =
            material.elasticity.as_ref().unwrap()
        else {
            panic!("reference fixture must use isotropic elasticity");
        };
        let shear = mechanical.youngs_modulus_pa.canonical_value()
            / (2.0 * (1.0 + mechanical.poisson_ratio.canonical_value()));
        let lambda = mechanical.youngs_modulus_pa.canonical_value()
            * mechanical.poisson_ratio.canonical_value()
            / ((1.0 + mechanical.poisson_ratio.canonical_value())
                * (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value()));
        let trace = strain[0] + strain[1] + strain[2];
        let expected = [
            lambda * trace + 2.0 * shear * strain[0],
            lambda * trace + 2.0 * shear * strain[1],
            lambda * trace + 2.0 * shear * strain[2],
            shear * strain[3],
            shear * strain[4],
            shear * strain[5],
        ];
        for point in 0..4 {
            let actual = &point_values[(element_index * 4 + point) * 6..][..6];
            for (actual, expected) in actual.iter().zip(expected) {
                let scale = expected.abs().max(1.0);
                assert!((actual - expected).abs() <= 1.0e-11 * scale);
            }
        }
    }

    let estimate = estimate_structural_recovery_error(
        &mesh,
        "elements",
        point_stress,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();
    assert_eq!(estimate.total_error, 0.0);
}

pub(super) fn solved_heterogeneous_tet10_stress_matches_independent_quadratic_constitutive_reference(
) {
    let mut model = fixture_model(FixtureId::MultiMaterialAssembly);
    let aluminum = model
        .material_assignments
        .iter_mut()
        .find(|assignment| {
            assignment.selected_region_id()
                == Some(runmat_meshing_core::fixtures::fixture_region("mid_aluminum").as_str())
        })
        .unwrap();
    aluminum.assigned_material_id = "mat_aluminum".into();
    aluminum.confidence = runmat_analysis_core::EvidenceConfidence::Verified;
    let mesh = runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh_with_order(
        ElementOrder::Tet10,
    );
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("heterogeneous Tet10 fixture should solve");
    let stress = result
        .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
        .expect("Tet10 solve must retain integration-point stress");
    assert_eq!(
        stress.shape,
        vec![mesh.topology.volume_elements.len() * 4, 6]
    );
    let stress_values = stress.as_host_f64().unwrap();
    let displacement = result
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let mut reference_energy = 0.0;
    let mut verified_elasticities = std::collections::BTreeSet::new();
    let mut peak_stress = 0.0_f64;
    let diagonal = (5.0 + 3.0 * 5.0_f64.sqrt()) / 20.0;
    let off_diagonal = (5.0 - 5.0_f64.sqrt()) / 20.0;
    let barycentric_gradients = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let local_edges = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];
    for (element_index, element) in mesh.topology.volume_elements.iter().enumerate() {
        let indices = element
            .node_ids
            .iter()
            .map(|id| {
                mesh.topology
                    .nodes
                    .iter()
                    .position(|node| node.node_id == *id)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let origin = mesh.topology.nodes[indices[0]]
            .coordinates_m
            .canonical_value();
        for (corner, offset) in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                mesh.topology.nodes[indices[corner]]
                    .coordinates_m
                    .canonical_value(),
                std::array::from_fn(|axis| origin[axis] + offset[axis])
            );
        }
        let assignment = model
            .material_assignments
            .iter()
            .find(|assignment| {
                assignment.selected_region_id()
                    == element.region_id.analysis_selection_id().ok().as_deref()
            })
            .unwrap();
        let material = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.assigned_material_id)
            .unwrap();
        let MaterialElasticityModel::IsotropicLinear(mechanical) =
            material.elasticity.as_ref().unwrap()
        else {
            panic!("reference fixture must use isotropic elasticity");
        };
        verified_elasticities.insert(mechanical.youngs_modulus_pa.canonical_value().to_bits());
        let shear = mechanical.youngs_modulus_pa.canonical_value()
            / (2.0 * (1.0 + mechanical.poisson_ratio.canonical_value()));
        let lambda = mechanical.youngs_modulus_pa.canonical_value()
            * mechanical.poisson_ratio.canonical_value()
            / ((1.0 + mechanical.poisson_ratio.canonical_value())
                * (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value()));
        let nodal: [[f64; 3]; 10] = std::array::from_fn(|local| {
            std::array::from_fn(|axis| displacement[indices[local] * 3 + axis])
        });
        for point in 0..4 {
            let mut barycentric = [off_diagonal; 4];
            barycentric[point] = diagonal;
            let mut gradient = [[0.0; 3]; 3];
            for corner in 0..4 {
                for (component, row) in gradient.iter_mut().enumerate() {
                    for (axis, value) in row.iter_mut().enumerate() {
                        *value += nodal[corner][component]
                            * (4.0 * barycentric[corner] - 1.0)
                            * barycentric_gradients[corner][axis];
                    }
                }
            }
            for (edge_index, [left, right]) in local_edges.into_iter().enumerate() {
                for (component, row) in gradient.iter_mut().enumerate() {
                    for (axis, value) in row.iter_mut().enumerate() {
                        *value += 4.0
                            * nodal[edge_index + 4][component]
                            * (barycentric[left] * barycentric_gradients[right][axis]
                                + barycentric[right] * barycentric_gradients[left][axis]);
                    }
                }
            }
            let strain = [
                gradient[0][0],
                gradient[1][1],
                gradient[2][2],
                gradient[1][2] + gradient[2][1],
                gradient[0][2] + gradient[2][0],
                gradient[0][1] + gradient[1][0],
            ];
            let trace = strain[0] + strain[1] + strain[2];
            let expected = [
                lambda * trace + 2.0 * shear * strain[0],
                lambda * trace + 2.0 * shear * strain[1],
                lambda * trace + 2.0 * shear * strain[2],
                shear * strain[3],
                shear * strain[4],
                shear * strain[5],
            ];
            let actual = &stress_values[(element_index * 4 + point) * 6..][..6];
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).abs() <= 1.0e-10 * expected.abs().max(1.0),
                    "element={element_index} point={point} actual={actual:e} expected={expected:e} difference={:e}", (actual - expected).abs());
                peak_stress = peak_stress.max(actual.abs());
            }
            reference_energy += expected
                .iter()
                .zip(strain)
                .map(|(stress, strain)| stress * strain)
                .sum::<f64>()
                / 48.0;
        }
    }
    let actual_energy = result
        .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    assert_eq!(verified_elasticities.len(), 3);
    assert!(peak_stress > 0.0);
    assert!(reference_energy > 0.0);
    assert!((actual_energy - reference_energy).abs() <= 1.0e-10 * reference_energy.abs().max(1.0));

    let mut reordered = model;
    reordered.materials.reverse();
    reordered.material_assignments.reverse();
    let reordered_result = crate::run_linear_static_with_options(
        &reordered,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("material declaration order must not change the Tet10 solve");
    assert_eq!(reordered_result.fields, result.fields);
}
