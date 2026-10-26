//! Hydrostatic pressure patch on connected linear and quadratic tetrahedra.

use runmat_meshing_core::fixtures::{
    connected_quadratic_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh,
};

use super::{
    assert_vectors_within_tolerance, baseline_linear_static_options, host_field,
    replace_fixture_excitations, solid_static_patch::traction_patch_model, ComputeBackend,
    FixtureExcitation, FixtureExcitationKind, LinearStaticSolveOptions, ParityTolerance,
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
};

#[test]
fn connected_tet4_and_tet10_pressure_patch_recovers_hydrostatic_affine_solution() {
    // E=100 Pa, nu=0.25 => 3K=200 Pa. u=0.06*x therefore has
    // epsilon=0.06*I, sigma=12*I and strain energy 0.18 J over V=1/6 m^3.
    // Uniform outward pressure +12 Pa on all four outer faces realizes
    // t=sigma*n. The existing six zero DOFs only remove rigid motion.
    for mesh in [
        connected_tetrahedron_solver_mesh(),
        connected_quadratic_tetrahedron_solver_mesh(),
    ] {
        let mut model = traction_patch_model();
        replace_fixture_excitations(
            &mut model,
            (0..4)
                .map(|face| FixtureExcitation {
                    id: format!("pressure_{face}"),
                    region_id: format!("face:{face}"),
                    kind: FixtureExcitationKind::Pressure(12.0),
                })
                .collect(),
        );
        let exact = mesh
            .topology
            .nodes
            .iter()
            .flat_map(|node| {
                node.coordinates_m
                    .canonical_value()
                    .map(|coordinate| 0.06 * coordinate)
            })
            .collect::<Vec<_>>();
        let summary = crate::assembly::assemble_structural_system(
            &model,
            &super::static_excitations(&model),
            Some(mesh.clone()),
            crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .expect("connected pressure patch must assemble");
        let mut expected_load = vec![0.0; exact.len()];
        for face in &mesh.topology.boundary_faces {
            let corners = face
                .node_ids
                .iter()
                .take(3)
                .map(|node_id| {
                    mesh.topology
                        .nodes
                        .iter()
                        .find(|node| node.node_id == *node_id)
                        .unwrap()
                        .coordinates_m
                        .canonical_value()
                })
                .collect::<Vec<_>>();
            let a = std::array::from_fn::<_, 3, _>(|i| corners[1][i] - corners[0][i]);
            let b = std::array::from_fn::<_, 3, _>(|i| corners[2][i] - corners[0][i]);
            let mut area_vector = [
                0.5 * (a[1] * b[2] - a[2] * b[1]),
                0.5 * (a[2] * b[0] - a[0] * b[2]),
                0.5 * (a[0] * b[1] - a[1] * b[0]),
            ];
            let face_centroid = std::array::from_fn::<_, 3, _>(|i| {
                (corners[0][i] + corners[1][i] + corners[2][i]) / 3.0
            });
            // The fixture occupies the unit tetrahedron, whose centroid is
            // (1/4,1/4,1/4). Face node winding is not an outward-normal contract.
            if (0..3)
                .map(|i| area_vector[i] * (face_centroid[i] - 0.25))
                .sum::<f64>()
                < 0.0
            {
                area_vector = area_vector.map(|component| -component);
            }
            // Straight-sided triangular shape functions integrate to A/3
            // at each Tet4 corner, or A/3 at each Tet10 midside node.
            let loaded_nodes = if face.node_ids.len() == 3 {
                &face.node_ids[..3]
            } else {
                &face.node_ids[3..6]
            };
            for node_id in loaded_nodes {
                let node_index = mesh
                    .topology
                    .nodes
                    .iter()
                    .position(|node| node.node_id == *node_id)
                    .unwrap();
                for axis in 0..3 {
                    expected_load[node_index * 3 + axis] += 12.0 * area_vector[axis] / 3.0;
                }
            }
        }
        assert_vectors_within_tolerance(
            &summary.physical_load,
            &expected_load,
            ParityTolerance {
                abs: 1.0e-10,
                rel: 1.0e-10,
            },
        );
        let internal = crate::algebra::apply_k(&summary.operator, &exact);
        assert_vectors_within_tolerance(
            &internal,
            &summary.physical_load,
            ParityTolerance {
                abs: 1.0e-9,
                rel: 1.0e-10,
            },
        );

        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh),
                ..baseline_linear_static_options()
            },
        )
        .expect("connected pressure patch must solve");
        assert_vectors_within_tolerance(
            host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
            &exact,
            ParityTolerance {
                abs: 1.0e-8,
                rel: 1.0e-8,
            },
        );
        for strain in host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN).chunks_exact(6) {
            assert_vectors_within_tolerance(
                strain,
                &[0.06, 0.06, 0.06, 0.0, 0.0, 0.0],
                ParityTolerance {
                    abs: 1.0e-8,
                    rel: 1.0e-8,
                },
            );
        }
        for stress in host_field(&result, FEA_FIELD_STRUCTURAL_STRESS).chunks_exact(6) {
            assert_vectors_within_tolerance(
                stress,
                &[12.0, 12.0, 12.0, 0.0, 0.0, 0.0],
                ParityTolerance {
                    abs: 1.0e-7,
                    rel: 1.0e-8,
                },
            );
        }
        assert!(host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE)
            .iter()
            .all(|force| force.abs() < 1.0e-8));
        let energy = host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0];
        assert!((energy - 0.18).abs() < 1.0e-8, "energy={energy}");
        assert!(result.structural_static_balance.as_ref().unwrap().accepted);
    }
}
