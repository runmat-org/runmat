//! Curved Tri6 pressure integration against an analytic isoparametric surface.

use super::*;
use runmat_meshing_core::CanonicalMeshingContract;

// Four-point Gauss-Legendre in each coordinate of the Duffy square-to-triangle
// map. Its polynomial exactness is independent of the production seven-point
// triangular rule and covers N_i times the curved surface Jacobian vector.
const GAUSS: [(f64, f64); 4] = [
    (0.069_431_844_202_973_71, 0.173_927_422_568_726_9),
    (0.330_009_478_207_571_87, 0.326_072_577_431_273_1),
    (0.669_990_521_792_428_1, 0.326_072_577_431_273_1),
    (0.930_568_155_797_026_2, 0.173_927_422_568_726_9),
];

#[test]
fn curved_tet10_pressure_preserves_analytic_resultant_moment_and_nodal_loads() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "curved_pressure".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Pressure(12.0),
        }],
    );
    let mut mesh = single_tetrahedron_solver_mesh_with_order_for_model(&model, ElementOrder::Tet10);
    let face_node_ids = mesh.topology.boundary_faces[1].node_ids.clone();
    // Face 1 has corners (0,0,0), (1,0,0), (0,0,1). Bend only its
    // 2-3 midside node outward in -y, by delta. The exact Tri6 map is
    // r(xi,eta) = (xi, -4*delta*xi*eta, eta).
    let delta = 0.05;
    let bent_node_id = face_node_ids[4];
    let bent_node = mesh
        .topology
        .nodes
        .iter_mut()
        .find(|node| node.node_id == bent_node_id)
        .unwrap();
    assert_eq!(bent_node.coordinates_m.canonical_value(), [0.5, 0.0, 0.5]);
    {
        let quantity_change = -delta;
        let quantity_index = 1;
        let quantity_slot = &mut bent_node.coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh.validate_canonical().unwrap();

    let mut expected_nodal_load = vec![0.0; mesh.topology.nodes.len() * 3];
    for (u, u_weight) in GAUSS {
        for (v, v_weight) in GAUSS {
            let xi = u;
            let eta = (1.0 - u) * v;
            let l1 = 1.0 - xi - eta;
            let shape = [
                l1 * (2.0 * l1 - 1.0),
                xi * (2.0 * xi - 1.0),
                eta * (2.0 * eta - 1.0),
                4.0 * l1 * xi,
                4.0 * xi * eta,
                4.0 * eta * l1,
            ];
            // dr/dxi x dr/deta, already outward for this -y face.
            let oriented_jacobian = [-4.0 * delta * eta, -1.0, -4.0 * delta * xi];
            let weight = u_weight * v_weight * (1.0 - u) * 12.0;
            for (&node_id, shape_value) in face_node_ids.iter().zip(shape) {
                let node_index = mesh
                    .topology
                    .nodes
                    .iter()
                    .position(|node| node.node_id == node_id)
                    .unwrap();
                for axis in 0..3 {
                    expected_nodal_load[3 * node_index + axis] +=
                        weight * shape_value * oriented_jacobian[axis];
                }
            }
        }
    }
    let expected_force = [-8.0 * delta, -6.0, -8.0 * delta];
    let moment_component = 12.0 * (1.0 / 6.0 + 4.0 * delta * delta / 15.0);
    let expected_moment = [moment_component, 0.0, -moment_component];
    let mut nodal_force = [0.0; 3];
    let mut nodal_moment = [0.0; 3];
    for (node, force) in mesh
        .topology
        .nodes
        .iter()
        .zip(expected_nodal_load.chunks_exact(3))
    {
        for axis in 0..3 {
            nodal_force[axis] += force[axis];
        }
        let [x, y, z] = node.coordinates_m.canonical_value();
        nodal_moment[0] += y * force[2] - z * force[1];
        nodal_moment[1] += z * force[0] - x * force[2];
        nodal_moment[2] += x * force[1] - y * force[0];
    }
    for axis in 0..3 {
        assert!((nodal_force[axis] - expected_force[axis]).abs() < 1.0e-10);
        assert!((nodal_moment[axis] - expected_moment[axis]).abs() < 1.0e-10);
    }

    let summary = crate::assembly::assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("curved Tet10 pressure must assemble");
    assert_vectors_within_tolerance(
        &summary.physical_load,
        &expected_nodal_load,
        ParityTolerance {
            abs: 1.0e-10,
            rel: 1.0e-10,
        },
    );

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..baseline_linear_static_options()
        },
    )
    .expect("curved Tet10 pressure must solve");
    let reaction = host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE);
    let mut reaction_force = [0.0; 3];
    let mut reaction_moment = [0.0; 3];
    for (node, force) in mesh.topology.nodes.iter().zip(reaction.chunks_exact(3)) {
        for axis in 0..3 {
            reaction_force[axis] += force[axis];
        }
        let [x, y, z] = node.coordinates_m.canonical_value();
        reaction_moment[0] += y * force[2] - z * force[1];
        reaction_moment[1] += z * force[0] - x * force[2];
        reaction_moment[2] += x * force[1] - y * force[0];
    }
    for axis in 0..3 {
        assert!((reaction_force[axis] + expected_force[axis]).abs() < 1.0e-8);
        assert!((reaction_moment[axis] + expected_moment[axis]).abs() < 1.0e-8);
    }
    assert!(result.structural_static_balance.as_ref().unwrap().accepted);
}
