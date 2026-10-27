use super::*;
use crate::physics::structural::contact::ContactPenaltyPolicy;
use crate::topology::{AnalysisProjection, BoundarySearchPolicy};
use runmat_analysis_core::{ContactFriction, ContactInterfaceModel};
use runmat_meshing_core::{fixtures, ElementOrder};
fn policy() -> BoundaryProjectionDifferentialPolicy {
    BoundaryProjectionDifferentialPolicy {
        stationarity_tolerance_m: 1e-10,
        minimum_barycentric: 1e-8,
        minimum_metric_ratio: 1e-10,
    }
}
fn near(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() < t, "{a} != {b}");
}
fn law() -> PenaltyContactLaw {
    PenaltyContactLaw::derive(
        &ContactInterfaceModel {
            friction: ContactFriction::Frictionless,
        },
        ContactPenaltyPolicy {
            normal_stiffness_pa_per_m: runmat_types::physical::TractionDisplacementStiffness::new(
                10000.,
            ),
            tangential_stiffness_pa_per_m: None,
        },
    )
    .unwrap()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[test]
fn original_normal_residual_has_closed_force_energy_and_moment_balance_on_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order))
                .unwrap();
        let positions: Vec<_> = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let surface = projection.boundary_surface_snapshot(0, &positions).unwrap();
        let point = surface.evaluate([1. / 3.; 3]).unwrap();
        let n = point.unit_normal;
        let q = std::array::from_fn(|i| point.position_m[i] - 0.01 * n[i]);
        let operator =
            NormalContactPoint::derive(surface, q, [1. / 3.; 3], 0.2, policy(), &law()).unwrap();
        let response = operator.contribution();
        near(response.nominal_pressure_pa.canonical_value(), 100., 1e-10);
        near(response.elastic_energy_j.canonical_value(), 0.1, 1e-12);
        for k in 0..3 {
            near(response.query_internal_force_n[k], -20. * n[k], 1e-10);
            near(
                response.query_internal_force_n[k]
                    + response
                        .primary_internal_force_n()
                        .iter()
                        .map(|f| f[k])
                        .sum::<f64>(),
                0.,
                1e-11,
            );
        }
        let mut moment = cross(q, response.query_internal_force_n);
        for (node, f) in projection.boundary_faces[0]
            .node_indices
            .iter()
            .zip(response.primary_internal_force_n())
        {
            let m = cross(positions[*node], *f);
            for k in 0..3 {
                moment[k] += m[k];
            }
        }
        for m in moment {
            near(m, 0., 1e-11);
        }
        for k in 0..3 {
            let shift = std::array::from_fn(|i| if i == k { 1. } else { 0. });
            let action = operator
                .tangent_action(shift, &vec![shift; point.shape_values().len()])
                .unwrap();
            for x in action
                .query_force_increment_n
                .iter()
                .chain(action.primary_force_increment_n().iter().flatten())
            {
                near(*x, 0., 1e-10);
            }
        }
    }
}
#[test]
fn complete_normal_geometric_tangent_matches_residual_changes_and_is_symmetric() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order))
                .unwrap();
        let positions: Vec<_> = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let face = &projection.boundary_faces[0];
        let surface = projection.boundary_surface_snapshot(0, &positions).unwrap();
        let point = surface.evaluate([1. / 3.; 3]).unwrap();
        let q = std::array::from_fn(|k| point.position_m[k] - 0.03 * point.unit_normal[k]);
        let operator =
            NormalContactPoint::derive(surface, q, [1. / 3.; 3], 0.2, policy(), &law()).unwrap();
        let dq = [0.01, -0.04, 0.03];
        let dx: Vec<_> = face
            .node_indices
            .iter()
            .map(|&i| {
                let p = positions[i];
                [
                    0.02 * p[0] - 0.03 * p[1],
                    0.04 * p[2],
                    -0.01 * p[0] + 0.02 * p[1],
                ]
            })
            .collect();
        let action = operator.tangent_action(dq, &dx).unwrap();
        for step in [1e-5, 1e-6] {
            let solve = |sign: f64| {
                let mut moved = positions.clone();
                for (&i, delta) in face.node_indices.iter().zip(&dx) {
                    for (value, change) in moved[i].iter_mut().zip(delta) {
                        *value += sign * step * change;
                    }
                }
                let query = std::array::from_fn(|k| q[k] + sign * step * dq[k]);
                let s = projection.boundary_surface_snapshot(0, &moved).unwrap();
                let closest = s
                    .closest_point(
                        query,
                        BoundarySearchPolicy {
                            distance_tolerance_m: 1e-10,
                            maximum_patches: 1000,
                            maximum_depth: 20,
                        },
                    )
                    .unwrap();
                *NormalContactPoint::derive(s, query, closest.barycentric, 0.2, policy(), &law())
                    .unwrap()
                    .contribution()
            };
            let plus = solve(1.);
            let minus = solve(-1.);
            for ((actual, a), b) in action
                .query_force_increment_n
                .iter()
                .chain(action.primary_force_increment_n().iter().flatten())
                .zip(
                    plus.query_internal_force_n
                        .iter()
                        .chain(plus.primary_internal_force_n().iter().flatten()),
                )
                .zip(
                    minus
                        .query_internal_force_n
                        .iter()
                        .chain(minus.primary_internal_force_n().iter().flatten()),
                )
            {
                near(*actual, (*a - *b) / (2. * step), 1e-6);
            }
        }
        let dofs = 3 * (1 + face.node_indices.len());
        let mut matrix = vec![vec![0.; dofs]; dofs];
        for column in 0..dofs {
            let mut dq = [0.; 3];
            let mut dx = vec![[0.; 3]; face.node_indices.len()];
            if column < 3 {
                dq[column] = 1.;
            } else {
                dx[(column - 3) / 3][(column - 3) % 3] = 1.;
            }
            let a = operator.tangent_action(dq, &dx).unwrap();
            for (row, v) in a
                .query_force_increment_n
                .iter()
                .chain(a.primary_force_increment_n().iter().flatten())
                .enumerate()
            {
                matrix[row][column] = *v;
            }
        }
        for i in 0..dofs {
            for j in 0..dofs {
                near(matrix[i][j], matrix[j][i], 1e-9);
            }
        }
    }
}

#[test]
fn normal_operator_and_tangent_rotate_objectively_and_use_no_trial_heap() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order))
                .unwrap();
        let positions: Vec<_> = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let surface = projection.boundary_surface_snapshot(0, &positions).unwrap();
        let point = surface.evaluate([1.0 / 3.0; 3]).unwrap();
        let query = std::array::from_fn(|k| point.position_m[k] - 0.03 * point.unit_normal[k]);
        let law = law();
        let (original, retained, peak) = crate::topology::measure_test_allocations(|| {
            NormalContactPoint::derive(surface, query, [1.0 / 3.0; 3], 0.2, policy(), &law).unwrap()
        });
        assert_eq!((retained, peak), (0, 0));
        let rotate = |p: [f64; 3]| [p[2], p[0], p[1]];
        let map = |p: [f64; 3]| {
            let p = rotate(p);
            [p[0] + 2.0, p[1] - 3.0, p[2] + 4.0]
        };
        let moved: Vec<_> = positions.iter().copied().map(map).collect();
        let rotated = NormalContactPoint::derive(
            projection.boundary_surface_snapshot(0, &moved).unwrap(),
            map(query),
            [1.0 / 3.0; 3],
            0.2,
            policy(),
            &law,
        )
        .unwrap();
        near(
            original.contribution().elastic_energy_j.canonical_value(),
            rotated.contribution().elastic_energy_j.canonical_value(),
            1e-12,
        );
        for (a, b) in rotate(original.contribution().query_internal_force_n)
            .into_iter()
            .zip(rotated.contribution().query_internal_force_n)
        {
            near(a, b, 1e-10);
        }
        let dq = [0.02, -0.01, 0.04];
        let dx = [[0.01, 0.02, -0.03]; 6];
        let count = point.shape_values().len();
        let (action, retained, peak) = crate::topology::measure_test_allocations(|| {
            original.tangent_action(dq, &dx[..count]).unwrap()
        });
        assert_eq!((retained, peak), (0, 0));
        let rotated_action = rotated
            .tangent_action(rotate(dq), &dx.map(rotate)[..count])
            .unwrap();
        for (a, b) in rotate(action.query_force_increment_n)
            .into_iter()
            .zip(rotated_action.query_force_increment_n)
        {
            near(a, b, 1e-9);
        }
        for (a, b) in action
            .primary_force_increment_n()
            .iter()
            .zip(rotated_action.primary_force_increment_n())
        {
            for (a, b) in rotate(*a).into_iter().zip(b) {
                near(a, *b, 1e-9);
            }
        }
        let token = crate::progress::FeaCancellationToken::new();
        token.cancel();
        let _guard =
            crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
                cancellation: token,
                ..Default::default()
            });
        let (result, retained, peak) =
            crate::topology::measure_test_allocations(|| original.tangent_action(dq, &dx[..count]));
        assert!(matches!(
            result,
            Err(NormalContactError::Geometry(BoundarySearchError::Cancelled))
        ));
        assert_eq!((retained, peak), (0, 0));
    }
}
