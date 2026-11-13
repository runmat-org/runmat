use super::*;
use crate::topology::quadrature::{
    tensor_tetrahedron_rule, visit_volume_rule, GAUSS_LEGENDRE_EIGHT_UNIT,
};
use runmat_meshing_core::{fixtures, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

const GRAD: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 2e-12, "{actual} != {expected}");
}

fn reference_coordinates(order: ElementOrder, curve: f64) -> Vec<[f64; 3]> {
    let mut nodes = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if order == ElementOrder::Tet10 {
        for [a, b] in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            nodes.push(std::array::from_fn(|axis| {
                (nodes[a][axis] + nodes[b][axis]) * 0.5
            }));
        }
    }
    for [r, s, t] in &mut nodes {
        *t += curve * *r * *s;
    }
    nodes
}

#[test]
fn both_geometry_orders_match_exact_affine_barycentric_integrals() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
        // Closed tetrahedral moments: ∫λᵢ=V/4, ∫λᵢ²=V/10,
        // ∫λᵢλⱼ=V/20. No numerical assembly is the reference.
        for (element_index, element) in projection.elements.iter().enumerate() {
            let stored = &space.elements[element_index];
            // Fixture coordinates may not be the unit reference tetrahedron.
            let coordinates = stored
                .corners
                .map(|index| projection.nodes[index].coordinates_m.canonical_value());
            let matrix = [
                super::super::element::subtract(coordinates[1], coordinates[0]),
                super::super::element::subtract(coordinates[2], coordinates[0]),
                super::super::element::subtract(coordinates[3], coordinates[0]),
            ];
            let inverse = super::super::element::invert(matrix).unwrap();
            let g = GRAD.map(|gradient| {
                std::array::from_fn::<_, 3, _>(|axis| {
                    (0..3).map(|r| inverse[axis][r] * gradient[r]).sum::<f64>()
                })
            });
            let volume = dot(matrix[0], cross(matrix[1], matrix[2])) / 6.0;
            close(volume, element.signed_volume_m3);
            let moments = space.element_moments(&projection, element_index).unwrap();
            let lambda = |i, j| volume / if i == j { 10.0 } else { 20.0 };
            for (i, [a, b]) in EDGES.into_iter().enumerate() {
                let sign_i = stored.signs[i];
                for (j, [c, d]) in EDGES.into_iter().enumerate() {
                    close(
                        moments.mass[i][j],
                        sign_i
                            * stored.signs[j]
                            * (lambda(a, c) * dot(g[b], g[d])
                                - lambda(a, d) * dot(g[b], g[c])
                                - lambda(b, c) * dot(g[a], g[d])
                                + lambda(b, d) * dot(g[a], g[c])),
                    );
                    close(
                        moments.curl_curl[i][j],
                        4.0 * volume
                            * sign_i
                            * stored.signs[j]
                            * dot(cross(g[a], g[b]), cross(g[c], g[d])),
                    );
                }
                for axis in 0..3 {
                    close(
                        moments.constant_vector_load[i][axis],
                        sign_i * volume * (g[b][axis] - g[a][axis]) / 4.0,
                    );
                }
                for (j, gradient) in g.iter().enumerate() {
                    close(
                        moments.gradient_coupling[i][j],
                        sign_i * volume * (dot(g[b], *gradient) - dot(g[a], *gradient)) / 4.0,
                    );
                }
            }
        }
    }
}

#[test]
fn curved_tet10_obeys_exact_covariant_and_curl_piola_maps() {
    let alpha = 0.6;
    let coordinates = reference_coordinates(ElementOrder::Tet10, alpha);
    for barycentric in [[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3]] {
        visit_volume_rule(
            ElementOrder::Tet10,
            &coordinates,
            [(barycentric, 1.0 / 6.0)],
            |point| {
                let actual = basis(point, [1.0; 6]);
                let [_, r, s, t] = barycentric;
                close(actual.position_m[0], r);
                close(actual.position_m[1], s);
                close(actual.position_m[2], t + alpha * r * s);
                close(actual.volume_weight_m3, 1.0 / 6.0);
                for (edge, [a, b]) in EDGES.into_iter().enumerate() {
                    let reference: [f64; 3] = std::array::from_fn(|axis| {
                        barycentric[a] * GRAD[b][axis] - barycentric[b] * GRAD[a][axis]
                    });
                    let expected_value = [
                        reference[0] - alpha * s * reference[2],
                        reference[1] - alpha * r * reference[2],
                        reference[2],
                    ];
                    let curl = cross(GRAD[a], GRAD[b]).map(|v| 2.0 * v);
                    let expected_curl = [
                        curl[0],
                        curl[1],
                        curl[2] + alpha * s * curl[0] + alpha * r * curl[1],
                    ];
                    for axis in 0..3 {
                        close(actual.values[edge][axis], expected_value[axis]);
                        close(actual.curls[edge][axis], expected_curl[axis]);
                    }
                }
            },
        )
        .unwrap();
    }
}

#[test]
fn edge_circulation_is_kronecker_on_straight_and_curved_geometry() {
    for (order, alpha) in [(ElementOrder::Tet4, 0.0), (ElementOrder::Tet10, 0.6)] {
        let coordinates = reference_coordinates(order, alpha);
        for (edge, [a, b]) in EDGES.into_iter().enumerate() {
            let mut circulation = [0.0; 6];
            let corners = reference_coordinates(ElementOrder::Tet4, 0.0);
            let delta = super::super::element::subtract(corners[b], corners[a]);
            for (t, weight) in GAUSS_LEGENDRE_EIGHT_UNIT {
                let mut barycentric = [0.0; 4];
                barycentric[a] = 1.0 - t;
                barycentric[b] = t;
                let r = barycentric[1];
                let s = barycentric[2];
                let tangent = [
                    delta[0],
                    delta[1],
                    delta[2] + alpha * (s * delta[0] + r * delta[1]),
                ];
                visit_volume_rule(order, &coordinates, [(barycentric, 1.0)], |point| {
                    for (i, value) in basis(point, [1.0; 6]).values.into_iter().enumerate() {
                        circulation[i] += weight * dot(value, tangent);
                    }
                })
                .unwrap();
            }
            for (i, value) in circulation.into_iter().enumerate() {
                close(value, if i == edge { 1.0 } else { 0.0 });
            }
        }
    }
}

#[test]
fn canonical_gradient_has_zero_curl_and_survives_node_storage_permutation() {
    let mesh = crate::topology::test_meshes::slab_mesh(2);
    let mut projection = AnalysisProjection::derive(&mesh).unwrap();
    let original = TetrahedralEdgeSpace::derive(&projection).unwrap();
    let identities: Vec<_> = original
        .edges()
        .iter()
        .map(|edge| edge.node_identities)
        .collect();
    let moments: Vec<_> = (0..projection.elements.len())
        .map(|i| original.element_moments(&projection, i).unwrap())
        .collect();
    let node_count = projection.nodes.len();
    projection.nodes.reverse();
    for element in &mut projection.elements {
        for index in &mut element.node_indices {
            *index = node_count - 1 - *index;
        }
    }
    let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
    assert_eq!(
        identities,
        space
            .edges()
            .iter()
            .map(|edge| edge.node_identities)
            .collect::<Vec<_>>()
    );
    let scalar: Vec<_> = projection
        .nodes
        .iter()
        .map(|node| {
            let [x, y, z] = node.coordinates_m.canonical_value();
            x * x + 3.0 * y - z
        })
        .collect();
    let coefficients = space.gradient(&scalar).unwrap();
    for (index, expected) in moments.into_iter().enumerate() {
        let actual = space.element_moments(&projection, index).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                close(actual.mass[i][j], expected.mass[i][j]);
                close(actual.curl_curl[i][j], expected.curl_curl[i][j]);
            }
        }
        let indices = space.element_edge_indices(index).unwrap();
        space
            .visit_element_quadrature(&projection, index, |point| {
                for axis in 0..3 {
                    close(
                        (0..6)
                            .map(|i| coefficients[indices[i]] * point.curls[i][axis])
                            .sum(),
                        0.0,
                    );
                    close(
                        (0..6)
                            .map(|i| coefficients[indices[i]] * point.values[i][axis])
                            .sum(),
                        space.elements[index]
                            .corners
                            .iter()
                            .enumerate()
                            .map(|(i, &node)| scalar[node] * point.corner_gradients[i][axis])
                            .sum(),
                    );
                }
            })
            .unwrap();
    }
}

#[test]
fn curved_forms_agree_with_separate_tensor_quadrature_for_polynomial_map() {
    let coordinates = reference_coordinates(ElementOrder::Tet10, 0.6);
    let integrate = |tensor| {
        let mut forms = [[[0.0; 6]; 6]; 2];
        let mut add = |point: VolumeQuadraturePoint| {
            let edge = basis(point, [1.0; 6]);
            for i in 0..6 {
                for j in 0..6 {
                    forms[0][i][j] += point.volume_weight_m3 * dot(edge.values[i], edge.values[j]);
                    forms[1][i][j] += point.volume_weight_m3 * dot(edge.curls[i], edge.curls[j]);
                }
            }
        };
        if tensor {
            visit_volume_rule(
                ElementOrder::Tet10,
                &coordinates,
                tensor_tetrahedron_rule(&GAUSS_LEGENDRE_EIGHT_UNIT),
                &mut add,
            )
            .unwrap();
        } else {
            crate::topology::quadrature::visit_volume_quadrature(
                ElementOrder::Tet10,
                &coordinates,
                &mut add,
            )
            .unwrap();
        }
        forms
    };
    let low = integrate(false);
    let high = integrate(true);
    for matrix in 0..2 {
        for i in 0..6 {
            for j in 0..6 {
                close(low[matrix][i][j], high[matrix][i][j]);
            }
        }
    }
}

#[test]
fn invalid_identity_arity_geometry_and_inputs_are_rejected() {
    let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut projection = AnalysisProjection::derive(&mesh).unwrap();
    let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
    assert!(space.gradient(&[]).is_err());
    assert!(space
        .gradient(&vec![f64::NAN; projection.nodes.len()])
        .is_err());
    assert!(space.element_moments(&projection, usize::MAX).is_err());
    let mut arity = projection.clone();
    arity.elements[0].node_indices.pop();
    assert!(TetrahedralEdgeSpace::derive(&arity).is_err());
    let mut corner = projection.clone();
    corner.elements[0].node_indices[1] = corner.elements[0].node_indices[0];
    assert!(TetrahedralEdgeSpace::derive(&corner).is_err());
    let mut unknown = projection.clone();
    unknown.elements[0].node_indices[0] = usize::MAX;
    assert!(TetrahedralEdgeSpace::derive(&unknown).is_err());
    projection.elements[0].stable_identity = StableDigest([42; 32]);
    assert!(space.element_moments(&projection, 0).is_err());
    projection.nodes[1].stable_identity = projection.nodes[0].stable_identity;
    assert!(TetrahedralEdgeSpace::derive(&projection).is_err());
    let folded = [[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(
        visit_volume_rule(ElementOrder::Tet4, &folded, [([0.25; 4], 1.0)], |_| {})
            .unwrap()
            .is_none()
    );
}

#[test]
fn shared_face_tangential_trace_is_single_valued_for_each_global_edge() {
    let mesh = crate::topology::test_meshes::slab_mesh(1);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
    let mut faces_checked = 0;
    for first in 0..space.elements.len() {
        for second in first + 1..space.elements.len() {
            let common: Vec<_> = space.elements[first]
                .corners
                .into_iter()
                .filter(|node| space.elements[second].corners.contains(node))
                .collect();
            if common.len() != 3 {
                continue;
            }
            let points: Vec<_> = common
                .iter()
                .map(|&node| projection.nodes[node].coordinates_m.canonical_value())
                .collect();
            let centroid: [f64; 3] = std::array::from_fn(|axis| {
                points.iter().map(|point| point[axis]).sum::<f64>() / 3.0
            });
            let tangent = super::super::element::subtract(points[1], points[0]);
            let mut traces = vec![vec![0.0; space.edges.len()]; 2];
            for (side, index) in [first, second].into_iter().enumerate() {
                let element = &space.elements[index];
                let coordinates: Vec<_> = element
                    .corners
                    .iter()
                    .map(|&node| projection.nodes[node].coordinates_m.canonical_value())
                    .collect();
                let matrix = [
                    super::super::element::subtract(coordinates[1], coordinates[0]),
                    super::super::element::subtract(coordinates[2], coordinates[0]),
                    super::super::element::subtract(coordinates[3], coordinates[0]),
                ];
                let inverse = super::super::element::invert(matrix).unwrap();
                let delta = super::super::element::subtract(centroid, coordinates[0]);
                let local: [f64; 3] = std::array::from_fn(|r| {
                    (0..3).map(|axis| delta[axis] * inverse[axis][r]).sum()
                });
                let barycentric = [
                    1.0 - local.iter().sum::<f64>(),
                    local[0],
                    local[1],
                    local[2],
                ];
                visit_volume_rule(
                    ElementOrder::Tet4,
                    &coordinates,
                    [(barycentric, 1.0)],
                    |point| {
                        for (i, value) in basis(point, element.signs).values.into_iter().enumerate()
                        {
                            traces[side][element.indices[i]] = dot(value, tangent);
                        }
                    },
                )
                .unwrap();
            }
            for (left, right) in traces[0].iter().zip(&traces[1]) {
                close(*left, *right);
            }
            faces_checked += 1;
        }
    }
    assert_eq!(faces_checked, 6);
}
