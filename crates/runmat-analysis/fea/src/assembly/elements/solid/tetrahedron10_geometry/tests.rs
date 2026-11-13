use super::*;
use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

fn nodes(origin: [f64; 3], scale: f64) -> [[f64; 3]; 10] {
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    .map(|point| std::array::from_fn(|axis| origin[axis] + scale * point[axis]));
    std::array::from_fn(|node| {
        if node < 4 {
            corners[node]
        } else {
            let [left, right] = TETRAHEDRON_MIDSIDE_EDGE_CORNERS[node - 4];
            std::array::from_fn(|axis| 0.5 * (corners[left][axis] + corners[right][axis]))
        }
    })
}

#[test]
fn affine_jacobian_and_shape_derivatives_are_independent_of_translation() {
    // Dyadic coordinates keep the translated geometry exactly representable.
    // The reference is the analytic derivative of Ni=lambda_i(2lambda_i-1)
    // and Nij=4lambda_i lambda_j in a scaled Cartesian simplex.
    let dlambda = [[-1.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for lambda in [
        [0.125, 0.25, 0.5, 0.125],
        // Also exercise the non-dyadic physical recovery rule, whose derivative
        // cancellation exposed the heterogeneous stress regression.
        crate::topology::quadrature::TETRAHEDRON_FOUR_POINT_RULE[0],
    ] {
        for scale in [0.5, 1.0, 2.0] {
            for origin in [[0.0; 3], [8.0, -16.0, 32.0], [2.0_f64.powi(40); 3]] {
                let geometry = nodes(origin, scale);
                assert!(is_affine(Tetrahedron10ElementGeometry {
                    nodes_m: geometry
                }));
                let (gradients, determinant) =
                    physical_shape_gradients_and_measure(geometry, lambda).unwrap();
                assert_eq!(determinant, scale.powi(3));
                for node in 0..10 {
                    let expected: [f64; 3] = if node < 4 {
                        dlambda[node].map(|value| (4.0 * lambda[node] - 1.0) * value / scale)
                    } else {
                        let [left, right] = TETRAHEDRON_MIDSIDE_EDGE_CORNERS[node - 4];
                        std::array::from_fn(|axis| {
                            4.0 * (lambda[left] * dlambda[right][axis]
                                + lambda[right] * dlambda[left][axis])
                                / scale
                        })
                    };
                    assert_eq!(gradients[node], expected);
                }
            }
        }
    }
}

#[test]
fn tiny_midside_curvature_is_not_classified_as_affine() {
    let mut geometry = nodes([0.0; 3], 1.0);
    geometry[4][1] += 2.0_f64.powi(-40);
    assert!(!is_affine(Tetrahedron10ElementGeometry {
        nodes_m: geometry
    }));
    validate_nodes(geometry).unwrap();
}
