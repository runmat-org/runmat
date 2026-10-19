use super::*;

fn rotate(v: V) -> V {
    [
        (3.0 * v[0] - 4.0 * v[1]) / 5.0,
        (4.0 * v[0] + 3.0 * v[1]) / 5.0,
        v[2],
    ]
}

#[test]
fn affine_physical_gradients_are_covariant_and_quadratic_truncation_refines() {
    let rows = [
        [1.0, 0.2, -0.1],
        [-0.3, 0.9, 0.4],
        [0.2, -0.4, 1.1],
        [-0.8, -0.5, -0.7],
    ];
    let expected = [0.7, -0.4, 0.2];
    for rotated in [false, true] {
        let map = |v| if rotated { rotate(v) } else { v };
        let gradient = map(expected);
        for scale_factor in [2.0_f64.powi(-30), 1.0, 2.0_f64.powi(30)] {
            let mut previous: Option<f64> = None;
            for refinement in [1.0, 0.5, 0.25] {
                let h = scale_factor * refinement;
                let d = rows.map(|v| scale(map(v), h));
                let metric = GradientMetric::derive(&d).unwrap();
                let weights = d.map(|row| metric.weight(row).unwrap());
                let affine = d.iter().zip(weights).fold([0.0; 3], |sum, (row, w)| {
                    add(sum, scale(w, dot(gradient, *row)))
                });
                assert!(norm(sub(affine, gradient)) < 1e-14);
                // Independent phi(x)=g.x+(a.x)^2/(2L), Hessian aa^T/L.
                // A nonsymmetric linear stencil has a first-order gradient defect.
                let a = map([0.4, 0.3, -0.2]);
                let quadratic = d.iter().zip(weights).fold([0.0; 3], |sum, (row, w)| {
                    add(
                        sum,
                        scale(
                            w,
                            dot(gradient, *row) + dot(a, *row).powi(2) / (2.0 * scale_factor),
                        ),
                    )
                });
                let error = norm(sub(quadratic, gradient));
                assert!(error > 1e-5);
                if let Some(before) = previous {
                    assert!((error / before - 0.5_f64).abs() < 1e-12);
                }
                previous = Some(error);
            }
        }
    }
}

#[test]
fn rank_loss_is_rejected_without_regularization_or_projection() {
    let metric =
        GradientMetric::derive(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.5, 0.0]]).unwrap();
    assert!(metric.weight([1.0, 0.0, 0.0]).is_err());
    assert!(GradientMetric::derive(&[[0.0; 3]]).is_err());
    assert!(GradientMetric::derive(&[[f64::INFINITY, 0.0, 0.0]]).is_err());
}
