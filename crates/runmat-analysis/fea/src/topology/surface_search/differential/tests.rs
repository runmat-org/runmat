use super::*;
fn policy() -> BoundaryProjectionDifferentialPolicy {
    BoundaryProjectionDifferentialPolicy {
        stationarity_tolerance_m: 1e-11,
        minimum_barycentric: 1e-8,
        minimum_metric_ratio: 1e-10,
    }
}
fn near(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() < t, "{a} != {b}");
}
fn nodes(a: f64) -> [[f64; 3]; 6] {
    [
        [0., 0., 0.],
        [1., 0., a],
        [0., 1., 0.],
        [0.5, 0., a / 4.],
        [0.5, 0.5, a / 4.],
        [0., 0.5, 0.],
    ]
}
// Independent scalar parabola stationarity: u-qx+2*a*u*(a*u*u-qz)=0.
fn oracle(a: f64, q: [f64; 3]) -> ([f64; 3], [f64; 3], f64) {
    let mut u = q[0];
    for _ in 0..12 {
        u -= (u - q[0] + 2. * a * u * (a * u * u - q[2]))
            / (1. + 6. * a * a * u * u - 2. * a * q[2]);
    }
    let length = (1. + 4. * a * a * u * u).sqrt();
    let n = [-2. * a * u / length, 0., 1. / length];
    let p = [u, q[1], a * u * u];
    let g = (0..3).map(|i| (q[i] - p[i]) * n[i]).sum();
    (p, n, g)
}
#[test]
fn curved_implicit_projection_variation_matches_independent_parabola_and_curvature() {
    let a = 0.4;
    let u = 0.25;
    let v = 0.3;
    let h = 0.1;
    let n = [-2. * a * u, 0., 1.];
    let l: f64 = norm(n);
    let n = n.map(|x| x / l);
    let q = std::array::from_fn(|i| [u, v, a * u * u][i] + h * n[i]);
    let surface = BoundarySurface::derive(BoundaryTriangleOrder::Tri6, &nodes(a)).unwrap();
    let velocity = [0.13, -0.04, 0.07];
    let da = 0.21;
    let dx = nodes(da);
    let dx = dx.map(|p| [0., 0., p[2]]);
    let result = surface
        .stationary_projection_variation(q, [1. - u - v, u, v], velocity, &dx, policy())
        .unwrap();
    near(
        result.signed_gap_increment_m,
        dot(n, subtract(velocity, [0., 0., da * u * u])),
        1e-13,
    );
    for step in [1e-5, 1e-6] {
        let plus = oracle(
            a + step * da,
            std::array::from_fn(|i| q[i] + step * velocity[i]),
        );
        let minus = oracle(
            a - step * da,
            std::array::from_fn(|i| q[i] - step * velocity[i]),
        );
        near(
            result.signed_gap_increment_m,
            (plus.2 - minus.2) / (2. * step),
            1e-8,
        );
        for k in 0..3 {
            near(
                result.point_position_increment_m[k],
                (plus.0[k] - minus.0[k]) / (2. * step),
                1e-8,
            );
            near(
                result.unit_normal_increment[k],
                (plus.1[k] - minus.1[k]) / (2. * step),
                1e-8,
            );
        }
    }
    assert!(result.unit_normal_increment[0].abs() > 0.01);
}
#[test]
fn gap_hessian_is_symmetric_and_rigid_translation_has_zero_variation() {
    let coordinates = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
    let surface = BoundarySurface::derive(BoundaryTriangleOrder::Tri3, &coordinates).unwrap();
    let q = [0.2, 0.3, -0.1];
    let b = [0.5, 0.2, 0.3];
    let mut hessian = [[0.; 12]; 12];
    for column in 0..12 {
        let mut dq = [0.; 3];
        let mut dx = [[0.; 3]; 3];
        if column < 3 {
            dq[column] = 1.;
        } else {
            dx[(column - 3) / 3][(column - 3) % 3] = 1.;
        }
        let v = surface
            .stationary_projection_variation(q, b, dq, &dx, policy())
            .unwrap();
        for (row, value) in v
            .gap_gradient_query_increment
            .iter()
            .chain(v.gap_gradient_nodes_increment().iter().flatten())
            .enumerate()
        {
            hessian[row][column] = *value;
        }
    }
    for i in 0..12 {
        for j in 0..12 {
            near(hessian[i][j], hessian[j][i], 1e-13);
        }
    }
    let shift = [0.2, -0.1, 0.3];
    let v = surface
        .stationary_projection_variation(q, b, shift, &[shift; 3], policy())
        .unwrap();
    near(v.signed_gap_increment_m, 0., 1e-14);
    for x in v.unit_normal_increment {
        near(x, 0., 1e-14);
    }
    let sum = std::array::from_fn::<_, 3, _>(|k| {
        v.gap_gradient_query[k] + v.gap_gradient_nodes().iter().map(|n| n[k]).sum::<f64>()
    });
    for x in sum {
        near(x, 0., 1e-14);
    }
}
#[test]
fn nonsmooth_nonstationary_and_focal_projections_cannot_supply_a_smooth_tangent() {
    let surface = BoundarySurface::derive(BoundaryTriangleOrder::Tri6, &nodes(0.4)).unwrap();
    let dx = [[0.; 3]; 6];
    assert!(matches!(
        surface.stationary_projection_variation(
            [0.2, 0.3, 0.1],
            [0.5, 0.2, 0.3],
            [0.; 3],
            &dx,
            policy()
        ),
        Err(BoundarySearchError::NonStationaryProjection)
    ));
    assert!(matches!(
        surface.stationary_projection_variation(
            [0., 0.3, 0.1],
            [0.7, 0., 0.3],
            [0.; 3],
            &dx,
            policy()
        ),
        Err(BoundarySearchError::NonSmoothProjection)
    ));
    // At u=.25 choose qz so 1+6*a²*u²-2*a*qz=0; the projection is focal.
    let a = 0.4;
    let u = 0.25;
    let qz = (1. + 6. * a * a * u * u) / (2. * a);
    let qx = u + 2. * a * u * (a * u * u - qz);
    assert!(matches!(
        surface.stationary_projection_variation(
            [qx, 0.3, qz],
            [0.45, u, 0.3],
            [0.; 3],
            &dx,
            policy()
        ),
        Err(BoundarySearchError::DegenerateProjection)
    ));
}

#[test]
fn mixed_projection_and_normal_derivatives_match_independent_parabola() {
    let a = 0.4;
    let u = 0.25;
    let v = 0.3;
    let h = 0.1;
    let n = [-2.0 * a * u, 0.0, 1.0];
    let n = n.map(|x| x / norm(n));
    let query = std::array::from_fn(|k| [u, v, a * u * u][k] + h * n[k]);
    let surface = BoundarySurface::derive(BoundaryTriangleOrder::Tri6, &nodes(a)).unwrap();
    let qa = [0.13, -0.04, 0.07];
    let qb = [-0.02, 0.06, -0.1];
    let da = 0.21;
    let db = -0.12;
    let xa = nodes(da).map(|p| [0.0, 0.0, p[2]]);
    let xb = nodes(db).map(|p| [0.0, 0.0, p[2]]);
    let directions = [
        BoundaryProjectionDirection {
            query_m: qa,
            nodes_m: &xa,
        },
        BoundaryProjectionDirection {
            query_m: qb,
            nodes_m: &xb,
        },
    ];
    let result = surface
        .stationary_projection_mixed_variation(query, [1.0 - u - v, u, v], directions, policy())
        .unwrap();
    let reverse = surface
        .stationary_projection_mixed_variation(
            query,
            [1.0 - u - v, u, v],
            [directions[1], directions[0]],
            policy(),
        )
        .unwrap();
    for (a, b) in result
        .unit_normal_mixed
        .into_iter()
        .zip(reverse.unit_normal_mixed)
    {
        near(a, b, 1e-12);
    }
    near(result.signed_gap_mixed_m, reverse.signed_gap_mixed_m, 1e-12);
    for step in [1e-3, 3e-4] {
        let eval = |s: f64, t: f64| {
            oracle(
                a + s * da + t * db,
                std::array::from_fn(|k| query[k] + s * qa[k] + t * qb[k]),
            )
        };
        let pp = eval(step, step);
        let pm = eval(step, -step);
        let mp = eval(-step, step);
        let mm = eval(-step, -step);
        for k in 0..3 {
            near(
                result.unit_normal_mixed[k],
                (pp.1[k] - pm.1[k] - mp.1[k] + mm.1[k]) / (4.0 * step * step),
                1e-6,
            );
        }
        near(
            result.parameter_mixed[0],
            (pp.0[0] - pm.0[0] - mp.0[0] + mm.0[0]) / (4.0 * step * step),
            1e-6,
        );
        near(
            result.signed_gap_mixed_m,
            (pp.2 - pm.2 - mp.2 + mm.2) / (4.0 * step * step),
            1e-6,
        );
    }
}
