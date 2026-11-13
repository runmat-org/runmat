//! Rational (NURBS) and periodic B-splines, and splines with hundreds of control points.

mod common;
use common::*;
use runmat_constraint::*;
use std::time::Instant;

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn fixed_points(s: &mut System, pts: &[[f64; 2]]) -> Vec<EntityId> {
    pts.iter()
        .map(|p| {
            let e = s.point2(p[0], p[1]);
            s.set_entity_flag(e, VarFlag::Fixed);
            e
        })
        .collect()
}

fn weights_of(s: &System, sp: EntityId) -> Vec<VarId> {
    match &s.entity(sp).kind {
        EntityKind::BSpline { weights, .. } => weights.clone(),
        _ => unreachable!(),
    }
}

/// A point constrained onto `curve` at parameter `t`; returns the point.
fn point_at(s: &mut System, curve: EntityId, t: f64, start: [f64; 2]) -> EntityId {
    let p = s.point2(start[0], start[1]);
    let c = s
        .constrain(Constraint::PointOnCurve { point: p, curve })
        .unwrap();
    let param = s.constraint(c).aux[0];
    s.constrain(Constraint::VarValue {
        var: param,
        value: t,
    })
    .unwrap();
    p
}

#[test]
fn nurbs_quarter_circle_is_exact() {
    let mut s = System::new();
    let cps = fixed_points(&mut s, &[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    let h = 0.5f64.sqrt();
    let sp = s.nurbs(2, &cps, &[1.0, h, 1.0]).unwrap();
    // Weights are created fixed: the spline has no DOF of its own.
    assert_eq!(weights_of(&s, sp).len(), 3);
    let mut pts = Vec::new();
    for k in 1..8 {
        let x = k as f64 / 8.0;
        let p = s.point2(x, 0.9);
        s.constrain(Constraint::PointOnCurve {
            point: p,
            curve: sp,
        })
        .unwrap();
        s.constrain(Constraint::VarValue {
            var: s.entity_vars(p)[0],
            value: x,
        })
        .unwrap();
        pts.push((p, x));
    }
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    for (p, x) in pts {
        let q = pt(&rep, &s, p);
        assert_close(q[1], (1.0 - x * x).sqrt(), 1e-9);
    }
}

#[test]
fn nurbs_weight_as_design_variable() {
    // The conics through P0 and P2 tangent to the control legs form a one-parameter family
    // in the middle weight; a point at 30° on the unit circle picks w1 = √2/2.
    let mut s = System::new();
    let cps = fixed_points(&mut s, &[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    let sp = s.nurbs(2, &cps, &[1.0, 1.0, 1.0]).unwrap();
    let w1 = weights_of(&s, sp)[1];
    s.set_flag(w1, VarFlag::Free);
    let a = 30f64.to_radians();
    let p = fixed_points(&mut s, &[[a.cos(), a.sin()]])[0];
    s.constrain(Constraint::PointOnCurve {
        point: p,
        curve: sp,
    })
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    assert_eq!(rep.entity_dof(sp), Some(0));
    assert_close(rep.value(w1), 0.5f64.sqrt(), 1e-9);
    // Fixing a spline holds its free weights too (and skips fixed ones): exact, no
    // redundancy.
    let mut s = System::new();
    let cps: Vec<EntityId> = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        .iter()
        .map(|p| s.point2(p[0], p[1]))
        .collect();
    let sp = s.nurbs(2, &cps, &[1.0, 0.6, 1.0]).unwrap();
    let w = weights_of(&s, sp);
    s.set_flag(w[1], VarFlag::Free);
    s.constrain(Constraint::Fix(sp)).unwrap();
    s.set_value(w[1], 0.9);
    s.set_value(s.entity_vars(cps[1])[0], 1.3);
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    assert_close(rep.value(w[1]), 0.6, 1e-12);
    assert_close(rep.value(s.entity_vars(cps[1])[0]), 1.0, 1e-12);
}

#[test]
fn periodic_spline_wraps_and_closes() {
    let mut s = System::new();
    let hex: Vec<[f64; 2]> = (0..6)
        .map(|k| {
            let a = k as f64 * std::f64::consts::PI / 3.0;
            [5.0 * a.cos(), 3.0 * a.sin()]
        })
        .collect();
    let cps = fixed_points(&mut s, &hex);
    let sp = s.periodic_bspline(3, &cps).unwrap();
    let a = point_at(&mut s, sp, 0.25, [0.0, 3.0]);
    let b = point_at(&mut s, sp, 1.25, [0.1, 2.9]);
    let z0 = point_at(&mut s, sp, 0.0, [4.0, 0.0]);
    let z1 = point_at(&mut s, sp, 1.0, [4.0, 0.1]);
    let zm = point_at(&mut s, sp, -0.5, [-4.0, 0.0]);
    let zh = point_at(&mut s, sp, 0.5, [-4.0, 0.1]);
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    assert_pt(pt(&rep, &s, a), pt(&rep, &s, b), 1e-9);
    assert_pt(pt(&rep, &s, z0), pt(&rep, &s, z1), 1e-9);
    assert_pt(pt(&rep, &s, zm), pt(&rep, &s, zh), 1e-9);
    // Knots (i - p)/n put t = 0 at the start of the span over P0..P3, where a uniform cubic
    // B-spline is (P0 + 4 P1 + P2) / 6.
    let q = pt(&rep, &s, z0);
    let (q0, q1, q2) = (hex[0], hex[1], hex[2]);
    let span_start = [
        (q0[0] + 4.0 * q1[0] + q2[0]) / 6.0,
        (q0[1] + 4.0 * q1[1] + q2[1]) / 6.0,
    ];
    assert_pt(q, span_start, 1e-9);
}

#[test]
fn periodic_spline_free_dof_and_validation() {
    let mut s = System::new();
    let cps: Vec<EntityId> = (0..8)
        .map(|k| {
            let a = k as f64 * std::f64::consts::PI / 4.0;
            s.point2(2.0 * a.cos(), a.sin())
        })
        .collect();
    let sp = s.periodic_bspline(3, &cps).unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.entity_dof(sp), Some(16));
    // No ends on a closed curve.
    let l = s.line_xy([0.0, 0.0], [1.0, 0.0]);
    assert!(matches!(
        s.constrain(Constraint::TangentAtEnd {
            a: sp,
            a_end: CurveEnd::Start,
            b: l,
            b_end: CurveEnd::Start
        }),
        Err(BuildError::WrongKind { .. })
    ));
    let c = s.point2(0.0, 0.0);
    let e0 = s.point2(1.0, 0.0);
    let e1 = s.point2(0.0, 1.0);
    let arc = s.arc(c, e0, e1).unwrap();
    assert!(matches!(
        s.constrain(Constraint::CurvatureContinuity {
            spline: sp,
            end: CurveEnd::End,
            arc
        }),
        Err(BuildError::BadSpline(_))
    ));
    // Repeating the first points (an explicit wrap) is refused: the wrap is implicit.
    let mut wrapped = cps.clone();
    wrapped.extend_from_slice(&cps[..3]);
    assert!(matches!(
        s.periodic_bspline(3, &wrapped),
        Err(BuildError::BadSpline(_))
    ));
    // Periodic knots need n + 2p + 1 values with periodic spacing.
    assert!(matches!(
        s.spline(
            3,
            &cps,
            SplineOptions {
                knots: Some(vec![0.0; 10]),
                periodic: true,
                ..SplineOptions::default()
            }
        ),
        Err(BuildError::BadSpline(_))
    ));
    // Knot intervals 0..3 and 11..14 repeat the fundamental ones (3..11); moving knot 1
    // breaks that.
    let mut k: Vec<f64> = (0..15).map(|i| i as f64).collect();
    k[1] = 1.5;
    assert!(matches!(
        s.spline(
            3,
            &cps,
            SplineOptions {
                knots: Some(k),
                periodic: true,
                ..SplineOptions::default()
            }
        ),
        Err(BuildError::BadSpline(_))
    ));
    // Weights: one per point, positive.
    assert!(matches!(
        s.nurbs(3, &cps, &[1.0; 3]),
        Err(BuildError::BadSpline(_))
    ));
    assert!(matches!(
        s.nurbs(3, &cps[..4], &[1.0, 0.0, 1.0, 1.0]),
        Err(BuildError::BadSpline(_))
    ));
}

/// A rational cubic Bézier point (homogeneous de Casteljau), for an independent check.
fn rational_bezier(p: &[[f64; 2]; 4], w: &[f64; 4], t: f64) -> [f64; 2] {
    let mut h: Vec<[f64; 3]> = (0..4)
        .map(|i| [p[i][0] * w[i], p[i][1] * w[i], w[i]])
        .collect();
    for r in 1..4 {
        for i in 0..4 - r {
            for k in 0..3 {
                h[i][k] = (1.0 - t) * h[i][k] + t * h[i + 1][k];
            }
        }
    }
    [h[0][0] / h[0][2], h[0][1] / h[0][2]]
}

#[test]
fn rational_g2_spline_into_arc() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let a0 = s.point2(10.0, 0.0);
    let a1 = s.point2(0.0, 10.0);
    let arc = s.arc(c, a0, a1).unwrap();
    s.constrain(Constraint::Fix(arc)).unwrap();
    let start = [[0.2, 10.1], [-4.0, 10.3], [-8.0, 8.0], [-12.0, 4.0]];
    let cps: Vec<EntityId> = start.iter().map(|p| s.point2(p[0], p[1])).collect();
    let w = [1.0, 0.7, 1.4, 1.0];
    let sp = s.nurbs(3, &cps, &w).unwrap();
    s.constrain(Constraint::Fix(cps[2])).unwrap();
    s.constrain(Constraint::Fix(cps[3])).unwrap();
    s.constrain(Constraint::Coincident(cps[0], a1)).unwrap();
    s.constrain(Constraint::CurvatureContinuity {
        spline: sp,
        end: CurveEnd::Start,
        arc,
    })
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    let p: [[f64; 2]; 4] = [
        pt(&rep, &s, cps[0]),
        pt(&rep, &s, cps[1]),
        pt(&rep, &s, cps[2]),
        pt(&rep, &s, cps[3]),
    ];
    // Curvature at t = 0 by one-sided differences of the rational curve itself.
    let h = 1e-4;
    let (c0, c1, c2) = (
        rational_bezier(&p, &w, 0.0),
        rational_bezier(&p, &w, h),
        rational_bezier(&p, &w, 2.0 * h),
    );
    let d1 = [
        (-3.0 * c0[0] + 4.0 * c1[0] - c2[0]) / (2.0 * h),
        (-3.0 * c0[1] + 4.0 * c1[1] - c2[1]) / (2.0 * h),
    ];
    let d2 = [
        (c0[0] - 2.0 * c1[0] + c2[0]) / (h * h),
        (c0[1] - 2.0 * c1[1] + c2[1]) / (h * h),
    ];
    let kappa = (d1[0] * d2[1] - d1[1] * d2[0]).abs() / d1[0].hypot(d1[1]).powi(3);
    assert_close(kappa, 0.1, 2e-3);
    // G1: the first leg is tangent to the arc at (0, 10).
    assert_close(p[1][1], 10.0, 1e-9);
}

/// Clamped uniform knots for `n` control points of degree `p` (as [`System::bspline`]).
fn clamped_knots(n: usize, p: usize) -> Vec<f64> {
    let interior = n - p - 1;
    let mut k = vec![0.0; p + 1];
    for i in 1..=interior {
        k.push(i as f64 / (interior + 1) as f64);
    }
    k.extend(std::iter::repeat_n(1.0, p + 1));
    k
}

/// Interpolates `n` fixed fit points with a cubic spline of `n` free control points at the
/// Greville abscissae: 2n control-point unknowns plus n parameters, all determined.
fn interpolation(n: usize) -> (System, EntityId, Vec<EntityId>, Vec<[f64; 2]>) {
    let p = 3;
    let knots = clamped_knots(n, p);
    let greville: Vec<f64> = (0..n)
        .map(|i| knots[i + 1..=i + p].iter().sum::<f64>() / p as f64)
        .collect();
    let mut s = System::new();
    let cps: Vec<EntityId> = greville.iter().map(|&g| s.point2(100.0 * g, 0.0)).collect();
    let sp = s.bspline(p, &cps).unwrap();
    let fit: Vec<[f64; 2]> = greville
        .iter()
        .map(|&g| [100.0 * g, 10.0 * (6.0 * g).sin() + 3.0 * (17.0 * g).cos()])
        .collect();
    let fp = fixed_points(&mut s, &fit);
    for (i, &f) in fp.iter().enumerate() {
        let c = s
            .constrain(Constraint::PointOnCurve {
                point: f,
                curve: sp,
            })
            .unwrap();
        let param = s.constraint(c).aux[0];
        s.constrain(Constraint::VarValue {
            var: param,
            value: greville[i],
        })
        .unwrap();
    }
    (s, sp, cps, fit)
}

#[test]
fn scale_five_hundred_control_point_interpolation() {
    let t = Instant::now();
    let (s, sp, _cps, _fit) = interpolation(500);
    let build = t.elapsed();
    assert_eq!(s.var_count(), 500 * 2 + 500 * 2 + 500);
    let t = Instant::now();
    let rep = solve(&s, &opts());
    let elapsed = t.elapsed();
    println!(
        "spline interpolation: 500 control points, {} unknowns: build {build:?}, solve+diagnose {elapsed:?}, {} iterations",
        rep.components.iter().map(|c| c.vars.len()).sum::<usize>(),
        rep.iterations
    );
    assert_eq!(
        rep.status,
        Status::WellDefined,
        "residual {}",
        rep.max_residual
    );
    assert_eq!(rep.dof, 0);
    assert_eq!(rep.entity_dof(sp), Some(0));
    assert!(rep.max_residual <= 1e-10);
    assert!(rep.iterations <= 10, "{} iterations", rep.iterations);
    let again = solve(&s, &opts());
    assert_eq!(bits(&rep.values), bits(&again.values));
    // The dog leg agrees.
    let dl = solve(
        &s,
        &SolveOptions {
            algorithm: Algorithm::DogLeg,
            ..opts()
        },
    );
    assert_eq!(dl.status, Status::WellDefined);
    for (a, b) in rep.values.iter().zip(&dl.values) {
        assert!((a - b).abs() <= 1e-7 * (1.0 + a.abs()));
    }
}

#[test]
fn scale_large_free_spline_and_fixed_spline() {
    // 300 free control points with 50 points dragged onto the curve: DOF stays exact.
    let mut s = System::new();
    let cps: Vec<EntityId> = (0..300)
        .map(|i| s.point2(i as f64, (i as f64 * 0.1).sin()))
        .collect();
    let sp = s.bspline(5, &cps).unwrap();
    for k in 0..50 {
        let x = 3.0 + 6.0 * k as f64;
        let p = s.point2(x, 0.5);
        s.constrain(Constraint::PointOnCurve {
            point: p,
            curve: sp,
        })
        .unwrap();
    }
    let t = Instant::now();
    let rep = solve(&s, &opts());
    println!(
        "free spline: 300 control points, 50 points on it: {:?}",
        t.elapsed()
    );
    assert!(rep.converged);
    assert_eq!(rep.status, Status::UnderDefined);
    // 600 + 50 × (2 + 1 − 2).
    assert_eq!(rep.dof, 650);
    assert_eq!(rep.entity_dof(sp), Some(600));

    // Fixing a 120-point NURBS (240 coordinates, fixed weights) is exact and not redundant.
    let mut s = System::new();
    let cps: Vec<EntityId> = (0..120)
        .map(|i| s.point2(i as f64, (i as f64 * 0.3).cos()))
        .collect();
    let w: Vec<f64> = (0..120).map(|i| 1.0 + 0.1 * (i % 3) as f64).collect();
    let sp = s.nurbs(3, &cps, &w).unwrap();
    s.constrain(Constraint::Fix(sp)).unwrap();
    let p = s.point2(40.2, 0.0);
    s.constrain(Constraint::PointOnCurve {
        point: p,
        curve: sp,
    })
    .unwrap();
    s.constrain(Constraint::VarValue {
        var: s.entity_vars(p)[0],
        value: 40.2,
    })
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{:?}", rep.redundant);
    assert!(rep.redundant.is_empty());
}
