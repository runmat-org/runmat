//! Ellipses and elliptical arcs: tangency to lines, circles and other ellipses, equal radii,
//! ellipse dimensions, elliptical arcs and G1 at their ends, with DOF and diagnosis checks.

mod common;
use common::*;
use runmat_constraint::*;
use std::f64::consts::FRAC_PI_2;

fn ell_vars(s: &System, e: EntityId) -> (EntityId, VarId, VarId, VarId) {
    match s.entity(e).kind {
        EntityKind::Ellipse {
            center,
            semi_major,
            semi_minor,
            rotation,
        }
        | EntityKind::EllipticalArc {
            center,
            semi_major,
            semi_minor,
            rotation,
            ..
        } => (center, semi_major, semi_minor, rotation),
        _ => unreachable!(),
    }
}

/// Holds an ellipse's shape and pose with explicit dimensions.
fn pin_ellipse(s: &mut System, e: EntityId, a: f64, b: f64, rot: f64) -> Vec<ConstraintId> {
    let (c, va, vb, vr) = ell_vars(s, e);
    vec![
        s.constrain(Constraint::Fix(c)).unwrap(),
        s.constrain(Constraint::VarValue { var: va, value: a })
            .unwrap(),
        s.constrain(Constraint::VarValue { var: vb, value: b })
            .unwrap(),
        s.constrain(Constraint::VarValue {
            var: vr,
            value: rot,
        })
        .unwrap(),
    ]
}

fn x_var(s: &System, p: EntityId) -> VarId {
    s.entity_vars(p)[0]
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn free_ellipse_and_elliptical_arc_dof() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let e = s.ellipse(c, 3.0, 1.0, 0.2).unwrap();
    let c2 = s.point2(10.0, 0.0);
    let p0 = s.point2(13.0, 0.1);
    let p1 = s.point2(10.1, 1.0);
    let arc = s.elliptical_arc(c2, 3.0, 1.0, 0.0, p0, p1).unwrap();
    let rep = solve(&s, &opts());
    assert!(rep.converged, "residual {}", rep.residual_norm);
    assert_eq!(rep.status, Status::UnderDefined);
    // Ellipse 5; elliptical arc: centre 2 + shape 3 + two ends (2 each) − 2 on-ellipse rows.
    assert_eq!(rep.dof, 12);
    assert_eq!(rep.entity_dof(e), Some(5));
    assert_eq!(rep.entity_dof(arc), Some(7));
    // The internal rows pulled the ends onto the ellipse.
    let (_, va, vb, vr) = ell_vars(&s, arc);
    let (a, b, r) = (rep.value(va), rep.value(vb), rep.value(vr));
    let cc = pt(&rep, &s, c2);
    for p in [p0, p1] {
        let q = pt(&rep, &s, p);
        let (dx, dy) = (q[0] - cc[0], q[1] - cc[1]);
        let (u, v) = (dx * r.cos() + dy * r.sin(), dy * r.cos() - dx * r.sin());
        assert_close((u / a).powi(2) + (v / b).powi(2), 1.0, 1e-9);
    }
}

#[test]
fn horizontal_line_tangent_to_rotated_ellipse() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let (a, b, rot) = (5.0, 2.0, 0.3);
    let e = s.ellipse(c, a, b, rot).unwrap();
    pin_ellipse(&mut s, e, a, b, rot);
    let p = s.point2(-6.0, 3.5);
    let q = s.point2(6.0, 3.6);
    let l = s.line(p, q).unwrap();
    s.constrain(Constraint::Horizontal(l)).unwrap();
    s.constrain(Constraint::VarValue {
        var: x_var(&s, p),
        value: -6.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: l,
        value: 12.0,
    })
    .unwrap();
    let t = s.constrain(Constraint::Tangent(l, e)).unwrap();
    assert!(
        s.constraint(t).aux.is_empty(),
        "line–ellipse needs no contact angle"
    );
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    // Support function along +y: h = sqrt(a² sin²r + b² cos²r); the line stays above.
    let h = ((a * rot.sin()).powi(2) + (b * rot.cos()).powi(2)).sqrt();
    assert_close(pt(&rep, &s, p)[1], h, 1e-9);
    assert_close(pt(&rep, &s, q)[1], h, 1e-9);
    assert!(rep.max_residual <= 1e-10);
}

#[test]
fn circle_tangent_to_ellipse_external_and_internal() {
    // External: a circle centred on the major axis at x = 7 touches the ellipse at (4, 0).
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let e = s.ellipse(c, 4.0, 2.0, 0.0).unwrap();
    pin_ellipse(&mut s, e, 4.0, 2.0, 0.0);
    let cc = s.point2(6.8, 0.2);
    let circle = s.circle(cc, 2.0).unwrap();
    s.constrain(Constraint::VarValue {
        var: x_var(&s, cc),
        value: 7.0,
    })
    .unwrap();
    s.constrain(Constraint::HorizontalPoints(cc, c)).unwrap();
    let t = s.constrain(Constraint::Tangent(circle, e)).unwrap();
    assert_eq!(s.constraint(t).aux.len(), 1);
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    let r = rep.value(*s.entity_vars(circle).last().unwrap());
    assert_close(r, 3.0, 1e-9);
    assert_close(rep.value(s.constraint(t).aux[0]).cos(), 1.0, 1e-9);

    // Internal: centred at (1, 0) inside the ellipse, the largest inscribed circle touches at
    // cos θ = a x₀ / (a² − b²) = 1/3: r = sqrt((a cos θ − 1)² + (b sin θ)²) = sqrt(33)/3.
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let e = s.ellipse(c, 4.0, 2.0, 0.0).unwrap();
    pin_ellipse(&mut s, e, 4.0, 2.0, 0.0);
    let cc = s.point2(1.0, 0.0);
    let circle = s.circle(cc, 1.8).unwrap();
    s.constrain(Constraint::Fix(cc)).unwrap();
    s.constrain(Constraint::Tangent(e, circle)).unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    let r = rep.value(*s.entity_vars(circle).last().unwrap());
    assert_close(r, 33f64.sqrt() / 3.0, 1e-9);
}

#[test]
fn ellipse_tangent_to_ellipse() {
    let mut s = System::new();
    let c1 = s.point2(0.0, 0.0);
    let e1 = s.ellipse(c1, 3.0, 1.0, 0.0).unwrap();
    pin_ellipse(&mut s, e1, 3.0, 1.0, 0.0);
    // A second ellipse standing upright (major axis vertical), sliding along the x axis.
    let c2 = s.point2(4.6, 0.0);
    let e2 = s.ellipse(c2, 2.0, 1.0, FRAC_PI_2).unwrap();
    let (_, a2, b2, r2) = ell_vars(&s, e2);
    for (v, val) in [(a2, 2.0), (b2, 1.0), (r2, FRAC_PI_2)] {
        s.constrain(Constraint::VarValue { var: v, value: val })
            .unwrap();
    }
    s.constrain(Constraint::HorizontalPoints(c1, c2)).unwrap();
    let t = s.constrain(Constraint::Tangent(e1, e2)).unwrap();
    assert_eq!(s.constraint(t).aux.len(), 2);
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    // Contact at (3, 0): the upright ellipse's half-width along x is its semi-minor 1.
    assert_pt(pt(&rep, &s, c2), [4.0, 0.0], 1e-9);
    let again = solve(&s, &opts());
    assert_eq!(bits(&rep.values), bits(&again.values));
}

#[test]
fn equal_radius_ellipses_and_dimensions() {
    let mut s = System::new();
    let c1 = s.point2(0.0, 0.0);
    let e1 = s.ellipse(c1, 5.5, 2.5, 0.0).unwrap();
    let c2 = s.point2(20.0, 0.0);
    let e2 = s.ellipse(c2, 4.0, 3.5, 0.4).unwrap();
    s.constrain(Constraint::Fix(c1)).unwrap();
    s.constrain(Constraint::VarValue {
        var: ell_vars(&s, e1).3,
        value: 0.0,
    })
    .unwrap();
    s.constrain(Constraint::EllipseDimension {
        curve: e1,
        measure: EllipseMeasure::SemiMajor,
        value: 5.0,
    })
    .unwrap();
    // Foci 8 apart with a = 5: b = 3.
    s.constrain(Constraint::EllipseDimension {
        curve: e1,
        measure: EllipseMeasure::FocalDistance,
        value: 8.0,
    })
    .unwrap();
    s.constrain(Constraint::EqualRadius(e1, e2)).unwrap();
    let minor = s
        .reference(Constraint::EllipseDimension {
            curve: e2,
            measure: EllipseMeasure::MinorDiameter,
            value: 0.0,
        })
        .unwrap();
    let rep = solve(&s, &opts());
    assert!(rep.converged);
    assert_eq!(rep.status, Status::UnderDefined);
    // e2 keeps its centre and rotation free.
    assert_eq!(rep.dof, 3);
    assert_eq!(rep.entity_dof(e1), Some(0));
    assert_eq!(rep.entity_dof(e2), Some(3));
    let (_, a2, b2, _) = ell_vars(&s, e2);
    assert_close(rep.value(a2), 5.0, 1e-9);
    assert_close(rep.value(b2), 3.0, 1e-9);
    assert_close(rep.measurement(minor).unwrap(), 6.0, 1e-9);

    // A third dimension on e2's semi-major is implied by the equality: redundant.
    let extra = s
        .constrain(Constraint::EllipseDimension {
            curve: e2,
            measure: EllipseMeasure::MajorDiameter,
            value: 10.0,
        })
        .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::OverDefined);
    assert_eq!(rep.redundant_ids(), vec![extra]);

    // And one that disagrees conflicts.
    let mut s2 = s.clone();
    let bad = s2
        .constrain(Constraint::EllipseDimension {
            curve: e2,
            measure: EllipseMeasure::SemiMinor,
            value: 2.0,
        })
        .unwrap();
    let rep = solve(&s2, &opts());
    assert_eq!(rep.status, Status::Conflicting);
    assert!(
        rep.conflict_set().contains(&bad),
        "{:?}",
        rep.conflict_set()
    );
}

#[test]
fn elliptical_arc_fully_defined_with_tangent_line_at_end() {
    let mut s = System::new();
    let c = s.point2(1.0, 2.0);
    let p0 = s.point2(4.7, 2.3);
    let p1 = s.point2(1.2, 3.8);
    let (a, b) = (4.0, 2.0);
    let arc = s.elliptical_arc(c, 3.7, 1.8, 0.1, p0, p1).unwrap();
    pin_ellipse(&mut s, arc, a, b, 0.0);
    // Start on the major axis, end on the minor axis: a quarter of the ellipse.
    s.constrain(Constraint::HorizontalPoints(c, p0)).unwrap();
    s.constrain(Constraint::VerticalPoints(c, p1)).unwrap();
    // A line leaving the end, G1 with the arc, 3 long.
    let q = s.point2(-1.5, 4.3);
    let l = s.line(p1, q).unwrap();
    s.constrain(Constraint::TangentAtEnd {
        a: arc,
        a_end: CurveEnd::End,
        b: l,
        b_end: CurveEnd::Start,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: l,
        value: 3.0,
    })
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    assert_pt(pt(&rep, &s, p0), [1.0 + a, 2.0], 1e-9);
    assert_pt(pt(&rep, &s, p1), [1.0, 2.0 + b], 1e-9);
    // At the top of the ellipse the tangent is horizontal; the line keeps its side (left).
    assert_pt(pt(&rep, &s, q), [-2.0, 2.0 + b], 1e-9);
}

#[test]
fn fix_elliptical_arc_is_exact_and_not_redundant() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let t0: f64 = 0.4;
    let t1: f64 = 2.2;
    let (a, b, rot): (f64, f64, f64) = (3.0, 1.5, 0.25);
    let on = |t: f64| {
        let (x, y) = (a * t.cos(), b * t.sin());
        [x * rot.cos() - y * rot.sin(), x * rot.sin() + y * rot.cos()]
    };
    let (q0, q1) = (on(t0), on(t1));
    let p0 = s.point2(q0[0], q0[1]);
    let p1 = s.point2(q1[0], q1[1]);
    let arc = s.elliptical_arc(c, a, b, rot, p0, p1).unwrap();
    s.constrain(Constraint::Fix(arc)).unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    assert!(rep.redundant.is_empty());
    assert_pt(pt(&rep, &s, p0), q0, 1e-12);
    assert_pt(pt(&rep, &s, p1), q1, 1e-12);
}

#[test]
fn point_on_elliptical_arc_and_line_tangent_to_arc() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let p0 = s.point2(3.0, 0.0);
    let p1 = s.point2(-3.0, 0.0);
    let arc = s.elliptical_arc(c, 3.0, 2.0, 0.0, p0, p1).unwrap();
    pin_ellipse(&mut s, arc, 3.0, 2.0, 0.0);
    s.constrain(Constraint::HorizontalPoints(c, p0)).unwrap();
    s.constrain(Constraint::HorizontalPoints(c, p1)).unwrap();
    // A point on the arc at x = 1.5 (upper half by starting position).
    let p = s.point2(1.5, 1.6);
    s.constrain(Constraint::PointOnCurve {
        point: p,
        curve: arc,
    })
    .unwrap();
    s.constrain(Constraint::VarValue {
        var: x_var(&s, p),
        value: 1.5,
    })
    .unwrap();
    // A horizontal line tangent to the arc, above it.
    let la = s.point2(-1.0, 2.4);
    let lb = s.point2(1.0, 2.5);
    let l = s.line(la, lb).unwrap();
    s.constrain(Constraint::Horizontal(l)).unwrap();
    s.constrain(Constraint::Tangent(l, arc)).unwrap();
    s.constrain(Constraint::VarValue {
        var: x_var(&s, la),
        value: -1.0,
    })
    .unwrap();
    s.constrain(Constraint::VarValue {
        var: x_var(&s, lb),
        value: 1.0,
    })
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined, "{rep:#?}");
    assert_pt(pt(&rep, &s, p), [1.5, 2.0 * (1.0 - 0.25f64).sqrt()], 1e-9);
    assert_close(pt(&rep, &s, la)[1], 2.0, 1e-9);
    // p0 and p1 on the major axis at ±a.
    assert_pt(pt(&rep, &s, p0), [3.0, 0.0], 1e-9);
    assert_pt(pt(&rep, &s, p1), [-3.0, 0.0], 1e-9);
}

#[test]
fn ellipse_validation() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let e = s.ellipse(c, 3.0, 1.0, 0.0).unwrap();
    let k = s.point2(5.0, 0.0);
    let circle = s.circle(k, 1.0).unwrap();
    let p = s.point2(1.0, 1.0);
    let q = s.point2(2.0, 1.0);
    assert!(matches!(
        s.constrain(Constraint::EqualRadius(e, circle)),
        Err(BuildError::WrongKind { .. })
    ));
    assert!(matches!(
        s.constrain(Constraint::EllipseDimension {
            curve: circle,
            measure: EllipseMeasure::SemiMajor,
            value: 1.0
        }),
        Err(BuildError::WrongKind { .. })
    ));
    assert!(matches!(
        s.constrain(Constraint::EllipseDimension {
            curve: e,
            measure: EllipseMeasure::SemiMajor,
            value: -1.0
        }),
        Err(BuildError::BadValue(_))
    ));
    assert!(matches!(
        s.elliptical_arc(c, 3.0, 0.0, 0.0, p, q),
        Err(BuildError::BadValue(_))
    ));
    assert!(matches!(
        s.constrain(Constraint::TangentAtEnd {
            a: e,
            a_end: CurveEnd::Start,
            b: circle,
            b_end: CurveEnd::Start
        }),
        Err(BuildError::WrongKind { .. })
    ));
}

#[test]
fn serde_round_trips_new_constraint_kinds_report() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let e = s.ellipse(c, 4.0, 2.0, 0.0).unwrap();
    pin_ellipse(&mut s, e, 4.0, 2.0, 0.0);
    let k = s.point2(7.0, 0.3);
    let circle = s.circle(k, 2.5).unwrap();
    s.constrain(Constraint::Tangent(circle, e)).unwrap();
    let rep = solve(&s, &opts());
    let json = serde_json::to_string(&rep).unwrap();
    let back: SolveReport = serde_json::from_str(&json).unwrap();
    assert_eq!(bits(&back.values), bits(&rep.values));
    assert_eq!(back.status, rep.status);
    assert_eq!(back.dof, rep.dof);
}
