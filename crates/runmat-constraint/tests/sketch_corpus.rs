//! The 2D sketch corpus: fully-defined profiles, under-defined sketches with known DOF,
//! over-defined and conflicting sketches with known constraint sets, and drag scenarios.
//!
//! Every case builds a [`System`], states the expected status and DOF, and checks geometry
//! where the answer is known in closed form. Cases are run one by one by `corpus_*` tests so a
//! failure names its case.

mod common;
use common::*;
use runmat_constraint::*;
use std::f64::consts::PI;

struct Expect {
    status: Status,
    dof: usize,
    /// Exactly these constraints are redundant (when `Some`).
    redundant: Option<Vec<ConstraintId>>,
    /// The conflict set must contain these.
    conflict_includes: Vec<ConstraintId>,
}

impl Expect {
    fn well() -> Self {
        Expect {
            status: Status::WellDefined,
            dof: 0,
            redundant: Some(vec![]),
            conflict_includes: vec![],
        }
    }
    fn under(dof: usize) -> Self {
        Expect {
            status: Status::UnderDefined,
            dof,
            redundant: Some(vec![]),
            conflict_includes: vec![],
        }
    }
    fn over(dof: usize, redundant: Vec<ConstraintId>) -> Self {
        Expect {
            status: Status::OverDefined,
            dof,
            redundant: Some(redundant),
            conflict_includes: vec![],
        }
    }
    fn conflict(includes: Vec<ConstraintId>) -> Self {
        Expect {
            status: Status::Conflicting,
            dof: usize::MAX,
            redundant: None,
            conflict_includes: includes,
        }
    }
}

type Check = Box<dyn Fn(&System, &SolveReport)>;

fn run_case(name: &str, s: System, e: Expect, check: Option<Check>) {
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, e.status, "{name}: status ({rep:#?})");
    if e.dof != usize::MAX {
        assert_eq!(rep.dof, e.dof, "{name}: dof");
    }
    if let Some(r) = &e.redundant {
        assert_eq!(&rep.redundant_ids(), r, "{name}: redundant set");
    }
    let set = rep.conflict_set();
    for c in &e.conflict_includes {
        assert!(set.contains(c), "{name}: conflict set {set:?} lacks {c}");
    }
    if e.status != Status::Conflicting {
        assert!(
            rep.converged,
            "{name}: not converged, residual {}",
            rep.residual_norm
        );
        assert!(
            rep.max_residual <= 1e-10,
            "{name}: residual {}",
            rep.max_residual
        );
    } else {
        assert!(!rep.converged, "{name}: conflicting but converged");
    }
    if let Some(check) = check {
        check(&s, &rep);
    }
    // Every case must also be deterministic and give the same answer without DM blocks.
    let again = solve(&s, &opts());
    assert_eq!(
        rep.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        again.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{name}: not deterministic"
    );
    let whole = solve(
        &s,
        &SolveOptions {
            decompose: false,
            ..opts()
        },
    );
    assert_eq!(
        whole.status, rep.status,
        "{name}: status without decomposition"
    );
    if rep.dof != usize::MAX {
        assert_eq!(whole.dof, rep.dof, "{name}: dof without decomposition");
    }
}

// ============================================================================ fully defined

fn fixed_rect(w: f64, h: f64) -> (System, Rect, Vec<ConstraintId>) {
    let mut s = System::new();
    let r = rect(&mut s, 0.3, -0.2, w * 0.9 + 1.0, h * 1.1 - 0.5);
    let mut ids = vec![s.constrain(Constraint::Fix(r.p[0])).unwrap()];
    ids.extend(square_up(&mut s, &r));
    ids.push(
        s.constrain(Constraint::Length {
            line: r.l[0],
            value: w,
        })
        .unwrap(),
    );
    ids.push(
        s.constrain(Constraint::Length {
            line: r.l[1],
            value: h,
        })
        .unwrap(),
    );
    (s, r, ids)
}

#[test]
fn corpus_01_horizontal_segment() {
    let mut s = System::new();
    let a = s.point2(1.0, 1.0);
    let b = s.point2(8.0, 2.5);
    let l = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Horizontal(l)).unwrap();
    s.constrain(Constraint::Length {
        line: l,
        value: 10.0,
    })
    .unwrap();
    run_case(
        "horizontal segment",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, b), [11.0, 1.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_02_rectangle() {
    let (s, r, _) = fixed_rect(40.0, 20.0);
    let p2 = r.p[2];
    let p0 = s.point(r.p[0]);
    run_case(
        "rectangle",
        s,
        Expect::well(),
        Some(Box::new(move |s, rep| {
            assert_pt(pt(rep, s, p2), [p0[0] + 40.0, p0[1] + 20.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_03_rectangle_from_coincident_segments() {
    let mut s = System::new();
    let corners = [[0.0, 0.0], [30.0, 1.0], [31.0, 12.0], [1.0, 11.0]];
    let mut lines = Vec::new();
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        lines.push(s.line_xy([a[0] + 0.2, a[1] - 0.1], [b[0] - 0.1, b[1] + 0.3]));
    }
    let ends = |s: &System, l: EntityId| match s.entity(l).kind {
        EntityKind::Line { start, end } => (start, end),
        _ => unreachable!(),
    };
    for i in 0..4 {
        let (_, e) = ends(&s, lines[i]);
        let (st, _) = ends(&s, lines[(i + 1) % 4]);
        s.constrain(Constraint::Coincident(e, st)).unwrap();
    }
    let (p0, _) = ends(&s, lines[0]);
    s.constrain(Constraint::Fix(p0)).unwrap();
    s.constrain(Constraint::Horizontal(lines[0])).unwrap();
    s.constrain(Constraint::Vertical(lines[1])).unwrap();
    s.constrain(Constraint::Horizontal(lines[2])).unwrap();
    s.constrain(Constraint::Vertical(lines[3])).unwrap();
    s.constrain(Constraint::Length {
        line: lines[0],
        value: 30.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: lines[1],
        value: 10.0,
    })
    .unwrap();
    run_case("coincident rectangle", s, Expect::well(), None);
}

#[test]
fn corpus_04_right_triangle_keeps_branch() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(2.8, 0.3);
    let c = s.point2(3.2, 3.5);
    let ab = s.line(a, b).unwrap();
    let bc = s.line(b, c).unwrap();
    let ca = s.line(c, a).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Horizontal(ab)).unwrap();
    s.constrain(Constraint::Length {
        line: ab,
        value: 3.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: bc,
        value: 4.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: ca,
        value: 5.0,
    })
    .unwrap();
    run_case(
        "3-4-5 triangle",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, b), [3.0, 0.0], 1e-9);
            assert_pt(pt(r, s, c), [3.0, 4.0], 1e-9); // the apex stays above the base
        })),
    );
}

#[test]
fn corpus_05_equilateral_triangle() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(9.0, 0.5);
    let c = s.point2(4.0, 7.0);
    let ab = s.line(a, b).unwrap();
    let bc = s.line(b, c).unwrap();
    let ca = s.line(c, a).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Horizontal(ab)).unwrap();
    s.constrain(Constraint::EqualLength(ab, bc)).unwrap();
    s.constrain(Constraint::EqualLength(bc, ca)).unwrap();
    s.constrain(Constraint::Length {
        line: ab,
        value: 10.0,
    })
    .unwrap();
    run_case(
        "equilateral",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, c), [5.0, 5.0 * 3f64.sqrt()], 1e-9)
        })),
    );
}

#[test]
fn corpus_06_circle_center_and_radius() {
    let mut s = System::new();
    let c = s.point2(1.0, 2.0);
    let k = s.circle(c, 3.0).unwrap();
    s.constrain(Constraint::Fix(c)).unwrap();
    s.constrain(Constraint::Radius {
        curve: k,
        value: 5.0,
    })
    .unwrap();
    run_case("circle radius", s, Expect::well(), None);
}

#[test]
fn corpus_07_circle_diameter_and_reference_radius() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let k = s.circle(c, 3.0).unwrap();
    s.constrain(Constraint::Fix(c)).unwrap();
    s.constrain(Constraint::Diameter {
        curve: k,
        value: 12.0,
    })
    .unwrap();
    let rr = s
        .reference(Constraint::Radius {
            curve: k,
            value: 1.0,
        })
        .unwrap();
    run_case(
        "circle diameter",
        s,
        Expect::well(),
        Some(Box::new(move |_, r| {
            assert_close(r.measurement(rr).unwrap(), 6.0, 1e-9)
        })),
    );
}

#[test]
fn corpus_08_obround_slot() {
    let (l, r) = (30.0, 6.0);
    let mut s = System::new();
    let c1 = s.point2(0.0, 0.0);
    let c2 = s.point2(l + 2.0, 0.5);
    let p1 = s.point2(0.3, r + 0.4);
    let p2 = s.point2(-0.2, -r + 0.3);
    let p3 = s.point2(l + 1.0, -r - 0.5);
    let p4 = s.point2(l + 2.5, r + 0.6);
    let left = s.arc(c1, p1, p2).unwrap();
    let right = s.arc(c2, p3, p4).unwrap();
    let bottom = s.line(p2, p3).unwrap();
    let top = s.line(p4, p1).unwrap();
    s.constrain(Constraint::Fix(c1)).unwrap();
    s.constrain(Constraint::HorizontalPoints(c1, c2)).unwrap();
    s.constrain(Constraint::TangentAtEnd {
        a: bottom,
        a_end: CurveEnd::Start,
        b: left,
        b_end: CurveEnd::End,
    })
    .unwrap();
    s.constrain(Constraint::TangentAtEnd {
        a: bottom,
        a_end: CurveEnd::End,
        b: right,
        b_end: CurveEnd::Start,
    })
    .unwrap();
    s.constrain(Constraint::TangentAtEnd {
        a: top,
        a_end: CurveEnd::Start,
        b: right,
        b_end: CurveEnd::End,
    })
    .unwrap();
    s.constrain(Constraint::TangentAtEnd {
        a: top,
        a_end: CurveEnd::End,
        b: left,
        b_end: CurveEnd::Start,
    })
    .unwrap();
    s.constrain(Constraint::EqualRadius(left, right)).unwrap();
    s.constrain(Constraint::Distance {
        a: c1,
        b: c2,
        value: l,
    })
    .unwrap();
    s.constrain(Constraint::Radius {
        curve: left,
        value: r,
    })
    .unwrap();
    run_case(
        "obround",
        s,
        Expect::well(),
        Some(Box::new(move |s, rep| {
            assert_pt(pt(rep, s, p1), [0.0, r], 1e-8);
            assert_pt(pt(rep, s, p2), [0.0, -r], 1e-8);
            assert_pt(pt(rep, s, p3), [l, -r], 1e-8);
            assert_pt(pt(rep, s, p4), [l, r], 1e-8);
        })),
    );
}

#[test]
fn corpus_09_concentric_circles() {
    let mut s = System::new();
    let c1 = s.point2(0.0, 0.0);
    let c2 = s.point2(0.4, -0.3);
    let k1 = s.circle(c1, 5.0).unwrap();
    let k2 = s.circle(c2, 2.0).unwrap();
    s.constrain(Constraint::Fix(k1)).unwrap();
    s.constrain(Constraint::Concentric(k1, k2)).unwrap();
    s.constrain(Constraint::Radius {
        curve: k2,
        value: 2.5,
    })
    .unwrap();
    run_case(
        "concentric",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, c2), [0.0, 0.0], 1e-12)
        })),
    );
}

#[test]
fn corpus_10_external_tangent_circles() {
    let mut s = System::new();
    let c1 = s.point2(0.0, 0.0);
    let c2 = s.point2(9.0, 0.7);
    let k1 = s.circle(c1, 4.0).unwrap();
    let k2 = s.circle(c2, 3.0).unwrap();
    s.constrain(Constraint::Fix(k1)).unwrap();
    s.constrain(Constraint::Radius {
        curve: k2,
        value: 3.0,
    })
    .unwrap();
    s.constrain(Constraint::HorizontalPoints(c1, c2)).unwrap();
    s.constrain(Constraint::Tangent(k1, k2)).unwrap();
    run_case(
        "external tangency",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, c2), [7.0, 0.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_11_internal_tangent_circles() {
    let mut s = System::new();
    let c1 = s.point2(0.0, 0.0);
    let c2 = s.point2(1.5, 0.3);
    let k1 = s.circle(c1, 5.0).unwrap();
    let k2 = s.circle(c2, 2.0).unwrap();
    s.constrain(Constraint::Fix(k1)).unwrap();
    s.constrain(Constraint::Radius {
        curve: k2,
        value: 2.0,
    })
    .unwrap();
    s.constrain(Constraint::HorizontalPoints(c1, c2)).unwrap();
    s.constrain(Constraint::Tangent(k2, k1)).unwrap();
    run_case(
        "internal tangency",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, c2), [3.0, 0.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_12_line_tangent_to_circle() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let k = s.circle(c, 4.0).unwrap();
    let a = s.point2(0.5, 4.6);
    let b = s.point2(9.0, 4.2);
    let l = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(k)).unwrap();
    s.constrain(Constraint::Horizontal(l)).unwrap();
    s.constrain(Constraint::Tangent(l, k)).unwrap();
    s.constrain(Constraint::VerticalPoints(a, c)).unwrap();
    s.constrain(Constraint::Length {
        line: l,
        value: 10.0,
    })
    .unwrap();
    run_case(
        "line tangent to circle",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, a), [0.0, 4.0], 1e-9);
            assert_pt(pt(r, s, b), [10.0, 4.0], 1e-9);
        })),
    );
}

#[test]
fn corpus_13_midpoint() {
    let mut s = System::new();
    let l = s.line_xy([0.0, 0.0], [10.0, 4.0]);
    let m = s.point2(3.0, 3.0);
    s.constrain(Constraint::Fix(l)).unwrap();
    s.constrain(Constraint::Midpoint { point: m, line: l })
        .unwrap();
    run_case(
        "midpoint",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, m), [5.0, 2.0], 1e-10)
        })),
    );
}

#[test]
fn corpus_14_symmetric_about_axis() {
    let mut s = System::new();
    let axis = s.line_xy([0.0, -5.0], [0.0, 5.0]);
    s.set_construction(axis, true);
    let a = s.point2(-3.0, 1.0);
    let b = s.point2(2.0, 1.4);
    s.constrain(Constraint::Fix(axis)).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Symmetric { a, b, axis }).unwrap();
    run_case(
        "symmetric",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, b), [3.0, 1.0], 1e-10)
        })),
    );
}

#[test]
fn corpus_15_collinear_segments() {
    let mut s = System::new();
    let l1 = s.line_xy([0.0, 0.0], [10.0, 0.0]);
    let l2 = s.line_xy([12.5, 0.6], [17.0, -0.4]);
    s.constrain(Constraint::Fix(l1)).unwrap();
    s.constrain(Constraint::Collinear(l1, l2)).unwrap();
    s.constrain(Constraint::Length {
        line: l2,
        value: 5.0,
    })
    .unwrap();
    let EntityKind::Line {
        start: l2a,
        end: l2b,
    } = s.entity(l2).kind
    else {
        unreachable!()
    };
    let EntityKind::Line { end: l1b, .. } = s.entity(l1).kind else {
        unreachable!()
    };
    s.constrain(Constraint::Distance {
        a: l1b,
        b: l2a,
        value: 2.0,
    })
    .unwrap();
    run_case(
        "collinear",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, l2a), [12.0, 0.0], 1e-9);
            assert_pt(pt(r, s, l2b), [17.0, 0.0], 1e-9);
        })),
    );
}

#[test]
fn corpus_16_angle_between_lines() {
    let mut s = System::new();
    let o = s.point2(0.0, 0.0);
    let a = s.point2(10.0, 0.0);
    let b = s.point2(4.0, 3.0);
    let l1 = s.line(o, a).unwrap();
    let l2 = s.line(o, b).unwrap();
    s.constrain(Constraint::Fix(l1)).unwrap();
    s.constrain(Constraint::Angle {
        a: l1,
        b: l2,
        value: PI / 6.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: l2,
        value: 6.0,
    })
    .unwrap();
    run_case(
        "angle",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, b), [6.0 * (PI / 6.0).cos(), 3.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_17_perpendicular_l_bracket() {
    let mut s = System::new();
    let o = s.point2(0.0, 0.0);
    let a = s.point2(20.0, 1.0);
    let b = s.point2(-1.0, 12.0);
    let l1 = s.line(o, a).unwrap();
    let l2 = s.line(o, b).unwrap();
    s.constrain(Constraint::Fix(o)).unwrap();
    s.constrain(Constraint::Horizontal(l1)).unwrap();
    s.constrain(Constraint::Perpendicular(l1, l2)).unwrap();
    s.constrain(Constraint::Length {
        line: l1,
        value: 20.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: l2,
        value: 12.0,
    })
    .unwrap();
    run_case(
        "L bracket",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, b), [0.0, 12.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_18_parallel_at_distance() {
    let mut s = System::new();
    let l1 = s.line_xy([0.0, 0.0], [10.0, 0.0]);
    let a = s.point2(0.4, 4.6);
    let b = s.point2(8.0, 5.5);
    let l2 = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(l1)).unwrap();
    s.constrain(Constraint::Parallel(l1, l2)).unwrap();
    s.constrain(Constraint::Distance {
        a,
        b: l1,
        value: 5.0,
    })
    .unwrap();
    let EntityKind::Line { start: l1a, .. } = s.entity(l1).kind else {
        unreachable!()
    };
    s.constrain(Constraint::VerticalPoints(a, l1a)).unwrap();
    s.constrain(Constraint::Length {
        line: l2,
        value: 8.0,
    })
    .unwrap();
    run_case(
        "parallel offset",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, b), [8.0, 5.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_19_point_on_circle_level_with_center() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let k = s.circle(c, 5.0).unwrap();
    let p = s.point2(-4.0, 0.6);
    s.constrain(Constraint::Fix(k)).unwrap();
    s.constrain(Constraint::PointOnCurve { point: p, curve: k })
        .unwrap();
    s.constrain(Constraint::HorizontalPoints(p, c)).unwrap();
    run_case(
        "point on circle",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, p), [-5.0, 0.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_20_arc_fully_defined() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let st = s.point2(5.0, 0.0);
    let en = s.point2(0.3, 4.4);
    s.arc(c, st, en).unwrap();
    s.constrain(Constraint::Fix(c)).unwrap();
    s.constrain(Constraint::Fix(st)).unwrap();
    s.constrain(Constraint::VerticalPoints(c, en)).unwrap();
    run_case(
        "arc",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, en), [0.0, 5.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_21_ellipse_and_point() {
    let mut s = System::new();
    let c = s.point2(1.0, 1.0);
    let e = s.ellipse(c, 4.5, 2.2, 0.1).unwrap();
    let EntityKind::Ellipse {
        semi_major,
        semi_minor,
        rotation,
        ..
    } = s.entity(e).kind
    else {
        unreachable!()
    };
    let p = s.point2(5.5, 1.2);
    s.constrain(Constraint::Fix(c)).unwrap();
    s.constrain(Constraint::VarValue {
        var: semi_major,
        value: 5.0,
    })
    .unwrap();
    s.constrain(Constraint::VarValue {
        var: semi_minor,
        value: 2.0,
    })
    .unwrap();
    s.constrain(Constraint::VarValue {
        var: rotation,
        value: 0.0,
    })
    .unwrap();
    s.constrain(Constraint::PointOnCurve { point: p, curve: e })
        .unwrap();
    s.constrain(Constraint::HorizontalPoints(p, c)).unwrap();
    run_case(
        "ellipse",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, p), [6.0, 1.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_22_point_on_fixed_spline() {
    let mut s = System::new();
    let cps: Vec<EntityId> = [[0.0, 0.0], [2.0, 4.0], [6.0, 4.0], [8.0, 0.0]]
        .iter()
        .map(|p| s.point2(p[0], p[1]))
        .collect();
    let sp = s.bspline(3, &cps).unwrap();
    for &c in &cps {
        s.constrain(Constraint::Fix(c)).unwrap();
    }
    let p = s.point2(3.9, 2.5);
    s.constrain(Constraint::PointOnCurve {
        point: p,
        curve: sp,
    })
    .unwrap();
    let EntityKind::Point2 { x, .. } = s.entity(p).kind else {
        unreachable!()
    };
    s.constrain(Constraint::VarValue { var: x, value: 4.0 })
        .unwrap();
    // A symmetric cubic Bézier: at x = 4 (t = 0.5) y = 3.
    run_case(
        "point on spline",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, p), [4.0, 3.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_23_g2_spline_into_arc() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let a0 = s.point2(10.0, 0.0);
    let a1 = s.point2(0.0, 10.0);
    let arc = s.arc(c, a0, a1).unwrap();
    s.constrain(Constraint::Fix(arc)).unwrap();
    let cps: Vec<EntityId> = [[0.2, 10.1], [-4.0, 10.3], [-8.0, 8.0], [-12.0, 4.0]]
        .iter()
        .map(|p| s.point2(p[0], p[1]))
        .collect();
    let sp = s.bspline(3, &cps).unwrap();
    s.constrain(Constraint::Fix(cps[2])).unwrap();
    s.constrain(Constraint::Fix(cps[3])).unwrap();
    s.constrain(Constraint::Coincident(cps[0], a1)).unwrap();
    s.constrain(Constraint::CurvatureContinuity {
        spline: sp,
        end: CurveEnd::Start,
        arc,
    })
    .unwrap();
    let p1 = cps[1];
    run_case(
        "G2 spline-arc",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            // κ(0) = (p-1)/p · cross(P1-P0, P2-P0)/|P1-P0|³ = 1/10 → |P1-P0| = sqrt(40/3).
            assert_pt(pt(r, s, p1), [-(40.0f64 / 3.0).sqrt(), 10.0], 1e-8);
        })),
    );
}

#[test]
fn corpus_24_tangent_arc_off_line_end() {
    let mut s = System::new();
    let l = s.line_xy([-10.0, 0.0], [0.0, 0.0]);
    let EntityKind::Line { end: le, .. } = s.entity(l).kind else {
        unreachable!()
    };
    let c = s.point2(0.3, 4.8);
    let st = s.point2(0.2, 0.1);
    let en = s.point2(5.2, 5.1);
    let arc = s.arc(c, st, en).unwrap();
    s.constrain(Constraint::Fix(l)).unwrap();
    s.constrain(Constraint::Coincident(st, le)).unwrap();
    s.constrain(Constraint::TangentAtEnd {
        a: l,
        a_end: CurveEnd::End,
        b: arc,
        b_end: CurveEnd::Start,
    })
    .unwrap();
    s.constrain(Constraint::Radius {
        curve: arc,
        value: 5.0,
    })
    .unwrap();
    s.constrain(Constraint::HorizontalPoints(c, en)).unwrap();
    run_case(
        "tangent arc",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, c), [0.0, 5.0], 1e-9);
            assert_pt(pt(r, s, en), [5.0, 5.0], 1e-9);
        })),
    );
}

#[test]
fn corpus_25_regular_hexagon_on_construction_circle() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let k = s.circle(c, 9.5).unwrap();
    s.set_construction(k, true);
    let pts: Vec<EntityId> = (0..6)
        .map(|i| {
            let a = i as f64 * PI / 3.0 + 0.05 * (i as f64).sin();
            let r = 10.0 + 0.3 * (i as f64).cos();
            s.point2(r * a.cos(), r * a.sin())
        })
        .collect();
    let lines: Vec<EntityId> = (0..6)
        .map(|i| s.line(pts[i], pts[(i + 1) % 6]).unwrap())
        .collect();
    s.constrain(Constraint::Fix(c)).unwrap();
    s.constrain(Constraint::Radius {
        curve: k,
        value: 10.0,
    })
    .unwrap();
    for &p in &pts {
        s.constrain(Constraint::PointOnCurve { point: p, curve: k })
            .unwrap();
    }
    for i in 1..6 {
        s.constrain(Constraint::EqualLength(lines[0], lines[i]))
            .unwrap();
    }
    s.constrain(Constraint::HorizontalPoints(pts[0], c))
        .unwrap();
    let p1 = pts[1];
    // Five chord equalities fix the five independent central angles; the horizontal pins the
    // rotation: 12 rows, 12 unknowns, no redundancy.
    run_case(
        "hexagon",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, p1), [5.0, 5.0 * 3f64.sqrt()], 1e-9)
        })),
    );
}

#[test]
fn corpus_26_coradial_arc() {
    let mut s = System::new();
    let c0 = s.point2(0.0, 0.0);
    let k = s.circle(c0, 6.0).unwrap();
    let c = s.point2(0.5, 0.2);
    let st = s.point2(6.2, 0.4);
    let en = s.point2(0.1, 5.4);
    let arc = s.arc(c, st, en).unwrap();
    s.constrain(Constraint::Fix(k)).unwrap();
    s.constrain(Constraint::Coradial(k, arc)).unwrap();
    s.constrain(Constraint::HorizontalPoints(st, c)).unwrap();
    s.constrain(Constraint::VerticalPoints(en, c)).unwrap();
    run_case(
        "coradial",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, st), [6.0, 0.0], 1e-9);
            assert_pt(pt(r, s, en), [0.0, 6.0], 1e-9);
        })),
    );
}

#[test]
fn corpus_27_distance_point_point() {
    let mut s = System::new();
    let a = s.point2(1.0, 1.0);
    let b = s.point2(4.0, 1.5);
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Distance { a, b, value: 5.0 })
        .unwrap();
    s.constrain(Constraint::HorizontalPoints(a, b)).unwrap();
    run_case(
        "distance",
        s,
        Expect::well(),
        Some(Box::new(move |s, r| {
            assert_pt(pt(r, s, b), [6.0, 1.0], 1e-9)
        })),
    );
}

#[test]
fn corpus_28_grounded_by_flags() {
    // Grounding through variable flags instead of Fix constraints.
    let mut s = System::new();
    let r = rect(&mut s, 0.0, 0.0, 9.0, 4.0);
    s.set_entity_flag(r.p[0], VarFlag::Fixed);
    square_up(&mut s, &r);
    s.constrain(Constraint::Length {
        line: r.l[0],
        value: 10.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: r.l[3],
        value: 5.0,
    })
    .unwrap();
    let p2 = r.p[2];
    run_case(
        "flag-grounded rectangle",
        s,
        Expect::well(),
        Some(Box::new(move |s, rep| {
            assert_pt(pt(rep, s, p2), [10.0, 5.0], 1e-9)
        })),
    );
}

// ============================================================================ under-defined

#[test]
fn corpus_29_free_point() {
    let mut s = System::new();
    s.point2(1.0, 2.0);
    run_case("free point", s, Expect::under(2), None);
}

#[test]
fn corpus_30_free_and_horizontal_line() {
    let mut s = System::new();
    s.line_xy([0.0, 0.0], [3.0, 1.0]);
    let l = s.line_xy([0.0, 5.0], [3.0, 6.0]);
    s.constrain(Constraint::Horizontal(l)).unwrap();
    run_case("lines", s, Expect::under(7), None);
}

#[test]
fn corpus_31_rectangle_without_dimensions() {
    let mut s = System::new();
    let r = rect(&mut s, 0.0, 0.0, 5.0, 3.0);
    s.constrain(Constraint::Fix(r.p[0])).unwrap();
    square_up(&mut s, &r);
    let (p0, p1, p2) = (r.p[0], r.p[1], r.p[2]);
    run_case(
        "undimensioned rectangle",
        s,
        Expect::under(2),
        Some(Box::new(move |_, rep| {
            assert_eq!(rep.entity_dof(p0), Some(0));
            assert_eq!(rep.entity_dof(p1), Some(1));
            assert_eq!(rep.entity_dof(p2), Some(2));
        })),
    );
}

#[test]
fn corpus_32_floating_rectangle() {
    let mut s = System::new();
    let r = rect(&mut s, 0.0, 0.0, 5.0, 3.0);
    square_up(&mut s, &r);
    s.constrain(Constraint::Length {
        line: r.l[0],
        value: 5.0,
    })
    .unwrap();
    run_case("floating rectangle", s, Expect::under(3), None);
}

#[test]
fn corpus_33_free_circle_and_arc() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    s.circle(c, 2.0).unwrap();
    let ac = s.point2(10.0, 0.0);
    let a0 = s.point2(12.0, 0.0);
    let a1 = s.point2(10.0, 2.0);
    let arc = s.arc(ac, a0, a1).unwrap();
    run_case(
        "free circle and arc",
        s,
        Expect::under(8),
        Some(Box::new(move |_, r| assert_eq!(r.entity_dof(arc), Some(5)))),
    );
}

#[test]
fn corpus_34_triangle_with_one_fixed_corner() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(4.0, 0.0);
    let c = s.point2(1.0, 3.0);
    let ab = s.line(a, b).unwrap();
    s.line(b, c).unwrap();
    s.line(c, a).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Length {
        line: ab,
        value: 4.0,
    })
    .unwrap();
    run_case("triangle one corner", s, Expect::under(3), None);
}

#[test]
fn corpus_35_rotating_link() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(3.0, 4.0);
    let l = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Length {
        line: l,
        value: 5.0,
    })
    .unwrap();
    run_case(
        "rotating link",
        s,
        Expect::under(1),
        Some(Box::new(move |_, r| {
            assert_eq!(r.entity_dof(a), Some(0));
            assert_eq!(r.entity_dof(b), Some(1));
            assert_eq!(r.entity_dof(l), Some(1));
        })),
    );
}

#[test]
fn corpus_36_point_on_fixed_circle() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let k = s.circle(c, 5.0).unwrap();
    let p = s.point2(3.0, 4.2);
    s.constrain(Constraint::Fix(k)).unwrap();
    s.constrain(Constraint::PointOnCurve { point: p, curve: k })
        .unwrap();
    run_case("point on circle", s, Expect::under(1), None);
}

#[test]
fn corpus_37_free_ellipse_and_spline() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    s.ellipse(c, 3.0, 1.0, 0.2).unwrap();
    let cps: Vec<EntityId> = (0..5).map(|i| s.point2(i as f64, (i % 2) as f64)).collect();
    let sp = s.bspline(3, &cps).unwrap();
    let p = s.point2(2.0, 0.6);
    s.constrain(Constraint::PointOnCurve {
        point: p,
        curve: sp,
    })
    .unwrap();
    // ellipse 5 + spline 10 + (point 2 + parameter 1 - 2 rows) = 16
    run_case(
        "free ellipse and spline",
        s,
        Expect::under(16),
        Some(Box::new(move |_, r| assert_eq!(r.entity_dof(sp), Some(10)))),
    );
}

#[test]
fn corpus_38_planar_four_bar() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let d = s.point2(4.0, 0.0);
    let b = s.point2(0.1, 1.05);
    let c = s.point2(3.85, 2.0);
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Fix(d)).unwrap();
    let crank = s.line(a, b).unwrap();
    let coupler = s.line(b, c).unwrap();
    let rocker = s.line(c, d).unwrap();
    s.constrain(Constraint::Length {
        line: crank,
        value: 1.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: coupler,
        value: 4.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: rocker,
        value: 2.0,
    })
    .unwrap();
    run_case(
        "four-bar",
        s,
        Expect::under(1),
        Some(Box::new(move |_, r| {
            assert_eq!(r.entity_dof(b), Some(1));
            assert_eq!(r.entity_dof(c), Some(1));
            assert_eq!(r.entity_dof(coupler), Some(1));
        })),
    );
}

#[test]
fn corpus_39_planar_slider_crank() {
    let mut s = System::new();
    let o = s.point2(0.0, 0.0);
    let a = s.point2(0.7, 0.75);
    let b = s.point2(3.6, 0.1);
    s.constrain(Constraint::Fix(o)).unwrap();
    let crank = s.line(o, a).unwrap();
    let rod = s.line(a, b).unwrap();
    s.constrain(Constraint::Length {
        line: crank,
        value: 1.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: rod,
        value: 3.0,
    })
    .unwrap();
    s.constrain(Constraint::HorizontalPoints(o, b)).unwrap();
    run_case("slider-crank", s, Expect::under(1), None);
}

#[test]
fn corpus_40_partially_dimensioned_rectangle() {
    let mut s = System::new();
    let r = rect(&mut s, 0.0, 0.0, 5.0, 3.0);
    s.constrain(Constraint::Fix(r.p[0])).unwrap();
    square_up(&mut s, &r);
    s.constrain(Constraint::Length {
        line: r.l[0],
        value: 6.0,
    })
    .unwrap();
    let p = r.p;
    let l = r.l;
    run_case(
        "width only",
        s,
        Expect::under(1),
        Some(Box::new(move |s, rep| {
            assert_eq!(rep.entity_dof(p[1]), Some(0));
            assert_eq!(rep.entity_dof(p[2]), Some(1));
            assert_eq!(rep.entity_dof(l[0]), Some(0));
            assert_eq!(rep.entity_dof(l[2]), Some(1));
            let v = s.entity_vars(p[2]);
            assert!(rep.determined[v[0].index()]);
            assert!(!rep.determined[v[1].index()]);
        })),
    );
}

// ============================================================================ over-defined

#[test]
fn corpus_41_rectangle_with_extra_parallel() {
    let (mut s, r, ids) = fixed_rect(40.0, 20.0);
    let par = s.constrain(Constraint::Parallel(r.l[0], r.l[2])).unwrap();
    let _ = ids;
    run_case("extra parallel", s, Expect::over(0, vec![par]), None);
}

#[test]
fn corpus_42_duplicate_horizontal() {
    let mut s = System::new();
    let l = s.line_xy([0.0, 0.0], [5.0, 1.0]);
    let h1 = s.constrain(Constraint::Horizontal(l)).unwrap();
    let h2 = s.constrain(Constraint::Horizontal(l)).unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(
        rep.redundant,
        vec![DependentConstraint {
            constraint: h2,
            depends_on: vec![h1]
        }]
    );
    run_case("duplicate horizontal", s, Expect::over(3, vec![h2]), None);
}

#[test]
fn corpus_43_equal_lengths_already_dimensioned() {
    let (mut s, r, _) = fixed_rect(40.0, 20.0);
    let w2 = s
        .constrain(Constraint::Length {
            line: r.l[2],
            value: 40.0,
        })
        .unwrap();
    let eq = s
        .constrain(Constraint::EqualLength(r.l[1], r.l[3]))
        .unwrap();
    run_case("dimensioned twice", s, Expect::over(0, vec![w2, eq]), None);
}

#[test]
fn corpus_44_horizontal_on_grounded_line() {
    let mut s = System::new();
    let l = s.line_xy([0.0, 2.0], [5.0, 2.0]);
    s.set_entity_flag(l, VarFlag::Fixed);
    let h = s.constrain(Constraint::Horizontal(l)).unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(
        rep.redundant,
        vec![DependentConstraint {
            constraint: h,
            depends_on: vec![]
        }]
    );
    run_case(
        "horizontal on grounded line",
        s,
        Expect::over(0, vec![h]),
        None,
    );
}

#[test]
fn corpus_45_triangle_with_consistent_angle() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(3.1, 0.2);
    let c = s.point2(2.8, 4.1);
    let ab = s.line(a, b).unwrap();
    let bc = s.line(b, c).unwrap();
    let ca = s.line(c, a).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Horizontal(ab)).unwrap();
    s.constrain(Constraint::Length {
        line: ab,
        value: 3.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: bc,
        value: 4.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: ca,
        value: 5.0,
    })
    .unwrap();
    let perp = s.constrain(Constraint::Perpendicular(ab, bc)).unwrap();
    run_case(
        "3-4-5 with right angle",
        s,
        Expect::over(0, vec![perp]),
        None,
    );
}

// ============================================================================ conflicting

#[test]
fn corpus_46_two_lengths_on_one_line() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(12.0, 0.0);
    let l = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Horizontal(l)).unwrap();
    let l10 = s
        .constrain(Constraint::Length {
            line: l,
            value: 10.0,
        })
        .unwrap();
    let l20 = s
        .constrain(Constraint::Length {
            line: l,
            value: 20.0,
        })
        .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.conflicting.len(), 1);
    assert_eq!(rep.conflicting[0].constraint, l20);
    assert!(rep.conflicting[0].depends_on.contains(&l10));
    assert!(rep.residual_norm > 1.0);
    run_case("two lengths", s, Expect::conflict(vec![l10, l20]), None);
}

#[test]
fn corpus_47_horizontal_vertical_and_length() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(3.0, 4.0);
    let l = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    let h = s.constrain(Constraint::Horizontal(l)).unwrap();
    let v = s.constrain(Constraint::Vertical(l)).unwrap();
    let len = s
        .constrain(Constraint::Length {
            line: l,
            value: 10.0,
        })
        .unwrap();
    run_case("h + v + length", s, Expect::conflict(vec![len, h, v]), None);
}

#[test]
fn corpus_48_fixed_points_wrong_distance() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(3.0, 0.0);
    let fa = s.constrain(Constraint::Fix(a)).unwrap();
    let fb = s.constrain(Constraint::Fix(b)).unwrap();
    let d = s
        .constrain(Constraint::Distance { a, b, value: 5.0 })
        .unwrap();
    run_case(
        "fixed points, wrong distance",
        s,
        Expect::conflict(vec![d, fa, fb]),
        None,
    );
}

#[test]
fn corpus_49_triangle_inequality_violated() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(10.0, 0.0);
    let c = s.point2(4.0, 1.0);
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Fix(b)).unwrap();
    let ac = s
        .constrain(Constraint::Distance {
            a,
            b: c,
            value: 3.0,
        })
        .unwrap();
    let bc = s
        .constrain(Constraint::Distance {
            a: b,
            b: c,
            value: 4.0,
        })
        .unwrap();
    run_case("3 + 4 < 10", s, Expect::conflict(vec![ac, bc]), None);
}

#[test]
fn corpus_50_rectangle_with_wrong_diagonal() {
    let (mut s, r, ids) = fixed_rect(40.0, 20.0);
    let diag = s
        .constrain(Constraint::Distance {
            a: r.p[0],
            b: r.p[2],
            value: 50.0,
        })
        .unwrap();
    let rep = solve(&s, &opts());
    let set = rep.conflict_set();
    // The diagonal fights the width and height.
    assert!(set.contains(&ids[5]) && set.contains(&ids[6]), "{set:?}");
    run_case("wrong diagonal", s, Expect::conflict(vec![diag]), None);
}

#[test]
fn corpus_51_concentric_but_apart() {
    let mut s = System::new();
    let c1 = s.point2(0.0, 0.0);
    let c2 = s.point2(4.0, 0.0);
    let k1 = s.circle(c1, 2.0).unwrap();
    let k2 = s.circle(c2, 1.0).unwrap();
    s.constrain(Constraint::Fix(c1)).unwrap();
    let con = s.constrain(Constraint::Concentric(k1, k2)).unwrap();
    let d = s
        .constrain(Constraint::Distance {
            a: c1,
            b: c2,
            value: 4.0,
        })
        .unwrap();
    run_case(
        "concentric and apart",
        s,
        Expect::conflict(vec![con, d]),
        None,
    );
}

// ============================================================================ drag

fn drag_case(s: &System, req: &DragRequest) -> SolveReport {
    let rep = drag(s, req, &opts());
    assert!(
        rep.converged,
        "drag did not converge: {}",
        rep.residual_norm
    );
    rep
}

#[test]
fn corpus_52_drag_free_point() {
    let mut s = System::new();
    let p = s.point2(0.0, 0.0);
    let rep = drag_case(&s, &DragRequest::point(&s, p, [3.0, -2.0]));
    assert_pt(pt(&rep, &s, p), [3.0, -2.0], 1e-12);
    assert_eq!(rep.drag.unwrap().distance_to_target, 0.0);
}

#[test]
fn corpus_53_drag_link_end() {
    let mut s = System::new();
    let a = s.point2(1.0, 1.0);
    let b = s.point2(6.0, 1.0);
    let l = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Length {
        line: l,
        value: 5.0,
    })
    .unwrap();
    let target = [1.0 + 3.0, 1.0 + 9.0];
    let rep = drag_case(&s, &DragRequest::point(&s, b, target));
    let n = 3f64.hypot(9.0);
    assert_pt(
        pt(&rep, &s, b),
        [1.0 + 5.0 * 3.0 / n, 1.0 + 5.0 * 9.0 / n],
        1e-9,
    );
    assert_pt(pt(&rep, &s, a), [1.0, 1.0], 1e-12);
}

#[test]
fn corpus_54_drag_point_on_circle() {
    let mut s = System::new();
    let c = s.point2(0.0, 0.0);
    let k = s.circle(c, 5.0).unwrap();
    let p = s.point2(5.0, 0.0);
    s.constrain(Constraint::Fix(k)).unwrap();
    s.constrain(Constraint::PointOnCurve { point: p, curve: k })
        .unwrap();
    let rep = drag_case(&s, &DragRequest::point(&s, p, [-6.0, 8.0]));
    assert_pt(pt(&rep, &s, p), [-3.0, 4.0], 1e-9);
    assert_close(rep.drag.unwrap().distance_to_target, 5.0, 1e-9);
}

#[test]
fn corpus_55_drag_rectangle_corner() {
    let mut s = System::new();
    let r = rect(&mut s, 0.0, 0.0, 5.0, 3.0);
    s.constrain(Constraint::Fix(r.p[0])).unwrap();
    square_up(&mut s, &r);
    let rep = drag_case(&s, &DragRequest::point(&s, r.p[2], [8.0, 6.0]));
    assert_pt(pt(&rep, &s, r.p[2]), [8.0, 6.0], 1e-9);
    assert_pt(pt(&rep, &s, r.p[1]), [8.0, 0.0], 1e-9);
    assert_pt(pt(&rep, &s, r.p[3]), [0.0, 6.0], 1e-9);
}

#[test]
fn corpus_56_drag_four_bar_coupler() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let d = s.point2(4.0, 0.0);
    let b = s.point2(0.0, 1.0);
    let c = s.point2(3.874, 1.997);
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Fix(d)).unwrap();
    s.constrain(Constraint::Distance { a, b, value: 1.0 })
        .unwrap();
    s.constrain(Constraint::Distance {
        a: b,
        b: c,
        value: 4.0,
    })
    .unwrap();
    s.constrain(Constraint::Distance {
        a: c,
        b: d,
        value: 2.0,
    })
    .unwrap();
    let rep = drag_case(&s, &DragRequest::point(&s, b, [0.8, 0.9]));
    let (pb, pc) = (pt(&rep, &s, b), pt(&rep, &s, c));
    assert_close(dist(pb, [0.0, 0.0]), 1.0, 1e-9);
    assert_close(dist(pb, pc), 4.0, 1e-9);
    assert_close(dist(pc, [4.0, 0.0]), 2.0, 1e-9);
    // The crank moved toward the target direction.
    assert!(pb[0] > 0.4, "{pb:?}");
}

#[test]
fn corpus_57_drag_respects_fixed_geometry() {
    let (s, r, _) = fixed_rect(10.0, 5.0);
    let before = solve(&s, &opts());
    let rep = drag_case(&s, &DragRequest::point(&s, r.p[2], [30.0, 30.0]));
    assert_eq!(rep.status, Status::WellDefined);
    assert_pt(pt(&rep, &s, r.p[2]), pt(&before, &s, r.p[2]), 1e-9);
}
