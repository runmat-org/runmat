//! Ellipses and elliptical arcs in sketches: the text form round-trips, the solver holds their
//! semi-axes, angle and tangency, and the solved ellipse becomes an exact kernel segment.

use runmat_cad::model::sketch::*;
use runmat_cad::model::FeatureKind;
use runmat_cad::sketch_solver::{ConstraintSketchSolver, SketchSolver};
use runmat_cad::syntax::{parse_part, print_part};
use runmat_cad::units::Units;
use runmat_cad::value::{Evaluator, NoDynamicValues, ValueEnv};
use runmat_geometry_kernel::op::SegmentKind;

fn part(body: &str) -> String {
    format!("runmat 0.7\nstd 1\n\npart p {{\n    sketch s1 on origin.xy {{\n{body}    }}\n}}\n")
}

fn sketch_of(text: &str) -> Sketch {
    let parsed = parse_part("p.part", text);
    assert!(
        parsed.diagnostics.is_empty(),
        "{text}\n{:#?}",
        parsed.diagnostics
    );
    let document = parsed.document.expect("parses");
    match &document.features[0].kind {
        FeatureKind::Sketch(s) => s.clone(),
        other => panic!("{other:?}"),
    }
}

fn solve(sketch: &Sketch) -> Result<SolvedSketch, SketchSolveFailure> {
    let units = Units;
    let ev = Evaluator {
        units: &units,
        dynamic: &NoDynamicValues,
    };
    ConstraintSketchSolver.solve(sketch, &ev, &ValueEnv::default())
}

#[test]
fn an_ellipse_and_an_elliptical_arc_print_as_they_are_written() {
    let text = part(concat!(
        "        e1 = ellipse(center = [10 mm, 5 mm], major = 20 mm, minor = 8 mm, angle = 30 deg)\n",
        "        e2 = ellipse(center = [0 mm, 0 mm], major = 6 mm, minor = 3 mm, start = [6 mm, 0 mm], end = [-6 mm, 0 mm], construction)\n",
        "        l1 = line(start = e2.end, end = e2.start)\n",
        "        constrain major_radius(e1) = 20 mm\n",
        "        constrain minor_radius(e1, id = m1) = 8 mm\n",
        "        constrain limit(major_radius(e2, value = 6 mm), min = 5 mm, max = 7 mm)\n",
        "        constrain horizontal(e2)\n",
        "        solved {\n",
        "            e1.center = [0.01, 0.005]\n",
        "            e1.major = 0.02\n",
        "            e1.minor = 0.008\n",
        "            e1.angle = 0.5235987755982988\n",
        "        }\n",
    ));
    let parsed = parse_part("p.part", &text);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    let document = parsed.document.unwrap();
    let printed = print_part(&document, &parsed.uses).unwrap();
    assert_eq!(printed, text);
    let s = sketch_of(&text);
    let SketchGeometry::Ellipse {
        start, end, angle, ..
    } = &s.entities[1].geometry
    else {
        panic!()
    };
    assert!(start.is_some() && end.is_some() && angle.is_none());
    assert_eq!(
        s.solved.scalars[&SolvedKey::Ellipse {
            entity: "e1".into(),
            measure: EllipseMeasure::Major
        }],
        0.02
    );
}

#[test]
fn an_ellipse_needs_both_ends_or_neither() {
    let text = part("        e1 = ellipse(center = [0 mm, 0 mm], major = 6 mm, minor = 3 mm, start = [6 mm, 0 mm])\n");
    let parsed = parse_part("p.part", &text);
    let message = format!("{:?}", parsed.diagnostics);
    assert!(message.contains("both `start` and `end`"), "{message}");
    let text = part("        e1 = ellipse(center = [0 mm, 0 mm], major = 6 mm)\n");
    let message = format!("{:?}", parse_part("p.part", &text).diagnostics);
    assert!(message.contains("`ellipse` needs `minor = …`"), "{message}");
}

#[test]
fn dimensions_and_a_direction_define_an_ellipse() {
    let s = sketch_of(&part(concat!(
        "        e1 = ellipse(center = [3 mm, 4 mm], major = 18 mm, minor = 9 mm, angle = 10 deg)\n",
        "        constrain fix(e1.center)\n",
        "        constrain major_radius(e1) = 25 mm\n",
        "        constrain minor_radius(e1) = 10 mm\n",
        "        constrain horizontal(e1)\n",
    )));
    let solved = solve(&s).expect("solves");
    assert_eq!(solved.status, SketchSolveStatus::WellDefined, "{solved:?}");
    assert_eq!(solved.dof, Some(0));
    let e = solved.ellipses[&"e1".into()];
    assert!((e.major - 0.025).abs() < 1e-9, "{e:?}");
    assert!((e.minor - 0.010).abs() < 1e-9, "{e:?}");
    assert!(e.angle.abs() < 1e-9, "{e:?}");
    // The cache carries the semi-axes and angle, and re-solving from it lands in place.
    let mut again = s.clone();
    again.solved = solved.cache.clone();
    assert_eq!(solve(&again).unwrap().ellipses, solved.ellipses);
    // Without the direction the ellipse can still turn: one degree of freedom left.
    let mut free = s.clone();
    free.constraints.pop();
    let solved = solve(&free).unwrap();
    assert_eq!(solved.status, SketchSolveStatus::UnderDefined);
    assert_eq!(solved.dof, Some(1));
    // Vertical turns the major axis a quarter turn.
    let mut upright = free.clone();
    upright.constraints.push(SketchConstraint {
        id: None,
        kind: SketchConstraintKind::Vertical {
            line: SketchTarget::entity("e1"),
        },
    });
    let e = solve(&upright).unwrap().ellipses[&"e1".into()];
    assert!(
        (e.angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
        "{e:?}"
    );
}

#[test]
fn a_line_tangent_to_an_ellipse_touches_it() {
    let s = sketch_of(&part(concat!(
        "        e1 = ellipse(center = [0 mm, 0 mm], major = 20 mm, minor = 10 mm)\n",
        "        l1 = line(start = [-30 mm, 12 mm], end = [30 mm, 12 mm])\n",
        "        constrain fix(e1.center)\n",
        "        constrain major_radius(e1) = 20 mm\n",
        "        constrain minor_radius(e1) = 10 mm\n",
        "        constrain horizontal(e1)\n",
        "        constrain horizontal(l1)\n",
        "        constrain tangent(l1, e1)\n",
    )));
    let solved = solve(&s).expect("solves");
    let y = solved.point(&"l1".into(), PointRole::Start).unwrap()[1];
    assert!(
        (y - 0.010).abs() < 1e-9,
        "the line sits on the top of the ellipse: {y}"
    );
}

#[test]
fn concentric_and_equal_work_with_ellipses() {
    let s = sketch_of(&part(concat!(
        "        c1 = circle(center = [5 mm, 5 mm], radius = 3 mm)\n",
        "        e1 = ellipse(center = [0 mm, 0 mm], major = 20 mm, minor = 10 mm)\n",
        "        e2 = ellipse(center = [40 mm, 0 mm], major = 12 mm, minor = 7 mm)\n",
        "        constrain fix(c1)\n",
        "        constrain concentric(e1, c1)\n",
        "        constrain equal(e1, e2)\n",
        "        constrain major_radius(e2) = 15 mm\n",
    )));
    let solved = solve(&s).expect("solves");
    let c = solved.point(&"e1".into(), PointRole::Center).unwrap();
    assert!(
        (c[0] - 0.005).abs() < 1e-9 && (c[1] - 0.005).abs() < 1e-9,
        "{c:?}"
    );
    let (a, b) = (solved.ellipses[&"e1".into()], solved.ellipses[&"e2".into()]);
    assert!(
        (a.major - 0.015).abs() < 1e-9 && (a.minor - b.minor).abs() < 1e-9,
        "{a:?} {b:?}"
    );
}

#[test]
fn an_elliptical_arc_keeps_its_ends_on_the_ellipse() {
    let s = sketch_of(&part(concat!(
        "        e1 = ellipse(center = [0 mm, 0 mm], major = 20 mm, minor = 10 mm, start = [20 mm, 0 mm], end = [-20 mm, 0 mm])\n",
        "        l1 = line(start = e1.end, end = e1.start)\n",
        "        constrain fix(e1.center)\n",
        "        constrain horizontal(e1)\n",
        "        constrain major_radius(e1) = 30 mm\n",
        "        constrain minor_radius(e1) = 12 mm\n",
        "        constrain horizontal(l1)\n",
    )));
    let solved = solve(&s).expect("solves");
    let (a, b) = (
        solved.point(&"e1".into(), PointRole::Start).unwrap(),
        solved.point(&"e1".into(), PointRole::End).unwrap(),
    );
    for p in [a, b] {
        let on = (p[0] / 0.030).powi(2) + (p[1] / 0.012).powi(2);
        assert!((on - 1.0).abs() < 1e-7, "{p:?} is off the ellipse ({on})");
    }
    // The half ellipse above the line, as an exact segment: the arc runs from start to end.
    let seg = solved.ellipse_segment(&"e1".into(), true).unwrap();
    assert!(matches!(seg.kind, SegmentKind::Ellipse { ratio, .. } if (ratio - 0.4).abs() < 1e-9));
    assert_eq!((seg.start, seg.end), (a, b));
}

#[test]
fn a_whole_ellipse_is_one_closed_segment_along_its_longer_axis() {
    // Written with the longer semi-axis as `minor`: the segment's major axis is still the longer.
    let s = sketch_of(&part(
        "        e1 = ellipse(center = [0 mm, 0 mm], major = 10 mm, minor = 20 mm, angle = 30 deg)\n",
    ));
    let solved = solve(&s).expect("solves");
    let seg = solved.ellipse_segment(&"e1".into(), false).unwrap();
    assert!(seg.is_closed_curve());
    let SegmentKind::Ellipse { major, ratio, .. } = seg.kind else {
        panic!()
    };
    assert!((major[0].hypot(major[1]) - 0.020).abs() < 1e-12 && (ratio - 0.5).abs() < 1e-12);
    let turn = major[1].atan2(major[0]).to_degrees();
    assert!((turn - 120.0).abs() < 1e-9, "{turn}");
    // Its sampled area is π·a·b.
    let ring = seg.polyline(2000);
    let area: f64 = ring
        .windows(2)
        .map(|w| (w[0][0] * w[1][1] - w[1][0] * w[0][1]) / 2.0)
        .sum();
    let want = std::f64::consts::PI * 0.010 * 0.020;
    assert!((area.abs() - want).abs() < want * 1e-4, "{area} vs {want}");
}
