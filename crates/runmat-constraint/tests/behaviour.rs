//! Cross-cutting behaviour: determinism, warm start, reference dimensions, validation, scale.

mod common;
use common::*;
use runmat_constraint::*;
use std::time::Instant;

/// A mixed system: a fully-defined plate, an under-defined linkage, a spline, a conflict-free
/// redundant pair, and a small assembly.
fn mixed() -> System {
    let mut s = System::new();
    let r = rect(&mut s, 0.2, 0.1, 38.0, 21.0);
    s.constrain(Constraint::Fix(r.p[0])).unwrap();
    square_up(&mut s, &r);
    s.constrain(Constraint::Length {
        line: r.l[0],
        value: 40.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: r.l[1],
        value: 20.0,
    })
    .unwrap();
    let c = s.point2(10.0, 10.0);
    let hole = s.circle(c, 3.0).unwrap();
    s.constrain(Constraint::Diameter {
        curve: hole,
        value: 8.0,
    })
    .unwrap();
    s.constrain(Constraint::Distance {
        a: c,
        b: r.l[3],
        value: 12.0,
    })
    .unwrap();
    s.constrain(Constraint::Distance {
        a: c,
        b: r.l[0],
        value: 10.0,
    })
    .unwrap();
    s.constrain(Constraint::Parallel(r.l[0], r.l[2])).unwrap();

    let a = s.point2(60.0, 0.0);
    let b = s.point2(60.5, 5.2);
    let k = s.point2(70.0, 6.0);
    let l1 = s.line(a, b).unwrap();
    let l2 = s.line(b, k).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Length {
        line: l1,
        value: 5.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: l2,
        value: 9.0,
    })
    .unwrap();

    let cps: Vec<EntityId> = (0..6)
        .map(|i| s.point2(80.0 + 3.0 * i as f64, (i as f64).sin()))
        .collect();
    let sp = s.bspline(3, &cps).unwrap();
    let on = s.point2(86.0, 1.0);
    s.constrain(Constraint::PointOnCurve {
        point: on,
        curve: sp,
    })
    .unwrap();
    s.constrain(Constraint::Fix(cps[0])).unwrap();

    let g = s.ground(Frame::IDENTITY);
    let gc = s.connector(g, Frame::IDENTITY).unwrap();
    let body = s.body(Frame::axis_angle([0.1, 0.2, 0.0], [0.0, 0.0, 1.0], 0.4));
    let bc = s.connector(body, Frame::IDENTITY).unwrap();
    s.constrain(Constraint::Mate(Mate::Revolute { a: gc, b: bc }))
        .unwrap();
    s
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn solve_is_bit_identical_across_runs() {
    let s = mixed();
    let a = solve(&s, &opts());
    let b = solve(&s, &opts());
    assert!(a.converged);
    assert_eq!(bits(&a.values), bits(&b.values));
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    // A clone of the system (fresh allocations) gives the same bits too.
    let c = solve(&s.clone(), &opts());
    assert_eq!(bits(&a.values), bits(&c.values));
}

#[test]
fn drag_is_bit_identical_across_runs() {
    let s = mixed();
    let p = EntityId::from_index(
        (0..s.entities().len())
            .find(|&i| {
                matches!(
                    s.entity(EntityId::from_index(i)).kind,
                    EntityKind::Point2 { .. }
                ) && i > 20
            })
            .unwrap(),
    );
    let req = DragRequest::point(&s, p, [65.0, 9.0]);
    let a = drag(&s, &req, &opts());
    let b = drag(&s, &req, &opts());
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn mixed_system_diagnosis() {
    let s = mixed();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::OverDefined);
    // linkage: b and k each on a circle (2); spline: 12 - 2 fixed, point + parameter 3 - 2
    // rows (11); revolute (1).
    assert_eq!(rep.dof, 2 + 11 + 1);
    assert_eq!(rep.redundant.len(), 1);
}

#[test]
fn warm_start_round_trip() {
    let s = mixed();
    let first = solve(&s, &opts());
    let warm = SolveOptions {
        warm_start: Some(WarmStart::from_report(&first)),
        ..opts()
    };
    let second = solve(&s, &warm);
    let w = second.warm_start.unwrap();
    assert!(w.cache_satisfied);
    assert!(w.matches);
    assert_eq!(w.max_deviation, 0.0);
    assert_eq!(second.iterations, 0);
    assert_eq!(bits(&first.values), bits(&second.values));
}

#[test]
fn warm_start_detects_a_changed_dimension() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(9.0, 0.5);
    let l = s.line(a, b).unwrap();
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Horizontal(l)).unwrap();
    s.constrain(Constraint::Length {
        line: l,
        value: 10.0,
    })
    .unwrap();
    let first = solve(&s, &opts());

    let mut s2 = System::new();
    let a2 = s2.point2(0.0, 0.0);
    let b2 = s2.point2(9.0, 0.5);
    let l2 = s2.line(a2, b2).unwrap();
    s2.constrain(Constraint::Fix(a2)).unwrap();
    s2.constrain(Constraint::Horizontal(l2)).unwrap();
    s2.constrain(Constraint::Length {
        line: l2,
        value: 12.0,
    })
    .unwrap();
    let rep = solve(
        &s2,
        &SolveOptions {
            warm_start: Some(WarmStart::from_report(&first)),
            ..opts()
        },
    );
    let w = rep.warm_start.unwrap();
    assert!(!w.cache_satisfied);
    assert!(!w.matches);
    assert_close(w.max_deviation, 2.0, 1e-9);
    assert_pt(pt(&rep, &s2, b2), [12.0, 0.0], 1e-9);
}

#[test]
fn reference_dimension_is_measured_not_enforced() {
    let mut s = System::new();
    let r = rect(&mut s, 0.0, 0.0, 38.0, 22.0);
    s.constrain(Constraint::Fix(r.p[0])).unwrap();
    square_up(&mut s, &r);
    s.constrain(Constraint::Length {
        line: r.l[0],
        value: 40.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: r.l[1],
        value: 20.0,
    })
    .unwrap();
    let diag = s
        .reference(Constraint::Distance {
            a: r.p[0],
            b: r.p[2],
            value: 0.0,
        })
        .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined);
    assert_close(rep.measurement(diag).unwrap(), 2000f64.sqrt(), 1e-9);
    assert_eq!(
        s.reference(Constraint::Horizontal(r.l[0])),
        Err(BuildError::NotADimension)
    );
}

#[test]
fn validation_rejects_bad_input() {
    let mut s = System::new();
    let p = s.point2(0.0, 0.0);
    let q = s.point2(1.0, 0.0);
    let l = s.line(p, q).unwrap();
    assert!(matches!(
        s.constrain(Constraint::Horizontal(p)),
        Err(BuildError::WrongKind { .. })
    ));
    assert_eq!(
        s.constrain(Constraint::Coincident(p, p)),
        Err(BuildError::SameEntity(p))
    );
    assert!(matches!(
        s.constrain(Constraint::Radius {
            curve: l,
            value: 1.0
        }),
        Err(BuildError::WrongKind { .. })
    ));
    assert!(matches!(
        s.constrain(Constraint::Length {
            line: l,
            value: f64::NAN
        }),
        Err(BuildError::BadValue(_))
    ));
    assert_eq!(
        s.constrain(Constraint::Fix(EntityId::from_index(99))),
        Err(BuildError::UnknownEntity(EntityId::from_index(99)))
    );
    assert!(matches!(
        s.bspline(3, &[p, q]),
        Err(BuildError::BadSpline(_))
    ));
    assert_eq!(s.constraints().len(), 0);
}

#[test]
fn not_converged_reports_residual() {
    let mut s = System::new();
    // Two circle constraints on one point: nonlinear, so one step cannot finish it.
    let a = s.point2(0.0, 0.0);
    let b = s.point2(10.0, 0.0);
    let c = s.point2(1.0, 7.0);
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Fix(b)).unwrap();
    s.constrain(Constraint::Distance {
        a,
        b: c,
        value: 6.0,
    })
    .unwrap();
    s.constrain(Constraint::Distance {
        a: b,
        b: c,
        value: 8.0,
    })
    .unwrap();
    let rep = solve(
        &s,
        &SolveOptions {
            max_iterations: 1,
            // The constructive pre-pass would place `c` exactly (two circles about fixed
            // points); this test is about the numerical iteration running out.
            clusters: false,
            ..opts()
        },
    );
    assert!(!rep.converged);
    assert!(rep.residual_norm > 0.0);
    assert!(matches!(
        rep.status,
        Status::NotConverged | Status::Conflicting
    ));
}

/// A rigid ladder truss `n` bays long: 4n unknowns in one connected component.
fn truss(s: &mut System, n: usize, y0: f64) {
    let mut bottom = Vec::new();
    let mut top = Vec::new();
    for i in 0..n {
        let j = |k: usize| 0.01 * ((k * 7919 % 13) as f64 - 6.0);
        bottom.push(s.point2(i as f64 + j(i), y0 + j(i + 3)));
        top.push(s.point2(i as f64 + j(i + 5), y0 + 1.0 + j(i + 1)));
    }
    s.constrain(Constraint::Fix(bottom[0])).unwrap();
    s.constrain(Constraint::HorizontalPoints(bottom[0], bottom[1]))
        .unwrap();
    let diag = 2f64.sqrt();
    for i in 0..n {
        s.constrain(Constraint::Distance {
            a: bottom[i],
            b: top[i],
            value: 1.0,
        })
        .unwrap();
        if i + 1 < n {
            s.constrain(Constraint::Distance {
                a: bottom[i],
                b: bottom[i + 1],
                value: 1.0,
            })
            .unwrap();
            s.constrain(Constraint::Distance {
                a: top[i],
                b: top[i + 1],
                value: 1.0,
            })
            .unwrap();
            s.constrain(Constraint::Distance {
                a: bottom[i],
                b: top[i + 1],
                value: diag,
            })
            .unwrap();
        }
    }
}

#[test]
fn scale_two_thousand_variable_component_and_many_small_ones() {
    let mut s = System::new();
    truss(&mut s, 500, 0.0); // 2,000 unknowns, connected
    for k in 0..250 {
        // 2,000 more unknowns in 250 independent fully-defined rectangles
        let r = rect(&mut s, 0.0, 10.0 + 3.0 * k as f64, 1.9, 1.1);
        s.constrain(Constraint::Fix(r.p[0])).unwrap();
        square_up(&mut s, &r);
        s.constrain(Constraint::Length {
            line: r.l[0],
            value: 2.0,
        })
        .unwrap();
        s.constrain(Constraint::Length {
            line: r.l[1],
            value: 1.0,
        })
        .unwrap();
    }
    let n = s.var_count();
    assert_eq!(n, 4000);
    let t = Instant::now();
    let rep = solve(&s, &opts());
    let solve_time = t.elapsed();
    assert!(rep.converged, "residual {}", rep.residual_norm);
    assert_eq!(rep.status, Status::WellDefined);
    assert_eq!(rep.dof, 0);
    assert_eq!(rep.components.len(), 251);
    let t = Instant::now();
    let fast = solve(
        &s,
        &SolveOptions {
            diagnose: false,
            ..opts()
        },
    );
    let no_diag = t.elapsed();
    assert!(fast.converged);
    // The same system solved without DM blocks: one 2,000-unknown sparse Newton system.
    let t = Instant::now();
    let whole = solve(
        &s,
        &SolveOptions {
            decompose: false,
            ..opts()
        },
    );
    let whole_time = t.elapsed();
    assert_eq!(whole.status, Status::WellDefined);
    assert_eq!(whole.components[0].blocks, 1);
    println!(
        "scale: whole-component (no DM) solve+diagnose {whole_time:?}, {} iterations",
        whole.iterations
    );
    println!(
        "scale: {n} variables, {} constraints, {} components: solve+diagnose {:?}, solve only {:?}, {} iterations",
        s.constraints().len(),
        rep.components.len(),
        solve_time,
        no_diag,
        rep.iterations
    );
}

#[test]
fn scale_under_defined_component_reports_entity_dof() {
    // A floppy chain: 600 points, 599 distances, first point fixed → 599 DOF.
    let mut s = System::new();
    let pts: Vec<EntityId> = (0..600)
        .map(|i| s.point2(i as f64, 0.3 * (i as f64 * 0.7).sin()))
        .collect();
    s.constrain(Constraint::Fix(pts[0])).unwrap();
    for i in 0..599 {
        s.constrain(Constraint::Distance {
            a: pts[i],
            b: pts[i + 1],
            value: 1.0,
        })
        .unwrap();
    }
    let t = Instant::now();
    let rep = solve(&s, &opts());
    println!(
        "chain: 1200 variables, 599 DOF, solve+diagnose {:?}",
        t.elapsed()
    );
    assert!(rep.converged);
    assert_eq!(rep.dof, 599);
    assert_eq!(rep.entity_dof(pts[0]), Some(0));
    assert_eq!(rep.entity_dof(pts[1]), Some(1));
    assert_eq!(rep.entity_dof(pts[300]), Some(2));
}
