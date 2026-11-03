//! Solver features: the active set for range limits, unit-aware scaling, the dog leg trust
//! region, the constructive rigid-cluster pre-pass, and serialisation of reports.

mod common;
use common::*;
use runmat_constraint::*;
use std::time::Instant;

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn body_position(s: &System, rep: &SolveReport, b: EntityId) -> [f64; 3] {
    let EntityKind::Body { position, .. } = s.entity(b).kind else {
        panic!()
    };
    position.map(|v| rep.value(v))
}

// ============================================================================ limits

#[test]
fn only_the_binding_limit_activates() {
    // Two lower bounds on one slider coordinate, both violated at the start. Activating both
    // would contradict (x = 2 and x = 1); the active set takes the binding one only.
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let gs = s
        .connector(g, Frame::with_z([0.0; 3], [1.0, 0.0, 0.0]))
        .unwrap();
    let b = s.body(Frame::at([0.0, 0.0, 0.0]));
    let bs = s
        .connector(b, Frame::with_z([0.0; 3], [1.0, 0.0, 0.0]))
        .unwrap();
    s.constrain(Constraint::Mate(Mate::Slider { a: gs, b: bs }))
        .unwrap();
    let lim = |s: &mut System, min: f64| {
        s.constrain(Constraint::Mate(Mate::Distance {
            a: gs,
            b: bs,
            mode: DistanceMode::PlanePoint,
            target: Target::Range { min, max: 10.0 },
        }))
        .unwrap()
    };
    let l2 = lim(&mut s, 2.0);
    let _l1 = lim(&mut s, 1.0);
    let rep = solve(&s, &opts());
    assert!(rep.converged, "{rep:#?}");
    assert_eq!(rep.status, Status::WellDefined);
    assert_eq!(rep.active_limits, vec![(l2, 2.0)]);
    assert!(rep.conflicting.is_empty());
    assert_close(body_position(&s, &rep, b)[0], 2.0, 1e-9);
}

#[test]
fn limit_released_when_another_takes_over() {
    // A body sliding in the ground plane (planar mate) from the origin, with two half-plane
    // limits: x ≥ 1 and (x + y)/√2 ≥ 2√2. The x bound is the most violated (relative to its
    // range) and activates first; the second then activates too, and at the vertex (1, 3) the
    // x bound's multiplier is negative: it holds the body back. Released, the least-motion
    // solution is (2, 2), where x ≥ 1 holds by itself.
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let gc = s.connector(g, Frame::IDENTITY).unwrap();
    let ax = s
        .connector(g, Frame::with_z([0.0; 3], [1.0, 0.0, 0.0]))
        .unwrap();
    let axy = s
        .connector(g, Frame::with_z([0.0; 3], [1.0, 1.0, 0.0]))
        .unwrap();
    let b = s.body(Frame::IDENTITY);
    let bc = s.connector(b, Frame::IDENTITY).unwrap();
    s.constrain(Constraint::Mate(Mate::Planar { a: gc, b: bc }))
        .unwrap();
    let lx = s
        .constrain(Constraint::Mate(Mate::Distance {
            a: ax,
            b: bc,
            mode: DistanceMode::PlanePoint,
            target: Target::Range {
                min: 1.0,
                max: 100.0,
            },
        }))
        .unwrap();
    let r = 2.0 * 2f64.sqrt();
    let lxy = s
        .constrain(Constraint::Mate(Mate::Distance {
            a: axy,
            b: bc,
            mode: DistanceMode::PlanePoint,
            target: Target::Range {
                min: r,
                max: 1000.0,
            },
        }))
        .unwrap();
    let rep = solve(&s, &opts());
    assert!(rep.converged, "{rep:#?}");
    assert_eq!(rep.active_limits, vec![(lxy, r)]);
    assert_eq!(rep.released_limits, vec![lx]);
    let p = body_position(&s, &rep, b);
    assert_close(p[0], 2.0, 1e-8);
    assert_close(p[1], 2.0, 1e-8);
    assert_close(p[2], 0.0, 1e-9);
    // Planar 3 DOF, one active bound.
    assert_eq!(rep.dof, 2);
    let again = solve(&s, &opts());
    assert_eq!(bits(&rep.values), bits(&again.values));
}

// ============================================================================ scaling

/// A triangle at kilometre-in-microns scale: two lengths of 1e7 and a 0.7 rad angle.
fn big_triangle() -> (System, EntityId) {
    let l = 1e7;
    let mut s = System::new();
    let p0 = s.point2(0.0, 0.0);
    let p1 = s.point2(0.9 * l, 0.05 * l);
    let p2 = s.point2(0.6 * l, 0.5 * l);
    let a = s.line(p0, p1).unwrap();
    let c = s.line(p0, p2).unwrap();
    s.constrain(Constraint::Fix(p0)).unwrap();
    s.constrain(Constraint::Horizontal(a)).unwrap();
    s.constrain(Constraint::Length { line: a, value: l })
        .unwrap();
    s.constrain(Constraint::Length { line: c, value: l })
        .unwrap();
    s.constrain(Constraint::Angle {
        a,
        b: c,
        value: 0.7,
    })
    .unwrap();
    (s, p2)
}

#[test]
fn scaling_conditions_mixed_length_angle_systems() {
    let (s, p2) = big_triangle();
    let scaled = solve(&s, &opts());
    assert_eq!(scaled.status, Status::WellDefined, "{scaled:#?}");
    assert_pt(
        pt(&scaled, &s, p2),
        [1e7 * 0.7f64.cos(), 1e7 * 0.7f64.sin()],
        1e-3,
    );
    let unscaled = solve(
        &s,
        &SolveOptions {
            scaling: false,
            ..opts()
        },
    );
    println!(
        "1e7 triangle: scaled {} iterations, unscaled {} iterations (converged {})",
        scaled.iterations, unscaled.iterations, unscaled.converged
    );
    assert!(scaled.iterations <= 15, "{}", scaled.iterations);
    // Unscaled, the angle row is 1e-7 the size of the length rows and the step crawls.
    assert!(!unscaled.converged || unscaled.iterations > 3 * scaled.iterations);
}

// ============================================================================ dog leg

#[test]
fn dog_leg_solves_and_diagnoses_like_newton() {
    let (s, p2) = big_triangle();
    let o = SolveOptions {
        algorithm: Algorithm::DogLeg,
        ..opts()
    };
    let rep = solve(&s, &o);
    assert_eq!(rep.status, Status::WellDefined);
    assert!(rep.components.iter().any(|c| c.method == Method::DogLeg));
    assert_pt(
        pt(&rep, &s, p2),
        [1e7 * 0.7f64.cos(), 1e7 * 0.7f64.sin()],
        1e-3,
    );
    let again = solve(&s, &o);
    assert_eq!(bits(&rep.values), bits(&again.values));

    // A contradiction is still diagnosed.
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(3.0, 0.0);
    let c = s.point2(1.0, 1.0);
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::Fix(b)).unwrap();
    let d1 = s
        .constrain(Constraint::Distance {
            a,
            b: c,
            value: 1.0,
        })
        .unwrap();
    let d2 = s
        .constrain(Constraint::Distance {
            a: b,
            b: c,
            value: 1.0,
        })
        .unwrap();
    let rep = solve(&s, &o);
    assert_eq!(rep.status, Status::Conflicting);
    let set = rep.conflict_set();
    assert!(set.contains(&d1) && set.contains(&d2), "{set:?}");
}

// ============================================================================ clusters

#[test]
fn clusters_construct_a_triangle_from_a_far_start() {
    // A 3-4-5 triangle dimensioned ten times larger than drawn.
    let mut s = System::new();
    let p0 = s.point2(0.0, 0.0);
    let p1 = s.point2(3.0, 0.2);
    let p2 = s.point2(0.4, 3.5);
    let base = s.line(p0, p1).unwrap();
    s.constrain(Constraint::Fix(p0)).unwrap();
    s.constrain(Constraint::Horizontal(base)).unwrap();
    s.constrain(Constraint::Length {
        line: base,
        value: 30.0,
    })
    .unwrap();
    s.constrain(Constraint::Distance {
        a: p0,
        b: p2,
        value: 40.0,
    })
    .unwrap();
    s.constrain(Constraint::Distance {
        a: p1,
        b: p2,
        value: 50.0,
    })
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined);
    assert_eq!(rep.iterations, 0, "the construction is exact");
    assert_eq!(rep.components[0].constructed_points, 2);
    assert_eq!(rep.components[0].clusters, 1);
    // Owen's rule: p2 stays on the side it was drawn on.
    assert_pt(pt(&rep, &s, p1), [30.0, 0.0], 1e-12);
    assert_pt(pt(&rep, &s, p2), [0.0, 40.0], 1e-9);
}

#[test]
fn clusters_construct_a_rectangle_by_loci() {
    let mut s = System::new();
    let r = rect(&mut s, 0.5, 0.3, 1.7, 0.4);
    s.constrain(Constraint::Fix(r.p[0])).unwrap();
    square_up(&mut s, &r);
    s.constrain(Constraint::Length {
        line: r.l[0],
        value: 80.0,
    })
    .unwrap();
    s.constrain(Constraint::Length {
        line: r.l[1],
        value: 25.0,
    })
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined);
    assert_eq!(rep.iterations, 0);
    assert_eq!(rep.components[0].constructed_points, 3);
    assert_pt(pt(&rep, &s, r.p[2]), [80.5, 25.3], 1e-9);
    assert_pt(pt(&rep, &s, r.p[3]), [0.5, 25.3], 1e-9);
}

/// A braced ladder truss of `n` bays with unit members, drawn at twice its size.
fn ladder(n: usize) -> System {
    let mut s = System::new();
    let mut bottom = Vec::new();
    let mut top = Vec::new();
    for i in 0..n {
        let j = |k: usize| 0.05 * ((k * 7919 % 13) as f64 - 6.0);
        bottom.push(s.point2(2.0 * i as f64 + j(i), j(i + 3)));
        top.push(s.point2(2.0 * i as f64 + j(i + 5), 2.0 + j(i + 1)));
    }
    s.constrain(Constraint::Fix(bottom[0])).unwrap();
    s.constrain(Constraint::HorizontalPoints(bottom[0], bottom[1]))
        .unwrap();
    let d = |s: &mut System, a, b, v| {
        s.constrain(Constraint::Distance { a, b, value: v })
            .unwrap();
    };
    for i in 0..n {
        d(&mut s, bottom[i], top[i], 1.0);
        if i + 1 < n {
            d(&mut s, bottom[i], bottom[i + 1], 1.0);
            d(&mut s, top[i], top[i + 1], 1.0);
            d(&mut s, bottom[i], top[i + 1], 2f64.sqrt());
        }
    }
    s
}

#[test]
fn clusters_construct_a_large_truss() {
    let s = ladder(300);
    let t = Instant::now();
    let rep = solve(&s, &opts());
    let with = t.elapsed();
    assert_eq!(rep.status, Status::WellDefined, "{}", rep.max_residual);
    assert!(rep.iterations <= 1, "{} iterations", rep.iterations);
    assert!(rep.components[0].constructed_points >= 598);
    let t = Instant::now();
    let without = solve(
        &s,
        &SolveOptions {
            clusters: false,
            ..opts()
        },
    );
    println!(
        "300-bay truss drawn at 2x: clusters {with:?} ({} iterations), numeric only {:?} ({} iterations, {:?})",
        rep.iterations,
        t.elapsed(),
        without.iterations,
        without.status
    );
    let again = solve(&s, &opts());
    assert_eq!(bits(&rep.values), bits(&again.values));
}

#[test]
fn clusters_place_a_floating_rigid_cluster() {
    let mut s = System::new();
    let a = s.point2(10.0, 10.0);
    let b = s.point2(10.4, 10.1);
    let c = s.point2(10.1, 10.5);
    for (p, q, v) in [(a, b, 4.0), (b, c, 5.0), (a, c, 3.0)] {
        s.constrain(Constraint::Distance {
            a: p,
            b: q,
            value: v,
        })
        .unwrap();
    }
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::UnderDefined);
    assert_eq!(rep.dof, 3);
    assert_eq!(rep.iterations, 0);
    assert_eq!(rep.components[0].constructed_points, 3);
    assert_close(dist(pt(&rep, &s, a), pt(&rep, &s, b)), 4.0, 1e-12);
    assert_close(dist(pt(&rep, &s, b), pt(&rep, &s, c)), 5.0, 1e-12);
    // Placed where it best fits the drawing: the centroid stays put.
    let cen = |r: &SolveReport| {
        let (pa, pb, pc) = (pt(r, &s, a), pt(r, &s, b), pt(r, &s, c));
        [(pa[0] + pb[0] + pc[0]) / 3.0, (pa[1] + pb[1] + pc[1]) / 3.0]
    };
    assert_pt(cen(&rep), [30.5 / 3.0, 30.6 / 3.0], 1e-9);
}

#[test]
fn clusters_leave_contradictions_to_the_diagnosis() {
    let mut s = System::new();
    let a = s.point2(0.0, 0.0);
    let b = s.point2(5.0, 0.0);
    let c = s.point2(2.0, 1.0);
    s.constrain(Constraint::Fix(a)).unwrap();
    s.constrain(Constraint::HorizontalPoints(a, b)).unwrap();
    let ids: Vec<_> = [(a, b, 10.0), (b, c, 2.0), (a, c, 3.0)]
        .into_iter()
        .map(|(p, q, v)| {
            s.constrain(Constraint::Distance {
                a: p,
                b: q,
                value: v,
            })
            .unwrap()
        })
        .collect();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::Conflicting);
    let set = rep.conflict_set();
    assert!(ids.iter().all(|c| set.contains(c)), "{set:?}");
}

// ============================================================================ serde

#[test]
fn report_and_options_round_trip_through_json() {
    let s = ladder(20);
    let o = SolveOptions {
        algorithm: Algorithm::DogLeg,
        warm_start: Some(WarmStart { values: vec![] }),
        ..opts()
    };
    let rep = solve(&s, &o);
    let json = serde_json::to_string(&rep).unwrap();
    let back: SolveReport = serde_json::from_str(&json).unwrap();
    assert_eq!(bits(&back.values), bits(&rep.values));
    assert_eq!(back.status, rep.status);
    assert_eq!(back.dof, rep.dof);
    assert_eq!(back.components.len(), rep.components.len());
    assert_eq!(back.components[0].method, rep.components[0].method);
    assert_eq!(back.entity_dof, rep.entity_dof);
    assert_eq!(back.redundant, rep.redundant);
    assert_eq!(back.warm_start, rep.warm_start);
    // Ids serialise as bare indices.
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(v["components"][0]["vars"][0].is_u64());

    let oj = serde_json::to_string(&o).unwrap();
    let ob: SolveOptions = serde_json::from_str(&oj).unwrap();
    assert_eq!(ob.algorithm, Algorithm::DogLeg);
    assert_eq!(ob.tolerance, o.tolerance);
    // Missing fields take their defaults.
    let partial: SolveOptions = serde_json::from_str(r#"{"tolerance": 1e-8}"#).unwrap();
    assert_eq!(partial.tolerance, 1e-8);
    assert_eq!(
        partial.max_iterations,
        SolveOptions::default().max_iterations
    );
}
