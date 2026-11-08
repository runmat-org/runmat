//! A fasten whose target turns a body far from its starting pose (a quarter turn, a half
//! turn, an arbitrary flip) lands exactly, and fastens that contradict each other are still
//! reported.

mod common;
use common::*;
use runmat_constraint::*;
use std::f64::consts::PI;

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let r = qmul(qmul(q, [0.0, v[0], v[1], v[2]]), [q[0], -q[1], -q[2], -q[3]]);
    [r[1], r[2], r[3]]
}

/// A connector's world origin and rotation after a solve.
fn world(s: &System, rep: &SolveReport, c: EntityId) -> ([f64; 3], [f64; 4]) {
    let EntityKind::Connector { body, local } = s.entity(c).kind else {
        panic!("a connector")
    };
    let EntityKind::Body {
        position,
        orientation,
    } = s.entity(body).kind
    else {
        panic!("a body")
    };
    let p = position.map(|v| rep.value(v));
    let q = orientation.map(|v| rep.value(v));
    let o = rotate(q, local.origin);
    ([p[0] + o[0], p[1] + o[1], p[2] + o[2]], qmul(q, local.rotation))
}

/// The two connectors coincide: same origin, same rotation (up to the quaternion's sign).
fn assert_coincide(s: &System, rep: &SolveReport, a: EntityId, b: EntityId) {
    let (oa, qa) = world(s, rep, a);
    let (ob, qb) = world(s, rep, b);
    for k in 0..3 {
        assert!((oa[k] - ob[k]).abs() < 1e-8, "origins {oa:?} vs {ob:?}");
    }
    let dot: f64 = (0..4).map(|k| qa[k] * qb[k]).sum();
    assert!(dot.abs() > 1.0 - 1e-10, "rotations {qa:?} vs {qb:?}");
}

/// A foot standing upright at `start`, fastened by a connector on its sole to a ground
/// connector at `target`. Returns the system and both connectors.
fn foot(target: Frame, start: Frame) -> (System, EntityId, EntityId) {
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let seat = s.connector(g, target).unwrap();
    let body = s.body(start);
    let sole = s.connector(body, Frame::at([0.02, -0.01, -0.05])).unwrap();
    s.constrain(Constraint::Mate(Mate::Fasten { a: seat, b: sole }))
        .unwrap();
    (s, seat, sole)
}

fn lands(target: Frame, start: Frame) {
    let (s, seat, sole) = foot(target, start);
    let rep = solve(&s, &opts());
    assert!(rep.converged, "status {:?}", rep.status);
    assert_eq!(rep.status, Status::WellDefined);
    assert!(rep.conflicting.is_empty() && rep.redundant.is_empty());
    assert_coincide(&s, &rep, seat, sole);
}

#[test]
fn a_quarter_turn_fasten_lands() {
    lands(
        Frame::axis_angle([0.3, 0.1, 0.2], [1.0, 0.0, 0.0], PI / 2.0),
        Frame::at([0.0, 0.0, 0.1]),
    );
}

#[test]
fn a_half_turn_fasten_lands() {
    // The connector's Z points down on a foot that starts upright.
    lands(
        Frame::with_z([0.4, -0.2, 0.3], [0.0, 0.0, -1.0]),
        Frame::at([0.0, 0.0, 0.1]),
    );
    // Exactly half a turn about each axis, from a start that matches the translation.
    for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]] {
        lands(
            Frame::axis_angle([0.0, 0.0, 0.0], axis, PI),
            Frame::at([0.0, 0.0, 0.0]),
        );
    }
}

#[test]
fn arbitrary_flips_land() {
    let axes = [[0.3, -0.8, 0.5], [-1.0, 0.2, 0.1], [0.0, 0.6, -0.8], [0.7, 0.7, 0.0]];
    let angles = [2.2, 2.9, PI, 3.1, -2.6];
    for (i, axis) in axes.iter().enumerate() {
        for (j, &angle) in angles.iter().enumerate() {
            let start = Frame::axis_angle(
                [0.01 * i as f64, -0.02 * j as f64, 0.05],
                [0.2, 0.9, -0.4],
                0.3 * j as f64,
            );
            lands(Frame::axis_angle([0.25, 0.5, -0.1], *axis, angle), start);
        }
    }
}

#[test]
fn a_chain_of_flipped_fastens_lands() {
    // Ground → A → B, each fasten a half turn from its start.
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let g_seat = s
        .connector(g, Frame::with_z([0.0, 0.0, 0.2], [0.0, 0.0, -1.0]))
        .unwrap();
    let a = s.body(Frame::at([0.0, 0.0, 0.0]));
    let a_bottom = s.connector(a, Frame::IDENTITY).unwrap();
    let a_top = s
        .connector(a, Frame::axis_angle([0.0, 0.1, 0.0], [0.0, 1.0, 0.0], PI))
        .unwrap();
    let b = s.body(Frame::at([0.5, 0.5, 0.5]));
    let b_foot = s.connector(b, Frame::at([0.0, 0.0, -0.03])).unwrap();
    s.constrain(Constraint::Mate(Mate::Fasten {
        a: g_seat,
        b: a_bottom,
    }))
    .unwrap();
    s.constrain(Constraint::Mate(Mate::Fasten {
        a: a_top,
        b: b_foot,
    }))
    .unwrap();
    let rep = solve(&s, &opts());
    assert_eq!(rep.status, Status::WellDefined);
    assert_coincide(&s, &rep, g_seat, a_bottom);
    assert_coincide(&s, &rep, a_top, b_foot);
}

#[test]
fn contradicting_fastens_are_still_reported() {
    // One body fastened to two ground seats a half turn apart: the second contradicts the first.
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let up = s.connector(g, Frame::at([0.0, 0.0, 0.1])).unwrap();
    let down = s
        .connector(g, Frame::with_z([0.0, 0.0, 0.1], [0.0, 0.0, -1.0]))
        .unwrap();
    let body = s.body(Frame::at([0.0, 0.0, 0.0]));
    let c = s.connector(body, Frame::IDENTITY).unwrap();
    s.constrain(Constraint::Mate(Mate::Fasten { a: up, b: c }))
        .unwrap();
    let second = s
        .constrain(Constraint::Mate(Mate::Fasten { a: down, b: c }))
        .unwrap();
    let rep = solve(&s, &opts());
    assert!(!rep.converged);
    assert_eq!(rep.status, Status::Conflicting);
    assert!(rep.conflicting.iter().any(|d| d.constraint == second));
}

#[test]
fn a_consistent_second_fasten_is_redundant() {
    // Two fastens that agree (both a half turn from the start): converged, the second implied.
    let mut s = System::new();
    let g = s.ground(Frame::IDENTITY);
    let seat = s
        .connector(g, Frame::with_z([0.0, 0.0, 0.1], [0.0, 0.0, -1.0]))
        .unwrap();
    let seat2 = s
        .connector(
            g,
            Frame::with_z([0.0, 0.0, 0.1], [0.0, 0.0, -1.0]),
        )
        .unwrap();
    let body = s.body(Frame::at([0.0, 0.0, 0.0]));
    let c = s.connector(body, Frame::IDENTITY).unwrap();
    s.constrain(Constraint::Mate(Mate::Fasten { a: seat, b: c }))
        .unwrap();
    let second = s
        .constrain(Constraint::Mate(Mate::Fasten { a: seat2, b: c }))
        .unwrap();
    let rep = solve(&s, &opts());
    assert!(rep.converged);
    assert_eq!(rep.status, Status::OverDefined);
    assert!(rep.redundant.iter().any(|d| d.constraint == second));
    assert_coincide(&s, &rep, seat, c);
}
