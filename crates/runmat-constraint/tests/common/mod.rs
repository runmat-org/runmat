//! Shared builders for the integration tests.
#![allow(dead_code)]

use runmat_constraint::*;

pub fn opts() -> SolveOptions {
    SolveOptions::default()
}

/// Four corner points and four lines (bottom, right, top, left), counter-clockwise from
/// `(x, y)`.
pub struct Rect {
    pub p: [EntityId; 4],
    pub l: [EntityId; 4],
}

pub fn rect(s: &mut System, x: f64, y: f64, w: f64, h: f64) -> Rect {
    let p = [s.point2(x, y), s.point2(x + w, y), s.point2(x + w, y + h), s.point2(x, y + h)];
    let l = [
        s.line(p[0], p[1]).unwrap(),
        s.line(p[1], p[2]).unwrap(),
        s.line(p[2], p[3]).unwrap(),
        s.line(p[3], p[0]).unwrap(),
    ];
    Rect { p, l }
}

/// Horizontal bottom/top, vertical sides.
pub fn square_up(s: &mut System, r: &Rect) -> Vec<ConstraintId> {
    vec![
        s.constrain(Constraint::Horizontal(r.l[0])).unwrap(),
        s.constrain(Constraint::Vertical(r.l[1])).unwrap(),
        s.constrain(Constraint::Horizontal(r.l[2])).unwrap(),
        s.constrain(Constraint::Vertical(r.l[3])).unwrap(),
    ]
}

pub fn pt(rep: &SolveReport, s: &System, p: EntityId) -> [f64; 2] {
    let v = s.entity_vars(p);
    [rep.value(v[0]), rep.value(v[1])]
}

pub fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

pub fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[track_caller]
pub fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(close(a, b, tol), "{a} != {b} (tol {tol})");
}

#[track_caller]
pub fn assert_pt(a: [f64; 2], b: [f64; 2], tol: f64) {
    assert!(dist(a, b) <= tol, "{a:?} != {b:?}");
}
