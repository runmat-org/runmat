//! Hand-drawn wires: the paths the lock stores between two pins, drawn as stored.
//!
//! - A stored wire (`wire main C1.1 U1.VIN ortho …` in the lock, [`crate::ids::SchWire`])
//!   draws from one pin's tip through its waypoints to the other pin's tip, in its mode.
//! - When a pin's tip isn't where the end segment can reach it in the wire's mode (a part
//!   moved), the segment leaves the pin straight out and turns once: across then up or down
//!   (`ortho`), or straight out then at 45° (`diagonal`). `any` draws it straight.
//! - A stored wire whose two pins aren't on one net any more (the connection was removed) isn't
//!   drawn; the lock drops it at the next edit ([`stale`]).
//! - The router treats a stored wire's pins as joined: the net's other pins route to the
//!   stored wire's tree like any other branch.

use super::symbol::PinSide;
use crate::elab::{CompInst, Netlist};
use crate::geom::{Coord, Point};
use crate::ids::{NamePath, SchWire, WireKey, WireMode};
use std::collections::{BTreeMap, BTreeSet};

/// A pin as its sheet's `connect` statements and the lock's `wire` records name it: the
/// component's local name and the pin (`U1.VIN`; `U1.A.1` where the plain name is ambiguous).
pub(crate) fn pin_ref(c: &CompInst, pin: usize) -> String {
    let p = &c.pins[pin];
    let plain = crate::elab::find_pins(&c.pins, &p.pin).map(|ix| ix.contains(&pin)).unwrap_or(false);
    if plain || p.gate.is_empty() {
        format!("{}.{}", c.path.leaf(), p.pin)
    } else {
        format!("{}.{}.{}", c.path.leaf(), p.gate, p.pin)
    }
}

/// A stored wire on one page, resolved to its two terms (indices into the page's terms).
pub(super) struct Hand {
    pub a: usize,
    pub b: usize,
    pub mode: WireMode,
    /// Waypoints in the page's drawing frame (nm, y up).
    pub points: Vec<Point>,
}

/// The recorded wires whose two pins are on one net in no instance of their sheet: the
/// connection was removed (or a pin or part went), so the record goes.
pub(crate) fn stale(nl: &Netlist, wires: &BTreeMap<WireKey, SchWire>) -> Vec<WireKey> {
    if wires.is_empty() {
        return Vec::new();
    }
    // (sheet, instance, pin ref) → net
    let mut nets: BTreeMap<(&str, NamePath, String), Option<usize>> = BTreeMap::new();
    for c in &nl.components {
        let instance = c.path.parent().unwrap_or_else(NamePath::root);
        for (i, p) in c.pins.iter().enumerate() {
            nets.insert((c.sheet.as_str(), instance.clone(), pin_ref(c, i)), p.net);
        }
    }
    let instances: BTreeSet<(&str, NamePath)> = nets.keys().map(|(s, i, _)| (*s, i.clone())).collect();
    wires
        .keys()
        .filter(|k| {
            !instances.iter().filter(|(s, _)| *s == k.sheet).any(|(s, i)| {
                let net = |r: &str| nets.get(&(*s, i.clone(), r.to_owned())).copied().flatten();
                net(&k.a).is_some() && net(&k.a) == net(&k.b)
            })
        })
        .cloned()
        .collect()
}

fn sign(v: Coord) -> Coord {
    v.signum()
}

/// The turn between `p` (leaving a pin on `side`, or `None` between waypoints) and `q`, when the
/// segment `p → q` doesn't run in `mode`.
fn elbow(p: Point, q: Point, side: Option<PinSide>, mode: WireMode) -> Option<Point> {
    let (dx, dy) = (q.x - p.x, q.y - p.y);
    let straight = dx == 0 || dy == 0;
    match mode {
        WireMode::Any => None,
        WireMode::Orthogonal if straight => None,
        WireMode::Diagonal if straight || dx.abs() == dy.abs() => None,
        WireMode::Orthogonal => {
            // Out of a pin along its side first; between waypoints, across first.
            let vertical_first = side.is_some_and(|s| !s.horizontal());
            Some(if vertical_first { Point::new(p.x, q.y) } else { Point::new(q.x, p.y) })
        }
        WireMode::Diagonal => {
            let horizontal_first = side.is_none_or(|s| s.horizontal());
            Some(if horizontal_first && dx.abs() >= dy.abs() {
                Point::new(p.x + sign(dx) * (dx.abs() - dy.abs()), p.y)
            } else if !horizontal_first && dy.abs() >= dx.abs() {
                Point::new(p.x, p.y + sign(dy) * (dy.abs() - dx.abs()))
            } else if dx.abs() >= dy.abs() {
                Point::new(q.x - sign(dx) * dy.abs(), p.y)
            } else {
                Point::new(p.x, q.y - sign(dy) * dx.abs())
            })
        }
    }
}

/// The polyline a stored wire draws, tip to tip.
pub(super) fn path(a: (Point, PinSide), points: &[Point], b: (Point, PinSide), mode: WireMode) -> Vec<Point> {
    let mut out = vec![a.0];
    let push = |out: &mut Vec<Point>, q: Point, side: Option<PinSide>| {
        let p = *out.last().expect("a start");
        if p == q {
            return;
        }
        if let Some(e) = elbow(p, q, side, mode) {
            if e != p && e != q {
                out.push(e);
            }
        }
        out.push(q);
    };
    let mut side = Some(a.1);
    for &q in points {
        push(&mut out, q, side.take());
    }
    // The last segment ends straight into pin b: turn so it arrives along b's side.
    let last = *out.last().expect("a start");
    if let Some(e) = elbow(b.0, last, Some(b.1), mode) {
        if e != last && e != b.0 {
            out.push(e);
        }
    }
    if *out.last().expect("a start") != b.0 {
        out.push(b.0);
    }
    // Drop points in the middle of a straight run.
    let mut clean: Vec<Point> = Vec::with_capacity(out.len());
    for p in out {
        if clean.len() >= 2 {
            let (u, v) = (clean[clean.len() - 2], clean[clean.len() - 1]);
            if (v.x - u.x) * (p.y - v.y) == (v.y - u.y) * (p.x - v.x) {
                clean.pop();
            }
        }
        clean.push(p);
    }
    clean
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: Coord = 1_270_000;

    fn p(x: Coord, y: Coord) -> Point {
        Point::new(x * G, y * G)
    }

    #[test]
    fn a_moved_pin_turns_once_into_the_kept_waypoints() {
        // Waypoints at (0, 10) and (10, 10); pin a moved to (2, 0) leaving up, pin b at (10, 0)
        // leaving up.
        let wire = path((p(2, 0), PinSide::Top), &[p(0, 10), p(10, 10)], (p(10, 0), PinSide::Top), WireMode::Orthogonal);
        assert_eq!(wire, vec![p(2, 0), p(2, 10), p(10, 10), p(10, 0)]);
        // Straight from a pin to its waypoint: no turn.
        let wire = path((p(0, 0), PinSide::Right), &[p(5, 0), p(5, 5)], (p(9, 5), PinSide::Left), WireMode::Orthogonal);
        assert_eq!(wire, vec![p(0, 0), p(5, 0), p(5, 5), p(9, 5)]);
    }

    #[test]
    fn diagonal_and_any_angle_modes_keep_their_segments() {
        let wire = path((p(0, 0), PinSide::Right), &[p(6, 2)], (p(10, 2), PinSide::Left), WireMode::Diagonal);
        assert_eq!(wire, vec![p(0, 0), p(4, 0), p(6, 2), p(10, 2)]);
        let wire = path((p(0, 0), PinSide::Right), &[p(6, 2)], (p(10, 2), PinSide::Left), WireMode::Any);
        assert_eq!(wire, vec![p(0, 0), p(6, 2), p(10, 2)]);
    }
}
