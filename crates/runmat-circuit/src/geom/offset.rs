//! Polygon offsetting with joins.
//!
//! A positive offset by `d` moves every ring edge `d` outward (to the right of edges, since
//! interior is on the left of counter-clockwise outers and clockwise holes) and joins
//! consecutive offset edges:
//!
//! - at a **convex** vertex the offset edges separate and the gap is closed by the join:
//!   [`Join::Round`] (an arc about the vertex, chords outside the true circle),
//!   [`Join::Miter`] (the offset lines' intersection, squared off beyond the limit), or
//!   [`Join::Square`] (cut perpendicular to the bisector at distance `d`);
//! - at a **reflex** vertex they overlap; the raw ring runs through the vertex itself, which
//!   leaves a small inverted loop.
//!
//! The raw rings are then unioned under the [`FillRule::Positive`] rule: inverted loops and
//! collapsed holes have non-positive winding and drop out, so the result is exact for any
//! input topology (rings that merge, holes that close). A negative offset is the positive
//! offset of the complement inside a margin box, subtracted from the set.
//!
//! Cost: linear in the vertex count plus one boolean of the raw rings, against the Minkowski
//! construction's union of one band and one disk per vertex.

use super::poly::{area2, boolean, boolean_with, rect_ring, BoolOp, FillRule, PolySet, Ring};
use super::{circle_segments, Coord, Point};

/// How convex corners are joined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Join {
    Round,
    /// Sharp corners up to `limit × d` from the vertex (KiCad and Clipper use 2), squared off
    /// beyond.
    Miter(f64),
    Square,
}

/// Offsets rings of any orientation (non-zero fill) by `d` (negative shrinks).
pub fn offset(rings: &[Ring], d: Coord, join: Join, tol: Coord) -> PolySet {
    offset_set(&boolean(rings, &[], BoolOp::Union), d, join, tol)
}

/// Offsets a canonical set (outers counter-clockwise, holes clockwise, as booleans return).
pub fn offset_set(set: &PolySet, d: Coord, join: Join, tol: Coord) -> PolySet {
    if d == 0 || set.is_empty() {
        return set.clone();
    }
    if d > 0 {
        let raw: Vec<Ring> = set
            .polys
            .iter()
            .flat_map(|p| p.rings())
            .map(|r| offset_ring(r, d as f64, join, tol))
            .filter(|r| r.len() >= 3)
            .collect();
        return boolean_with(&raw, &[], BoolOp::Union, FillRule::Positive);
    }
    // Erosion: grow the complement inside a box that stays clear of the set.
    let e = -d;
    let frame = set.bbox().inflate(2 * e + 1000);
    let mut comp: Vec<Ring> = vec![rect_ring(frame)];
    for p in &set.polys {
        for r in p.rings() {
            comp.push(r.iter().rev().copied().collect());
        }
    }
    let raw: Vec<Ring> = comp
        .iter()
        .map(|r| offset_ring(r, e as f64, join, tol))
        .filter(|r| r.len() >= 3)
        .collect();
    let grown = boolean_with(&raw, &[], BoolOp::Union, FillRule::Positive);
    boolean(&set.rings(), &grown.rings(), BoolOp::Difference)
}

fn unit(a: Point, b: Point) -> Option<(f64, f64)> {
    let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
    let l = dx.hypot(dy);
    (l > 0.0).then(|| (dx / l, dy / l))
}

/// The raw offset of one oriented ring by `d > 0` (may self-intersect).
pub fn offset_ring(r: &[Point], d: f64, join: Join, tol: Coord) -> Ring {
    let mut pts: Vec<Point> = r.to_vec();
    pts.dedup();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    let n = pts.len();
    if n < 3 || area2(&pts) == 0 {
        return Vec::new();
    }
    let at = |v: Point, x: f64, y: f64| Point::from_f(v.x as f64 + x, v.y as f64 + y);
    let segs = circle_segments(d as Coord, tol).max(8) as f64;
    let mut out: Ring = Vec::with_capacity(n * 3);
    for i in 0..n {
        let (p, v, q) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
        let (Some(e1), Some(e2)) = (unit(p, v), unit(v, q)) else {
            continue;
        };
        // Right-hand normals: outward for the ring's interior on the left.
        let n1 = (e1.1, -e1.0);
        let n2 = (e2.1, -e2.0);
        let cross = e1.0 * e2.1 - e1.1 * e2.0;
        let dot = e1.0 * e2.0 + e1.1 * e2.1;
        let a = at(v, n1.0 * d, n1.1 * d);
        let b = at(v, n2.0 * d, n2.1 * d);
        if cross.abs() < 1e-12 && dot > 0.0 {
            out.push(a);
            continue;
        }
        if cross < 0.0 && !(cross.abs() < 1e-12 && dot < 0.0) {
            // Reflex: through the vertex.
            out.push(a);
            out.push(v);
            out.push(b);
            continue;
        }
        // Convex (or a full reversal): the turn from n1 to n2 is counter-clockwise by theta.
        let a1 = n1.1.atan2(n1.0);
        let mut theta = n2.1.atan2(n2.0) - a1;
        while theta <= 0.0 {
            theta += std::f64::consts::TAU;
        }
        if theta > std::f64::consts::PI + 1e-9 {
            theta = std::f64::consts::PI;
        }
        let square = |out: &mut Ring| {
            let t = d * (theta / 4.0).tan();
            out.push(at(v, n1.0 * d + e1.0 * t, n1.1 * d + e1.1 * t));
            out.push(at(v, n2.0 * d - e2.0 * t, n2.1 * d - e2.1 * t));
        };
        match join {
            Join::Round if d / (theta / 2.0).cos() - d <= tol as f64 => {
                // A shallow turn: the miter point is within tolerance of the arc (and outside
                // it), so one vertex replaces the arc.
                let bis = a1 + theta / 2.0;
                let m = d / (theta / 2.0).cos();
                out.push(at(v, m * bis.cos(), m * bis.sin()));
            }
            Join::Round => {
                let k = ((segs * theta / std::f64::consts::TAU).ceil() as usize).max(1);
                let step = theta / k as f64;
                let rr = d / (step / 2.0).cos();
                out.push(a);
                for j in 0..k {
                    let t = a1 + step * (j as f64 + 0.5);
                    out.push(at(v, rr * t.cos(), rr * t.sin()));
                }
                out.push(b);
            }
            Join::Miter(limit) => {
                let half = theta / 2.0;
                let m = d / half.cos();
                if half.cos() > 1e-9 && m <= limit.max(1.0) * d {
                    let bis = a1 + half;
                    out.push(at(v, m * bis.cos(), m * bis.sin()));
                } else {
                    square(&mut out);
                }
            }
            Join::Square => square(&mut out),
        }
    }
    out.dedup();
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;
    use crate::geom::poly::{boolean, BoolOp};
    use crate::geom::Rect;

    fn sq(x: f64, y: f64, s: f64) -> Ring {
        rect_ring(Rect::new(Point::mm(x, y), Point::mm(x + s, y + s)))
    }

    #[test]
    fn joins_on_a_square() {
        let s = vec![sq(0.0, 0.0, 2.0)];
        let pi = std::f64::consts::PI;
        let round = offset(&s, mm(0.5), Join::Round, 500);
        let want = (4.0 + 4.0 + pi * 0.25) * 1e12;
        assert_eq!(round.polys.len(), 1);
        assert!(
            (round.area() - want).abs() / want < 0.005,
            "{}",
            round.area()
        );
        assert!(round.area() >= want, "chords lie outside the true offset");
        let miter = offset(&s, mm(0.5), Join::Miter(2.0), 500);
        assert!((miter.area() - 9.0e12).abs() < 1e7, "{}", miter.area());
        assert_eq!(miter.polys[0].outer.len(), 4);
        let square = offset(&s, mm(0.5), Join::Square, 500);
        // Corners cut at 0.5 mm from the vertex along the diagonal: each removes the
        // triangle beyond the cut, of area (d·√2 − d)².
        let cut = 4.0 * (0.5 * (2f64.sqrt() - 1.0)).powi(2);
        let want = 9.0e12 - cut * 1e12;
        assert!(
            (square.area() - want).abs() / want < 1e-4,
            "{}",
            square.area()
        );
        assert_eq!(square.polys[0].outer.len(), 8);
    }

    #[test]
    fn reflex_corners_and_merging() {
        // An L shape: one reflex corner.
        let l = vec![vec![
            Point::mm(0.0, 0.0),
            Point::mm(4.0, 0.0),
            Point::mm(4.0, 1.0),
            Point::mm(1.0, 1.0),
            Point::mm(1.0, 4.0),
            Point::mm(0.0, 4.0),
        ]];
        let m = offset(&l, mm(0.25), Join::Miter(2.0), 500);
        // Mitred L grows to the L of the offset outline: 4.5×1.5 + 1.5×3 = 11.25 mm².
        assert_eq!(m.polys.len(), 1);
        assert!((m.area() - 11.25e12).abs() < 1e7, "{}", m.area());
        // Two squares 0.4 mm apart merge when grown by 0.25 mm.
        let two = vec![sq(0.0, 0.0, 1.0), sq(1.4, 0.0, 1.0)];
        let g = offset(&two, mm(0.25), Join::Round, 500);
        assert_eq!(g.polys.len(), 1);
        // A frame whose 0.4 mm hole closes when grown by 0.25 mm.
        let frame = boolean(
            &[sq(0.0, 0.0, 3.0)],
            &[sq(1.3, 1.3, 0.4)],
            BoolOp::Difference,
        );
        let g = offset_set(&frame, mm(0.25), Join::Miter(2.0), 500);
        assert_eq!(g.polys.len(), 1);
        assert!(g.polys[0].holes.is_empty());
        assert!(
            (g.area() - 3.5f64.powi(2) * 1e12).abs() < 1e7,
            "{}",
            g.area()
        );
    }

    #[test]
    fn erosion_matches_dilation_inverse() {
        let s = vec![sq(0.0, 0.0, 2.0)];
        let e = offset(&s, -mm(0.5), Join::Round, 500);
        assert!((e.area() - 1.0e12).abs() < 1e7, "{}", e.area());
        let gone = offset(&s, -mm(1.01), Join::Round, 500);
        assert!(gone.is_empty());
        // A hole grows when the set shrinks.
        let frame = boolean(
            &[sq(0.0, 0.0, 4.0)],
            &[sq(1.5, 1.5, 1.0)],
            BoolOp::Difference,
        );
        let e = offset_set(&frame, -mm(0.25), Join::Miter(2.0), 500);
        assert_eq!(e.polys.len(), 1);
        assert_eq!(e.polys[0].holes.len(), 1);
        assert!((e.area() - (3.5f64.powi(2) - 1.5f64.powi(2)) * 1e12).abs() < 1e7);
        // Thin neck splits under erosion.
        let neck = boolean(
            &[
                sq(0.0, 0.0, 2.0),
                sq(3.0, 0.0, 2.0),
                rect_ring(Rect::new(Point::mm(2.0, 0.95), Point::mm(3.0, 1.05))),
            ],
            &[],
            BoolOp::Union,
        );
        assert_eq!(offset_set(&neck, -mm(0.1), Join::Round, 500).polys.len(), 2);
    }

    #[test]
    fn many_vertices_is_fast_and_exact() {
        // A 2 000-vertex circle grown by 0.1 mm is the larger circle (within tolerance).
        let c = crate::geom::poly::circle_ring(Point::ORIGIN, mm(5.0), 10);
        assert!(c.len() >= 400);
        let g = offset(&[c], mm(0.1), Join::Round, 500);
        let want = std::f64::consts::PI * 5.1f64.powi(2) * 1e12;
        assert!((g.area() - want).abs() / want < 1e-3, "{}", g.area());
    }
}
