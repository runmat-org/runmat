//! Copper and graphic primitives as "core plus radius" shapes.
//!
//! Every primitive is the set of points within `radius` of a core: a point (circle), a
//! segment (track, oval pad), or filled rings (rectangles, polygons; a rounded rectangle is
//! its inner rectangle plus the corner radius). The distance between two primitives is then
//! the core distance minus both radii, which is exact for circles, tracks, ovals and rounded
//! rectangles; arcs are flattened to segments within [`ARC_TOL`](super::ARC_TOL).

use super::poly::{boolean, capsule_ring, circle_ring, inflate, ring_contains, BoolOp, Ring};
use super::{orient, Coord, Point, Rect, Transform};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Core {
    Point(Point),
    Segment(Point, Point),
    /// Filled region (non-zero rule); outers CCW, holes CW.
    Poly(Vec<Ring>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Prim {
    pub core: Core,
    pub radius: Coord,
}

impl Prim {
    pub fn circle(c: Point, r: Coord) -> Prim {
        Prim {
            core: Core::Point(c),
            radius: r,
        }
    }
    pub fn segment(a: Point, b: Point, width: Coord) -> Prim {
        Prim {
            core: Core::Segment(a, b),
            radius: width / 2,
        }
    }
    pub fn polygon(rings: Vec<Ring>) -> Prim {
        Prim {
            core: Core::Poly(rings),
            radius: 0,
        }
    }
    /// Axis rectangle `w × h` about the origin with corner radius `r`, then placed by `t`.
    pub fn round_rect(w: Coord, h: Coord, r: Coord, t: &Transform) -> Prim {
        let r = r.clamp(0, w.min(h) / 2);
        let (hw, hh) = (w / 2 - r, h / 2 - r);
        if hw <= 0 && hh <= 0 {
            return Prim::circle(t.apply(Point::ORIGIN), r);
        }
        if hw <= 0 || hh <= 0 {
            let (a, b) = if hw <= 0 {
                (Point::new(0, -hh), Point::new(0, hh))
            } else {
                (Point::new(-hw, 0), Point::new(hw, 0))
            };
            return Prim {
                core: Core::Segment(t.apply(a), t.apply(b)),
                radius: r,
            };
        }
        let pts = [
            Point::new(-hw, -hh),
            Point::new(hw, -hh),
            Point::new(hw, hh),
            Point::new(-hw, hh),
        ];
        let mut ring: Ring = pts.iter().map(|p| t.apply(*p)).collect();
        if super::poly::area2(&ring) < 0 {
            ring.reverse();
        }
        Prim {
            core: Core::Poly(vec![ring]),
            radius: r,
        }
    }
    pub fn rect(w: Coord, h: Coord, t: &Transform) -> Prim {
        Prim::round_rect(w, h, 0, t)
    }
    /// Oval (obround) `w × h`: a capsule along the longer axis.
    pub fn oval(w: Coord, h: Coord, t: &Transform) -> Prim {
        Prim::round_rect(w, h, w.min(h) / 2, t)
    }
    /// Polygon from local points, placed by `t`.
    pub fn local_polygon(points: &[Point], t: &Transform) -> Prim {
        let mut ring: Ring = points.iter().map(|p| t.apply(*p)).collect();
        if super::poly::area2(&ring) < 0 {
            ring.reverse();
        }
        Prim::polygon(vec![ring])
    }

    /// Polygon with holes from local points, placed by `t` (outer counter-clockwise, holes
    /// clockwise whatever the transform's handedness).
    pub fn local_polygon_with_holes(points: &[Point], holes: &[Vec<Point>], t: &Transform) -> Prim {
        let place = |r: &[Point], ccw: bool| {
            let mut ring: Ring = r.iter().map(|p| t.apply(*p)).collect();
            if (super::poly::area2(&ring) > 0) != ccw {
                ring.reverse();
            }
            ring
        };
        let mut rings = vec![place(points, true)];
        rings.extend(
            holes
                .iter()
                .filter(|h| h.len() >= 3)
                .map(|h| place(h, false)),
        );
        Prim::polygon(rings)
    }

    pub fn bbox(&self) -> Rect {
        let core = match &self.core {
            Core::Point(p) => Rect::new(*p, *p),
            Core::Segment(a, b) => Rect::new(*a, *b),
            Core::Poly(rs) => Rect::from_points(rs.iter().flatten().copied()),
        };
        core.inflate(self.radius)
    }

    /// The primitive grown by `grow` as rings (CCW outers), edges outside the true shape.
    pub fn outline(&self, grow: Coord, tol: Coord) -> Vec<Ring> {
        let r = self.radius + grow;
        match &self.core {
            Core::Point(p) => vec![circle_ring(*p, r.max(1), tol)],
            Core::Segment(a, b) => vec![capsule_ring(*a, *b, r.max(1), tol)],
            Core::Poly(rs) => {
                if r <= 0 {
                    rs.clone()
                } else {
                    inflate(rs, r, tol).rings()
                }
            }
        }
    }

    /// A representative interior point.
    pub fn anchor(&self) -> Point {
        match &self.core {
            Core::Point(p) => *p,
            Core::Segment(a, _) => *a,
            Core::Poly(rs) => {
                let s = super::poly::PolySet::from_rings(rs.clone());
                interior_point(&s.rings()).unwrap_or(rs[0][0])
            }
        }
    }

    /// Gap between the two shapes: core distance minus both radii. Negative or zero when they
    /// touch or overlap.
    pub fn gap(&self, o: &Prim) -> f64 {
        core_distance(&self.core, &o.core) - self.radius as f64 - o.radius as f64
    }

    /// The nearest points of the two shapes' edges, `(on self, on o)`: the ends of the gap a
    /// clearance finding measures. Shapes that touch or overlap give one point twice.
    pub fn closest(&self, o: &Prim) -> (Point, Point) {
        let (p, q) = core_closest(&self.core, &o.core);
        let d = p.dist(q);
        let gap = d - self.radius as f64 - o.radius as f64;
        if d == 0.0 || gap <= 0.0 {
            return (p, p);
        }
        let (ux, uy) = ((q.x - p.x) as f64 / d, (q.y - p.y) as f64 / d);
        let at = |c: Point, r: f64| Point::new(c.x + (ux * r).round() as Coord, c.y + (uy * r).round() as Coord);
        (at(p, self.radius as f64), at(q, -(o.radius as f64)))
    }

    /// Whether `p` lies inside (or on) the shape.
    pub fn contains(&self, p: Point) -> bool {
        let c = Prim::circle(p, 0);
        self.gap(&c) <= 0.0
    }
}

/// A point strictly inside a filled ring set, found by probing ring vertices nudged inward.
pub fn interior_point(rings: &[Ring]) -> Option<Point> {
    let set = boolean(rings, &[], BoolOp::Union);
    for poly in &set.polys {
        let r = &poly.outer;
        let n = r.len();
        for i in 0..n {
            let a = r[(i + n - 1) % n];
            let b = r[i];
            let c = r[(i + 1) % n];
            if orient(a, b, c) <= 0 {
                continue;
            }
            // Centroid of a convex corner triangle lies inside the outer ring.
            let p = Point::new((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3);
            if poly.contains(p) == Some(true) {
                return Some(p);
            }
        }
    }
    None
}

pub fn point_segment_distance(p: Point, a: Point, b: Point) -> f64 {
    let (px, py) = p.to_f();
    let (ax, ay) = a.to_f();
    let (bx, by) = b.to_f();
    let (dx, dy) = (bx - ax, by - ay);
    let l2 = dx * dx + dy * dy;
    if l2 == 0.0 {
        return (px - ax).hypot(py - ay);
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / l2).clamp(0.0, 1.0);
    (px - ax - t * dx).hypot(py - ay - t * dy)
}

/// Exact test that `p` lies on the closed segment `a–b`.
pub fn point_on_segment(p: Point, a: Point, b: Point) -> bool {
    orient(a, b, p) == 0
        && p.x >= a.x.min(b.x)
        && p.x <= a.x.max(b.x)
        && p.y >= a.y.min(b.y)
        && p.y <= a.y.max(b.y)
}

/// Exact test that two closed segments share a point.
pub fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let d1 = orient(a, b, c);
    let d2 = orient(a, b, d);
    let d3 = orient(c, d, a);
    let d4 = orient(c, d, b);
    let on = |p: Point, q: Point, r: Point| {
        r.x >= p.x.min(q.x) && r.x <= p.x.max(q.x) && r.y >= p.y.min(q.y) && r.y <= p.y.max(q.y)
    };
    if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)) {
        return true;
    }
    (d1 == 0 && on(a, b, c))
        || (d2 == 0 && on(a, b, d))
        || (d3 == 0 && on(c, d, a))
        || (d4 == 0 && on(c, d, b))
}

pub fn segment_distance(a: Point, b: Point, c: Point, d: Point) -> f64 {
    if segments_intersect(a, b, c, d) {
        return 0.0;
    }
    point_segment_distance(a, c, d)
        .min(point_segment_distance(b, c, d))
        .min(point_segment_distance(c, a, b))
        .min(point_segment_distance(d, a, b))
}

fn poly_contains(rs: &[Ring], p: Point) -> bool {
    let mut w = 0;
    for r in rs {
        match ring_contains(r, p) {
            None => return true,
            Some(true) => w += if super::poly::area2(r) > 0 { 1 } else { -1 },
            Some(false) => {}
        }
    }
    w != 0
}

fn poly_edges(rs: &[Ring]) -> impl Iterator<Item = (Point, Point)> + '_ {
    rs.iter()
        .flat_map(|r| (0..r.len()).map(move |i| (r[i], r[(i + 1) % r.len()])))
}

pub fn core_distance(x: &Core, y: &Core) -> f64 {
    match (x, y) {
        (Core::Point(p), Core::Point(q)) => p.dist(*q),
        (Core::Point(p), Core::Segment(a, b)) | (Core::Segment(a, b), Core::Point(p)) => {
            point_segment_distance(*p, *a, *b)
        }
        (Core::Segment(a, b), Core::Segment(c, d)) => segment_distance(*a, *b, *c, *d),
        (Core::Poly(rs), Core::Point(p)) | (Core::Point(p), Core::Poly(rs)) => {
            if poly_contains(rs, *p) {
                0.0
            } else {
                poly_edges(rs)
                    .map(|(a, b)| point_segment_distance(*p, a, b))
                    .fold(f64::INFINITY, f64::min)
            }
        }
        (Core::Poly(rs), Core::Segment(a, b)) | (Core::Segment(a, b), Core::Poly(rs)) => {
            if poly_contains(rs, *a) || poly_contains(rs, *b) {
                return 0.0;
            }
            poly_edges(rs)
                .map(|(c, d)| segment_distance(*a, *b, c, d))
                .fold(f64::INFINITY, f64::min)
        }
        (Core::Poly(r1), Core::Poly(r2)) => {
            if r2
                .first()
                .and_then(|r| r.first())
                .is_some_and(|p| poly_contains(r1, *p))
                || r1
                    .first()
                    .and_then(|r| r.first())
                    .is_some_and(|p| poly_contains(r2, *p))
            {
                return 0.0;
            }
            let mut best = f64::INFINITY;
            for (a, b) in poly_edges(r1) {
                for (c, d) in poly_edges(r2) {
                    best = best.min(segment_distance(a, b, c, d));
                    if best == 0.0 {
                        return 0.0;
                    }
                }
            }
            best
        }
    }
}

fn closest_on_segment(p: Point, a: Point, b: Point) -> Point {
    let (px, py) = p.to_f();
    let (ax, ay) = a.to_f();
    let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
    let l2 = dx * dx + dy * dy;
    if l2 == 0.0 {
        return a;
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / l2).clamp(0.0, 1.0);
    Point::new(a.x + (t * dx).round() as Coord, a.y + (t * dy).round() as Coord)
}

/// The nearest pair of points between two segments (from the four end projections; crossing
/// segments are at distance zero and take the nearer projection).
fn segments_closest(a: Point, b: Point, c: Point, d: Point) -> (Point, Point) {
    [
        (a, closest_on_segment(a, c, d)),
        (b, closest_on_segment(b, c, d)),
        (closest_on_segment(c, a, b), c),
        (closest_on_segment(d, a, b), d),
    ]
    .into_iter()
    .min_by(|x, y| x.0.dist(x.1).total_cmp(&y.0.dist(y.1)))
    .unwrap()
}

fn core_edges(c: &Core) -> Vec<(Point, Point)> {
    match c {
        Core::Point(p) => vec![(*p, *p)],
        Core::Segment(a, b) => vec![(*a, *b)],
        Core::Poly(rs) => poly_edges(rs).collect(),
    }
}

/// The nearest points of two cores' edges (a point or segment is its own edge).
fn core_closest(x: &Core, y: &Core) -> (Point, Point) {
    let mut best: Option<(f64, (Point, Point))> = None;
    for (a, b) in core_edges(x) {
        for (c, d) in core_edges(y) {
            let pq = segments_closest(a, b, c, d);
            let dd = pq.0.dist(pq.1);
            if best.is_none_or(|(bd, _)| dd < bd) {
                best = Some((dd, pq));
            }
        }
    }
    best.map(|b| b.1).unwrap_or((Point::new(0, 0), Point::new(0, 0)))
}

/// Points along a circular arc from `start` sweeping `sweep_deg` (CCW positive) about
/// `center`, within `tol` of the true arc, including both ends.
pub fn arc_points(center: Point, start: Point, sweep_deg: f64, tol: Coord) -> Vec<Point> {
    let r = start.dist(center);
    let n = ((super::circle_segments(r as Coord, tol) as f64) * sweep_deg.abs() / 360.0)
        .ceil()
        .max(1.0) as usize;
    let (sx, sy) = ((start.x - center.x) as f64, (start.y - center.y) as f64);
    let a0 = sy.atan2(sx);
    let sw = sweep_deg.to_radians();
    (0..=n)
        .map(|i| {
            if i == 0 {
                return start;
            }
            let t = a0 + sw * i as f64 / n as f64;
            Point::from_f(center.x as f64 + r * t.cos(), center.y as f64 + r * t.sin())
        })
        .collect()
}

/// Centre and sweep of the arc through `start`, `mid`, `end` (KiCad's three-point form).
pub fn arc_from_three(start: Point, mid: Point, end: Point) -> Option<(Point, f64)> {
    let (ax, ay) = start.to_f();
    let (bx, by) = mid.to_f();
    let (cx, cy) = end.to_f();
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-6 {
        return None;
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    let ang = |x: f64, y: f64| (y - uy).atan2(x - ux);
    let (t0, t1, t2) = (ang(ax, ay), ang(bx, by), ang(cx, cy));
    let ccw = |from: f64, to: f64| (to - from).rem_euclid(std::f64::consts::TAU);
    let sweep = if ccw(t0, t1) < ccw(t0, t2) {
        ccw(t0, t2)
    } else {
        -(std::f64::consts::TAU - ccw(t0, t2))
    };
    Some((Point::from_f(ux, uy), sweep.to_degrees()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    #[test]
    fn gaps() {
        let t1 = Prim::segment(Point::mm(0.0, 0.0), Point::mm(10.0, 0.0), mm(0.2));
        let t2 = Prim::segment(Point::mm(0.0, 0.5), Point::mm(10.0, 0.5), mm(0.2));
        assert!((t1.gap(&t2) - mm(0.3) as f64).abs() < 1.0);
        let pad = Prim::rect(
            mm(1.0),
            mm(1.0),
            &Transform::translation(Point::mm(5.0, 1.5)),
        );
        assert!((t2.gap(&pad) - mm(0.4) as f64).abs() < 1.0);
        let rr = Prim::round_rect(
            mm(1.0),
            mm(1.0),
            mm(0.25),
            &Transform::translation(Point::mm(5.0, 1.5)),
        );
        assert!((t2.gap(&rr) - mm(0.4) as f64).abs() < 1.0);
        // Corner of a rotated square towards a circle.
        let rot = Prim::rect(
            mm(2.0),
            mm(2.0),
            &Transform::new(Point::ORIGIN, 45.0, false),
        );
        let c = Prim::circle(Point::mm(3.0, 0.0), mm(0.5));
        assert!((rot.gap(&c) - (3.0 - 2f64.sqrt() - 0.5) * 1e6).abs() < 5.0);
        assert!(rot.contains(Point::mm(0.5, 0.5)));
        assert!(!rot.contains(Point::mm(1.2, 1.2)));
        // Overlap.
        assert!(t1.gap(&Prim::circle(Point::mm(5.0, 0.0), mm(0.1))) < 0.0);
    }

    #[test]
    fn arcs() {
        let (c, sw) = arc_from_three(
            Point::mm(1.0, 0.0),
            Point::mm(0.0, 1.0),
            Point::mm(-1.0, 0.0),
        )
        .unwrap();
        assert!(c.x.abs() < 2 && c.y.abs() < 2);
        assert!((sw - 180.0).abs() < 1e-6);
        let (_, sw) = arc_from_three(
            Point::mm(1.0, 0.0),
            Point::mm(0.0, -1.0),
            Point::mm(-1.0, 0.0),
        )
        .unwrap();
        assert!((sw + 180.0).abs() < 1e-6);
        let pts = arc_points(Point::ORIGIN, Point::mm(1.0, 0.0), 90.0, 1000);
        assert_eq!(*pts.last().unwrap(), Point::mm(0.0, 1.0));
    }
}
