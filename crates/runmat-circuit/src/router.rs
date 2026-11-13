//! Interactive-routing primitives: track shaping and walkaround (push-and-shove is P5 and
//! builds on these).
//!
//! **Shaping.** A track from `a` to the cursor `b` is drawn in one of two geometries:
//! [`Geometry::Octilinear`] (segments at multiples of 45°) or [`Geometry::Orthogonal`] (90°).
//! [`shape`] gives the two-segment trace for a [`Posture`] (straight-then-diagonal or
//! diagonal-then-straight; horizontal-then-vertical or the reverse). [`mitre`] chamfers right
//! angles into two 45° corners, [`round_corners`] replaces corners with tangent arcs, and
//! [`simplify`] merges collinear runs.
//!
//! **Obstacles.** Each copper item of another net on the layer becomes a convex *hull*: the
//! item grown by the rule clearance plus half the track width, bounded by eight half-planes
//! (axis and diagonal directions) for 45° routing, or four for 90°. A path that does not enter
//! any hull's interior keeps every clearance (hulls contain the true grown shapes). The board
//! edge and track keepouts are hulls too. Zone fills of other nets are not obstacles: pours
//! are refilled around the new track.
//!
//! **Walkaround.** [`walkaround`] finds a path around the hulls in the chosen geometry:
//! - [`WalkMode::Shortest`]: Dijkstra over hull corners, joining corners with shaped
//!   (45° or 90°) connections, minimising track length;
//! - [`WalkMode::Hug`]: the interactive walkaround: go around the obstacles clockwise or
//!   counter-clockwise, tight to their hulls (the shortest path whose corners all lie on one
//!   side of the line from start to target). [`WalkMode::HugBest`] tries both turns and
//!   keeps the shorter.
//!
//! Every result is optimised (corners whose neighbours can be joined directly are dropped)
//! and checked; [`plan`] wraps obstacles, walkaround and shaping for one board edit, ready
//! for [`crate::route::insert_track`].

use crate::board_rules::{props, BItem, BoardRules, TrackProbe};
use crate::geom::shape::{arc_from_three, Prim};
use crate::geom::{Coord, Point, Rect};
use crate::layer_set::CopperIndex;
use crate::layout::{Board, NetId, TrackGeom};
use crate::spatial::Grid;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Allowed segment directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry {
    /// Multiples of 45°.
    Octilinear,
    /// Multiples of 90°.
    Orthogonal,
}

/// Which leg of a two-segment trace comes first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Posture {
    /// 45°: the axis-parallel leg first; 90°: the horizontal leg first.
    StraightFirst,
    /// 45°: the diagonal leg first; 90°: the vertical leg first.
    DiagonalFirst,
}

/// The two-segment trace from `a` to `b` (one segment when `a → b` is already allowed).
pub fn shape(a: Point, b: Point, geometry: Geometry, posture: Posture) -> Vec<Point> {
    let d = b - a;
    if d.x == 0 || d.y == 0 || (geometry == Geometry::Octilinear && d.x.abs() == d.y.abs()) {
        return if a == b { vec![a] } else { vec![a, b] };
    }
    let corner = match geometry {
        Geometry::Orthogonal => match posture {
            Posture::StraightFirst => Point::new(b.x, a.y),
            Posture::DiagonalFirst => Point::new(a.x, b.y),
        },
        Geometry::Octilinear => {
            let m = d.x.abs().min(d.y.abs());
            let diag = Point::new(m * d.x.signum(), m * d.y.signum());
            match posture {
                Posture::DiagonalFirst => a + diag,
                Posture::StraightFirst => b - diag,
            }
        }
    };
    vec![a, corner, b]
}

/// Whether the direction `a → b` is allowed in `geometry`.
pub fn allowed(a: Point, b: Point, geometry: Geometry) -> bool {
    let d = b - a;
    d.x == 0 || d.y == 0 || (geometry == Geometry::Octilinear && d.x.abs() == d.y.abs())
}

/// Length of a path, nm.
pub fn path_length(path: &[Point]) -> f64 {
    path.windows(2).map(|w| w[0].dist(w[1])).sum()
}

/// Drops repeated points and merges collinear runs that continue in the same direction.
pub fn simplify(path: &[Point]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::with_capacity(path.len());
    for &p in path {
        if out.last() == Some(&p) {
            continue;
        }
        while out.len() >= 2 {
            let (u, v) = (out[out.len() - 2], out[out.len() - 1]);
            if crate::geom::orient(u, v, p) == 0 && (v - u).dot(p - v) >= 0 {
                out.pop();
            } else {
                break;
            }
        }
        out.push(p);
    }
    out
}

/// Chamfers every right (or sharper) corner by cutting `chamfer` nm back along each leg,
/// leaving two 45° corners. Legs shorter than twice the chamfer are cut proportionally.
pub fn mitre(path: &[Point], chamfer: Coord) -> Vec<Point> {
    let path = simplify(path);
    if path.len() < 3 || chamfer <= 0 {
        return path;
    }
    let mut out = vec![path[0]];
    for i in 1..path.len() - 1 {
        let (p, v, q) = (path[i - 1], path[i], path[i + 1]);
        let (d1, d2) = (v - p, q - v);
        // Only corners of 90° or sharper (dot ≤ 0) are chamfered.
        if d1.dot(d2) > 0 {
            out.push(v);
            continue;
        }
        let (l1, l2) = (d1.len(), d2.len());
        let c = (chamfer as f64).min(l1 / 2.0).min(l2 / 2.0);
        let back = Point::from_f(
            v.x as f64 - d1.x as f64 / l1 * c,
            v.y as f64 - d1.y as f64 / l1 * c,
        );
        let fwd = Point::from_f(
            v.x as f64 + d2.x as f64 / l2 * c,
            v.y as f64 + d2.y as f64 / l2 * c,
        );
        out.push(back);
        out.push(fwd);
    }
    out.push(*path.last().unwrap());
    simplify(&out)
}

/// The path as track geometry with every corner replaced by a tangent arc of `radius`
/// (smaller where the legs are short): straight segments joined by three-point arcs.
pub fn round_corners(path: &[Point], radius: Coord) -> Vec<TrackGeom> {
    let path = simplify(path);
    let mut out = Vec::new();
    if path.len() < 2 {
        return out;
    }
    let mut start = path[0];
    for i in 1..path.len() {
        let v = path[i];
        if i + 1 == path.len() || radius <= 0 {
            out.push(TrackGeom::Segment { a: start, b: v });
            start = v;
            continue;
        }
        let (p, q) = (path[i - 1], path[i + 1]);
        let (u1, u2) = (unit(v - p), unit(q - v));
        let cos = (u1.0 * u2.0 + u1.1 * u2.1).clamp(-1.0, 1.0);
        let turn = cos.acos(); // exterior angle
        if turn < 1e-9 {
            continue;
        }
        // Tangent length for the radius; limited to half of each leg.
        let mut t = radius as f64 * (turn / 2.0).tan();
        let lim = (v.dist(p) / 2.0).min(v.dist(q) / 2.0);
        if start != p {
            t = t.min(v.dist(start));
        }
        t = t.min(lim);
        let r = t / (turn / 2.0).tan();
        let a = Point::from_f(v.x as f64 - u1.0 * t, v.y as f64 - u1.1 * t);
        let b = Point::from_f(v.x as f64 + u2.0 * t, v.y as f64 + u2.1 * t);
        // Arc midpoint: from the vertex towards the centre by (r / cos(turn/2) − r).
        let bis = unit_f(u2.0 - u1.0, u2.1 - u1.1);
        let k = r / (turn / 2.0).cos() - r;
        let mid = Point::from_f(v.x as f64 + bis.0 * k, v.y as f64 + bis.1 * k);
        if start != a {
            out.push(TrackGeom::Segment { a: start, b: a });
        }
        if arc_from_three(a, mid, b).is_some() {
            out.push(TrackGeom::Arc {
                start: a,
                mid,
                end: b,
            });
        } else {
            out.push(TrackGeom::Segment { a, b });
        }
        start = b;
    }
    out
}

fn unit(d: Point) -> (f64, f64) {
    unit_f(d.x as f64, d.y as f64)
}

fn unit_f(x: f64, y: f64) -> (f64, f64) {
    let l = x.hypot(y);
    if l == 0.0 {
        (0.0, 0.0)
    } else {
        (x / l, y / l)
    }
}

/// A convex hull bounded by axis and (for 45°) diagonal half-planes:
/// `x ∈ [x0, x1]`, `y ∈ [y0, y1]`, `x + y ∈ [s0, s1]`, `x − y ∈ [d0, d1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hull {
    pub x0: Coord,
    pub x1: Coord,
    pub y0: Coord,
    pub y1: Coord,
    pub s0: Coord,
    pub s1: Coord,
    pub d0: Coord,
    pub d1: Coord,
}

impl Hull {
    /// The hull of `prim` grown by `grow`.
    pub fn of(prim: &Prim, grow: Coord, geometry: Geometry) -> Hull {
        let r = (prim.radius + grow) as f64;
        let pts: Vec<Point> = match &prim.core {
            crate::geom::shape::Core::Point(p) => vec![*p],
            crate::geom::shape::Core::Segment(a, b) => vec![*a, *b],
            crate::geom::shape::Core::Poly(rs) => rs.iter().flatten().copied().collect(),
        };
        let ext = |f: &dyn Fn(Point) -> f64, k: f64| {
            let lo = pts.iter().map(|p| f(*p)).fold(f64::INFINITY, f64::min) - r * k;
            let hi = pts.iter().map(|p| f(*p)).fold(f64::NEG_INFINITY, f64::max) + r * k;
            // Outward by a nanometre so integer vertices never fall inside the true shape.
            (lo.floor() as Coord - 1, hi.ceil() as Coord + 1)
        };
        let (x0, x1) = ext(&|p| p.x as f64, 1.0);
        let (y0, y1) = ext(&|p| p.y as f64, 1.0);
        let (s0, s1, d0, d1) = match geometry {
            Geometry::Octilinear => {
                let s = ext(&|p| (p.x + p.y) as f64, std::f64::consts::SQRT_2);
                let d = ext(&|p| (p.x - p.y) as f64, std::f64::consts::SQRT_2);
                (s.0, s.1, d.0, d.1)
            }
            Geometry::Orthogonal => (x0 + y0, x1 + y1, x0 - y1, x1 - y0),
        };
        Hull {
            x0,
            x1,
            y0,
            y1,
            s0: s0.max(x0 + y0),
            s1: s1.min(x1 + y1),
            d0: d0.max(x0 - y1),
            d1: d1.min(x1 - y0),
        }
    }
    pub fn bbox(&self) -> Rect {
        Rect::new(Point::new(self.x0, self.y0), Point::new(self.x1, self.y1))
    }
    /// Corners counter-clockwise from the bottom edge (duplicates removed).
    pub fn corners(&self) -> Vec<Point> {
        let h = self;
        let mut v = vec![
            Point::new(h.s0 - h.y0, h.y0),
            Point::new(h.d1 + h.y0, h.y0),
            Point::new(h.x1, h.x1 - h.d1),
            Point::new(h.x1, h.s1 - h.x1),
            Point::new(h.s1 - h.y1, h.y1),
            Point::new(h.d0 + h.y1, h.y1),
            Point::new(h.x0, h.x0 - h.d0),
            Point::new(h.x0, h.s0 - h.x0),
        ];
        v.dedup();
        if v.len() > 1 && v.first() == v.last() {
            v.pop();
        }
        v
    }
    /// Strictly inside (the boundary is free).
    pub fn contains(&self, p: Point) -> bool {
        let (s, d) = (p.x + p.y, p.x - p.y);
        p.x > self.x0
            && p.x < self.x1
            && p.y > self.y0
            && p.y < self.y1
            && s > self.s0
            && s < self.s1
            && d > self.d0
            && d < self.d1
    }
    /// Whether the segment `a–b` passes through the interior (touching is allowed).
    pub fn blocks(&self, a: Point, b: Point) -> bool {
        // Clip the parameter range against each open half-plane `lo < f(p) < hi`.
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let fs: [(f64, f64, Coord, Coord); 4] = [
            (a.x as f64, (b.x - a.x) as f64, self.x0, self.x1),
            (a.y as f64, (b.y - a.y) as f64, self.y0, self.y1),
            (
                (a.x + a.y) as f64,
                ((b.x + b.y) - (a.x + a.y)) as f64,
                self.s0,
                self.s1,
            ),
            (
                (a.x - a.y) as f64,
                ((b.x - b.y) - (a.x - a.y)) as f64,
                self.d0,
                self.d1,
            ),
        ];
        for (f0, df, lo, hi) in fs {
            let (lo, hi) = (lo as f64, hi as f64);
            if df == 0.0 {
                if f0 <= lo || f0 >= hi {
                    return false;
                }
                continue;
            }
            let (ta, tb) = ((lo - f0) / df, (hi - f0) / df);
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
            if t1 - t0 <= 1e-12 {
                return false;
            }
        }
        // A positive-length piece inside; ignore pieces shorter than a nanometre.
        (t1 - t0) * a.dist(b) > 0.5
    }
}

/// Hulls of everything a track of one net may not cross on one layer.
#[derive(Clone, Debug)]
pub struct Obstacles {
    pub geometry: Geometry,
    pub hulls: Vec<(BItem, Hull)>,
    grid: Grid<u32>,
}

impl Obstacles {
    pub fn new(geometry: Geometry, hulls: Vec<(BItem, Hull)>) -> Obstacles {
        let mut grid = Grid::new(crate::geom::mm(1.0));
        for (k, (_, h)) in hulls.iter().enumerate() {
            grid.insert(k as u32, h.bbox());
        }
        Obstacles {
            geometry,
            hulls,
            grid,
        }
    }
    /// The first hull (by index) the segment passes through.
    pub fn blocker(&self, a: Point, b: Point) -> Option<usize> {
        let mut hit: Option<usize> = None;
        self.grid.visit(&Rect::new(a, b), |k, _| {
            let k = k as usize;
            if hit.is_none_or(|h| k < h) && self.hulls[k].1.blocks(a, b) {
                hit = Some(k);
            }
        });
        hit
    }
    pub fn segment_clear(&self, a: Point, b: Point) -> bool {
        self.blocker(a, b).is_none()
    }
    pub fn path_clear(&self, path: &[Point]) -> bool {
        path.windows(2).all(|w| self.segment_clear(w[0], w[1]))
    }
    pub fn inside(&self, p: Point) -> bool {
        let mut hit = false;
        self.grid.visit(&Rect::new(p, p), |k, _| {
            hit |= self.hulls[k as usize].1.contains(p);
        });
        hit
    }
    /// A clear shaped connection `a → b`, trying both postures (straight-first first).
    pub fn shaped(&self, a: Point, b: Point) -> Option<Vec<Point>> {
        [Posture::StraightFirst, Posture::DiagonalFirst]
            .into_iter()
            .map(|p| shape(a, b, self.geometry, p))
            .find(|s| self.path_clear(s))
    }
}

/// Builds the obstacles for routing `net` on `layer` with `width` inside `region`: other-net
/// pads, tracks and vias grown by the pair clearance plus half the width, track keepouts on
/// the layer, and the board outline grown by the edge clearance.
pub fn obstacles(
    board: &Board,
    rules: &BoardRules,
    net: Option<NetId>,
    layer: CopperIndex,
    width: Coord,
    region: Rect,
    geometry: Geometry,
) -> Obstacles {
    let probe = rules.probe(board, TrackProbe { net, layer, width });
    let mut hulls: Vec<(BItem, Hull)> = Vec::new();
    let mut add = |item: BItem, item_net: Option<NetId>, p: &Prim| {
        if net.is_some() && item_net == net {
            return;
        }
        let grow = rules
            .probe_len(&probe, item, props::CLEARANCE)
            .max(board.limits.min_clearance)
            + width / 2;
        if p.bbox().inflate(grow).intersects(&region) {
            hulls.push((item, Hull::of(p, grow, geometry)));
        }
    };
    for p in board.all_pads() {
        if p.layers.contains(&layer) {
            add(BItem::Pad(p.footprint, p.index as u32), p.net, &p.prim);
        }
    }
    for t in board.tracks.values() {
        if t.layer == layer {
            for p in t.prims() {
                add(BItem::Track(t.id), t.net, &p);
            }
        }
    }
    for v in board.vias.values() {
        if v.layers().any(|l| l == layer) {
            add(BItem::Via(v.id), v.net, &Prim::circle(v.at, v.diameter / 2));
        }
    }
    for k in board.keepouts.values() {
        if k.rules.tracks && (k.layers.is_empty() || k.layers.contains(&layer)) {
            let p = Prim::polygon(vec![k.outline.outer.clone()]);
            hulls.push((BItem::Edge, Hull::of(&p, width / 2, geometry)));
        }
    }
    let edge = rules.probe_len(&probe, BItem::Edge, props::EDGE_CLEARANCE) + width / 2;
    for r in board.outline.rings() {
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            let p = Prim::segment(a, b, 0);
            if p.bbox().inflate(edge).intersects(&region) {
                // A thin hull per edge: axis edges exactly, others by their own hull.
                hulls.push((BItem::Edge, Hull::of(&p, edge, geometry)));
            }
        }
    }
    Obstacles::new(geometry, hulls)
}

/// How [`walkaround`] goes around obstacles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkMode {
    /// Shortest shaped path over hull corners.
    Shortest,
    /// Around the obstacles clockwise (`true`) or counter-clockwise, tight to their hulls.
    Hug { clockwise: bool },
    /// Hug both ways and keep the shorter result.
    HugBest,
}

/// A path from `a` to `b` around `obs` in its geometry, or `None` if an end is blocked or no
/// path is found. The path is simplified and every segment is allowed and clear.
pub fn walkaround(obs: &Obstacles, a: Point, b: Point, mode: WalkMode) -> Option<Vec<Point>> {
    if obs.inside(a) || obs.inside(b) {
        return None;
    }
    if let Some(s) = obs.shaped(a, b) {
        return Some(simplify(&s));
    }
    let path = match mode {
        WalkMode::Shortest => shortest(obs, a, b)?,
        WalkMode::Hug { clockwise } => hug(obs, a, b, clockwise)?,
        WalkMode::HugBest => {
            let cw = hug(obs, a, b, true);
            let ccw = hug(obs, a, b, false);
            match (cw, ccw) {
                (Some(x), Some(y)) => {
                    if path_length(&y) < path_length(&x) {
                        y
                    } else {
                        x
                    }
                }
                (x, y) => x.or(y)?,
            }
        }
    };
    let path = optimise(obs, &path);
    debug_assert!(obs.path_clear(&path));
    (obs.path_clear(&path) && path.windows(2).all(|w| allowed(w[0], w[1], obs.geometry)))
        .then_some(path)
}

/// Drops corners whose neighbours have a clear shaped connection no longer than the detour.
pub fn optimise(obs: &Obstacles, path: &[Point]) -> Vec<Point> {
    let mut p = simplify(path);
    let mut changed = true;
    while changed && p.len() > 2 {
        changed = false;
        let mut i = 0;
        while i + 2 < p.len() {
            let (u, w) = (p[i], p[i + 2]);
            let detour = p[i].dist(p[i + 1]) + p[i + 1].dist(p[i + 2]);
            if let Some(s) = obs.shaped(u, w) {
                if path_length(&s) + 1e-6 < detour || (s.len() == 2 && s[1] == w) {
                    let mut next = p[..=i].to_vec();
                    next.extend_from_slice(&s[1..]);
                    next.extend_from_slice(&p[i + 3..]);
                    let next = simplify(&next);
                    if next.len() < p.len() || path_length(&next) + 1e-6 < path_length(&p) {
                        p = next;
                        changed = true;
                        continue;
                    }
                }
            }
            i += 1;
        }
    }
    p
}

fn shortest(obs: &Obstacles, a: Point, b: Point) -> Option<Vec<Point>> {
    shortest_by(obs, a, b, |_| true)
}

/// Dijkstra over `a`, `b` and the hull corners outside every hull that `keep` accepts.
fn shortest_by(
    obs: &Obstacles,
    a: Point,
    b: Point,
    keep: impl Fn(Point) -> bool,
) -> Option<Vec<Point>> {
    let mut pts = vec![a, b];
    for (_, h) in &obs.hulls {
        for c in h.corners() {
            if keep(c) && !obs.inside(c) {
                pts.push(c);
            }
        }
    }
    pts.sort_by_key(|p| (p.x, p.y));
    pts.dedup();
    let ia = pts.iter().position(|p| *p == a)?;
    let ib = pts.iter().position(|p| *p == b)?;
    let n = pts.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<(usize, Vec<Point>)>> = vec![None; n];
    let mut done = vec![false; n];
    dist[ia] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0u64, ia)));
    while let Some(Reverse((_, u))) = heap.pop() {
        if done[u] {
            continue;
        }
        done[u] = true;
        if u == ib {
            break;
        }
        for v in 0..n {
            if done[v] {
                continue;
            }
            // Lower bound first: the geometry's own metric.
            let d = pts[v] - pts[u];
            let (dx, dy) = (d.x.abs() as f64, d.y.abs() as f64);
            let lb = match obs.geometry {
                Geometry::Orthogonal => dx + dy,
                Geometry::Octilinear => dx.max(dy) + (std::f64::consts::SQRT_2 - 1.0) * dx.min(dy),
            };
            if dist[u] + lb >= dist[v] {
                continue;
            }
            if let Some(s) = obs.shaped(pts[u], pts[v]) {
                let nd = dist[u] + path_length(&s);
                if nd < dist[v] {
                    dist[v] = nd;
                    prev[v] = Some((u, s));
                    heap.push(Reverse((nd as u64, v)));
                }
            }
        }
    }
    if !dist[ib].is_finite() {
        return None;
    }
    let mut legs: Vec<Vec<Point>> = Vec::new();
    let mut cur = ib;
    while let Some((u, s)) = &prev[cur] {
        legs.push(s.clone());
        cur = *u;
    }
    legs.reverse();
    let mut out = vec![a];
    for l in legs {
        out.extend_from_slice(&l[1..]);
    }
    Some(simplify(&out))
}

fn hug(obs: &Obstacles, a: Point, b: Point, clockwise: bool) -> Option<Vec<Point>> {
    // Going around clockwise keeps the obstacles on the right: every corner on the left of
    // the line a → b (and counter-clockwise the reverse).
    shortest_by(obs, a, b, |c| {
        let o = crate::geom::orient(a, b, c);
        if clockwise {
            o >= 0
        } else {
            o <= 0
        }
    })
}

/// A planned interactive route: the centreline and its track geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub path: Vec<Point>,
    pub tracks: Vec<TrackGeom>,
    pub length: f64,
}

/// Corner treatment for [`plan`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Corners {
    Sharp,
    /// Chamfer right angles by this much (90° routing becomes 45° corners).
    Mitre(Coord),
    /// Tangent arcs of this radius.
    Arc(Coord),
}

/// Plans a track for `net` on `layer` from `a` to `b`: obstacles in a window around the two
/// points, a walkaround in `mode`, then the corner treatment. Corner treatments that would
/// collide fall back to sharp corners.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    board: &Board,
    rules: &BoardRules,
    net: Option<NetId>,
    layer: CopperIndex,
    width: Coord,
    a: Point,
    b: Point,
    geometry: Geometry,
    mode: WalkMode,
    corners: Corners,
) -> Option<Plan> {
    let margin = crate::geom::mm(10.0) + a.dist(b) as Coord / 2;
    let region = Rect::new(a, b).inflate(margin);
    let obs = obstacles(board, rules, net, layer, width, region, geometry);
    let path = walkaround(&obs, a, b, mode)?;
    let tracks = match corners {
        Corners::Mitre(c) => {
            let m = mitre(&path, c);
            if obs.path_clear(&m) {
                segments(&m)
            } else {
                segments(&path)
            }
        }
        Corners::Arc(r) => {
            let t = round_corners(&path, r);
            // The arcs cut inside each corner: check their flattened centrelines.
            let clear = t.iter().all(|g| match *g {
                TrackGeom::Segment { a, b } => obs.segment_clear(a, b),
                TrackGeom::Arc { start, mid, end } => match arc_from_three(start, mid, end) {
                    Some((c, sw)) => {
                        crate::geom::shape::arc_points(c, start, sw, crate::geom::ARC_TOL)
                            .windows(2)
                            .all(|w| obs.segment_clear(w[0], w[1]))
                    }
                    None => obs.segment_clear(start, end),
                },
            });
            if t.is_empty() || !clear {
                segments(&path)
            } else {
                t
            }
        }
        Corners::Sharp => segments(&path),
    };
    let length = tracks
        .iter()
        .map(|g| match *g {
            TrackGeom::Segment { a, b } => a.dist(b),
            TrackGeom::Arc { start, mid, end } => match arc_from_three(start, mid, end) {
                Some((c, sw)) => c.dist(start) * sw.abs().to_radians(),
                None => start.dist(end),
            },
        })
        .sum();
    Some(Plan {
        path,
        tracks,
        length,
    })
}

fn segments(path: &[Point]) -> Vec<TrackGeom> {
    path.windows(2)
        .map(|w| TrackGeom::Segment { a: w[0], b: w[1] })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    fn obs_of(prims: &[Prim], grow: Coord, g: Geometry) -> Obstacles {
        Obstacles::new(
            g,
            prims
                .iter()
                .map(|p| (BItem::Edge, Hull::of(p, grow, g)))
                .collect(),
        )
    }

    #[test]
    fn shapes_and_postures() {
        let (a, b) = (Point::mm(0.0, 0.0), Point::mm(5.0, 2.0));
        assert_eq!(
            shape(a, b, Geometry::Octilinear, Posture::StraightFirst),
            vec![a, Point::mm(3.0, 0.0), b]
        );
        assert_eq!(
            shape(a, b, Geometry::Octilinear, Posture::DiagonalFirst),
            vec![a, Point::mm(2.0, 2.0), b]
        );
        assert_eq!(
            shape(a, b, Geometry::Orthogonal, Posture::StraightFirst),
            vec![a, Point::mm(5.0, 0.0), b]
        );
        assert_eq!(
            shape(a, b, Geometry::Orthogonal, Posture::DiagonalFirst),
            vec![a, Point::mm(0.0, 2.0), b]
        );
        let d = Point::mm(3.0, -3.0);
        assert_eq!(
            shape(a, d, Geometry::Octilinear, Posture::StraightFirst),
            vec![a, d]
        );
        // Mitre a right angle into two 45° corners.
        let m = mitre(&[a, Point::mm(5.0, 0.0), Point::mm(5.0, 5.0)], mm(1.0));
        assert_eq!(
            m,
            vec![
                a,
                Point::mm(4.0, 0.0),
                Point::mm(5.0, 1.0),
                Point::mm(5.0, 5.0)
            ]
        );
        assert!(m
            .windows(2)
            .all(|w| allowed(w[0], w[1], Geometry::Octilinear)));
        // Round a right angle: segment, quarter arc of radius 1 mm, segment.
        let r = round_corners(&[a, Point::mm(5.0, 0.0), Point::mm(5.0, 5.0)], mm(1.0));
        assert_eq!(r.len(), 3);
        let TrackGeom::Arc { start, mid, end } = r[1] else {
            panic!("{r:?}")
        };
        let (c, sw) = arc_from_three(start, mid, end).unwrap();
        assert!((c.dist(start) - 1e6).abs() < 10.0 && (sw.abs() - 90.0).abs() < 0.01);
        assert_eq!(
            simplify(&[a, a, Point::mm(1.0, 0.0), Point::mm(2.0, 0.0)]).len(),
            2
        );
    }

    #[test]
    fn hulls_contain_grown_shapes() {
        for g in [Geometry::Octilinear, Geometry::Orthogonal] {
            let c = Prim::circle(Point::mm(1.0, 1.0), mm(0.3));
            let h = Hull::of(&c, mm(0.2), g);
            // Every point of the grown circle is inside or on the hull.
            for k in 0..64 {
                let t = k as f64 / 64.0 * std::f64::consts::TAU;
                let p = Point::from_f(1e6 + 0.5e6 * t.cos(), 1e6 + 0.5e6 * t.sin());
                let inside_or_on = p.x >= h.x0
                    && p.x <= h.x1
                    && p.y >= h.y0
                    && p.y <= h.y1
                    && p.x + p.y >= h.s0
                    && p.x + p.y <= h.s1
                    && p.x - p.y >= h.d0
                    && p.x - p.y <= h.d1;
                assert!(inside_or_on, "{g:?} {p:?}");
            }
            let n = h.corners().len();
            assert_eq!(n, if g == Geometry::Octilinear { 8 } else { 4 });
            // A segment through the middle is blocked; one along an edge is not.
            assert!(h.blocks(Point::mm(-1.0, 1.0), Point::mm(3.0, 1.0)));
            let e = h.corners();
            assert!(!h.blocks(e[0], e[1]));
        }
    }

    #[test]
    fn walkaround_modes_route_around_a_wall() {
        // A vertical wall of vias between a and b with a gap at the top.
        let mut prims = Vec::new();
        for i in 0..10 {
            prims.push(Prim::circle(Point::mm(5.0, i as f64 * 0.8), mm(0.3)));
        }
        let (a, b) = (Point::mm(0.0, 2.0), Point::mm(10.0, 2.0));
        for g in [Geometry::Octilinear, Geometry::Orthogonal] {
            let obs = obs_of(&prims, mm(0.25), g);
            let mut lengths = Vec::new();
            for mode in [
                WalkMode::Shortest,
                WalkMode::Hug { clockwise: true },
                WalkMode::Hug { clockwise: false },
                WalkMode::HugBest,
            ] {
                let Some(p) = walkaround(&obs, a, b, mode) else {
                    lengths.push(f64::INFINITY);
                    continue;
                };
                assert_eq!((p[0], *p.last().unwrap()), (a, b), "{g:?} {mode:?}");
                assert!(obs.path_clear(&p), "{g:?} {mode:?} {p:?}");
                assert!(
                    p.windows(2).all(|w| allowed(w[0], w[1], g)),
                    "{g:?} {mode:?}"
                );
                lengths.push(path_length(&p));
            }
            // Shortest is no longer than any hugging result; best-of-two is the better hug.
            assert!(lengths[0].is_finite());
            assert!(
                lengths[1..].iter().all(|l| *l >= lengths[0] - 1.0),
                "{g:?} {lengths:?}"
            );
            assert!(lengths[3] <= lengths[1].min(lengths[2]) + 1.0);
            assert!(lengths[3].is_finite());
        }
        // Blocked end.
        let obs = obs_of(&prims, mm(0.25), Geometry::Octilinear);
        assert!(walkaround(&obs, Point::mm(5.0, 0.0), b, WalkMode::Shortest).is_none());
    }
}
