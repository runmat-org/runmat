//! Exact polygon booleans and offsets on integer coordinates.
//!
//! # Algorithm
//!
//! 1. **Edges.** Every ring edge of both operands becomes an edge stored bottom-to-top in
//!    `(y, x)` order with a winding contribution per operand (+1 when the ring runs upward,
//!    −1 when downward). Rings may have any orientation and may overlap; each operand is
//!    filled by the non-zero rule.
//! 2. **Arrangement.** Edges are split at every crossing, T-junction and collinear overlap,
//!    found with exact `i128` orientation tests. Crossing points are rounded to the integer
//!    grid; splitting repeats until no new crossings appear, which handles the rare
//!    rounding-induced crossings. Identical edges are merged by summing contributions, so
//!    shared boundaries cancel exactly.
//! 3. **Classification.** A sweep over the distinct `y` values keeps the active edges
//!    ordered by `x` inside each slab (exact rational comparison). Since no two edges cross
//!    inside a slab, one prefix sum gives the winding of both operands on each side of every
//!    edge. An edge is a result boundary when the boolean predicate differs on its two sides.
//!    Horizontal edges are classified by querying the windings just below and just above.
//! 4. **Rings.** Boundary edges are directed with the result's interior on the left and
//!    linked into rings, taking the leftmost turn at shared vertices so rings touching at a
//!    point separate. Positive-area rings are outers (counter-clockwise); negative-area rings
//!    are holes, each assigned to the smallest outer containing it.
//!
//! The result is canonical for a given input: no hashing, fixed sort orders, and each ring
//! starts at its lowest `(y, x)` vertex.

use super::{circle_segments, orient, Coord, Point, Rect};
use std::cmp::Ordering;
use std::collections::BTreeMap;

pub type Ring = Vec<Point>;

/// A polygon with holes: outer counter-clockwise, holes clockwise.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Polygon {
    pub outer: Ring,
    pub holes: Vec<Ring>,
}

/// A set of disjoint polygons (a multipolygon), the result type of every boolean.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PolySet {
    pub polys: Vec<Polygon>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    Union,
    Intersection,
    Difference,
    Xor,
}

/// Twice the signed area (positive for counter-clockwise).
pub fn area2(r: &[Point]) -> i128 {
    let n = r.len();
    let mut s = 0i128;
    for i in 0..n {
        s += r[i].cross(r[(i + 1) % n]);
    }
    s
}

pub fn ring_area(r: &[Point]) -> f64 {
    area2(r) as f64 / 2.0
}

pub fn ring_bbox(r: &[Point]) -> Rect {
    Rect::from_points(r.iter().copied())
}

/// Point location against a ring: `Some(true)` inside, `Some(false)` outside, `None` on the
/// boundary. Uses non-zero winding, so orientation does not matter.
pub fn ring_contains(r: &[Point], p: Point) -> Option<bool> {
    let n = r.len();
    let mut w = 0i32;
    for i in 0..n {
        let a = r[i];
        let b = r[(i + 1) % n];
        let o = orient(a, b, p);
        if o == 0
            && p.x >= a.x.min(b.x)
            && p.x <= a.x.max(b.x)
            && p.y >= a.y.min(b.y)
            && p.y <= a.y.max(b.y)
        {
            return None;
        }
        if a.y <= p.y {
            if b.y > p.y && o > 0 {
                w += 1;
            }
        } else if b.y <= p.y && o < 0 {
            w -= 1;
        }
    }
    Some(w != 0)
}

impl Polygon {
    pub fn from_ring(mut r: Ring) -> Polygon {
        if area2(&r) < 0 {
            r.reverse();
        }
        Polygon {
            outer: r,
            holes: Vec::new(),
        }
    }
    pub fn area(&self) -> f64 {
        ring_area(&self.outer) + self.holes.iter().map(|h| ring_area(h)).sum::<f64>()
    }
    pub fn bbox(&self) -> Rect {
        ring_bbox(&self.outer)
    }
    /// `Some(true)` strictly inside the filled area, `None` on any boundary.
    pub fn contains(&self, p: Point) -> Option<bool> {
        match ring_contains(&self.outer, p) {
            Some(true) => {}
            other => return other,
        }
        for h in &self.holes {
            match ring_contains(h, p) {
                Some(true) => return Some(false),
                None => return None,
                _ => {}
            }
        }
        Some(true)
    }
    pub fn rings(&self) -> impl Iterator<Item = &Ring> {
        std::iter::once(&self.outer).chain(self.holes.iter())
    }
}

impl PolySet {
    pub fn new() -> Self {
        PolySet::default()
    }
    pub fn from_rings(rings: Vec<Ring>) -> Self {
        boolean(&rings, &[], BoolOp::Union)
    }
    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
    }
    pub fn area(&self) -> f64 {
        self.polys.iter().map(|p| p.area()).sum()
    }
    pub fn bbox(&self) -> Rect {
        self.polys
            .iter()
            .fold(Rect::EMPTY, |r, p| r.union(p.bbox()))
    }
    /// All rings with their orientation (outers CCW, holes CW), suitable as an operand.
    pub fn rings(&self) -> Vec<Ring> {
        self.polys.iter().flat_map(|p| p.rings().cloned()).collect()
    }
    pub fn contains(&self, p: Point) -> Option<bool> {
        let mut on = false;
        for poly in &self.polys {
            if !poly.bbox().contains(p) {
                continue;
            }
            match poly.contains(p) {
                Some(true) => return Some(true),
                None => on = true,
                _ => {}
            }
        }
        if on {
            None
        } else {
            Some(false)
        }
    }
    pub fn op(&self, other: &PolySet, op: BoolOp) -> PolySet {
        boolean(&self.rings(), &other.rings(), op)
    }
    pub fn union(&self, o: &PolySet) -> PolySet {
        self.op(o, BoolOp::Union)
    }
    pub fn intersection(&self, o: &PolySet) -> PolySet {
        self.op(o, BoolOp::Intersection)
    }
    pub fn difference(&self, o: &PolySet) -> PolySet {
        self.op(o, BoolOp::Difference)
    }
    pub fn xor(&self, o: &PolySet) -> PolySet {
        self.op(o, BoolOp::Xor)
    }
    pub fn vertex_count(&self) -> usize {
        self.polys
            .iter()
            .map(|p| p.rings().map(|r| r.len()).sum::<usize>())
            .sum()
    }
}

// ---------------------------------------------------------------- engine

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct E {
    a: Point,
    b: Point,
    w: [i32; 2],
}

fn ylt(p: Point, q: Point) -> bool {
    (p.y, p.x) < (q.y, q.x)
}

fn make_edge(p: Point, q: Point, w: [i32; 2]) -> Option<E> {
    if p == q {
        return None;
    }
    if ylt(p, q) {
        Some(E { a: p, b: q, w })
    } else {
        Some(E {
            a: q,
            b: p,
            w: [-w[0], -w[1]],
        })
    }
}

fn round_div(n: i128, d: i128) -> i128 {
    let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

fn on_segment_interior(a: Point, b: Point, p: Point) -> bool {
    p != a
        && p != b
        && orient(a, b, p) == 0
        && p.x >= a.x.min(b.x)
        && p.x <= a.x.max(b.x)
        && p.y >= a.y.min(b.y)
        && p.y <= a.y.max(b.y)
}

/// Adds the split points two edges induce on each other.
fn intersect(e: &E, f: &E, se: &mut Vec<Point>, sf: &mut Vec<Point>) {
    let d1 = orient(e.a, e.b, f.a);
    let d2 = orient(e.a, e.b, f.b);
    let d3 = orient(f.a, f.b, e.a);
    let d4 = orient(f.a, f.b, e.b);
    if d1 == 0 && d2 == 0 {
        for p in [f.a, f.b] {
            if on_segment_interior(e.a, e.b, p) {
                se.push(p);
            }
        }
        for p in [e.a, e.b] {
            if on_segment_interior(f.a, f.b, p) {
                sf.push(p);
            }
        }
        return;
    }
    if d1 != 0 && d2 != 0 && d3 != 0 && d4 != 0 && (d1 > 0) != (d2 > 0) && (d3 > 0) != (d4 > 0) {
        let r = e.b - e.a;
        let s = f.b - f.a;
        let num = (f.a - e.a).cross(s);
        let den = r.cross(s);
        let x = e.a.x as i128 + round_div(r.x as i128 * num, den);
        let y = e.a.y as i128 + round_div(r.y as i128 * num, den);
        let p = Point::new(x as Coord, y as Coord);
        if p != e.a && p != e.b {
            se.push(p);
        }
        if p != f.a && p != f.b {
            sf.push(p);
        }
        return;
    }
    if d1 == 0 && on_segment_interior(e.a, e.b, f.a) {
        se.push(f.a);
    }
    if d2 == 0 && on_segment_interior(e.a, e.b, f.b) {
        se.push(f.b);
    }
    if d3 == 0 && on_segment_interior(f.a, f.b, e.a) {
        sf.push(e.a);
    }
    if d4 == 0 && on_segment_interior(f.a, f.b, e.b) {
        sf.push(e.b);
    }
}

fn split_pass(edges: &[E]) -> Option<Vec<E>> {
    let n = edges.len();
    let mut order: Vec<usize> = (0..n).collect();
    let minx = |e: &E| e.a.x.min(e.b.x);
    let maxx = |e: &E| e.a.x.max(e.b.x);
    order.sort_by_key(|&i| (minx(&edges[i]), i));
    let mut splits: Vec<Vec<Point>> = vec![Vec::new(); n];
    let mut any = false;
    for oi in 0..n {
        let i = order[oi];
        let ei = edges[i];
        let mxi = maxx(&ei);
        let (ylo, yhi) = (ei.a.y, ei.b.y);
        for &j in &order[oi + 1..] {
            let ej = edges[j];
            if minx(&ej) > mxi {
                break;
            }
            if ej.b.y < ylo || ej.a.y > yhi {
                continue;
            }
            let (mut si, mut sj) = (Vec::new(), Vec::new());
            intersect(&ei, &ej, &mut si, &mut sj);
            if !si.is_empty() || !sj.is_empty() {
                any = true;
                splits[i].extend(si);
                splits[j].extend(sj);
            }
        }
    }
    if !any {
        return None;
    }
    let mut out = Vec::with_capacity(n + n / 4);
    for (i, e) in edges.iter().enumerate() {
        if splits[i].is_empty() {
            out.push(*e);
            continue;
        }
        let d = e.b - e.a;
        let mut pts = std::mem::take(&mut splits[i]);
        pts.sort_by_key(|p| ((*p - e.a).dot(d), *p));
        pts.dedup();
        let mut prev = e.a;
        for p in pts.into_iter().chain(std::iter::once(e.b)) {
            // Pieces keep the original direction's contribution.
            if let Some(x) = make_edge(prev, p, e.w) {
                out.push(x);
            }
            prev = p;
        }
    }
    Some(out)
}

/// `x` of edge `e` at doubled height `y2`, as a fraction `(num, den)` with `den > 0`.
fn x_at2(e: &E, y2: i128) -> (i128, i128) {
    let dy = (e.b.y - e.a.y) as i128;
    let dx = (e.b.x - e.a.x) as i128;
    (
        2 * e.a.x as i128 * dy + dx * (y2 - 2 * e.a.y as i128),
        2 * dy,
    )
}

fn cmp_at(e: &E, f: &E, y2: i128) -> Ordering {
    let (n1, d1) = x_at2(e, y2);
    let (n2, d2) = x_at2(f, y2);
    (n1 * d2).cmp(&(n2 * d1)).then_with(|| {
        // Equal at the slab middle only for degenerate residue; order by slope dx/dy.
        let s1 = (e.b.x - e.a.x) as i128 * (f.b.y - f.a.y) as i128;
        let s2 = (f.b.x - f.a.x) as i128 * (e.b.y - e.a.y) as i128;
        s1.cmp(&s2).then(e.cmp(f))
    })
}

/// `2·x_at(y) < s2` for an edge spanning `y` (horizontal-query helper).
fn left_of(e: &E, y: Coord, s2: i128) -> bool {
    let dy = (e.b.y - e.a.y) as i128;
    let dx = (e.b.x - e.a.x) as i128;
    let num = 2 * (e.a.x as i128 * dy + dx * (y - e.a.y) as i128);
    num < s2 * dy
}

fn pred(op: BoolOp, w: [i32; 2]) -> bool {
    let a = w[0] != 0;
    let b = w[1] != 0;
    match op {
        BoolOp::Union => a || b,
        BoolOp::Intersection => a && b,
        BoolOp::Difference => a && !b,
        BoolOp::Xor => a != b,
    }
}

/// Boolean of two ring sets, each filled by the non-zero rule.
pub fn boolean(a: &[Ring], b: &[Ring], op: BoolOp) -> PolySet {
    let mut edges = Vec::new();
    for (k, set) in [a, b].iter().enumerate() {
        for r in set.iter() {
            let n = r.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let mut w = [0, 0];
                w[k] = 1;
                if let Some(e) = make_edge(r[i], r[(i + 1) % n], w) {
                    edges.push(e);
                }
            }
        }
    }
    let boundary = classify(edges, op);
    build_rings(boundary)
}

fn classify(mut edges: Vec<E>, op: BoolOp) -> Vec<(Point, Point)> {
    for _ in 0..16 {
        match split_pass(&edges) {
            Some(e) => edges = e,
            None => break,
        }
    }
    // Merge identical edges.
    edges.sort_by(|x, y| (x.a, x.b).cmp(&(y.a, y.b)));
    let mut merged: Vec<E> = Vec::with_capacity(edges.len());
    for e in edges {
        if let Some(last) = merged.last_mut() {
            if last.a == e.a && last.b == e.b {
                last.w[0] += e.w[0];
                last.w[1] += e.w[1];
                continue;
            }
        }
        merged.push(e);
    }
    let (horiz, mut vert): (Vec<E>, Vec<E>) = merged.into_iter().partition(|e| e.a.y == e.b.y);
    vert.retain(|e| e.w != [0, 0]);
    let mut horiz = horiz;
    horiz.sort_by_key(|e| (e.a.y, e.a.x, e.b.x));
    vert.sort_by_key(|e| (e.a.y, e.a.x, e.b.y, e.b.x));

    let mut ys: Vec<Coord> = vert
        .iter()
        .flat_map(|e| [e.a.y, e.b.y])
        .chain(horiz.iter().map(|h| h.a.y))
        .collect();
    ys.sort_unstable();
    ys.dedup();

    let mut out = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let mut next = 0usize;
    let mut hi = 0usize; // horizontals are sorted by (y, x)
    let wind = |active: &[usize], y: Coord, s2: i128| -> [i32; 2] {
        let mut w = [0, 0];
        for &i in active {
            let e = &vert[i];
            if left_of(e, y, s2) {
                w[0] -= e.w[0];
                w[1] -= e.w[1];
            }
        }
        w
    };
    for (k, &y) in ys.iter().enumerate() {
        let h_start = hi;
        while hi < horiz.len() && horiz[hi].a.y == y {
            hi += 1;
        }
        let hs = &horiz[h_start..hi];
        let below: Vec<[i32; 2]> = hs
            .iter()
            .map(|h| wind(&active, y, h.a.x as i128 + h.b.x as i128))
            .collect();
        active.retain(|&i| vert[i].b.y != y);
        let mut new_edges = Vec::new();
        while next < vert.len() && vert[next].a.y == y {
            new_edges.push(next);
            next += 1;
        }
        if !new_edges.is_empty() && k + 1 < ys.len() {
            let y2 = y as i128 + ys[k + 1] as i128;
            for &ni in &new_edges {
                let pos = active
                    .binary_search_by(|&ai| cmp_at(&vert[ai], &vert[ni], y2))
                    .unwrap_or_else(|p| p);
                active.insert(pos, ni);
            }
            let mut w = [0i32, 0];
            for &ai in &active {
                let e = &vert[ai];
                let left = w;
                let right = [w[0] - e.w[0], w[1] - e.w[1]];
                if e.a.y == y {
                    let il = pred(op, left);
                    let ir = pred(op, right);
                    if il != ir {
                        out.push(if il { (e.a, e.b) } else { (e.b, e.a) });
                    }
                }
                w = right;
            }
        }
        for (h, wb) in hs.iter().zip(below) {
            let wa = wind(&active, y, h.a.x as i128 + h.b.x as i128);
            let ib = pred(op, wb);
            let ia = pred(op, wa);
            if ib != ia {
                // Interior above means interior on the left of a rightward edge.
                out.push(if ia { (h.a, h.b) } else { (h.b, h.a) });
            }
        }
    }
    out
}

/// `true` when direction `d1` is a more counter-clockwise turn from `din` than `d2`.
fn more_left(din: Point, d1: Point, d2: Point) -> bool {
    let half = |d: Point| {
        let c = din.cross(d);
        if c > 0 || (c == 0 && din.dot(d) < 0) {
            0
        } else {
            1
        }
    };
    let (h1, h2) = (half(d1), half(d2));
    if h1 != h2 {
        return h1 < h2;
    }
    d2.cross(d1) > 0
}

fn simplify_ring(r: &mut Ring) {
    let mut changed = true;
    while changed && r.len() >= 3 {
        changed = false;
        let n = r.len();
        let mut keep = Vec::with_capacity(n);
        for i in 0..n {
            let p = r[(i + n - 1) % n];
            let c = r[i];
            let q = r[(i + 1) % n];
            if orient(p, c, q) == 0 && (c - p).dot(q - c) >= 0 {
                changed = true;
                continue;
            }
            keep.push(c);
        }
        if changed {
            // Remove one point per pass at a time is unnecessary: collinear runs are
            // independent of each other.
            *r = keep;
        }
    }
    if r.len() < 3 {
        r.clear();
        return;
    }
    let start = r
        .iter()
        .enumerate()
        .min_by_key(|(_, p)| (p.y, p.x))
        .map(|(i, _)| i)
        .unwrap();
    r.rotate_left(start);
}

fn build_rings(mut edges: Vec<(Point, Point)>) -> PolySet {
    edges.sort();
    let mut by_start: BTreeMap<Point, Vec<usize>> = BTreeMap::new();
    for (i, (a, _)) in edges.iter().enumerate() {
        by_start.entry(*a).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut rings: Vec<Ring> = Vec::new();
    for s in 0..edges.len() {
        if used[s] {
            continue;
        }
        used[s] = true;
        let start = edges[s].0;
        let mut ring = vec![start];
        let mut cur = s;
        let mut ok = false;
        for _ in 0..edges.len() + 1 {
            let (a, b) = edges[cur];
            if b == start {
                ok = true;
                break;
            }
            ring.push(b);
            let din = b - a;
            let mut best: Option<usize> = None;
            if let Some(cands) = by_start.get(&b) {
                for &c in cands {
                    if used[c] {
                        continue;
                    }
                    let d = edges[c].1 - b;
                    best = match best {
                        None => Some(c),
                        Some(bi) if more_left(din, d, edges[bi].1 - b) => Some(c),
                        keep => keep,
                    };
                }
            }
            match best {
                Some(c) => {
                    used[c] = true;
                    cur = c;
                }
                None => break,
            }
        }
        if ok {
            simplify_ring(&mut ring);
            if ring.len() >= 3 && area2(&ring) != 0 {
                rings.push(ring);
            }
        }
    }
    assemble(rings)
}

fn assemble(rings: Vec<Ring>) -> PolySet {
    let (mut outers, holes): (Vec<Ring>, Vec<Ring>) = rings.into_iter().partition(|r| area2(r) > 0);
    outers.sort_by(|a, b| area2(a).cmp(&area2(b)).then_with(|| a.cmp(b)));
    let boxes: Vec<Rect> = outers.iter().map(|r| ring_bbox(r)).collect();
    let mut polys: Vec<Polygon> = outers
        .iter()
        .map(|o| Polygon {
            outer: o.clone(),
            holes: Vec::new(),
        })
        .collect();
    for h in holes {
        let hb = ring_bbox(&h);
        let mut placed = false;
        'outer: for (oi, o) in outers.iter().enumerate() {
            let ob = boxes[oi];
            if hb.min.x < ob.min.x
                || hb.min.y < ob.min.y
                || hb.max.x > ob.max.x
                || hb.max.y > ob.max.y
            {
                continue;
            }
            // Probe edge midpoints (doubled grid) until one is strictly in or out.
            let n = h.len();
            for i in 0..n {
                let a = h[i];
                let b = h[(i + 1) % n];
                let o2: Vec<Point> = o.iter().map(|p| Point::new(p.x * 2, p.y * 2)).collect();
                match ring_contains(&o2, a + b) {
                    Some(true) => {
                        polys[oi].holes.push(h.clone());
                        placed = true;
                        break 'outer;
                    }
                    Some(false) => continue 'outer,
                    None => {}
                }
            }
            // Every probe on the boundary: the hole coincides with this outer's boundary.
            polys[oi].holes.push(h.clone());
            placed = true;
            break;
        }
        let _ = placed;
    }
    for p in &mut polys {
        p.holes.sort();
    }
    polys.sort();
    PolySet { polys }
}

// ---------------------------------------------------------------- shapes and offsets

/// A regular polygon circumscribing the circle (every edge is at distance ≥ `r` from the
/// centre), counter-clockwise.
pub fn circle_ring(c: Point, r: Coord, tol: Coord) -> Ring {
    let n = circle_segments(r, tol);
    let rr = r as f64 / (std::f64::consts::PI / n as f64).cos();
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            Point::from_f(c.x as f64 + rr * t.cos(), c.y as f64 + rr * t.sin())
        })
        .collect()
}

/// The rectangle covering segment `a–b` widened by `d` on both sides (no end caps), CCW.
pub fn edge_band(a: Point, b: Point, d: Coord) -> Ring {
    let (dx, dy) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
    let l = dx.hypot(dy);
    if l == 0.0 {
        return Vec::new();
    }
    let (nx, ny) = (-dy / l * d as f64, dx / l * d as f64);
    let off = |p: Point, s: f64| Point::from_f(p.x as f64 + s * nx, p.y as f64 + s * ny);
    let r = vec![off(a, -1.0), off(b, -1.0), off(b, 1.0), off(a, 1.0)];
    if area2(&r) < 0 {
        r.into_iter().rev().collect()
    } else {
        r
    }
}

/// A capsule (stadium): the set within `r` of segment `a–b`, as a CCW ring whose edges lie
/// outside the true capsule.
pub fn capsule_ring(a: Point, b: Point, r: Coord, tol: Coord) -> Ring {
    if a == b {
        return circle_ring(a, r, tol);
    }
    let n = circle_segments(r, tol) / 2;
    let rr = r as f64 / (std::f64::consts::PI / (2 * n) as f64).cos();
    let ang = ((b.y - a.y) as f64).atan2((b.x - a.x) as f64);
    let mut out = Vec::with_capacity(2 * n + 2);
    let half = std::f64::consts::FRAC_PI_2;
    for (c, base) in [(b, ang - half), (a, ang + half)] {
        // Endpoints of each semicircle sit on the true radius so the straight sides are tangent.
        out.push(Point::from_f(
            c.x as f64 + r as f64 * base.cos(),
            c.y as f64 + r as f64 * base.sin(),
        ));
        for i in 0..n {
            let t = base + std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
            out.push(Point::from_f(
                c.x as f64 + rr * t.cos(),
                c.y as f64 + rr * t.sin(),
            ));
        }
        out.push(Point::from_f(
            c.x as f64 + r as f64 * (base + std::f64::consts::PI).cos(),
            c.y as f64 + r as f64 * (base + std::f64::consts::PI).sin(),
        ));
    }
    out.dedup();
    if out.first() == out.last() {
        out.pop();
    }
    out
}

/// Minkowski sum of the filled rings with a disk of radius `d` (an outward offset with round
/// joins), as a union of the rings, one band per edge and one disk per vertex.
pub fn inflate(rings: &[Ring], d: Coord, tol: Coord) -> PolySet {
    if d <= 0 {
        return boolean(rings, &[], BoolOp::Union);
    }
    let mut parts: Vec<Ring> = rings.to_vec();
    for r in rings {
        let n = r.len();
        for i in 0..n {
            let a = r[i];
            let b = r[(i + 1) % n];
            let band = edge_band(a, b, d);
            if !band.is_empty() {
                parts.push(band);
            }
            parts.push(circle_ring(a, d, tol));
        }
    }
    boolean(&parts, &[], BoolOp::Union)
}

/// Minkowski erosion by a disk of radius `d`: the points at least `d` inside the set.
pub fn deflate(set: &PolySet, d: Coord, tol: Coord) -> PolySet {
    if d <= 0 {
        return set.clone();
    }
    let mut cutters: Vec<Ring> = Vec::new();
    for r in set.rings() {
        let n = r.len();
        for i in 0..n {
            let a = r[i];
            let b = r[(i + 1) % n];
            let band = edge_band(a, b, d);
            if !band.is_empty() {
                cutters.push(band);
            }
            cutters.push(circle_ring(a, d, tol));
        }
    }
    boolean(&set.rings(), &cutters, BoolOp::Difference)
}

/// Morphological opening: removes every part narrower than `2·d` (erode then dilate).
pub fn open(set: &PolySet, d: Coord, tol: Coord) -> PolySet {
    let e = deflate(set, d, tol);
    if e.is_empty() {
        return e;
    }
    let grown = inflate(&e.rings(), d, tol);
    // Dilation of the erosion stays inside the original up to the polygon tolerance.
    grown.intersection(set)
}

pub fn rect_ring(r: Rect) -> Ring {
    vec![
        r.min,
        Point::new(r.max.x, r.min.y),
        r.max,
        Point::new(r.min.x, r.max.y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    fn sq(x: f64, y: f64, s: f64) -> Ring {
        rect_ring(Rect::new(Point::mm(x, y), Point::mm(x + s, y + s)))
    }

    #[test]
    fn basic_ops() {
        let a = vec![sq(0.0, 0.0, 2.0)];
        let b = vec![sq(1.0, 1.0, 2.0)];
        let u = boolean(&a, &b, BoolOp::Union);
        assert_eq!(u.polys.len(), 1);
        assert!((u.area() - 7.0e12).abs() < 1.0);
        assert_eq!(u.polys[0].outer.len(), 8);
        let i = boolean(&a, &b, BoolOp::Intersection);
        assert!((i.area() - 1.0e12).abs() < 1.0);
        let d = boolean(&a, &b, BoolOp::Difference);
        assert!((d.area() - 3.0e12).abs() < 1.0);
        let x = boolean(&a, &b, BoolOp::Xor);
        assert!((x.area() - 6.0e12).abs() < 1.0);
        assert_eq!(x.polys.len(), 2);
    }

    #[test]
    fn holes_and_shared_edges() {
        let a = vec![sq(0.0, 0.0, 10.0)];
        let b = vec![sq(2.0, 2.0, 2.0), sq(6.0, 6.0, 2.0)];
        let d = boolean(&a, &b, BoolOp::Difference);
        assert_eq!(d.polys.len(), 1);
        assert_eq!(d.polys[0].holes.len(), 2);
        assert!((d.area() - 92.0e12).abs() < 1.0);
        // Adjacent squares sharing an edge union into one rectangle with 4 vertices.
        let u = boolean(&[sq(0.0, 0.0, 1.0), sq(1.0, 0.0, 1.0)], &[], BoolOp::Union);
        assert_eq!(u.polys.len(), 1);
        assert_eq!(u.polys[0].outer.len(), 4);
        // Touching at a corner: two separate polygons.
        let u = boolean(&[sq(0.0, 0.0, 1.0), sq(1.0, 1.0, 1.0)], &[], BoolOp::Union);
        assert_eq!(u.polys.len(), 2);
        // Identical operands: difference is empty, xor is empty.
        assert!(boolean(&a, &a, BoolOp::Difference).is_empty());
        assert!(boolean(&a, &a, BoolOp::Xor).is_empty());
        // A hole whose island sits inside it.
        let ring = boolean(&a, &[sq(1.0, 1.0, 8.0)], BoolOp::Difference);
        let island = boolean(&ring.rings(), &[sq(3.0, 3.0, 4.0)], BoolOp::Union);
        assert_eq!(island.polys.len(), 2);
        assert!((island.area() - (36.0e12 + 16.0e12)).abs() < 1.0);
    }

    #[test]
    fn crossing_and_overlap_union() {
        // A plus sign from two crossing bars, plus a self-overlapping duplicate.
        let h = rect_ring(Rect::new(Point::mm(-3.0, -0.5), Point::mm(3.0, 0.5)));
        let v = rect_ring(Rect::new(Point::mm(-0.5, -3.0), Point::mm(0.5, 3.0)));
        let u = boolean(&[h.clone(), v.clone(), h.clone()], &[], BoolOp::Union);
        assert_eq!(u.polys.len(), 1);
        assert_eq!(u.polys[0].outer.len(), 12);
        assert!((u.area() - 11.0e12).abs() < 1.0);
        // Rotated square (non-axis edges) intersecting a square.
        let dia = vec![
            Point::mm(0.0, -2.0),
            Point::mm(2.0, 0.0),
            Point::mm(0.0, 2.0),
            Point::mm(-2.0, 0.0),
        ];
        let i = boolean(&[dia], &[sq(0.0, 0.0, 3.0)], BoolOp::Intersection);
        assert!((i.area() - 2.0e12).abs() < 10.0, "{}", i.area());
    }

    #[test]
    fn circles_and_offsets() {
        let c = circle_ring(Point::ORIGIN, mm(1.0), 1000);
        let a = ring_area(&c);
        let exact = std::f64::consts::PI * 1e12;
        assert!(a >= exact && a < exact * 1.01, "{a}");
        // Inflating a 2 mm square by 0.5 mm: area = 4 + 4·2·0.5 + π·0.25.
        let s = vec![sq(0.0, 0.0, 2.0)];
        let inf = inflate(&s, mm(0.5), 500);
        let want = (4.0 + 4.0 + std::f64::consts::PI * 0.25) * 1e12;
        assert_eq!(inf.polys.len(), 1);
        assert!((inf.area() - want).abs() / want < 0.005, "{}", inf.area());
        // Deflating it back gives (approximately) the square.
        let back = deflate(&inf, mm(0.5), 500);
        assert!(
            (back.area() - 4.0e12).abs() / 4.0e12 < 0.01,
            "{}",
            back.area()
        );
        // Opening removes a thin neck: two squares joined by a 0.1 mm bridge.
        let shape = boolean(
            &[
                sq(0.0, 0.0, 2.0),
                sq(3.0, 0.0, 2.0),
                rect_ring(Rect::new(Point::mm(2.0, 0.95), Point::mm(3.0, 1.05))),
            ],
            &[],
            BoolOp::Union,
        );
        assert_eq!(shape.polys.len(), 1);
        let opened = open(&shape, mm(0.1), 500);
        assert_eq!(opened.polys.len(), 2);
    }

    #[test]
    fn many_overlapping_disks_union() {
        let mut rings = Vec::new();
        for i in 0..200 {
            let x = (i % 20) as f64 * 0.75;
            let y = (i / 20) as f64 * 0.75;
            rings.push(circle_ring(Point::mm(x, y), mm(0.5), 1000));
        }
        let u = boolean(&rings, &[], BoolOp::Union);
        assert_eq!(u.polys.len(), 1);
        // Holes appear between four disks whose gaps do not close.
        assert!(!u.polys[0].holes.is_empty());
        let again = boolean(&u.rings(), &[], BoolOp::Union);
        assert_eq!(again, u, "union of a result is a fixed point");
    }
}
