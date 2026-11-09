//! Exact polygon booleans on integer coordinates (offsets are in [`super::offset`]).
//!
//! # Algorithm
//!
//! 1. **Edges.** Every ring edge of both operands becomes an edge stored bottom-to-top in
//!    `(y, x)` order with a winding contribution per operand (+1 when the ring runs upward,
//!    −1 when downward). Rings may have any orientation and may overlap; each operand is
//!    filled by the non-zero rule.
//! 2. **Arrangement.** Edges are bucketed in a uniform grid sized to the edge count and split
//!    at every crossing, T-junction and collinear overlap among edges sharing a cell,
//!    found with exact `i128` orientation tests. Crossing points are rounded to the integer
//!    grid; splitting repeats until no new crossings appear, which handles the rare
//!    rounding-induced crossings. Identical edges are merged by summing contributions, so
//!    shared boundaries cancel exactly.
//! 3. **Classification.** A sweep over the distinct `y` values keeps the active edges
//!    ordered by `x` inside each slab (exact rational comparison). Since no edge crosses or
//!    starts inside another, the winding just right of an edge is constant over its life: it
//!    is computed once, from its left neighbour, when the edge enters the sweep. An edge is a
//!    result boundary when the predicate (under the operand's [`FillRule`]) differs on its
//!    two sides. Horizontal edges are classified by the windings just below and just above,
//!    found by binary search.
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

/// How an operand's windings fill: non-zero (any orientation) or positive (only regions wound
/// counter-clockwise, which polygon offsetting needs to drop its inverted loops).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FillRule {
    #[default]
    NonZero,
    Positive,
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

/// Finds every pair of edges whose boxes meet, by bucketing edges into a uniform grid sized
/// to the edge count (a pair is tested once, in the first cell both boxes share).
fn candidate_pairs(edges: &[E], mut f: impl FnMut(usize, usize)) {
    let n = edges.len();
    if n < 2 {
        return;
    }
    let bb = |e: &E| Rect::new(e.a, e.b);
    let all = edges.iter().fold(Rect::EMPTY, |r, e| r.union(bb(e)));
    let (w, h) = (
        (all.width() as f64).max(1.0),
        (all.height() as f64).max(1.0),
    );
    // About two edges per cell; cells no smaller than the median edge.
    let mut lens: Vec<Coord> = edges
        .iter()
        .map(|e| (e.b.x - e.a.x).abs().max(e.b.y - e.a.y))
        .collect();
    let mid = lens.len() / 2;
    let median = *lens.select_nth_unstable(mid).1 as f64;
    let cell = ((w * h / n as f64 * 2.0).sqrt()).max(median).max(1.0);
    let gx = ((w / cell).ceil() as usize).clamp(1, 4096);
    let gy = ((h / cell).ceil() as usize).clamp(1, 4096);
    let cx = |x: Coord| (((x - all.min.x) as f64 / w * gx as f64) as usize).min(gx - 1);
    let cy = |y: Coord| (((y - all.min.y) as f64 / h * gy as f64) as usize).min(gy - 1);
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); gx * gy];
    let mut ranges: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(n);
    for (i, e) in edges.iter().enumerate() {
        let r = bb(e);
        let (x0, y0, x1, y1) = (cx(r.min.x), cy(r.min.y), cx(r.max.x), cy(r.max.y));
        for y in y0..=y1 {
            for x in x0..=x1 {
                cells[y * gx + x].push(i as u32);
            }
        }
        ranges.push((x0, y0, x1, y1));
    }
    for (c, list) in cells.iter().enumerate() {
        let (x, y) = (c % gx, c / gx);
        for (k, &i) in list.iter().enumerate() {
            let (i, ri) = (i as usize, ranges[i as usize]);
            let bi = bb(&edges[i]);
            for &j in &list[k + 1..] {
                let (j, rj) = (j as usize, ranges[j as usize]);
                // Reference cell: the first cell both ranges share.
                if x != ri.0.max(rj.0) || y != ri.1.max(rj.1) {
                    continue;
                }
                if bi.intersects(&bb(&edges[j])) {
                    f(i, j);
                }
            }
        }
    }
}

fn split_pass(edges: &[E]) -> Option<Vec<E>> {
    let n = edges.len();
    let mut splits: Vec<Vec<Point>> = vec![Vec::new(); n];
    let mut any = false;
    candidate_pairs(edges, |i, j| {
        let (mut si, mut sj) = (Vec::new(), Vec::new());
        intersect(&edges[i], &edges[j], &mut si, &mut sj);
        if !si.is_empty() || !sj.is_empty() {
            any = true;
            splits[i].extend(si);
            splits[j].extend(sj);
        }
    });
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

fn pred(op: BoolOp, fill: FillRule, w: [i32; 2]) -> bool {
    let (a, b) = match fill {
        FillRule::NonZero => (w[0] != 0, w[1] != 0),
        FillRule::Positive => (w[0] > 0, w[1] > 0),
    };
    match op {
        BoolOp::Union => a || b,
        BoolOp::Intersection => a && b,
        BoolOp::Difference => a && !b,
        BoolOp::Xor => a != b,
    }
}

/// Boolean of two ring sets, each filled by the non-zero rule.
pub fn boolean(a: &[Ring], b: &[Ring], op: BoolOp) -> PolySet {
    boolean_with(a, b, op, FillRule::NonZero)
}

/// Boolean of two ring sets filled by `fill`.
pub fn boolean_with(a: &[Ring], b: &[Ring], op: BoolOp, fill: FillRule) -> PolySet {
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
    let boundary = classify(edges, op, fill);
    build_rings(boundary)
}

fn classify(mut edges: Vec<E>, op: BoolOp, fill: FillRule) -> Vec<(Point, Point)> {
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

    // Sweep. `active` holds the edges spanning the current slab ordered by x; `right[i]` is
    // the winding just right of edge `i`, constant over the edge's life because no edge
    // crosses or starts in the interior of another.
    let mut out = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let mut right: Vec<[i32; 2]> = vec![[0, 0]; vert.len()];
    let mut ends: Vec<usize> = (0..vert.len()).collect();
    ends.sort_by_key(|&i| (vert[i].b.y, i));
    let mut next_end = 0usize;
    let mut next = 0usize;
    let mut hi = 0usize; // horizontals are sorted by (y, x)
                         // Winding at doubled abscissa `s2` on the line `y` among `active` (edges spanning `y`).
    let wind = |active: &[usize], right: &[[i32; 2]], y: Coord, s2: i128| -> [i32; 2] {
        let k = active.partition_point(|&i| left_of(&vert[i], y, s2));
        if k == 0 {
            [0, 0]
        } else {
            right[active[k - 1]]
        }
    };
    for (k, &y) in ys.iter().enumerate() {
        let h_start = hi;
        while hi < horiz.len() && horiz[hi].a.y == y {
            hi += 1;
        }
        let hs = &horiz[h_start..hi];
        let below: Vec<[i32; 2]> = hs
            .iter()
            .map(|h| wind(&active, &right, y, h.a.x as i128 + h.b.x as i128))
            .collect();
        // Remove edges ending here: each is found by its order in the slab below.
        if k > 0 {
            let y2 = ys[k - 1] as i128 + y as i128;
            while next_end < ends.len() && vert[ends[next_end]].b.y == y {
                let e = ends[next_end];
                next_end += 1;
                if let Ok(pos) = active.binary_search_by(|&ai| cmp_at(&vert[ai], &vert[e], y2)) {
                    active.remove(pos);
                }
            }
        }
        let first_new = next;
        while next < vert.len() && vert[next].a.y == y {
            next += 1;
        }
        if first_new < next && k + 1 < ys.len() {
            let y2 = y as i128 + ys[k + 1] as i128;
            for ni in first_new..next {
                let pos = active
                    .binary_search_by(|&ai| cmp_at(&vert[ai], &vert[ni], y2))
                    .unwrap_or_else(|p| p);
                active.insert(pos, ni);
            }
            // Windings left to right, after every new edge is in place (a new edge's left
            // neighbour may be another new edge).
            let mut pos: Vec<usize> = (first_new..next)
                .map(|ni| {
                    active
                        .binary_search_by(|&ai| cmp_at(&vert[ai], &vert[ni], y2))
                        .expect("inserted edge")
                })
                .collect();
            pos.sort_unstable();
            for p in pos {
                let ni = active[p];
                let left = if p == 0 { [0, 0] } else { right[active[p - 1]] };
                let e = &vert[ni];
                let r = [left[0] - e.w[0], left[1] - e.w[1]];
                right[ni] = r;
                let il = pred(op, fill, left);
                let ir = pred(op, fill, r);
                if il != ir {
                    out.push(if il { (e.a, e.b) } else { (e.b, e.a) });
                }
            }
        }
        for (h, wb) in hs.iter().zip(below) {
            let wa = wind(&active, &right, y, h.a.x as i128 + h.b.x as i128);
            let ib = pred(op, fill, wb);
            let ia = pred(op, fill, wa);
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

pub(crate) fn build_rings(mut edges: Vec<(Point, Point)>) -> PolySet {
    edges.sort();
    // Edges sorted by start: the edges leaving a point are one run.
    let starting = |edges: &[(Point, Point)], p: Point| {
        let lo = edges.partition_point(|e| e.0 < p);
        let hi = lo + edges[lo..].partition_point(|e| e.0 == p);
        lo..hi
    };
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
            {
                for c in starting(&edges, b) {
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
    // Large outers answer containment from an edge index over their doubled ring, built once.
    let mut doubled: Vec<Option<(Ring, Option<crate::spatial::EdgeIndex>)>> = vec![None; 0];
    doubled.resize_with(outers.len(), || None);
    for h in holes {
        let hb = ring_bbox(&h);
        'outer: for (oi, o) in outers.iter().enumerate() {
            let ob = boxes[oi];
            if hb.min.x < ob.min.x
                || hb.min.y < ob.min.y
                || hb.max.x > ob.max.x
                || hb.max.y > ob.max.y
            {
                continue;
            }
            let (o2, index) = doubled[oi].get_or_insert_with(|| {
                let o2: Ring = o.iter().map(|p| Point::new(p.x * 2, p.y * 2)).collect();
                let index = (o2.len() > 256).then(|| {
                    let b = ring_bbox(&o2);
                    let cell = ((b.width().max(b.height()) as f64) / (o2.len() as f64).sqrt())
                        .max(1.0) as Coord;
                    crate::spatial::EdgeIndex::new([&o2], cell)
                });
                (o2, index)
            });
            // Probe edge midpoints (doubled grid) until one is strictly in or out.
            let n = h.len();
            for i in 0..n {
                let m = h[i] + h[(i + 1) % n];
                let inside = match index {
                    Some(ix) => ix.contains(m),
                    None => ring_contains(o2, m),
                };
                match inside {
                    Some(true) => {
                        polys[oi].holes.push(h.clone());
                        break 'outer;
                    }
                    Some(false) => continue 'outer,
                    None => {}
                }
            }
            // Every probe on the boundary: the hole coincides with this outer's boundary.
            polys[oi].holes.push(h.clone());
            break;
        }
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

/// Outward offset by `d` with round joins: the points within `d` of the filled rings (any
/// orientation, non-zero fill). Arc chords lie outside the true offset. See
/// [`super::offset`] for other joins.
pub fn inflate(rings: &[Ring], d: Coord, tol: Coord) -> PolySet {
    super::offset::offset(rings, d, super::offset::Join::Round, tol)
}

/// Erosion by `d` with round joins: the points at least `d` inside the set.
pub fn deflate(set: &PolySet, d: Coord, tol: Coord) -> PolySet {
    super::offset::offset_set(set, -d, super::offset::Join::Round, tol)
}

/// Morphological opening: removes every part narrower than `2·d` (erode then dilate).
pub fn open(set: &PolySet, d: Coord, tol: Coord) -> PolySet {
    let e = deflate(set, d, tol);
    if e.is_empty() {
        return e;
    }
    let grown = super::offset::offset_set(&e, d, super::offset::Join::Round, tol);
    // Dilation of the erosion stays inside the original up to the polygon tolerance.
    grown.intersection(set)
}

/// A polygon with holes as one contour with *cut-ins*: each hole is joined by a vertical
/// segment from its lowest vertex down to the nearest boundary below (the outer ring or
/// another hole), traversed in both directions. This is the form Gerber regions use for
/// holes (no clear polarity needed); IPC-2581 and ODB++ carry holes natively.
///
/// The cut-in point on a slanted edge is rounded to the nanometre grid.
pub fn cut_in_contour(p: &Polygon) -> Ring {
    /// Per (ring, edge): (distance² along the edge, cut-in point, hole ring).
    type Bridges = BTreeMap<(usize, usize), Vec<(i128, Point, usize)>>;
    if p.holes.is_empty() {
        return p.outer.clone();
    }
    let rings: Vec<&Ring> = std::iter::once(&p.outer).chain(p.holes.iter()).collect();
    let mut grid: crate::spatial::Grid<u32> = crate::spatial::Grid::new(super::mm(1.0));
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (r, ring) in rings.iter().enumerate() {
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            grid.insert(edges.len() as u32, Rect::new(a, b));
            edges.push((r, i));
        }
    }
    let bottom = p.bbox().min.y - 1;
    // Bridges per (ring, edge): (distance along the edge, cut-in point, hole ring).
    let mut bridges: Bridges = BTreeMap::new();
    let mut start: Vec<usize> = vec![0; rings.len()];
    for h in 1..rings.len() {
        let ring = rings[h];
        let vi = (0..ring.len())
            .min_by_key(|&i| (ring[i].y, ring[i].x))
            .unwrap();
        start[h] = vi;
        let v = ring[vi];
        let mut best: Option<(i128, i128, usize, Point)> = None; // (y·den numerator order)
        grid.visit(
            &Rect::new(Point::new(v.x, bottom), Point::new(v.x, v.y)),
            |k, _| {
                let (r, i) = edges[k as usize];
                if r == h {
                    return;
                }
                let er = rings[r];
                let (a, b) = (er[i], er[(i + 1) % er.len()]);
                let (lo, hi) = (a.x.min(b.x), a.x.max(b.x));
                if v.x < lo || v.x > hi {
                    return;
                }
                // Height of the edge at x = v.x (top endpoint for a vertical edge).
                let hit = if a.x == b.x {
                    Point::new(v.x, a.y.max(b.y))
                } else {
                    let y = a.y as i128
                        + round_div(
                            (v.x - a.x) as i128 * (b.y - a.y) as i128,
                            (b.x - a.x) as i128,
                        );
                    Point::new(v.x, y as Coord)
                };
                if hit.y >= v.y {
                    return;
                }
                let key = (hit.y as i128, -(k as i128));
                if best.is_none_or(|bb| (key.0, key.1) > (bb.0, bb.1)) {
                    best = Some((key.0, key.1, k as usize, hit));
                }
            },
        );
        let Some((_, _, k, hit)) = best else {
            continue;
        };
        let (r, i) = edges[k];
        let a = rings[r][i];
        let t = (hit - a).dot(hit - a);
        bridges.entry((r, i)).or_default().push((t, hit, h));
    }
    for v in bridges.values_mut() {
        v.sort();
    }
    fn emit(r: usize, rings: &[&Ring], start: &[usize], bridges: &Bridges, out: &mut Ring) {
        let ring = rings[r];
        let n = ring.len();
        for k in 0..n {
            let i = (start[r] + k) % n;
            out.push(ring[i]);
            if let Some(bs) = bridges.get(&(r, i)) {
                for &(_, hit, h) in bs {
                    out.push(hit);
                    emit(h, rings, start, bridges, out);
                    out.push(rings[h][start[h]]);
                    out.push(hit);
                }
            }
        }
    }
    let mut out = Vec::new();
    emit(0, &rings, &start, &bridges, &mut out);
    out.dedup();
    out
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
    fn cut_in_contours_keep_holes() {
        let a = vec![sq(0.0, 0.0, 10.0)];
        let holes = vec![
            sq(2.0, 2.0, 2.0),
            sq(2.5, 6.0, 1.0),
            circle_ring(Point::mm(7.0, 7.0), mm(1.0), 1000),
            // A slanted hole whose ray lands on a slanted edge of the hole below.
            vec![
                Point::mm(6.2, 2.0),
                Point::mm(8.0, 3.3),
                Point::mm(6.9, 4.4),
            ],
        ];
        let set = boolean(&a, &holes, BoolOp::Difference);
        assert_eq!(set.polys.len(), 1);
        let c = cut_in_contour(&set.polys[0]);
        // The contour fills exactly the polygon (cut-ins cancel).
        let again = boolean(std::slice::from_ref(&c), &[], BoolOp::Union);
        assert_eq!(again.polys.len(), 1);
        assert_eq!(again.polys[0].holes.len(), 4);
        assert!(
            (again.area() - set.area()).abs() < 1e6,
            "{} {}",
            again.area(),
            set.area()
        );
        assert!((ring_area(&c) - set.area()).abs() < 1e6);
        // Cut-ins are vertical.
        let n = c.len();
        let mut verticals = 0;
        for i in 0..n {
            let (p, q) = (c[i], c[(i + 1) % n]);
            for j in 0..n {
                if (c[j], c[(j + 1) % n]) == (q, p) && p.x == q.x {
                    verticals += 1;
                }
            }
        }
        assert!(verticals >= 8, "{verticals}");
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
