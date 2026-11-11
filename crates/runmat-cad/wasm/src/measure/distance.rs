//! The minimum distance between two picked items, each a set of primitives in world
//! coordinates: points (a vertex), segments (an edge's polyline) and triangles (a face's or a
//! body's tessellation). The answer carries the two nearest points, which the pinned dimension
//! joins.
//!
//! Primitives are grouped into clusters with boxes; cluster pairs are visited nearest box first
//! and a pair is skipped once its box gap exceeds the best distance found, so two components
//! with tens of thousands of triangles each measure in a few thousand primitive pairs.

use crate::edit::assembly::frames::{add, cross, dot, norm, scale, sub, V3};

/// What an item is made of, in world coordinates (metres).
#[derive(Clone, Debug, Default)]
pub struct Primitives {
    pub points: Vec<V3>,
    pub segments: Vec<[V3; 2]>,
    pub triangles: Vec<[V3; 3]>,
}

/// The nearest points of two items and their distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    pub distance: f64,
    pub a: V3,
    pub b: V3,
}

#[derive(Clone, Copy, Debug)]
enum Prim {
    Point(V3),
    Segment([V3; 2]),
    Triangle([V3; 3]),
}

impl Prim {
    fn corners(&self) -> &[V3] {
        match self {
            Prim::Point(p) => std::slice::from_ref(p),
            Prim::Segment(s) => s,
            Prim::Triangle(t) => t,
        }
    }
}

struct Cluster {
    lo: V3,
    hi: V3,
    prims: Vec<Prim>,
}

const CLUSTER: usize = 48;

fn clusters(p: &Primitives) -> Vec<Cluster> {
    let mut all: Vec<Prim> = p.points.iter().map(|q| Prim::Point(*q)).collect();
    all.extend(p.segments.iter().map(|s| Prim::Segment(*s)));
    all.extend(p.triangles.iter().map(|t| Prim::Triangle(*t)));
    let mut out = Vec::new();
    split(all, &mut out);
    out
}

fn bounds(prims: &[Prim]) -> (V3, V3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for c in prims.iter().flat_map(|p| p.corners()) {
        for i in 0..3 {
            lo[i] = lo[i].min(c[i]);
            hi[i] = hi[i].max(c[i]);
        }
    }
    (lo, hi)
}

/// Splits along the box's longest side at the median centroid until each cluster is small.
fn split(mut prims: Vec<Prim>, out: &mut Vec<Cluster>) {
    let (lo, hi) = bounds(&prims);
    if prims.len() <= CLUSTER {
        if !prims.is_empty() {
            out.push(Cluster { lo, hi, prims });
        }
        return;
    }
    let axis = (0..3).max_by(|&i, &j| (hi[i] - lo[i]).total_cmp(&(hi[j] - lo[j]))).unwrap_or(0);
    let key = |p: &Prim| p.corners().iter().map(|c| c[axis]).sum::<f64>() / p.corners().len() as f64;
    let mid = prims.len() / 2;
    prims.select_nth_unstable_by(mid, |a, b| key(a).total_cmp(&key(b)));
    let rest = prims.split_off(mid);
    split(prims, out);
    split(rest, out);
}

fn box_gap(a: &Cluster, b: &Cluster) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        let d = (a.lo[i] - b.hi[i]).max(b.lo[i] - a.hi[i]).max(0.0);
        s += d * d;
    }
    s.sqrt()
}

/// The minimum distance between `a` and `b`, or None when either is empty.
pub fn nearest(a: &Primitives, b: &Primitives) -> Option<Nearest> {
    let (ca, cb) = (clusters(a), clusters(b));
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(ca.len() * cb.len());
    for (i, x) in ca.iter().enumerate() {
        for (j, y) in cb.iter().enumerate() {
            pairs.push((box_gap(x, y), i, j));
        }
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut best: Option<Nearest> = None;
    for (gap, i, j) in pairs {
        if best.is_some_and(|b| gap > b.distance) {
            break;
        }
        for p in &ca[i].prims {
            for q in &cb[j].prims {
                let n = prim_pair(p, q);
                if best.is_none_or(|b| n.distance < b.distance) {
                    best = Some(n);
                }
            }
        }
    }
    best
}

fn flip(n: Nearest) -> Nearest {
    Nearest { distance: n.distance, a: n.b, b: n.a }
}

fn prim_pair(p: &Prim, q: &Prim) -> Nearest {
    match (p, q) {
        (Prim::Point(a), Prim::Point(b)) => Nearest { distance: norm(sub(*a, *b)), a: *a, b: *b },
        (Prim::Point(a), Prim::Segment(s)) => point_segment(*a, s),
        (Prim::Segment(s), Prim::Point(a)) => flip(point_segment(*a, s)),
        (Prim::Point(a), Prim::Triangle(t)) => point_triangle(*a, t),
        (Prim::Triangle(t), Prim::Point(a)) => flip(point_triangle(*a, t)),
        (Prim::Segment(s), Prim::Segment(r)) => segment_segment(s, r),
        (Prim::Segment(s), Prim::Triangle(t)) => segment_triangle(s, t),
        (Prim::Triangle(t), Prim::Segment(s)) => flip(segment_triangle(s, t)),
        (Prim::Triangle(t), Prim::Triangle(u)) => triangle_triangle(t, u),
    }
}

fn closer(a: Nearest, b: Nearest) -> Nearest {
    if b.distance < a.distance {
        b
    } else {
        a
    }
}

pub fn point_segment(p: V3, s: &[V3; 2]) -> Nearest {
    let d = sub(s[1], s[0]);
    let l2 = dot(d, d);
    let t = if l2 > 0.0 { (dot(sub(p, s[0]), d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let q = add(s[0], scale(d, t));
    Nearest { distance: norm(sub(p, q)), a: p, b: q }
}

/// The nearest point of triangle `t` to `p` (the region test of Ericson, Real-Time Collision
/// Detection, 5.1.5).
pub fn point_triangle(p: V3, t: &[V3; 3]) -> Nearest {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let at = |q: V3| Nearest { distance: norm(sub(p, q)), a: p, b: q };
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return at(a);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return at(b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return at(add(a, scale(ab, d1 / (d1 - d3))));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return at(c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return at(add(a, scale(ac, d2 / (d2 - d6))));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return at(add(b, scale(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))));
    }
    let denom = va + vb + vc;
    if denom.abs() < 1e-300 {
        // A degenerate triangle: its longest side stands in.
        return [[a, b], [b, c], [c, a]].iter().map(|s| point_segment(p, s)).fold(at(a), closer);
    }
    let (v, w) = (vb / denom, vc / denom);
    at(add(a, add(scale(ab, v), scale(ac, w))))
}

/// The nearest points of two segments (Ericson 5.1.9).
pub fn segment_segment(s: &[V3; 2], r: &[V3; 2]) -> Nearest {
    let (d1, d2) = (sub(s[1], s[0]), sub(r[1], r[0]));
    let rr = sub(s[0], r[0]);
    let (a, e, f) = (dot(d1, d1), dot(d2, d2), dot(d2, rr));
    let eps = 1e-30;
    let (sc, tc);
    if a <= eps && e <= eps {
        return Nearest { distance: norm(sub(s[0], r[0])), a: s[0], b: r[0] };
    }
    if a <= eps {
        sc = 0.0;
        tc = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, rr);
        if e <= eps {
            tc = 0.0;
            sc = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            let mut s0 = if denom > eps * a * e { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t0 = 0.0;
                s0 = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t0 = 1.0;
                s0 = ((b - c) / a).clamp(0.0, 1.0);
            }
            sc = s0;
            tc = t0;
        }
    }
    let (p, q) = (add(s[0], scale(d1, sc)), add(r[0], scale(d2, tc)));
    Nearest { distance: norm(sub(p, q)), a: p, b: q }
}

/// Where segment `s` crosses triangle `t`, if it does.
fn crossing(s: &[V3; 2], t: &[V3; 3]) -> Option<V3> {
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let (da, db) = (dot(n, sub(s[0], t[0])), dot(n, sub(s[1], t[0])));
    if da * db > 0.0 || (da - db).abs() < 1e-300 {
        return None;
    }
    let p = add(s[0], scale(sub(s[1], s[0]), da / (da - db)));
    let inside = (0..3).all(|i| dot(n, cross(sub(t[(i + 1) % 3], t[i]), sub(p, t[i]))) >= 0.0);
    inside.then_some(p)
}

pub fn segment_triangle(s: &[V3; 2], t: &[V3; 3]) -> Nearest {
    if let Some(p) = crossing(s, t) {
        return Nearest { distance: 0.0, a: p, b: p };
    }
    let mut best = point_triangle(s[0], t);
    best = closer(best, point_triangle(s[1], t));
    for i in 0..3 {
        best = closer(best, segment_segment(s, &[t[i], t[(i + 1) % 3]]));
    }
    best
}

pub fn triangle_triangle(t: &[V3; 3], u: &[V3; 3]) -> Nearest {
    let mut best = segment_triangle(&[t[0], t[1]], u);
    for i in 1..3 {
        best = closer(best, segment_triangle(&[t[i], t[(i + 1) % 3]], u));
    }
    for i in 0..3 {
        best = closer(best, flip(segment_triangle(&[u[i], u[(i + 1) % 3]], t)));
    }
    best
}
