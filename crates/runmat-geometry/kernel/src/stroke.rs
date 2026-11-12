//! The outline of polylines drawn with a round pen: the closed loops of lines and arcs that bound
//! every point within `radius` of a stroke. Sketch text uses it to turn single-line glyphs into
//! profiles that extrude and cut.
//!
//! # Method
//!
//! The region is the union of a disk at every stroke vertex and a rectangle along every stroke
//! segment (as wide as the pen). Its boundary is made of pieces of those shapes' boundaries:
//!
//! 1. Each disk contributes its circle and each rectangle its two long sides, oriented with the
//!    shape inside on the left (circles counter-clockwise).
//! 2. Every boundary curve is split where it meets another (line–line, line–circle and
//!    circle–circle intersections, tangent points included).
//! 3. A piece stays when its midpoint lies strictly inside no shape. Two pieces that coincide
//!    running the same way count once; running opposite ways, the region lies on both sides and
//!    both go.
//! 4. The kept pieces link end to end into loops, taking the leftmost turn where several leave
//!    one point, so loops that touch at a point stay separate. Consecutive pieces of one line
//!    or one circle merge.
//!
//! Outer loops run counter-clockwise and holes clockwise; every loop is exact lines and arcs
//! (or one whole circle, for a dot).

use crate::ids::SketchEntityId;
use crate::op::{ProfileSegment, SegmentKind};
use std::f64::consts::TAU;

type P = [f64; 2];

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: P, b: P) -> P {
    [a[0] + b[0], a[1] + b[1]]
}
fn mul(a: P, s: f64) -> P {
    [a[0] * s, a[1] * s]
}
fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: P, b: P) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn len(a: P) -> f64 {
    a[0].hypot(a[1])
}
fn dist(a: P, b: P) -> f64 {
    len(sub(a, b))
}

/// A boundary curve of one shape: a directed line from `a` to `b`, or a counter-clockwise
/// circle.
#[derive(Clone, Copy, Debug)]
enum Curve {
    Line { a: P, b: P },
    Circle { c: P },
}

/// A piece of a boundary curve: a line from `a` to `b`, an arc counter-clockwise about `c` from
/// `a` to `b`, or a whole circle about `c`.
#[derive(Clone, Copy, Debug)]
enum Piece {
    Line { a: P, b: P },
    Arc { c: P, a: P, b: P },
    Circle { c: P },
}

impl Piece {
    fn start(&self, r: f64) -> P {
        match *self {
            Piece::Line { a, .. } | Piece::Arc { a, .. } => a,
            Piece::Circle { c } => [c[0] + r, c[1]],
        }
    }
    fn end(&self, r: f64) -> P {
        match *self {
            Piece::Line { b, .. } | Piece::Arc { b, .. } => b,
            Piece::Circle { c } => [c[0] + r, c[1]],
        }
    }
    fn mid(&self, r: f64) -> P {
        match *self {
            Piece::Line { a, b } => mul(add(a, b), 0.5),
            Piece::Arc { c, a, b } => {
                let t0 = (a[1] - c[1]).atan2(a[0] - c[0]);
                let t1 = (b[1] - c[1]).atan2(b[0] - c[0]);
                let t = t0 + (t1 - t0).rem_euclid(TAU) / 2.0;
                [c[0] + r * t.cos(), c[1] + r * t.sin()]
            }
            Piece::Circle { c } => [c[0] - r, c[1]],
        }
    }
    /// The direction the piece leaves its start and arrives at its end.
    fn tangents(&self) -> (P, P) {
        match *self {
            Piece::Line { a, b } => {
                let d = sub(b, a);
                let d = mul(d, 1.0 / len(d).max(1e-300));
                (d, d)
            }
            Piece::Arc { c, a, b } => {
                let t = |p: P| {
                    let v = sub(p, c);
                    let v = mul(v, 1.0 / len(v).max(1e-300));
                    [-v[1], v[0]]
                };
                (t(a), t(b))
            }
            Piece::Circle { .. } => ([0.0, 1.0], [0.0, 1.0]),
        }
    }
}

/// Parameters along `curve` where it meets `other`: a fraction of a line, an angle on a circle.
fn meets(curve: &Curve, other: &Curve, r: f64, eps: f64, out: &mut Vec<f64>) {
    match (*curve, *other) {
        (Curve::Line { a, b }, Curve::Line { a: p, b: q }) => {
            let (d, e) = (sub(b, a), sub(q, p));
            let denom = cross(d, e);
            let l = len(d);
            if denom.abs() <= 1e-12 * l * len(e) {
                // Parallel: where the other's ends lie on this line.
                for x in [p, q] {
                    let t = dot(sub(x, a), d) / (l * l);
                    if cross(d, sub(x, a)).abs() / l <= eps && (0.0..=1.0).contains(&t) {
                        out.push(t);
                    }
                }
                return;
            }
            let t = cross(sub(p, a), e) / denom;
            let u = cross(sub(p, a), d) / denom;
            let (et, eu) = (eps / l, eps / len(e));
            if (-et..=1.0 + et).contains(&t) && (-eu..=1.0 + eu).contains(&u) {
                out.push(t.clamp(0.0, 1.0));
            }
        }
        (Curve::Line { a, b }, Curve::Circle { c }) => {
            for p in line_circle(a, b, c, r, eps) {
                let d = sub(b, a);
                out.push((dot(sub(p, a), d) / dot(d, d)).clamp(0.0, 1.0));
            }
        }
        (Curve::Circle { c }, Curve::Line { a, b }) => {
            for p in line_circle(a, b, c, r, eps) {
                out.push((p[1] - c[1]).atan2(p[0] - c[0]));
            }
        }
        (Curve::Circle { c }, Curve::Circle { c: k }) => {
            let d = dist(c, k);
            if d <= eps || d > 2.0 * r + eps {
                return;
            }
            let base = (k[1] - c[1]).atan2(k[0] - c[0]);
            let half = (d / (2.0 * r)).min(1.0).acos();
            out.push(base - half);
            if half > eps / r {
                out.push(base + half);
            }
        }
    }
}

/// Where the segment `a`–`b` meets the circle about `c` (a tangent point once).
fn line_circle(a: P, b: P, c: P, r: f64, eps: f64) -> Vec<P> {
    let d = sub(b, a);
    let l = len(d);
    let u = mul(d, 1.0 / l);
    let s = dot(sub(c, a), u);
    let foot = add(a, mul(u, s));
    let h = dist(foot, c);
    if h > r + eps {
        return Vec::new();
    }
    let w = (r * r - h * h).max(0.0).sqrt();
    let along = if w <= eps { vec![s] } else { vec![s - w, s + w] };
    along
        .into_iter()
        .filter(|t| (-eps..=l + eps).contains(t))
        .map(|t| add(a, mul(u, t.clamp(0.0, l))))
        .collect()
}

/// The closed loops bounding every point within `radius` of the `strokes` (polylines; a
/// one-point polyline is a dot), each segment tagged with `curve`. Empty when there is nothing
/// to draw or `radius` isn't positive.
pub fn stroke_outline(
    strokes: &[Vec<[f64; 2]>],
    radius: f64,
    curve: &SketchEntityId,
) -> Vec<Vec<ProfileSegment>> {
    let r = radius;
    if !(r > 0.0 && r.is_finite()) {
        return Vec::new();
    }
    let eps = 1e-7 * r;
    // The shapes: a disk at each distinct vertex, a rectangle along each distinct segment.
    let mut disks: Vec<P> = Vec::new();
    let mut segments: Vec<(P, P)> = Vec::new();
    for stroke in strokes {
        for &p in stroke {
            if !disks.iter().any(|&q| dist(p, q) <= eps) {
                disks.push(p);
            }
        }
        for w in stroke.windows(2) {
            let (p, q) = (w[0], w[1]);
            let same = |&(s, t): &(P, P)| {
                (dist(s, p) <= eps && dist(t, q) <= eps) || (dist(s, q) <= eps && dist(t, p) <= eps)
            };
            if dist(p, q) > eps && !segments.iter().any(same) {
                segments.push((p, q));
            }
        }
    }
    let mut curves: Vec<Curve> = disks.iter().map(|&c| Curve::Circle { c }).collect();
    for &(p, q) in &segments {
        let d = sub(q, p);
        let n = mul([-d[1], d[0]], r / len(d));
        curves.push(Curve::Line {
            a: add(q, n),
            b: add(p, n),
        });
        curves.push(Curve::Line {
            a: sub(p, n),
            b: sub(q, n),
        });
    }
    let bbox = |c: &Curve| match *c {
        Curve::Line { a, b } => [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])],
        Curve::Circle { c } => [c[0] - r, c[1] - r, c[0] + r, c[1] + r],
    };
    let boxes: Vec<[f64; 4]> = curves.iter().map(bbox).collect();
    let touch = |i: usize, j: usize| {
        let (a, b) = (boxes[i], boxes[j]);
        a[0] <= b[2] + eps && b[0] <= a[2] + eps && a[1] <= b[3] + eps && b[1] <= a[3] + eps
    };
    // Strictly inside a shape: a disk, or a rectangle's interior.
    let inside = |m: P| {
        disks.iter().any(|&c| dist(m, c) < r - eps)
            || segments.iter().any(|&(p, q)| {
                let d = sub(q, p);
                let l = len(d);
                let t = dot(sub(m, p), d) / l;
                t > eps && t < l - eps && cross(d, sub(m, p)).abs() / l < r - eps
            })
    };

    // Split every curve where others meet it, and keep the pieces on the region's boundary.
    let mut kept: Vec<Piece> = Vec::new();
    for (i, curve) in curves.iter().enumerate() {
        let mut at = Vec::new();
        for (j, other) in curves.iter().enumerate() {
            if i != j && touch(i, j) {
                meets(curve, other, r, eps, &mut at);
            }
        }
        let pieces: Vec<Piece> = match *curve {
            Curve::Line { a, b } => {
                at.extend([0.0, 1.0]);
                at.sort_by(f64::total_cmp);
                let l = dist(a, b);
                at.dedup_by(|x, y| (*x - *y).abs() * l <= eps);
                let point = |t: f64| add(a, mul(sub(b, a), t));
                at.windows(2)
                    .map(|w| Piece::Line {
                        a: point(w[0]),
                        b: point(w[1]),
                    })
                    .collect()
            }
            Curve::Circle { c } => {
                let mut angles: Vec<f64> = at.into_iter().map(|t| t.rem_euclid(TAU)).collect();
                angles.sort_by(f64::total_cmp);
                angles.dedup_by(|x, y| (*x - *y).abs() * r <= eps);
                if angles.len() > 1 && (angles[0] + TAU - angles[angles.len() - 1]) * r <= eps {
                    angles.pop();
                }
                let point = |t: f64| [c[0] + r * t.cos(), c[1] + r * t.sin()];
                match angles.len() {
                    0 => vec![Piece::Circle { c }],
                    n => (0..n)
                        .map(|k| Piece::Arc {
                            c,
                            a: point(angles[k]),
                            b: point(angles[(k + 1) % n]),
                        })
                        .collect(),
                }
            }
        };
        for piece in pieces {
            if matches!(piece, Piece::Arc { a, b, .. } if dist(a, b) <= eps) {
                // One cut point only: the whole circle less nothing.
                if let Piece::Arc { c, .. } = piece {
                    if !inside(Piece::Circle { c }.mid(r)) {
                        kept.push(piece);
                    }
                }
                continue;
            }
            if !inside(piece.mid(r)) {
                kept.push(piece);
            }
        }
    }
    // Coincident pieces: the same way once; opposite ways, neither.
    let mut drop = vec![false; kept.len()];
    for i in 0..kept.len() {
        for j in i + 1..kept.len() {
            if drop[i] || drop[j] {
                continue;
            }
            let (a, b) = (&kept[i], &kept[j]);
            let (sa, ea, sb, eb) = (a.start(r), a.end(r), b.start(r), b.end(r));
            let midway = dist(a.mid(r), b.mid(r)) <= eps;
            if midway && dist(sa, sb) <= eps && dist(ea, eb) <= eps {
                drop[j] = true;
            } else if midway && dist(sa, eb) <= eps && dist(ea, sb) <= eps {
                drop[i] = true;
                drop[j] = true;
            }
        }
    }
    let pieces: Vec<Piece> = kept
        .into_iter()
        .zip(drop)
        .filter(|(_, d)| !d)
        .map(|(p, _)| p)
        .collect();

    // Link the pieces into loops.
    let mut used = vec![false; pieces.len()];
    let mut loops: Vec<Vec<Piece>> = Vec::new();
    for first in 0..pieces.len() {
        if used[first] {
            continue;
        }
        used[first] = true;
        let mut chain = vec![pieces[first]];
        if matches!(pieces[first], Piece::Circle { .. }) {
            loops.push(chain);
            continue;
        }
        let origin = pieces[first].start(r);
        loop {
            let last = *chain.last().expect("non-empty");
            let end = last.end(r);
            let incoming = last.tangents().1;
            let next = (0..pieces.len())
                .filter(|&k| !used[k] && dist(pieces[k].start(r), end) <= 10.0 * eps)
                .max_by(|&x, &y| {
                    let turn = |k: usize| {
                        let out = pieces[k].tangents().0;
                        cross(incoming, out).atan2(dot(incoming, out))
                    };
                    turn(x).total_cmp(&turn(y))
                });
            match next {
                Some(k) => {
                    used[k] = true;
                    chain.push(pieces[k]);
                }
                None => break,
            }
            if dist(chain.last().expect("non-empty").end(r), origin) <= 10.0 * eps {
                break;
            }
        }
        loops.push(chain);
    }

    loops
        .into_iter()
        .filter_map(|chain| {
            let n = chain.len();
            if n == 1 {
                if let Piece::Circle { c } = chain[0] {
                    return Some(vec![ProfileSegment {
                        curve: curve.clone(),
                        kind: SegmentKind::Circle { center: c, radius: r },
                        start: [c[0] + r, c[1]],
                        end: [c[0] + r, c[1]],
                    }]);
                }
            }
            // An open chain is a defect of the input (a cusp the method can't link); skip it.
            if dist(chain[n - 1].end(r), chain[0].start(r)) > 10.0 * eps {
                return None;
            }
            Some(merge(chain, r, eps, curve))
        })
        .collect()
}

/// The loop's pieces as profile segments, consecutive pieces of one line or one circle merged
/// and every piece starting exactly where the previous one ends.
fn merge(chain: Vec<Piece>, r: f64, eps: f64, curve: &SketchEntityId) -> Vec<ProfileSegment> {
    let same_carrier = |x: &Piece, y: &Piece| match (x, y) {
        (Piece::Arc { c, .. }, Piece::Arc { c: k, .. }) => dist(*c, *k) <= eps,
        (Piece::Line { a, b }, Piece::Line { a: p, b: q }) => {
            let (d, e) = (sub(*b, *a), sub(*q, *p));
            cross(d, e).abs() <= 1e-9 * len(d) * len(e)
                && dot(d, e) > 0.0
                && cross(d, sub(*p, *a)).abs() / len(d) <= eps
        }
        _ => false,
    };
    let mut merged: Vec<Piece> = Vec::new();
    for piece in chain {
        match merged.last_mut() {
            Some(last) if same_carrier(last, &piece) => match (last, piece) {
                (Piece::Arc { b, .. }, Piece::Arc { b: nb, .. })
                | (Piece::Line { b, .. }, Piece::Line { b: nb, .. }) => *b = nb,
                _ => unreachable!("same carrier"),
            },
            _ => merged.push(piece),
        }
    }
    // The loop may start partway along a carrier: join its last piece onto its first.
    if merged.len() > 1 && same_carrier(&merged[merged.len() - 1], &merged[0]) {
        let last = merged.pop().expect("non-empty");
        match (&mut merged[0], last) {
            (Piece::Arc { a, .. }, Piece::Arc { a: la, .. })
            | (Piece::Line { a, .. }, Piece::Line { a: la, .. }) => *a = la,
            _ => unreachable!("same carrier"),
        }
    }
    // A loop that merged into one arc all the way round is a circle.
    if let [Piece::Arc { c, .. }] = merged.as_slice() {
        let c = *c;
        return vec![ProfileSegment {
            curve: curve.clone(),
            kind: SegmentKind::Circle { center: c, radius: r },
            start: [c[0] + r, c[1]],
            end: [c[0] + r, c[1]],
        }];
    }
    let n = merged.len();
    let starts: Vec<P> = merged.iter().map(|p| p.start(r)).collect();
    merged
        .iter()
        .enumerate()
        .map(|(k, piece)| {
            let (start, end) = (starts[k], starts[(k + 1) % n]);
            let kind = match piece {
                Piece::Line { .. } => SegmentKind::Line,
                Piece::Arc { c, .. } => SegmentKind::Arc {
                    center: *c,
                    clockwise: false,
                },
                Piece::Circle { .. } => unreachable!("a circle is its own loop"),
            };
            ProfileSegment {
                curve: curve.clone(),
                kind,
                start,
                end,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
