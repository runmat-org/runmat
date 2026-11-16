//! Where Measure draws two parallel planar items' normal distance: along the common normal,
//! between two points that lie on the items themselves.
//!
//! Both items are projected along the normal onto one plane. Where the projections overlap,
//! the line stands at the overlap's centroid (the area centroid of overlapping faces, the
//! length centroid where an edge crosses a face, the mean of vertices inside a face), moved to
//! the nearest overlap point when the centroid falls outside it (an L-shaped overlap). Where
//! they don't overlap, the line joins the closest pair of points between the items. Either way
//! both ends lie on the picked geometry.
//!
//! The Desktop draws a .fea study's face-to-face normal distance by the same rule
//! (`normalSpan` in desktop measure-format.ts), from the picked faces' mesh triangles.

use super::distance::Primitives;
use crate::edit::assembly::frames::{add, cross, dot, perpendicular, scale, unit, V3};

type P2 = [f64; 2];

fn sub2(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross2(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn dist2(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The closest point of segment `s` to `p`.
fn closest_on_segment(p: P2, s: [P2; 2]) -> P2 {
    let d = sub2(s[1], s[0]);
    let l = d[0] * d[0] + d[1] * d[1];
    let t = if l > 0.0 {
        (((p[0] - s[0][0]) * d[0] + (p[1] - s[0][1]) * d[1]) / l).clamp(0.0, 1.0)
    } else {
        0.0
    };
    [s[0][0] + t * d[0], s[0][1] + t * d[1]]
}

/// The closest pair between two segments (their ends against each other; segments that cross
/// overlap, which [`normal_span`] finds first).
fn closest_segments(a: [P2; 2], b: [P2; 2]) -> (P2, P2) {
    let candidates = [
        (a[0], closest_on_segment(a[0], b)),
        (a[1], closest_on_segment(a[1], b)),
        (closest_on_segment(b[0], a), b[0]),
        (closest_on_segment(b[1], a), b[1]),
    ];
    candidates
        .into_iter()
        .min_by(|x, y| dist2(x.0, x.1).total_cmp(&dist2(y.0, y.1)))
        .expect("four candidates")
}

/// A counter-clockwise triangle (or `None` when degenerate).
fn ccw(t: [P2; 3]) -> Option<[P2; 3]> {
    let area = cross2(sub2(t[1], t[0]), sub2(t[2], t[0]));
    if area.abs() < 1e-18 {
        None
    } else if area > 0.0 {
        Some(t)
    } else {
        Some([t[0], t[2], t[1]])
    }
}

fn inside(p: P2, t: &[P2; 3]) -> bool {
    (0..3).all(|i| cross2(sub2(t[(i + 1) % 3], t[i]), sub2(p, t[i])) >= -1e-15)
}

/// Polygon `poly` clipped to the counter-clockwise triangle `t` (Sutherland–Hodgman).
fn clip(poly: Vec<P2>, t: &[P2; 3]) -> Vec<P2> {
    let mut out = poly;
    for i in 0..3 {
        let (e0, e1) = (t[i], t[(i + 1) % 3]);
        let side = |p: P2| cross2(sub2(e1, e0), sub2(p, e0));
        let input = std::mem::take(&mut out);
        for k in 0..input.len() {
            let (p, q) = (input[k], input[(k + 1) % input.len()]);
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let s = sp / (sp - sq);
                out.push([p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])]);
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// A polygon's area and centroid.
fn area_centroid(poly: &[P2]) -> (f64, P2) {
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for k in 0..poly.len() {
        let (p, q) = (poly[k], poly[(k + 1) % poly.len()]);
        let c = cross2(p, q);
        a += c;
        cx += (p[0] + q[0]) * c;
        cy += (p[1] + q[1]) * c;
    }
    a /= 2.0;
    if a.abs() < 1e-30 {
        return (0.0, poly[0]);
    }
    (a, [cx / (6.0 * a), cy / (6.0 * a)])
}

/// The part of segment `s` inside the counter-clockwise triangle `t`.
fn clip_segment(s: [P2; 2], t: &[P2; 3]) -> Option<[P2; 2]> {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let d = sub2(s[1], s[0]);
    for i in 0..3 {
        let (e0, e1) = (t[i], t[(i + 1) % 3]);
        let e = sub2(e1, e0);
        let at = cross2(e, sub2(s[0], e0));
        let rate = cross2(e, d);
        if rate.abs() < 1e-30 {
            if at < 0.0 {
                return None;
            }
            continue;
        }
        let x = -at / rate;
        if rate > 0.0 {
            lo = lo.max(x);
        } else {
            hi = hi.min(x);
        }
    }
    (hi > lo + 1e-12).then(|| {
        let at = |x: f64| [s[0][0] + x * d[0], s[0][1] + x * d[1]];
        [at(lo), at(hi)]
    })
}

/// Where the projections overlap, by dimension: areas, lengths, points.
#[derive(Default)]
struct Overlap {
    areas: Vec<Vec<P2>>,
    lengths: Vec<[P2; 2]>,
    points: Vec<P2>,
}

impl Overlap {
    /// The overlap's centroid, moved onto the overlap when it falls outside.
    fn point(&self) -> Option<P2> {
        let weighted = |items: Vec<(f64, P2)>| -> Option<P2> {
            let total: f64 = items.iter().map(|i| i.0).sum();
            (total > 0.0).then(|| {
                let x = items.iter().map(|i| i.0 * i.1[0]).sum::<f64>() / total;
                let y = items.iter().map(|i| i.0 * i.1[1]).sum::<f64>() / total;
                [x, y]
            })
        };
        let nearest = |c: P2, candidates: Vec<P2>| {
            candidates
                .into_iter()
                .min_by(|a, b| dist2(*a, c).total_cmp(&dist2(*b, c)))
        };
        if let Some(c) = weighted(
            self.areas
                .iter()
                .map(|p| {
                    let (a, c) = area_centroid(p);
                    (a.abs(), c)
                })
                .collect(),
        ) {
            let on = |p: &Vec<P2>| {
                (1..p.len().saturating_sub(1))
                    .any(|k| ccw([p[0], p[k], p[k + 1]]).is_some_and(|t| inside(c, &t)))
            };
            if self.areas.iter().any(on) {
                return Some(c);
            }
            return nearest(
                c,
                self.areas
                    .iter()
                    .flat_map(|p| (0..p.len()).map(move |k| [p[k], p[(k + 1) % p.len()]]))
                    .map(|e| closest_on_segment(c, e))
                    .collect(),
            );
        }
        if let Some(c) = weighted(
            self.lengths
                .iter()
                .map(|s| {
                    (
                        dist2(s[0], s[1]),
                        [(s[0][0] + s[1][0]) / 2.0, (s[0][1] + s[1][1]) / 2.0],
                    )
                })
                .collect(),
        ) {
            return nearest(
                c,
                self.lengths
                    .iter()
                    .map(|s| closest_on_segment(c, *s))
                    .collect(),
            );
        }
        if self.points.is_empty() {
            return None;
        }
        let n = self.points.len() as f64;
        let c = [
            self.points.iter().map(|p| p[0]).sum::<f64>() / n,
            self.points.iter().map(|p| p[1]).sum::<f64>() / n,
        ];
        nearest(c, self.points.clone())
    }
}

/// An item projected on the plane: its triangles, segments and points, and its height along the
/// normal (the mean of its points' heights; a planar item has one).
struct Flat {
    triangles: Vec<[P2; 3]>,
    segments: Vec<[P2; 2]>,
    points: Vec<P2>,
    height: f64,
}

fn flatten(p: &Primitives, u: V3, v: V3, n: V3) -> Flat {
    let at = |x: V3| [dot(x, u), dot(x, v)];
    let (mut sum, mut count) = (0.0, 0usize);
    let mut h = |x: V3| {
        sum += dot(x, n);
        count += 1;
    };
    for t in &p.triangles {
        t.iter().for_each(|&x| h(x));
    }
    for s in &p.segments {
        s.iter().for_each(|&x| h(x));
    }
    p.points.iter().for_each(|&x| h(x));
    Flat {
        triangles: p
            .triangles
            .iter()
            .filter_map(|t| ccw([at(t[0]), at(t[1]), at(t[2])]))
            .collect(),
        segments: p.segments.iter().map(|s| [at(s[0]), at(s[1])]).collect(),
        points: p.points.iter().map(|&x| at(x)).collect(),
        height: if count > 0 { sum / count as f64 } else { 0.0 },
    }
}

fn bounds(points: &[P2]) -> [f64; 4] {
    points.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |b, p| {
            [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ]
        },
    )
}

fn apart(a: [f64; 4], b: [f64; 4]) -> bool {
    a[2] < b[0] || b[2] < a[0] || a[3] < b[1] || b[3] < a[1]
}

/// The two ends of the normal-distance line between items `a` and `b`, which lie in parallel
/// planes with unit normal `normal`: a point on `a` and a point on `b`. `None` when either item
/// has no geometry.
pub fn normal_span(a: &Primitives, b: &Primitives, normal: V3) -> Option<(V3, V3)> {
    let n = unit(normal);
    let u = perpendicular(n);
    let v = cross(n, u);
    let (fa, fb) = (flatten(a, u, v, n), flatten(b, u, v, n));
    let mut overlap = Overlap::default();
    for ta in &fa.triangles {
        let ba = bounds(ta);
        for tb in &fb.triangles {
            if apart(ba, bounds(tb)) {
                continue;
            }
            let piece = clip(ta.to_vec(), tb);
            if piece.len() >= 3 && area_centroid(&piece).0.abs() > 1e-24 {
                overlap.areas.push(piece);
            }
        }
    }
    for (segments, triangles) in [(&fa.segments, &fb.triangles), (&fb.segments, &fa.triangles)] {
        for s in segments {
            for t in triangles {
                if let Some(piece) = clip_segment(*s, t) {
                    overlap.lengths.push(piece);
                }
            }
        }
    }
    for (points, triangles) in [(&fa.points, &fb.triangles), (&fb.points, &fa.triangles)] {
        for p in points {
            if triangles.iter().any(|t| inside(*p, t)) {
                overlap.points.push(*p);
            }
        }
    }
    let lift = |p: P2, h: f64| add(add(scale(u, p[0]), scale(v, p[1])), scale(n, h));
    if let Some(p) = overlap.point() {
        return Some((lift(p, fa.height), lift(p, fb.height)));
    }
    // No overlap: the closest pair, from each item's outline (triangle edges and segments) and
    // points.
    let outline = |f: &Flat| -> Vec<[P2; 2]> {
        f.triangles
            .iter()
            .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
            .chain(f.segments.iter().copied())
            .chain(f.points.iter().map(|&p| [p, p]))
            .collect()
    };
    let (oa, ob) = (outline(&fa), outline(&fb));
    let mut best: Option<(f64, P2, P2)> = None;
    for sa in &oa {
        for sb in &ob {
            let (p, q) = closest_segments(*sa, *sb);
            let d = dist2(p, q);
            if best.is_none_or(|b| d < b.0) {
                best = Some((d, p, q));
            }
        }
    }
    best.map(|(_, p, q)| (lift(p, fa.height), lift(q, fb.height)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64, z: f64) -> Primitives {
        let (a, b, c, d) = ([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]);
        Primitives {
            triangles: vec![[a, b, c], [a, c, d]],
            ..Default::default()
        }
    }

    fn close(a: V3, b: V3) {
        assert!((0..3).all(|i| (a[i] - b[i]).abs() < 1e-9), "{a:?} vs {b:?}");
    }

    #[test]
    fn overlapping_faces_take_the_overlap_centroid() {
        // A wide base top at z = 0 and a narrow wall top at z = 37 over its right end.
        let base = square(0.0, 0.0, 60.0, 40.0, 0.0);
        let wall = square(50.0, 0.0, 60.0, 40.0, 37.0);
        let (from, to) = normal_span(&base, &wall, [0.0, 0.0, 1.0]).unwrap();
        close(from, [55.0, 20.0, 0.0]);
        close(to, [55.0, 20.0, 37.0]);
    }

    #[test]
    fn an_l_shaped_overlap_moves_its_centroid_onto_the_overlap() {
        // Face a: an L (two rectangles); face b covers it all. The L's centroid lies outside it.
        let mut l = square(0.0, 0.0, 10.0, 1.0, 0.0);
        l.triangles
            .extend(square(0.0, 1.0, 1.0, 10.0, 0.0).triangles);
        let b = square(-5.0, -5.0, 20.0, 20.0, 3.0);
        let (from, to) = normal_span(&l, &b, [0.0, 0.0, 1.0]).unwrap();
        assert!((to[2] - 3.0).abs() < 1e-12 && from[2].abs() < 1e-12);
        let on_l = (from[0] <= 10.0 && from[1] <= 1.0 + 1e-9)
            || (from[0] <= 1.0 + 1e-9 && from[1] <= 10.0);
        assert!(on_l, "{from:?} is off the L");
        close([from[0], from[1], 0.0], [to[0], to[1], 0.0]);
    }

    #[test]
    fn faces_that_dont_overlap_take_their_closest_pair() {
        let a = square(0.0, 0.0, 10.0, 10.0, 0.0);
        let b = square(20.0, 0.0, 30.0, 10.0, 5.0);
        let (from, to) = normal_span(&a, &b, [0.0, 0.0, 1.0]).unwrap();
        assert!((from[0] - 10.0).abs() < 1e-9 && (to[0] - 20.0).abs() < 1e-9);
        assert!(from[2].abs() < 1e-12 && (to[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn an_edge_over_a_face_takes_the_part_inside() {
        let face = square(0.0, 0.0, 10.0, 10.0, 0.0);
        let edge = Primitives {
            segments: vec![[[5.0, 5.0, 8.0], [25.0, 5.0, 8.0]]],
            ..Default::default()
        };
        let (from, to) = normal_span(&edge, &face, [0.0, 0.0, 1.0]).unwrap();
        close(from, [7.5, 5.0, 8.0]);
        close(to, [7.5, 5.0, 0.0]);
    }
}
