//! Where Measure draws two parallel planar items' normal distance. The dimension line always
//! runs along the common normal between the items' two planes, so its drawn length equals its
//! label.
//!
//! Both items are projected along the normal onto one plane.
//!
//! - Where the projections overlap, the line stands at the overlap's centroid (the area centroid
//!   of overlapping faces, the length centroid where an edge crosses a face, the mean of vertices
//!   inside a face), moved to the nearest overlap point when the centroid falls outside it (an
//!   L-shaped overlap). Both ends lie on the items, and no extension lines are needed.
//! - Where they don't overlap, the line stands at the second item's point closest to the first
//!   item's projection. When a whole stretch ties for closest (two parallel edges), it stands at
//!   that stretch's middle. Its end on the first item's plane lies off that item, so an extension
//!   line joins that end to the nearest point of the item's outline, as ISO 129-1 extension
//!   lines carry a dimension off its feature. Items whose projections touch along an edge
//!   measure at the shared edge's middle with no extension lines.
//!
//! The Desktop draws a .fea study's face-to-face normal distance by the same rule
//! (`normalSpan` in desktop measure-span.ts), from the picked faces' mesh triangles.

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

/// A normal distance's drawing: the dimension line from `a`'s plane (`from`) to `b`'s plane (`to`)
/// along the normal, and the extension lines that join an end lying off its item to that item's
/// outline, each `[point on the item, dimension line end]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub from: V3,
    pub to: V3,
    pub extension_lines: Vec<[V3; 2]>,
}

/// The part of segment `s` within `r` of segment `other`, as parameters along `s`, around its
/// point at parameter `t0` (which is within `r`). The distance along `s` is convex, so the part
/// is one interval.
fn within_of(s: [P2; 2], other: [P2; 2], t0: f64, r: f64) -> [f64; 2] {
    let at = |t: f64| {
        [
            s[0][0] + t * (s[1][0] - s[0][0]),
            s[0][1] + t * (s[1][1] - s[0][1]),
        ]
    };
    let far = |t: f64| {
        let p = at(t);
        dist2(p, closest_on_segment(p, other)) > r
    };
    // Bisect between a point inside (`inner`) and an end outside.
    let edge = |end: f64| {
        if !far(end) {
            return end;
        }
        let (mut inner, mut outer) = (t0, end);
        for _ in 0..60 {
            let mid = 0.5 * (inner + outer);
            if far(mid) {
                outer = mid;
            } else {
                inner = mid;
            }
        }
        inner
    };
    [edge(0.0), edge(1.0)]
}

/// Parameter intervals merged where they overlap.
fn union(mut intervals: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    intervals.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut out: Vec<[f64; 2]> = Vec::new();
    for i in intervals {
        match out.last_mut() {
            Some(last) if i[0] <= last[1] => last[1] = last[1].max(i[1]),
            _ => out.push(i),
        }
    }
    out
}

/// An item's outline: its triangles' boundary edges (the edges only one triangle has, either way
/// round), its segments, and its points as zero-length segments.
fn outline(f: &Flat) -> Vec<[P2; 2]> {
    let edges: Vec<[P2; 2]> = f
        .triangles
        .iter()
        .flat_map(|t| [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]])
        .collect();
    let key = |e: &[P2; 2]| {
        let bits = |p: P2| [p[0].to_bits(), p[1].to_bits()];
        let (x, y) = (bits(e[0]), bits(e[1]));
        if x <= y {
            (x, y)
        } else {
            (y, x)
        }
    };
    let mut count = std::collections::HashMap::new();
    for e in &edges {
        *count.entry(key(e)).or_insert(0usize) += 1;
    }
    edges
        .iter()
        .filter(|e| count[&key(e)] == 1)
        .copied()
        .chain(f.segments.iter().copied())
        .chain(f.points.iter().map(|&p| [p, p]))
        .collect()
}

/// The parameter of `p` along segment `s`.
fn parameter(p: P2, s: [P2; 2]) -> f64 {
    let d = sub2(s[1], s[0]);
    let l = d[0] * d[0] + d[1] * d[1];
    if l > 0.0 {
        (((p[0] - s[0][0]) * d[0] + (p[1] - s[0][1]) * d[1]) / l).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The nearest point of an outline to `p`.
fn nearest_on(outline: &[[P2; 2]], p: P2) -> Option<P2> {
    outline
        .iter()
        .map(|s| closest_on_segment(p, *s))
        .min_by(|x, y| dist2(*x, p).total_cmp(&dist2(*y, p)))
}

/// The normal-distance drawing between items `a` and `b`, which lie in parallel planes with
/// normal `normal` (see the module notes). `None` when either item has no geometry.
pub fn normal_span(a: &Primitives, b: &Primitives, normal: V3) -> Option<Span> {
    let n = unit(normal);
    let u = perpendicular(n);
    let v = cross(n, u);
    let (fa, fb) = (flatten(a, u, v, n), flatten(b, u, v, n));
    // Overlaps smaller than a millionth of the items' size, squared, are rounding slivers where
    // two projections only touch.
    let all: Vec<P2> = [&fa, &fb]
        .iter()
        .flat_map(|f| {
            f.triangles
                .iter()
                .flatten()
                .chain(f.segments.iter().flatten())
                .chain(&f.points)
        })
        .copied()
        .collect();
    let box_ = bounds(&all);
    let size = (box_[2] - box_[0]).hypot(box_[3] - box_[1]);
    let sliver = (1e-6 * size).powi(2);
    let mut overlap = Overlap::default();
    for ta in &fa.triangles {
        let ba = bounds(ta);
        for tb in &fb.triangles {
            if apart(ba, bounds(tb)) {
                continue;
            }
            let piece = clip(ta.to_vec(), tb);
            if piece.len() >= 3 && area_centroid(&piece).0.abs() > sliver {
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
        return Some(Span {
            from: lift(p, fa.height),
            to: lift(p, fb.height),
            extension_lines: Vec::new(),
        });
    }
    // No overlap: the second item's closest region to the first, from the items' outlines.
    let (oa, ob) = (outline(&fa), outline(&fb));
    let pairs: Vec<(f64, P2, usize, [P2; 2])> = oa
        .iter()
        .flat_map(|sa| {
            ob.iter().enumerate().map(move |(j, sb)| {
                let (_, q) = closest_segments(*sa, *sb);
                let p = closest_on_segment(q, *sa);
                (dist2(p, q), q, j, *sa)
            })
        })
        .collect();
    let &(least, closest, ..) = pairs.iter().min_by(|x, y| x.0.total_cmp(&y.0))?;
    // Ties within a millionth of the items' size (the display mesh's f32 corners differ by
    // that much along one straight edge).
    let r = least + 1e-9 + 1e-6 * size;
    // The closest region: on each of `b`'s segments, the union of the stretches within `r` of
    // one of `a`'s.
    let mut stretches: Vec<Vec<[f64; 2]>> = vec![Vec::new(); ob.len()];
    for &(d, q, j, sa) in &pairs {
        if d <= r {
            stretches[j].push(within_of(ob[j], sa, parameter(q, ob[j]), r));
        }
    }
    let mut region = Overlap::default();
    for (j, intervals) in stretches.into_iter().enumerate() {
        let s = ob[j];
        let at = |t: f64| {
            [
                s[0][0] + t * (s[1][0] - s[0][0]),
                s[0][1] + t * (s[1][1] - s[0][1]),
            ]
        };
        for [lo, hi] in union(intervals) {
            let piece = [at(lo), at(hi)];
            if dist2(piece[0], piece[1]) > 0.0 {
                region.lengths.push(piece);
            } else {
                region.points.push(piece[0]);
            }
        }
    }
    // A stretch shorter than a thousandth of the items' size is the tolerance around one
    // closest point (a corner), not a tie: the line stands at that point.
    let tied: f64 = region.lengths.iter().map(|s| dist2(s[0], s[1])).sum();
    let q = if tied < 1e-3 * size {
        closest
    } else {
        region.point()?
    };
    let (from, to) = (lift(q, fa.height), lift(q, fb.height));
    // Each end off its item reaches the item's outline along that item's plane.
    let mut extension_lines = Vec::new();
    for (o, end, h) in [(&oa, from, fa.height), (&ob, to, fb.height)] {
        let p = nearest_on(o, q)?;
        if dist2(p, q) > r - least {
            extension_lines.push([lift(p, h), end]);
        }
    }
    Some(Span {
        from,
        to,
        extension_lines,
    })
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
        let span = normal_span(&base, &wall, [0.0, 0.0, 1.0]).unwrap();
        close(span.from, [55.0, 20.0, 0.0]);
        close(span.to, [55.0, 20.0, 37.0]);
        // Both ends lie on the faces: no extension lines.
        assert!(span.extension_lines.is_empty());
    }

    #[test]
    fn an_l_shaped_overlap_moves_its_centroid_onto_the_overlap() {
        // Face a: an L (two rectangles); face b covers it all. The L's centroid lies outside it.
        let mut l = square(0.0, 0.0, 10.0, 1.0, 0.0);
        l.triangles
            .extend(square(0.0, 1.0, 1.0, 10.0, 0.0).triangles);
        let b = square(-5.0, -5.0, 20.0, 20.0, 3.0);
        let span = normal_span(&l, &b, [0.0, 0.0, 1.0]).unwrap();
        let (from, to) = (span.from, span.to);
        assert!(span.extension_lines.is_empty());
        assert!((to[2] - 3.0).abs() < 1e-12 && from[2].abs() < 1e-12);
        let on_l = (from[0] <= 10.0 && from[1] <= 1.0 + 1e-9)
            || (from[0] <= 1.0 + 1e-9 && from[1] <= 10.0);
        assert!(on_l, "{from:?} is off the L");
        close([from[0], from[1], 0.0], [to[0], to[1], 0.0]);
    }

    /// The rule's drafting checks: the dimension line runs along `n` and is as long as the
    /// planes are apart; every extension line starts on its item's outline and ends at a
    /// dimension-line end in that item's plane.
    fn drafted(span: &Span, n: V3, gap: f64) {
        let d = crate::edit::assembly::frames::sub(span.to, span.from);
        let along = cross(d, n);
        assert!(
            along.iter().all(|c| c.abs() < 1e-9),
            "{span:?} leaves the normal"
        );
        assert!(
            (crate::edit::assembly::frames::norm(d) - gap).abs() < 1e-9,
            "{span:?}"
        );
        for [on, end] in &span.extension_lines {
            assert!(*end == span.from || *end == span.to, "{span:?}");
            assert!(
                (dot(*on, n) - dot(*end, n)).abs() < 1e-9,
                "{span:?} leaves the plane"
            );
        }
    }

    #[test]
    fn faces_that_dont_overlap_stand_the_line_on_the_second_and_extend_to_the_first() {
        // a: x 0..10; b: x 20..30, 5 above. The line stands at b's nearest edge (x = 20) and an
        // extension line in a's plane runs back to a's edge (x = 10).
        let a = square(0.0, 0.0, 10.0, 10.0, 0.0);
        let b = square(20.0, 0.0, 30.0, 10.0, 5.0);
        let n = [0.0, 0.0, 1.0];
        let span = normal_span(&a, &b, n).unwrap();
        drafted(&span, n, 5.0);
        // The whole edge x = 20 ties for closest: the line stands at its middle.
        close(span.from, [20.0, 5.0, 0.0]);
        close(span.to, [20.0, 5.0, 5.0]);
        assert_eq!(span.extension_lines.len(), 1);
        close(span.extension_lines[0][0], [10.0, 5.0, 0.0]);
        close(span.extension_lines[0][1], span.from);
    }

    #[test]
    fn a_tie_takes_the_middle_of_the_shorter_side() {
        // b's facing edge (y 2..6) is shorter than a's (y 0..10): the line stands at y = 4.
        let a = square(0.0, 0.0, 10.0, 10.0, 0.0);
        let b = square(20.0, 2.0, 30.0, 6.0, -3.0);
        let n = [0.0, 0.0, 1.0];
        let span = normal_span(&a, &b, n).unwrap();
        drafted(&span, n, 3.0);
        close(span.to, [20.0, 4.0, -3.0]);
        close(span.extension_lines[0][0], [10.0, 4.0, 0.0]);
    }

    #[test]
    fn a_diagonal_offset_extends_to_the_first_faces_corner() {
        // b sits beyond a's corner (10, 10): its closest point is its corner (20, 15), and the
        // extension line runs to a's corner.
        let a = square(0.0, 0.0, 10.0, 10.0, 0.0);
        let b = square(20.0, 15.0, 30.0, 25.0, 7.0);
        let n = [0.0, 0.0, 1.0];
        let span = normal_span(&a, &b, n).unwrap();
        drafted(&span, n, 7.0);
        close(span.to, [20.0, 15.0, 7.0]);
        close(span.extension_lines[0][0], [10.0, 10.0, 0.0]);
    }

    #[test]
    fn faces_touching_along_an_edge_measure_at_its_middle() {
        // A step: the projections share the edge x = 10, y 0..10.
        let a = square(0.0, 0.0, 10.0, 10.0, 0.0);
        let b = square(10.0, 0.0, 20.0, 10.0, 4.0);
        let n = [0.0, 0.0, 1.0];
        let span = normal_span(&a, &b, n).unwrap();
        drafted(&span, n, 4.0);
        close(span.from, [10.0, 5.0, 0.0]);
        close(span.to, [10.0, 5.0, 4.0]);
        assert!(span.extension_lines.is_empty(), "{span:?}");
    }

    #[test]
    fn a_tilted_normal_keeps_the_line_along_it() {
        // The same separated squares turned about X: the rule doesn't depend on the frame.
        let (c, s) = (0.6_f64, 0.8_f64);
        let turn = |p: Primitives| Primitives {
            triangles: p
                .triangles
                .iter()
                .map(|t| t.map(|x| [x[0], c * x[1] - s * x[2], s * x[1] + c * x[2]]))
                .collect(),
            ..Default::default()
        };
        let a = turn(square(0.0, 0.0, 10.0, 10.0, 0.0));
        let b = turn(square(20.0, 0.0, 30.0, 10.0, 5.0));
        let n = [0.0, -s, c];
        let span = normal_span(&a, &b, n).unwrap();
        drafted(&span, n, 5.0);
        assert_eq!(span.extension_lines.len(), 1);
        assert!(
            (span.extension_lines[0][0][0] - 10.0).abs() < 1e-9,
            "{span:?}"
        );
    }

    #[test]
    fn an_edge_over_a_face_takes_the_part_inside() {
        let face = square(0.0, 0.0, 10.0, 10.0, 0.0);
        let edge = Primitives {
            segments: vec![[[5.0, 5.0, 8.0], [25.0, 5.0, 8.0]]],
            ..Default::default()
        };
        let span = normal_span(&edge, &face, [0.0, 0.0, 1.0]).unwrap();
        close(span.from, [7.5, 5.0, 8.0]);
        close(span.to, [7.5, 5.0, 0.0]);
        assert!(span.extension_lines.is_empty());
    }
}
