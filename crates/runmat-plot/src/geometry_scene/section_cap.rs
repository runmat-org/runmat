//! Section caps: the solid's cut face where a section plane crosses a closed mesh, drawn as a
//! section view shows it. Without caps a cut model shows its hollow inside, and edges
//! behind the cut read as if they were in front.
//!
//! Each triangle that crosses the plane gives one segment of the cut. The segment is oriented
//! with the solid on its left (from the triangle's outward winding), so the segments of a closed
//! body form loops wound counter-clockwise around material and clockwise around holes, as a
//! fuselage skin's ring of material does. Only segments that join into closed loops are kept: an
//! open surface has no inside to fill. The loops fill by the nonzero winding rule, so bodies that
//! overlap union instead of punching holes in each other, through a scanline over the loops'
//! vertices: between two consecutive vertex heights no segment starts or ends, and each span with
//! nonzero winding is one trapezoid.
//!
//! The cap takes the color of the body it cuts, tinted toward [`SECTION_CAP_TINT`] so the cut
//! face reads apart from the model's own faces, and faces the removed side of the plane.

use crate::core::Vertex;
use glam::{Vec2, Vec3, Vec4};
use std::collections::HashMap;

/// The color a cut face leans toward: a warm grey, as a drafting section's tint.
pub const SECTION_CAP_TINT: [f32; 4] = [0.93, 0.56, 0.38, 1.0];
/// How far a cut face's color leans toward [`SECTION_CAP_TINT`].
pub const SECTION_CAP_TINT_SHARE: f32 = 0.38;

/// One oriented segment of the cut, in plane coordinates, with the color of the face it cut.
#[derive(Clone, Copy, Debug)]
struct Segment {
    a: Vec2,
    b: Vec2,
    color: [f32; 4],
}

/// The plane's coordinates: `u` and `v` span it with `u × v = normal`.
#[derive(Clone, Copy)]
struct PlaneFrame {
    normal: Vec3,
    origin: Vec3,
    u: Vec3,
    v: Vec3,
}

impl PlaneFrame {
    fn new(normal: Vec3, origin: Vec3) -> Self {
        let helper = if normal.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let u = helper.cross(normal).normalize_or_zero();
        let v = normal.cross(u);
        Self {
            normal,
            origin,
            u,
            v,
        }
    }

    fn to_plane(self, p: Vec3) -> Vec2 {
        let d = p - self.origin;
        Vec2::new(d.dot(self.u), d.dot(self.v))
    }

    fn to_world(self, p: Vec2) -> Vec3 {
        self.origin + self.u * p.x + self.v * p.y
    }
}

/// The cap triangles (three vertices each, unindexed) where the plane `(normal, origin)` cuts the
/// closed meshes among `triangles`. The kept side is where `(p − origin) · normal ≥ 0`.
pub(super) fn section_cap(
    triangles: &[[Vertex; 3]],
    (normal, origin): (Vec3, Vec3),
) -> Vec<Vertex> {
    let frame = PlaneFrame::new(normal, origin);
    let mut segments = Vec::new();
    let mut extent = 0.0f32;
    for tri in triangles {
        if let Some(segment) = cut_triangle(tri, &frame) {
            extent = extent
                .max(segment.a.abs().max_element())
                .max(segment.b.abs().max_element());
            segments.push(segment);
        }
    }
    if segments.is_empty() {
        return Vec::new();
    }
    // Endpoints shared by neighbouring triangles are computed from the same edge, so they agree
    // to rounding; the tolerance scales with the cut's size.
    let tolerance = (extent * 2.0e-6).max(1.0e-9);
    let closed = closed_segments(&segments, tolerance);
    let face_normal = (-normal).to_array();
    fill_nonzero(&closed)
        .into_iter()
        .flat_map(|(points, color)| {
            let tinted = Vec4::from_array(color)
                .lerp(Vec4::from_array(SECTION_CAP_TINT), SECTION_CAP_TINT_SHARE);
            let color = [tinted.x, tinted.y, tinted.z, color[3]];
            points.into_iter().map(move |p| Vertex {
                position: frame.to_world(p).to_array(),
                color,
                normal: face_normal,
                tex_coords: [0.0, 0.0],
            })
        })
        .collect()
}

/// The segment where the plane crosses one triangle, oriented with the solid on its left.
fn cut_triangle(tri: &[Vertex; 3], frame: &PlaneFrame) -> Option<Segment> {
    let p = tri.map(|v| Vec3::from_array(v.position));
    let d = p.map(|q| (q - frame.origin).dot(frame.normal));
    let side = d.map(|x| x >= 0.0);
    if side[0] == side[1] && side[1] == side[2] {
        return None;
    }
    let mut points = Vec::with_capacity(2);
    for (i, j) in [(0, 1), (1, 2), (2, 0)] {
        if side[i] != side[j] {
            // From the edge's endpoints in one fixed order, so both triangles on the edge
            // compute the same point to the bit.
            let (a, b) = if p[i].to_array() < p[j].to_array() {
                (i, j)
            } else {
                (j, i)
            };
            let t = d[a] / (d[a] - d[b]);
            points.push(p[a].lerp(p[b], t));
        }
    }
    let (a, b) = (frame.to_plane(points[0]), frame.to_plane(points[1]));
    if (b - a).length_squared() <= 0.0 {
        return None;
    }
    // The solid lies behind the triangle's outward face: its in-plane inward direction.
    let outward = (p[1] - p[0]).cross(p[2] - p[0]);
    let inward3 = -(outward - frame.normal * outward.dot(frame.normal));
    let inward = Vec2::new(inward3.dot(frame.u), inward3.dot(frame.v));
    let along = b - a;
    let left = along.perp_dot(inward) > 0.0;
    let color = tri[0].color;
    Some(if left {
        Segment { a, b, color }
    } else {
        Segment { a: b, b: a, color }
    })
}

/// The segments that join end to start into closed loops; dangling chains (an open surface's
/// cut) are dropped.
fn closed_segments(segments: &[Segment], tolerance: f32) -> Vec<Segment> {
    let key = |p: Vec2| {
        (
            (p.x / tolerance).round() as i64,
            (p.y / tolerance).round() as i64,
        )
    };
    // Each point's segment count in and out; a point on a closed loop balances them.
    let mut balance: HashMap<(i64, i64), i32> = HashMap::new();
    let mut ends = Vec::with_capacity(segments.len());
    for s in segments {
        let (ka, kb) = (key(s.a), key(s.b));
        *balance.entry(ka).or_default() += 1;
        *balance.entry(kb).or_default() -= 1;
        ends.push((ka, kb));
    }
    // Drop segments touching an unbalanced point until none is left: what stays are loops.
    let mut keep = vec![true; segments.len()];
    let mut by_point: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, (ka, kb)) in ends.iter().enumerate() {
        by_point.entry(*ka).or_default().push(i);
        by_point.entry(*kb).or_default().push(i);
    }
    let mut stack: Vec<(i64, i64)> = balance
        .iter()
        .filter(|(_, b)| **b != 0)
        .map(|(k, _)| *k)
        .collect();
    while let Some(point) = stack.pop() {
        if balance.get(&point).copied().unwrap_or(0) == 0 {
            continue;
        }
        for &i in by_point.get(&point).map(Vec::as_slice).unwrap_or(&[]) {
            let excess = balance.get(&point).copied().unwrap_or(0);
            // Only segments that leave (excess out) or reach (excess in) the point settle it.
            if !keep[i] || excess == 0 || (excess > 0) != (ends[i].0 == point) {
                continue;
            }
            keep[i] = false;
            let (ka, kb) = ends[i];
            *balance.entry(ka).or_default() -= 1;
            *balance.entry(kb).or_default() += 1;
            for k in [ka, kb] {
                if k != point && balance.get(&k).copied().unwrap_or(0) != 0 {
                    stack.push(k);
                }
            }
        }
    }
    segments
        .iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(s, _)| *s)
        .collect()
}

/// Triangles (plane coordinates, three points each) filling where the segments' winding number
/// is nonzero, each with the color of the segment on its span's left.
fn fill_nonzero(segments: &[Segment]) -> Vec<(Vec<Vec2>, [f32; 4])> {
    // Segments that span heights, lowest end first, with +1 going up and −1 going down.
    let mut edges: Vec<(Vec2, Vec2, i32, [f32; 4])> = segments
        .iter()
        .filter(|s| s.a.y != s.b.y)
        .map(|s| {
            if s.a.y < s.b.y {
                (s.a, s.b, 1, s.color)
            } else {
                (s.b, s.a, -1, s.color)
            }
        })
        .collect();
    if edges.is_empty() {
        return Vec::new();
    }
    edges.sort_by(|x, y| x.0.y.total_cmp(&y.0.y));
    let mut heights: Vec<f32> = edges.iter().flat_map(|e| [e.0.y, e.1.y]).collect();
    heights.sort_by(f32::total_cmp);
    heights.dedup();
    let x_at = |e: &(Vec2, Vec2, i32, [f32; 4]), y: f32| {
        let t = (y - e.0.y) / (e.1.y - e.0.y);
        e.0.x + (e.1.x - e.0.x) * t
    };
    let mut out = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let mut next = 0;
    for pair in heights.windows(2) {
        let (y0, y1) = (pair[0], pair[1]);
        active.retain(|&i| edges[i].1.y > y0);
        while next < edges.len() && edges[next].0.y <= y0 {
            if edges[next].1.y > y0 {
                active.push(next);
            }
            next += 1;
        }
        if active.len() < 2 {
            continue;
        }
        let mid = 0.5 * (y0 + y1);
        let mut row: Vec<(f32, usize)> =
            active.iter().map(|&i| (x_at(&edges[i], mid), i)).collect();
        row.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut winding = 0;
        let mut open: Option<usize> = None;
        for (_, i) in row {
            let before = winding;
            winding += edges[i].2;
            if before == 0 && winding != 0 {
                open = Some(i);
            } else if before != 0 && winding == 0 {
                if let Some(l) = open.take() {
                    let (l0, l1) = (x_at(&edges[l], y0), x_at(&edges[l], y1));
                    let (r0, r1) = (x_at(&edges[i], y0), x_at(&edges[i], y1));
                    // Clockwise in (u, v), whose u × v is the plane's normal: each triangle
                    // faces the removed side, toward the viewer, as its vertex normal does. The
                    // shader lights a back face from behind, so a cap wound the other way
                    // reads dark.
                    let quad = [
                        Vec2::new(l0, y0),
                        Vec2::new(r1, y1),
                        Vec2::new(r0, y0),
                        Vec2::new(l0, y0),
                        Vec2::new(l1, y1),
                        Vec2::new(r1, y1),
                    ];
                    out.push((quad.to_vec(), edges[l].3));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(p: [f32; 3]) -> Vertex {
        Vertex {
            position: p,
            color: [0.5, 0.5, 0.5, 1.0],
            normal: [0.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0],
        }
    }

    /// A closed box, outward wound, from `lo` to `hi`.
    fn cube(lo: [f32; 3], hi: [f32; 3]) -> Vec<[Vertex; 3]> {
        let c = |i: usize| {
            [
                if i & 1 == 0 { lo[0] } else { hi[0] },
                if i & 2 == 0 { lo[1] } else { hi[1] },
                if i & 4 == 0 { lo[2] } else { hi[2] },
            ]
        };
        // Faces as quads, counter-clockwise seen from outside.
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        quads
            .iter()
            .flat_map(|q| {
                [
                    [v(c(q[0])), v(c(q[1])), v(c(q[2]))],
                    [v(c(q[0])), v(c(q[2])), v(c(q[3]))],
                ]
            })
            .collect()
    }

    fn area(cap: &[Vertex]) -> f32 {
        cap.chunks(3)
            .map(|t| {
                let p = t
                    .iter()
                    .map(|v| Vec3::from_array(v.position))
                    .collect::<Vec<_>>();
                0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
            })
            .sum()
    }

    #[test]
    fn a_cut_box_is_capped_with_its_cross_section() {
        let cap = section_cap(
            &cube([0.0; 3], [2.0, 3.0, 4.0]),
            (Vec3::Z, Vec3::new(0.0, 0.0, 1.5)),
        );
        assert!((area(&cap) - 6.0).abs() < 1e-4, "cap area {}", area(&cap));
        for vertex in &cap {
            assert!((vertex.position[2] - 1.5).abs() < 1e-5);
            assert_eq!(vertex.normal, [0.0, 0.0, -1.0]);
            assert_ne!(vertex.color, [0.5, 0.5, 0.5, 1.0], "the cut face is tinted");
        }
        // Each triangle's winding faces where its normal points (the removed side), so the
        // viewer sees its front face.
        for t in cap.chunks(3) {
            let p = t
                .iter()
                .map(|v| Vec3::from_array(v.position))
                .collect::<Vec<_>>();
            let facing = (p[1] - p[0]).cross(p[2] - p[0]);
            assert!(
                facing.dot(Vec3::from_array(t[0].normal)) > 0.0,
                "cap triangle faces {facing:?}"
            );
        }
    }

    #[test]
    fn a_hollow_ring_keeps_its_hole_and_overlapping_bodies_union() {
        // A square tube: an outer box with an inner box wound inward (the cavity).
        let mut tube = cube([0.0; 3], [4.0, 4.0, 4.0]);
        tube.extend(
            cube([1.0, 1.0, 0.0], [3.0, 3.0, 4.0])
                .into_iter()
                .map(|[a, b, c]| [a, c, b]),
        );
        let cap = section_cap(&tube, (Vec3::Z, Vec3::new(0.0, 0.0, 2.0)));
        assert!((area(&cap) - 12.0).abs() < 1e-4, "ring area {}", area(&cap));
        // Two overlapping boxes fill their union once.
        let mut pair = cube([0.0; 3], [2.0, 2.0, 2.0]);
        pair.extend(cube([1.0, 0.0, 0.0], [3.0, 2.0, 2.0]));
        let cap = section_cap(&pair, (Vec3::Z, Vec3::new(0.0, 0.0, 1.0)));
        assert!((area(&cap) - 6.0).abs() < 1e-4, "union area {}", area(&cap));
    }

    #[test]
    fn an_open_surface_has_no_cap() {
        // One face of the box alone: its cut is a dangling segment.
        let face = cube([0.0; 3], [2.0, 2.0, 2.0])[4..6].to_vec();
        assert!(section_cap(&face, (Vec3::Z, Vec3::new(0.0, 0.0, 1.0))).is_empty());
    }
}
