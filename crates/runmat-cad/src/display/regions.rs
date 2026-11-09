//! A sketch's closed regions, for picking a region as a profile in the model view: every closed
//! loop of the sketch's profile curves, nested by containment. A loop inside an even number of
//! others bounds a region; the loops directly inside it are its holes (the rule the kernels
//! use to make faces from a profile). Each region carries its loops' curve ids (outer first,
//! as `ProfileRef.loops` takes them) and a triangulation in the sketch plane.

use crate::model::sketch::{Sketch, SolvedSketch};
use crate::regen::engine::profile_loops;
use crate::regen::kernel::{ProfileSegment, SegmentKind};
use serde::{Deserialize, Serialize};

/// A closed region of a sketch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchRegionDisplay {
    /// The region's loops by curve id: the outer loop, then its holes.
    pub loops: Vec<Vec<String>>,
    /// Boundary points in the sketch plane (metres), the triangles' corners.
    pub points: Vec<[f32; 2]>,
    /// Triangles covering the region, as indices into `points`.
    pub triangles: Vec<[u32; 3]>,
}

fn polygon(l: &[ProfileSegment]) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = Vec::new();
    for s in l {
        let n = match s.kind {
            SegmentKind::Line => 1,
            _ if s.is_closed_curve() => 64,
            _ => 24,
        };
        let pts = s.polyline(n);
        for p in pts {
            if out.last().is_none_or(|q: &[f64; 2]| (q[0] - p[0]).abs() + (q[1] - p[1]).abs() > 1e-12) {
                out.push(p);
            }
        }
    }
    if out.len() > 1 {
        let (a, b) = (out[0], out[out.len() - 1]);
        if (a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 1e-12 {
            out.pop();
        }
    }
    out
}

fn signed_area(p: &[[f64; 2]]) -> f64 {
    (0..p.len())
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % p.len()]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        / 2.0
}

fn inside(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut c = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + n - 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            c = !c;
        }
    }
    c
}

/// The closed regions of `sketch`; none when its profile curves don't all close.
pub fn sketch_regions(sketch: &Sketch, solved: &SolvedSketch) -> Vec<SketchRegionDisplay> {
    let Ok(loops) = profile_loops(sketch, solved) else {
        return Vec::new();
    };
    let polys: Vec<Vec<[f64; 2]>> = loops.iter().map(|l| polygon(l)).collect();
    if polys.iter().any(|p| p.len() < 3) {
        return Vec::new();
    }
    // A point well inside each loop's own area: the middle of its first edge, nudged inwards.
    let probe = |p: &[[f64; 2]]| -> [f64; 2] {
        let (a, b) = (p[0], p[1]);
        let m = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
        let s = if signed_area(p) > 0.0 { 1.0 } else { -1.0 };
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = (dx * dx + dy * dy).sqrt().max(1e-300);
        let e = 1e-7_f64.max(l * 1e-4);
        [m[0] - s * dy / l * e, m[1] + s * dx / l * e]
    };
    let contains = |outer: usize, inner: usize| -> bool {
        outer != inner
            && signed_area(&polys[outer]).abs() > signed_area(&polys[inner]).abs()
            && inside(polys[inner][0], &polys[outer])
            && inside(probe(&polys[inner]), &polys[outer])
    };
    let depth: Vec<usize> = (0..polys.len())
        .map(|i| (0..polys.len()).filter(|o| contains(*o, i)).count())
        .collect();
    let mut out = Vec::new();
    for o in (0..polys.len()).filter(|i| depth[*i] % 2 == 0) {
        let holes: Vec<usize> = (0..polys.len())
            .filter(|h| depth[*h] == depth[o] + 1 && contains(o, *h))
            .collect();
        let mut points: Vec<[f64; 2]> = Vec::new();
        let mut index_loops: Vec<Vec<usize>> = Vec::new();
        for (k, l) in std::iter::once(o).chain(holes.iter().copied()).enumerate() {
            let mut p = polys[l].clone();
            // Outer counter-clockwise, holes clockwise.
            if (signed_area(&p) > 0.0) != (k == 0) {
                p.reverse();
            }
            let base = points.len();
            index_loops.push((base..base + p.len()).collect());
            points.extend(p);
        }
        let Some(triangles) = crate::regen::fake::triangulate_polygon(&points, &index_loops) else {
            continue;
        };
        out.push(SketchRegionDisplay {
            loops: std::iter::once(o)
                .chain(holes)
                .map(|l| loops[l].iter().map(|s| s.curve.0.clone()).collect())
                .collect(),
            points: points.iter().map(|p| [p[0] as f32, p[1] as f32]).collect(),
            triangles: triangles.iter().map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect(),
        });
    }
    out
}
