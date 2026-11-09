//! Fake-kernel geometry analyses beyond mass properties and draft: section properties (exact
//! on grid planes, sampled on any other plane), wall thickness by inward rays, curvature per
//! face, zebra continuity per edge, deviation between two bodies and undercuts along a pull.

use super::features::{rects_of_face, rects_of_solid, RectSet};
use super::*;
use crate::evaluate::{AnalysisResult, EdgeContinuity, FaceValue, SectionProperties};

fn solid<'a>(bodies: &[&'a BodyHandle], k: usize) -> Result<&'a FakeSolid, KernelError> {
    bodies
        .get(k)
        .and_then(|h| h.downcast::<FakeSolid>())
        .ok_or(KernelError::ForeignOperand { operand: k as u32 })
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-300);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dotv(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The first hit beyond `eps` of a ray against rects.
fn first_hit(rects: &RectSet, o: [f64; 3], d: [f64; 3], eps: f64) -> Option<f64> {
    let mut best: Option<f64> = None;
    for (axis, off, r) in rects {
        if d[*axis].abs() < 1e-12 {
            continue;
        }
        let t = (off - o[*axis]) / d[*axis];
        if t <= eps {
            continue;
        }
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        if (0..3).all(|a| a == *axis || (p[a] >= r.lo[a] - 1e-12 && p[a] <= r.hi[a] + 1e-12))
            && best.map(|b| t < b).unwrap_or(true)
        {
            best = Some(t);
        }
    }
    best
}

/// Sample points on a face with their outward normals (one per facet rect).
fn face_samples(s: &FakeSolid, f: usize) -> Vec<([f64; 3], [f64; 3])> {
    s.faces[f]
        .rect_planes()
        .into_iter()
        .map(|(p, r)| {
            let c = [
                (r.lo[0] + r.hi[0]) / 2.0,
                (r.lo[1] + r.hi[1]) / 2.0,
                (r.lo[2] + r.hi[2]) / 2.0,
            ];
            let mut n = [0.0; 3];
            n[p.axis] = p.dir as f64;
            (c, n)
        })
        .collect()
}

fn scale_of(s: &FakeSolid) -> f64 {
    s.bounds()
        .map(|(lo, hi)| (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max))
        .unwrap_or(1.0)
}

pub(super) fn section(
    bodies: &[&BodyHandle],
    origin: [f64; 3],
    normal: [f64; 3],
) -> Result<AnalysisResult, KernelError> {
    let n = norm(normal);
    let seed = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = norm([
        seed[0] - n[0] * dotv(seed, n),
        seed[1] - n[1] * dotv(seed, n),
        seed[2] - n[2] * dotv(seed, n),
    ]);
    let e2 = [
        n[1] * e1[2] - n[2] * e1[1],
        n[2] * e1[0] - n[0] * e1[2],
        n[0] * e1[1] - n[1] * e1[0],
    ];
    // (area, Σu, Σv, Σu², Σv², Σuv) in the section frame (origin, e1, e2).
    let mut m = [0.0f64; 6];
    let mut add = |a: f64, u: f64, v: f64, ju: f64, jv: f64| {
        m[0] += a;
        m[1] += a * u;
        m[2] += a * v;
        m[3] += a * u * u + ju;
        m[4] += a * v * v + jv;
        m[5] += a * u * v;
    };
    for k in 0..bodies.len() {
        let s = solid(bodies, k)?;
        let d = s.dims();
        match axis_of(n) {
            Some((ax, _)) => {
                // Exact: the slab of cells the plane passes through.
                let g = &s.grid[ax];
                let c = origin[ax];
                if g.len() < 2 || c <= g[0] || c >= g[g.len() - 1] {
                    continue;
                }
                let layer = g.partition_point(|x| *x <= c) - 1;
                for z in 0..d[2] {
                    for y in 0..d[1] {
                        for x in 0..d[0] {
                            let cell = [x, y, z];
                            if cell[ax] != layer || !s.occ[x + d[0] * (y + d[1] * z)] {
                                continue;
                            }
                            let lo = [s.grid[0][x], s.grid[1][y], s.grid[2][z]];
                            let hi = [s.grid[0][x + 1], s.grid[1][y + 1], s.grid[2][z + 1]];
                            let mid = [
                                (lo[0] + hi[0]) / 2.0 - origin[0],
                                (lo[1] + hi[1]) / 2.0 - origin[1],
                                (lo[2] + hi[2]) / 2.0 - origin[2],
                            ];
                            let w = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
                            let wu: f64 = (0..3).map(|a| w[a] * e1[a].abs()).sum();
                            let wv: f64 = (0..3).map(|a| w[a] * e2[a].abs()).sum();
                            let a = wu * wv;
                            add(
                                a,
                                dotv(mid, e1),
                                dotv(mid, e2),
                                a * wu * wu / 12.0,
                                a * wv * wv / 12.0,
                            );
                        }
                    }
                }
            }
            None => {
                // Any other plane: sampled on a lattice over the body's extent.
                let Some((lo, hi)) = s.bounds() else { continue };
                let half = (0..3).map(|a| (hi[a] - lo[a]).abs()).sum::<f64>();
                let steps = 240usize;
                let h = 2.0 * half / steps as f64;
                for i in 0..steps {
                    for j in 0..steps {
                        let u = -half + (i as f64 + 0.5) * h;
                        let v = -half + (j as f64 + 0.5) * h;
                        let p = [
                            origin[0] + u * e1[0] + v * e2[0],
                            origin[1] + u * e1[1] + v * e2[1],
                            origin[2] + u * e1[2] + v * e2[2],
                        ];
                        if s.contains(p) {
                            add(h * h, u, v, h.powi(4) / 12.0, h.powi(4) / 12.0);
                        }
                    }
                }
            }
        }
    }
    if m[0] <= 0.0 {
        return Err(KernelError::Failed {
            message: "the plane does not cut the body".into(),
        });
    }
    let (cu, cv) = (m[1] / m[0], m[2] / m[0]);
    Ok(AnalysisResult::SectionProperties(SectionProperties {
        area: m[0],
        centroid: [
            origin[0] + cu * e1[0] + cv * e2[0],
            origin[1] + cu * e1[1] + cv * e2[1],
            origin[2] + cu * e1[2] + cv * e2[2],
        ],
        // About the section's own axes through its centroid: Ixx = ∫v², Iyy = ∫u².
        ixx: m[4] - m[0] * cv * cv,
        iyy: m[3] - m[0] * cu * cu,
        ixy: m[5] - m[0] * cu * cv,
    }))
}

/// Wall thickness per face: rays from each facet inward to the next face of the body.
fn thickness_values(s: &FakeSolid) -> Vec<FaceValue> {
    let rects = rects_of_solid(s);
    let eps = scale_of(s) * 1e-9;
    let mut out = Vec::new();
    for f in 0..s.faces.len() {
        let mut lo = f64::INFINITY;
        let mut hi: f64 = 0.0;
        for (c, n) in face_samples(s, f) {
            if let Some(t) = first_hit(&rects, c, [-n[0], -n[1], -n[2]], eps) {
                lo = lo.min(t);
                hi = hi.max(t);
            }
        }
        if lo.is_finite() {
            out.push(FaceValue {
                face: f as u32,
                min: lo,
                max: hi,
            });
        }
    }
    out
}

pub(super) fn thickness(
    bodies: &[&BodyHandle],
    min: f64,
    max: Option<f64>,
) -> Result<AnalysisResult, KernelError> {
    let s = solid(bodies, 0)?;
    let faces = thickness_values(s)
        .into_iter()
        .filter(|v| v.min < min || max.map(|m| v.max > m).unwrap_or(false))
        .collect();
    Ok(AnalysisResult::Thickness { faces })
}

pub(super) fn curvature(bodies: &[&BodyHandle]) -> Result<AnalysisResult, KernelError> {
    let s = solid(bodies, 0)?;
    let mut faces = Vec::new();
    for (f, t) in s.topology.faces.iter().enumerate() {
        let (lo, hi) = match t.geom.surface {
            SurfaceKind::Plane { .. } => (0.0, 0.0),
            SurfaceKind::Cylinder { radius, .. } => (0.0, 1.0 / radius.max(1e-300)),
            SurfaceKind::Other => {
                // The turn of the facet normals over the face's extent.
                let samples = face_samples(s, f);
                let mut k: f64 = 0.0;
                for (i, a) in samples.iter().enumerate() {
                    for b in &samples[i + 1..] {
                        let turn = dotv(a.1, b.1).clamp(-1.0, 1.0).acos();
                        let d = ((a.0[0] - b.0[0]).powi(2)
                            + (a.0[1] - b.0[1]).powi(2)
                            + (a.0[2] - b.0[2]).powi(2))
                        .sqrt();
                        if turn > 1e-9 && d > 0.0 {
                            k = k.max(turn / d);
                        }
                    }
                }
                (0.0, k)
            }
        };
        faces.push(FaceValue {
            face: f as u32,
            min: lo,
            max: hi,
        });
    }
    Ok(AnalysisResult::Curvature { faces })
}

pub(super) fn zebra(bodies: &[&BodyHandle]) -> Result<AnalysisResult, KernelError> {
    let s = solid(bodies, 0)?;
    let blend = |f: u32| !s.faces[f as usize].is_grid();
    let g2 = |f: u32| {
        s.decorations
            .iter()
            .any(|d| d.g2 && d.pieces.iter().any(|p| p.2 == f))
    };
    let edges = s
        .topology
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let [a, b] = e.faces;
            // A blend's end arc meets the face across its end at a crease (G0).
            let end_arc = [(a, b), (b, a)].iter().any(|&(f, other)| {
                blend(f)
                    && s.decorations.iter().any(|d| {
                        d.pieces.iter().any(|p| p.2 == f)
                            && s.faces[other as usize]
                                .plane
                                .is_some_and(|pl| pl.axis == d.along)
                    })
            });
            let c = if end_arc {
                EdgeContinuity::G0
            } else if blend(a) || blend(b) {
                if g2(a) || g2(b) {
                    EdgeContinuity::G2
                } else {
                    EdgeContinuity::G1
                }
            } else {
                EdgeContinuity::G0
            };
            (i as u32, c)
        })
        .collect();
    Ok(AnalysisResult::Zebra { edges })
}

pub(super) fn deviation(bodies: &[&BodyHandle]) -> Result<AnalysisResult, KernelError> {
    let (a, b) = (solid(bodies, 0)?, solid(bodies, 1)?);
    let target: Vec<Rect> = b
        .faces
        .iter()
        .flat_map(|f| f.rect_planes().into_iter().map(|(_, r)| r))
        .collect();
    let mut max: f64 = 0.0;
    let mut sum = 0.0;
    let mut n = 0usize;
    for f in 0..a.faces.len() {
        for (c, _) in face_samples(a, f) {
            let d = target
                .iter()
                .map(|r| rect_distance(c, r))
                .fold(f64::INFINITY, f64::min);
            if d.is_finite() {
                max = max.max(d);
                sum += d * d;
                n += 1;
            }
        }
    }
    if n == 0 {
        return Err(KernelError::Failed {
            message: "nothing to compare".into(),
        });
    }
    Ok(AnalysisResult::Deviation {
        max,
        rms: (sum / n as f64).sqrt(),
    })
}

pub(super) fn undercut(
    bodies: &[&BodyHandle],
    pull: [f64; 3],
) -> Result<AnalysisResult, KernelError> {
    let s = solid(bodies, 0)?;
    let p = norm(pull);
    let eps = scale_of(s) * 1e-9;
    let mut faces = Vec::new();
    for f in 0..s.faces.len() {
        let others: RectSet = (0..s.faces.len() as u32)
            .filter(|g| *g as usize != f)
            .flat_map(|g| rects_of_face(s, g))
            .collect();
        // Each facet releases along the pull, or against it when it faces that way; a face is
        // an undercut where the body blocks that path.
        let blocked = face_samples(s, f).into_iter().any(|(c, n)| {
            let along = dotv(n, p);
            if along.abs() < 1e-9 {
                return false;
            }
            let d = if along > 0.0 {
                p
            } else {
                [-p[0], -p[1], -p[2]]
            };
            first_hit(&others, c, d, eps).is_some()
        });
        if blocked {
            faces.push(f as u32);
        }
    }
    Ok(AnalysisResult::Undercut { faces })
}
