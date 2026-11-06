//! Undercut analysis (CAD-36): which faces a two-part mould can't release along the pull.
//!
//! Each facet of the part's tessellation releases the way it faces: toward the pull (with the
//! cavity) when its normal leans toward it, away from it (with the core) when it leans away,
//! and either way when it is parallel to the pull. A facet is undercut when the part itself
//! stands in front of it along every way it can release: a ray from the facet along that
//! direction hits another facet. A face is undercut when more than half a percent of its area
//! is; such faces need a side action, a lifter or a change to the part.
//!
//! The rays test only the facets whose outline, seen along the pull, covers the ray's cell in a
//! grid over the part, so the test stays fast on large tessellations.

use runmat_cad::display::{FaceDisplay, PartDisplay};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UndercutBand {
    /// Releases toward the pull, unobstructed.
    Cavity,
    /// Releases away from the pull, unobstructed.
    Core,
    /// The part blocks part of the face along every way it can release.
    Undercut,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FaceUndercut {
    pub face: String,
    pub label: String,
    /// The share of the face's area that is undercut, 0 to 1.
    pub share: f64,
    pub band: UndercutBand,
}

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A ray from `o` along `d` hits the triangle `t` beyond `min` (Möller–Trumbore).
fn hits(o: V3, d: V3, t: &[V3; 3], min: f64) -> bool {
    let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-18 {
        return false;
    }
    let inv = 1.0 / det;
    let s = sub(o, t[0]);
    let u = dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    dot(e2, q) * inv > min
}

/// Triangles binned by their outline seen along the pull.
struct Grid {
    tris: Vec<[V3; 3]>,
    cells: Vec<Vec<usize>>,
    n: usize,
    lo: [f64; 2],
    step: [f64; 2],
    axes: [V3; 2],
}

impl Grid {
    fn new(tris: Vec<[V3; 3]>, pull: V3) -> Grid {
        let helper = if pull[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = cross(pull, helper);
        let lu = dot(u, u).sqrt();
        let u = [u[0] / lu, u[1] / lu, u[2] / lu];
        let v = cross(pull, u);
        let flat = |p: V3| [dot(p, u), dot(p, v)];
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for t in &tris {
            for p in t {
                let f = flat(*p);
                for k in 0..2 {
                    lo[k] = lo[k].min(f[k]);
                    hi[k] = hi[k].max(f[k]);
                }
            }
        }
        let n = ((tris.len() as f64).sqrt().ceil() as usize).clamp(1, 256);
        let step = [0, 1].map(|k| ((hi[k] - lo[k]) / n as f64).max(1e-12));
        let mut cells = vec![Vec::new(); n * n];
        let cell = |x: f64, k: usize| (((x - lo[k]) / step[k]).floor().max(0.0) as usize).min(n - 1);
        for (i, t) in tris.iter().enumerate() {
            let f = t.map(flat);
            let (a0, a1) = (cell(f.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min), 0), cell(f.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max), 0));
            let (b0, b1) = (cell(f.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min), 1), cell(f.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max), 1));
            for a in a0..=a1 {
                for b in b0..=b1 {
                    cells[a * n + b].push(i);
                }
            }
        }
        Grid { tris, cells, n, lo, step, axes: [u, v] }
    }

    /// Whether a ray from `o` along `d` (±pull) hits a triangle other than `own`.
    fn blocked(&self, o: V3, d: V3, own: usize, min: f64) -> bool {
        let f = [dot(o, self.axes[0]), dot(o, self.axes[1])];
        let cell = |k: usize| (((f[k] - self.lo[k]) / self.step[k]).floor().max(0.0) as usize).min(self.n - 1);
        self.cells[cell(0) * self.n + cell(1)].iter().any(|&i| i != own && hits(o, d, &self.tris[i], min))
    }
}

fn point(face: &FaceDisplay, i: u32) -> V3 {
    let k = i as usize * 3;
    [f64::from(face.positions[k]), f64::from(face.positions[k + 1]), f64::from(face.positions[k + 2])]
}

/// Every face's undercut against `pull` (unit). `label` names a face for people.
pub fn undercut_faces(display: &PartDisplay, pull: V3, label: &dyn Fn(u32, &FaceDisplay) -> String) -> Vec<FaceUndercut> {
    let faces: Vec<(u32, &FaceDisplay)> = display.bodies.iter().flat_map(|b| b.faces.iter().map(move |f| (b.index, f))).collect();
    let mut tris = Vec::new();
    let mut owner = Vec::new();
    for (at, (_, face)) in faces.iter().enumerate() {
        for t in face.triangles.chunks_exact(3).filter(|t| t.iter().all(|&i| (i as usize) * 3 + 2 < face.positions.len())) {
            tris.push([point(face, t[0]), point(face, t[1]), point(face, t[2])]);
            owner.push(at);
        }
    }
    let size = display.bounds.as_ref().map(|b| (0..3).map(|k| f64::from(b.max[k] - b.min[k])).fold(0.0, f64::max)).unwrap_or(1.0);
    // Rays start a hair off the facet and ignore hits closer than that, so neighbours sharing
    // an edge don't block each other.
    let lift = size * 1e-6;
    let back = [-pull[0], -pull[1], -pull[2]];
    let grid = Grid::new(tris, pull);
    let mut area = vec![0.0; faces.len()];
    let mut blocked = vec![0.0; faces.len()];
    for (i, t) in grid.tris.iter().enumerate() {
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let a = dot(n, n).sqrt();
        if a < 1e-24 {
            continue;
        }
        let along = dot(n, pull) / a;
        let c = [0, 1, 2].map(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
        let off = |d: V3| [c[0] + d[0] * lift, c[1] + d[1] * lift, c[2] + d[2] * lift];
        let ways: &[V3] = if along > 1e-3 { &[pull] } else if along < -1e-3 { std::slice::from_ref(&back) } else { &[pull, back] };
        let ways: Vec<V3> = ways.to_vec();
        let stuck = ways.iter().all(|d| grid.blocked(off(*d), *d, i, lift));
        area[owner[i]] += a;
        if stuck {
            blocked[owner[i]] += a;
        }
    }
    faces
        .iter()
        .enumerate()
        .filter(|(k, _)| area[*k] > 0.0)
        .map(|(k, (body, face))| {
            let share = blocked[k] / area[k];
            let mean: f64 = face
                .triangles
                .chunks_exact(3)
                .filter(|t| t.iter().all(|&i| (i as usize) * 3 + 2 < face.positions.len()))
                .map(|t| dot(cross(sub(point(face, t[1]), point(face, t[0])), sub(point(face, t[2]), point(face, t[0]))), pull))
                .sum();
            FaceUndercut {
                face: face.reference.clone().unwrap_or_else(|| format!("face {}", face.index)),
                label: label(*body, face),
                share,
                band: if share > 0.005 {
                    UndercutBand::Undercut
                } else if mean >= 0.0 {
                    UndercutBand::Cavity
                } else {
                    UndercutBand::Core
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_cad::display::{BodyDisplay, Bounds};

    /// An axis-aligned box's six faces, outward.
    fn boxed(lo: V3, hi: V3, first: u32) -> Vec<FaceDisplay> {
        let c = |x: usize, y: usize, z: usize| [if x == 1 { hi[0] } else { lo[0] }, if y == 1 { hi[1] } else { lo[1] }, if z == 1 { hi[2] } else { lo[2] }];
        // Each quad wound counter-clockwise seen from outside.
        let quads = [
            [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)],
            [c(0, 0, 0), c(0, 1, 0), c(1, 1, 0), c(1, 0, 0)],
            [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)],
            [c(0, 1, 0), c(0, 1, 1), c(1, 1, 1), c(1, 1, 0)],
            [c(0, 0, 0), c(0, 0, 1), c(0, 1, 1), c(0, 1, 0)],
            [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)],
        ];
        quads
            .iter()
            .enumerate()
            .map(|(k, q)| FaceDisplay {
                index: first + k as u32,
                positions: q.iter().flatten().map(|v| *v as f32).collect(),
                triangles: vec![0, 1, 2, 0, 2, 3],
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn a_ledge_under_an_overhang_is_undercut_and_open_faces_release() {
        // A base block with a second block hovering over part of it: the base's top under
        // the overhang can't release up, and the overhang's underside can't release down.
        let mut faces = boxed([0.0, 0.0, 0.0], [10.0, 10.0, 2.0], 0);
        faces.extend(boxed([5.0, 0.0, 5.0], [10.0, 10.0, 7.0], 6));
        let display = PartDisplay {
            bodies: vec![BodyDisplay { faces, ..Default::default() }],
            bounds: Some(Bounds { min: [0.0; 3], max: [10.0, 10.0, 7.0] }),
            ..Default::default()
        };
        let out = undercut_faces(&display, [0.0, 0.0, 1.0], &|_, f| format!("Face {}", f.index + 1));
        let band = |i: u32| out.iter().find(|f| f.face == format!("face {i}")).unwrap();
        // The base's top: half under the overhang.
        assert_eq!(band(0).band, UndercutBand::Undercut);
        assert!((band(0).share - 0.5).abs() < 0.01, "{}", band(0).share);
        // The overhang's underside blocks against the base below; its top is clear.
        assert_eq!(band(7).band, UndercutBand::Undercut);
        assert_eq!(band(6).band, UndercutBand::Cavity);
        // The base's bottom releases with the core.
        assert_eq!(band(1).band, UndercutBand::Core);
    }
}
