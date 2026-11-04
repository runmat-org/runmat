//! Display tessellation: a body as triangles per face, polylines per edge and points per
//! vertex, indexed the same way as the body's [`Topology`](crate::history::Topology).
//!
//! The design layer joins these meshes with the persistent names it holds per face, edge and
//! vertex index, so a pick on a rendered triangle maps back to a named entity. Kernels draw
//! with their own tessellator: the test kernel lays out its grid faces exactly, and the OCCT
//! backend meshes exact B-rep under the same tolerances.

use crate::op::{BodyHandle, KernelError};
use serde::{Deserialize, Serialize};

/// How finely curved geometry is tessellated (lengths in metres, angles in radians).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TessellationOptions {
    /// The largest distance between a curved surface and its triangles.
    pub chord_tolerance: f64,
    /// The largest angle between the normals at the ends of one segment of a curve.
    pub angle_tolerance: f64,
}

impl Default for TessellationOptions {
    fn default() -> Self {
        Self {
            chord_tolerance: 1.0e-4,
            angle_tolerance: 8.0_f64.to_radians(),
        }
    }
}

/// One face's triangles. Positions are not shared with other faces, so each face shades on
/// its own normals and a crease between faces stays sharp.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FaceMesh {
    pub positions: Vec<[f64; 3]>,
    /// Counter-clockwise seen from outside the solid.
    pub triangles: Vec<[u32; 3]>,
}

impl FaceMesh {
    /// The total area of the triangles.
    pub fn area(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.positions[i as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
            })
            .sum()
    }
}

/// A body's display mesh, in step with its topology: `faces[i]` is face `i`, `edges[i]` edge
/// `i`, `vertices[i]` vertex `i`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyMesh {
    pub faces: Vec<FaceMesh>,
    /// Each edge as a polyline along the drawn surface (empty when the kernel draws none).
    pub edges: Vec<Vec<[f64; 3]>>,
    pub vertices: Vec<[f64; 3]>,
}

impl BodyMesh {
    pub fn triangle_count(&self) -> usize {
        self.faces.iter().map(|f| f.triangles.len()).sum()
    }

    /// The axis-aligned bounds of every face position, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut any = false;
        for p in self.faces.iter().flat_map(|f| &f.positions) {
            any = true;
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        any.then_some((lo, hi))
    }
}

/// A kernel that can draw the bodies it made.
pub trait Tessellator {
    fn tessellate(
        &self,
        body: &BodyHandle,
        options: &TessellationOptions,
    ) -> Result<BodyMesh, KernelError>;
}
