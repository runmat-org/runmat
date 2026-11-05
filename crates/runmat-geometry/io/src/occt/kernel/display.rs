//! Display tessellation on OCCT: [`Tessellator`] for [`OcctKernel`].
//!
//! BRepMesh meshes each body's exact B-rep (`exact_kernel_display.cc`). The deflections follow
//! the body's size: the chordal deflection is a fixed fraction of the body's diagonal (never
//! coarser than the caller's tolerance scaled to that size), and the angular deflection is the
//! caller's. Faces, edges and vertices come back in the body's summary order, so `faces[i]` is
//! topology face `i` and the design layer's names, picks and drawings line up with the mesh.

use super::{backend, OcctBody, OcctKernel};
use runmat_geometry_kernel::display::{BodyMesh, FaceMesh, TessellationOptions, Tessellator};
use runmat_geometry_kernel::history::Topology;
use runmat_geometry_kernel::op::{BodyHandle, KernelError};

/// The chordal deflection as a fraction of the body's diagonal.
const CHORD_OF_SIZE: f64 = 5.0e-4;
/// The finest chordal deflection, in metres (a micrometre).
const CHORD_MIN: f64 = 1.0e-6;

/// The diagonal of the faces' bounding boxes, in metres.
pub(crate) fn diagonal(topology: &Topology) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for f in &topology.faces {
        for a in 0..3 {
            lo[a] = lo[a].min(f.geom.bbox_min[a]);
            hi[a] = hi[a].max(f.geom.bbox_max[a]);
        }
    }
    let d: f64 = (0..3).map(|a| (hi[a] - lo[a]).powi(2)).sum::<f64>().sqrt();
    if d.is_finite() {
        d
    } else {
        0.0
    }
}

/// The chordal deflection for a body of diagonal `size`: a fixed fraction of its size, at
/// most ten times the caller's tolerance, at least [`CHORD_MIN`].
pub(crate) fn chord_for(size: f64, options: &TessellationOptions) -> f64 {
    (size * CHORD_OF_SIZE)
        .min(options.chord_tolerance * 10.0)
        .max(CHORD_MIN)
}

fn triples(flat: &[f64]) -> Vec<[f64; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

impl Tessellator for OcctKernel {
    fn tessellate(
        &self,
        body: &BodyHandle,
        options: &TessellationOptions,
    ) -> Result<BodyMesh, KernelError> {
        let b = body
            .downcast::<OcctBody>()
            .ok_or(KernelError::ForeignOperand { operand: 0 })?;
        let chord = chord_for(diagonal(&b.topology), options);
        let angle = options.angle_tolerance.clamp(1.0_f64.to_radians(), 0.5);
        let d = backend::tessellate(b.id, chord, angle)?;
        if d.faces.len() != b.topology.faces.len() || d.edges.len() != b.topology.edges.len() {
            return Err(KernelError::Failed {
                message: "OCCT display mesh is out of step with the body's topology".into(),
            });
        }
        Ok(BodyMesh {
            faces: d
                .faces
                .into_iter()
                .map(|(positions, triangles)| FaceMesh {
                    positions: triples(&positions),
                    triangles: triangles
                        .chunks_exact(3)
                        .map(|t| [t[0], t[1], t[2]])
                        .collect(),
                })
                .collect(),
            edges: d.edges.iter().map(|e| triples(e)).collect(),
            vertices: triples(&d.vertices),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chord_follows_the_body_size() {
        let o = TessellationOptions::default();
        // A 100 mm part: 0.05 mm; a 2 m skin: capped at ten times the caller's 0.1 mm.
        assert!((chord_for(0.1, &o) - 5.0e-5).abs() < 1e-12);
        assert!((chord_for(2.0, &o) - 1.0e-3).abs() < 1e-12);
        assert_eq!(chord_for(0.0, &o), CHORD_MIN);
    }
}
