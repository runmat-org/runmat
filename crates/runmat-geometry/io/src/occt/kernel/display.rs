//! Display tessellation on OCCT: [`Tessellator`] for [`OcctKernel`].
//!
//! BRepMesh meshes each body's exact B-rep (`exact_kernel_display.cc`). The deflections follow
//! the body's size: the chordal deflection is the caller's fraction of the body's diagonal (never
//! coarser than the caller's tolerance scaled to that size), and the angular deflection gives
//! the caller's turn per segment ([`brep_mesh_angle`]). Faces, edges and vertices come back in the body's summary order, so `faces[i]` is
//! topology face `i` and the design layer's names, picks and drawings line up with the mesh.

use super::{backend, OcctBody, OcctKernel};
use runmat_geometry_kernel::display::{BodyMesh, FaceMesh, TessellationOptions, Tessellator};
use runmat_geometry_kernel::history::Topology;
use runmat_geometry_kernel::op::{BodyHandle, KernelError};

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

/// The chordal deflection a large body may reach, as a fraction of its diagonal (a 3 m rib to
/// 0.6 mm; larger bodies stop at [`CHORD_MAX`]), under a pixel when the body fills the view (normals stay exact, so shading
/// is smooth at any density).
const CHORD_OF_SIZE_LARGE: f64 = 2.0e-4;

/// The largest chordal deflection any body gets, 1.5 mm. A thin wall's chords sag inside its
/// true surface by up to the deflection, so structure under a 3 mm skin (an airliner's frames
/// and stringers, a wing box's ribs under 4 mm skins) pokes through a skin meshed to 5 mm; at
/// 1.5 mm it stays under (with the edge depth offset, 0.9 mm at 60 m), and a 10 m doubly curved
/// fairing costs under twice the triangles it had at 2e-4 of its size. Normals stay exact.
const CHORD_MAX: f64 = 1.5e-3;

/// The chordal deflection for a body of diagonal `size`: the caller's fraction of its size, at
/// most ten times the caller's tolerance (or [`CHORD_OF_SIZE_LARGE`] of the size, when that is
/// larger: metre-scale bodies), never past [`CHORD_MAX`], at least [`CHORD_MIN`].
pub(crate) fn chord_for(size: f64, options: &TessellationOptions) -> f64 {
    (size * options.size_fraction)
        .min((options.chord_tolerance * 10.0).max(size * CHORD_OF_SIZE_LARGE))
        .min(options.chord_max.unwrap_or(CHORD_MAX))
        .max(CHORD_MIN)
}

/// The angular deflection BRepMesh takes for the contract's angle. BRepMesh bounds the angle
/// between each chord and the curve's tangent at its ends, which is half the turn across the
/// segment; the contract bounds the turn itself (the angle between the normals at a segment's
/// ends), so OCCT gets twice it.
pub(crate) fn brep_mesh_angle(options: &TessellationOptions) -> f64 {
    (options.angle_tolerance * 2.0).clamp(1.0_f64.to_radians(), 1.0)
}

/// The surface normals pointing out of the solid, the way the triangles wind; empty when they
/// are missing or don't match the nodes.
fn oriented(
    mut normals: Vec<[f64; 3]>,
    positions: &[[f64; 3]],
    triangles: &[[u32; 3]],
) -> Vec<[f64; 3]> {
    if normals.len() != positions.len() {
        return Vec::new();
    }
    let mut agree = 0.0;
    for t in triangles {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for i in t {
            let m = normals[*i as usize];
            agree += n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
        }
    }
    if agree < 0.0 {
        for m in &mut normals {
            *m = [-m[0], -m[1], -m[2]];
        }
    }
    normals
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
        let angle = brep_mesh_angle(options);
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
                .map(|(positions, triangles, normals)| {
                    let positions = triples(&positions);
                    let triangles: Vec<[u32; 3]> = triangles
                        .chunks_exact(3)
                        .map(|t| [t[0], t[1], t[2]])
                        .collect();
                    let normals = oriented(triples(&normals), &positions, &triangles);
                    FaceMesh {
                        positions,
                        triangles,
                        normals,
                    }
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
        // A 10 m fairing: 1.5 mm, the ceiling, so a skin's chords never sag through the structure under it.
        assert!((chord_for(10.0, &o) - CHORD_MAX).abs() < 1e-12);
        assert!((chord_for(40.0, &o) - CHORD_MAX).abs() < 1e-12);
        assert_eq!(chord_for(0.0, &o), CHORD_MIN);
        // Coarser callers (assembly display: 0.5 mm tolerance, 0.2 % of size) stop there too.
        let coarse = TessellationOptions { chord_tolerance: 5.0e-4, size_fraction: 2.0e-3, ..o };
        assert!((chord_for(8.0, &coarse) - CHORD_MAX).abs() < 1e-12);
        assert!((chord_for(0.3, &coarse) - 6.0e-4).abs() < 1e-12);
    }

    #[test]
    fn brep_mesh_gets_the_tangent_angle_for_the_contract_turn() {
        let o = TessellationOptions::default();
        assert!((brep_mesh_angle(&o) - 16.0_f64.to_radians()).abs() < 1e-12);
    }
}
