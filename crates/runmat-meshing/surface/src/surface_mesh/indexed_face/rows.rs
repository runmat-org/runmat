//! Borrow only the common numerical rows; neither face nor global mesh is copied.
use crate::*;
use runmat_geometry_core::PersistentEntityId;
#[derive(Clone, Copy)]
pub(super) struct MeshRows<'a> {
    pub schema_version: u16,
    pub source_face_id: &'a PersistentEntityId,
    pub nodes: &'a [ExactFaceMeshNode],
    pub triangles: &'a [ExactFaceMeshTriangle],
    pub boundary_segments: &'a [ExactFaceMeshBoundarySegment],
    pub joined_chart_cuts: &'a [ExactFaceMeshJoinedCut],
    pub maximum_chordal_deviation_m: runmat_types::physical::Length,
    pub maximum_normal_deviation_rad: f64,
}
impl<'a> MeshRows<'a> {
    pub fn face(mesh: &'a ExactFaceMesh) -> Self {
        Self {
            schema_version: mesh.schema_version,
            source_face_id: &mesh.source_face_id,
            nodes: &mesh.nodes,
            triangles: &mesh.triangles,
            boundary_segments: &mesh.boundary_segments,
            joined_chart_cuts: &mesh.joined_chart_cuts,
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(
                mesh.maximum_chordal_deviation_m.canonical_value(),
            ),
            maximum_normal_deviation_rad: mesh.maximum_normal_deviation_rad,
        }
    }
    pub fn surface(mesh: &'a ExactSurfaceMesh) -> Result<Self, super::ExactSurfaceMeshError> {
        Ok(Self {
            schema_version: crate::EXACT_FACE_MESH_SCHEMA_VERSION,
            source_face_id: mesh
                .face_ids
                .first()
                .ok_or_else(|| super::invalid("global surface has no canonical face owner"))?,
            nodes: &mesh.nodes,
            triangles: &mesh.triangles,
            boundary_segments: &mesh.boundary_segments,
            joined_chart_cuts: &[],
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(
                mesh.maximum_chordal_deviation_m.canonical_value(),
            ),
            maximum_normal_deviation_rad: mesh.maximum_normal_deviation_rad,
        })
    }
}
