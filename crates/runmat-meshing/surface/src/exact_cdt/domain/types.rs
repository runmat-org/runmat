use runmat_geometry_core::PersistentEntityId;

use crate::{ExactFaceDelaunayTriangle, ExactFaceRecoveredSegment};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceTrimmedDelaunay {
    pub source_face_id: PersistentEntityId,
    pub triangles: Vec<ExactFaceDelaunayTriangle>,
    pub boundary_segments: Vec<ExactFaceRecoveredSegment>,
    pub removed_exterior_triangle_count: u64,
    pub removed_hole_triangle_count: u64,
}
