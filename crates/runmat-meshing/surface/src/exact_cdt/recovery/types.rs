use runmat_geometry_core::PersistentEntityId;

use crate::ExactFaceDelaunayTriangle;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceConstrainedDelaunay {
    pub source_face_id: PersistentEntityId,
    pub triangles: Vec<ExactFaceDelaunayTriangle>,
    pub protected_segments: Vec<ExactFaceRecoveredSegment>,
    pub recovery_edge_flip_count: u64,
    pub cavity_retriangulation_count: u64,
    pub delaunay_restoration_flip_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceRecoveredSegment {
    pub pslg_segment_index: u32,
    pub source_coedge_id: PersistentEntityId,
    pub source_edge_id: PersistentEntityId,
    /// Oriented exactly as the source PSLG segment.
    pub vertex_indices: [u32; 2],
}
