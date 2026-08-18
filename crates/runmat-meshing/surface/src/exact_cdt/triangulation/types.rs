use runmat_geometry_core::PersistentEntityId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceDelaunay {
    pub source_face_id: PersistentEntityId,
    pub triangles: Vec<ExactFaceDelaunayTriangle>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExactFaceDelaunayTriangle {
    /// Counterclockwise vertex indices rotated to begin at the smallest index.
    pub vertex_indices: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactFaceDelaunayOptions {
    pub maximum_triangles: usize,
    pub maximum_predicate_evaluations: u64,
    pub cancellation_check_interval: u64,
}

impl Default for ExactFaceDelaunayOptions {
    fn default() -> Self {
        Self {
            maximum_triangles: 10_000_000,
            maximum_predicate_evaluations: 100_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceDelaunayError {
    pub kind: ExactFaceDelaunayErrorKind,
    pub source_face_id: PersistentEntityId,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceDelaunayErrorKind {
    InvalidPslg,
    InvalidOptions,
    InvalidTopology,
    ResourceLimit,
    Cancelled,
}

impl ExactFaceDelaunayError {
    pub(super) fn new(
        kind: ExactFaceDelaunayErrorKind,
        source_face_id: &PersistentEntityId,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            source_face_id: source_face_id.clone(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for ExactFaceDelaunayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face Delaunay {:?} for {:?}: {}",
            self.kind, self.source_face_id, self.reason
        )
    }
}

impl std::error::Error for ExactFaceDelaunayError {}
