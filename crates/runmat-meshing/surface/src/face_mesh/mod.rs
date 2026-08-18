mod identity;
mod join;
mod lineage;
mod types;

pub use join::{join_exact_face_charts, validate_exact_face_mesh};
pub use types::{
    ExactFaceJoinContext, ExactFaceJoinError, ExactFaceJoinErrorKind, ExactFaceJoinOptions,
    ExactFaceMesh, ExactFaceMeshBoundarySegment, ExactFaceMeshEdgeParameter, ExactFaceMeshNode,
    ExactFaceMeshNodeUse, ExactFaceMeshTriangle,
};
