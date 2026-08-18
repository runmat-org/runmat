mod admission;
mod identity;
mod join;
mod lineage;
mod types;

pub(crate) use admission::validate_exact_face_mesh_contract;
pub use join::{join_exact_face_charts, validate_exact_face_mesh};
pub use types::{
    ExactFaceJoinContext, ExactFaceJoinError, ExactFaceJoinErrorKind, ExactFaceJoinOptions,
    ExactFaceMesh, ExactFaceMeshBoundarySegment, ExactFaceMeshEdgeParameter,
    ExactFaceMeshJoinedCut, ExactFaceMeshNode, ExactFaceMeshNodeUse, ExactFaceMeshTriangle,
    EXACT_FACE_MESH_SCHEMA_VERSION,
};
