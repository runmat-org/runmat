mod admission;
mod identity;
mod join;
mod lineage;
mod storage;
mod types;

pub(crate) use admission::valid_triangle_measures;
pub(crate) use admission::validate_exact_face_mesh_contract_with_cancellation;
pub(crate) use identity::{canonical_triangle, exact_face_triangle_id};
pub use join::{join_exact_face_charts, validate_exact_face_mesh};
pub use types::{
    ExactFaceJoinContext, ExactFaceJoinError, ExactFaceJoinErrorKind, ExactFaceJoinOptions,
    ExactFaceMesh, ExactFaceMeshBoundarySegment, ExactFaceMeshEdgeParameter,
    ExactFaceMeshJoinedCut, ExactFaceMeshNode, ExactFaceMeshNodeUse, ExactFaceMeshTriangle,
    EXACT_FACE_MESH_SCHEMA_VERSION,
};
