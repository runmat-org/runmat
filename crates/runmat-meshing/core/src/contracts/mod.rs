pub const MESHING_CONTRACT_SCHEMA_VERSION: &str = "meshing-contracts/v1";

pub mod cad;
pub mod curve;
pub mod pipeline;
pub mod plc;
pub mod size;
pub mod solve;
pub mod stage;
pub mod surface;
pub mod tetrahedron;

pub use cad::{
    CadEdgeContract, CadEvaluatorCapabilities, CadFaceContract, CadModel, CadShellContract,
    CadVertexContract, CadVolumeContract,
};
pub use curve::{CurveMesh, CurveMeshElement, CurveMeshNode};
pub use pipeline::{
    validate_meshing_stage_order, MeshingStageArtifacts, MeshingStageContractError,
};
pub use plc::{
    PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, ProtectedBoundaryComplex,
};
pub use size::SizingFieldContract;
pub use solve::SolveReadinessReport;
pub use stage::{MeshingStage, StageEvidence, StageEvidenceStatus, TopologyEntityId};
pub use surface::{SurfaceMesh, SurfaceMeshNode, SurfaceMeshTriangle};
pub use tetrahedron::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh, TetrahedronMeshNode,
};
