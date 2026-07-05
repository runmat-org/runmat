pub const MESHING_CONTRACT_SCHEMA_VERSION: &str = "meshing-contracts/v1";

pub mod artifact;
pub mod backend;
pub mod boundary;
pub mod cad;
pub mod curve;
pub mod material;
pub mod options;
pub mod pipeline;
pub mod plc;
pub mod provenance;
pub mod size;
pub mod solve;
pub mod stage;
pub mod surface;
pub mod tetrahedron;
pub mod topology;

pub use artifact::{
    analysis_mesh_field_topology, AnalysisBoundaryEdge, AnalysisBoundaryFace,
    AnalysisFieldTopologyDescriptor, AnalysisFieldTopologyLocation, AnalysisMeshArtifact,
    AnalysisMeshNode, AnalysisVolumeElement, MeshBackendSummary,
    ANALYSIS_MESH_BOUNDARY_EDGE_TOPOLOGY_ID, ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID,
    ANALYSIS_MESH_FIELD_TOPOLOGY_ID, TETRAHEDRON4_FIELD_ELEMENT_KIND, TRI3_FIELD_ELEMENT_KIND,
};
pub use backend::{select_volume_backend, MeshBackendKind, MeshBackendSelection};
pub use boundary::{BoundaryMeshInput, BoundaryMeshTriangle};
pub use cad::{
    CadEdgeContract, CadEvaluatorCapabilities, CadFaceContract, CadModel, CadShellContract,
    CadVertexContract, CadVolumeContract,
};
pub use curve::{CurveMesh, CurveMeshElement, CurveMeshNode};
pub use material::{DEFAULT_MATERIAL_REGION_ID, UNCLASSIFIED_MATERIAL_REGION_ID};
pub use options::{
    AdaptiveMeshingOptions, MeshElementOrder, MeshKindRequest, MeshProfile, MeshRefinementOptions,
    MeshTargetSize, MeshValidationPolicyOptions, RefinementConvergenceOptions,
    RefinementFocusLevel, RefinementFocusOptions, RefinementIndicatorMode,
    RefinementIndicatorOverrides, RefinementStrategy, VolumeMeshingOptions,
};
pub use pipeline::{
    validate_meshing_stage_order, MeshingStageArtifacts, MeshingStageContractError,
};
pub use plc::{
    PlcFacet, PlcNode, PlcProtectedEdge, PlcValidationSummary, ProtectedBoundaryComplex,
};
pub use provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind};
pub use size::SizingFieldContract;
pub use solve::SolveReadinessReport;
pub use stage::{MeshingStage, StageEvidence, StageEvidenceStatus, TopologyEntityId};
pub use surface::{SurfaceMesh, SurfaceMeshNode, SurfaceMeshTriangle};
pub use tetrahedron::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh, TetrahedronMeshNode,
};
pub use topology::{BoundaryElementKind, VolumeElementKind};
