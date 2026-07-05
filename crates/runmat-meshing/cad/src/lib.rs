//! CAD topology, evaluator, region-map, and healing stages.

pub const CRATE_PURPOSE: &str =
    "CAD topology, evaluators, region mapping, and bounded healing for meshing";

pub use runmat_geometry_core::{CadCurveEvaluationSample, CadCurveEvaluationSampleSource};

mod math;

pub mod eval;
pub mod heal;
pub mod region_map;
pub mod topology;

pub use eval::{
    build_cad_evaluation_model, build_cad_evaluation_model_with_provider, project_to_face,
    summarize_cad_evaluation, CadEvaluationError, CadEvaluationModel, CadEvaluationReport,
    CadEvaluationSource, CadFaceEvaluationFrame, CadFaceEvaluationRequest,
    CadFaceEvaluatorProvider, CadFaceProjection, NoopCadFaceEvaluatorProvider,
};
pub use topology::{
    build_cad_topology, extract_source_topology, validate_cad_topology_model, CadEdge, CadEntityId,
    CadEntityKind, CadFace, CadLoop, CadShell, CadTopologyError, CadTopologyModel,
    CadTopologyReport, CadTopologySource, CadVertex, CadVolume, SourceTopologyEdge,
    SourceTopologyError, SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex,
};
