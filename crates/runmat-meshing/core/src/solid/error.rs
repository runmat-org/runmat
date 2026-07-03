use crate::{
    cad::{eval::CadEvaluationError, topology::CadTopologyError},
    curve::CurveDiscretizationError,
    plc::build::PlcBuildError,
    source_topology::SourceTopologyError,
    surface::{recovery::SurfaceRecoveryError, validate::SurfaceValidationError},
    tetrahedron::{generate::TetrahedronGenerationError, recover::TetrahedronRecoveryError},
    validation::AnalysisMeshValidationError,
};

#[derive(Debug, Clone, PartialEq)]
pub enum SolidMeshError {
    Topology(SourceTopologyError),
    CadTopology(CadTopologyError),
    CadEvaluation(CadEvaluationError),
    Curve(CurveDiscretizationError),
    Surface(crate::surface::SurfaceDiscretizationError),
    SurfaceValidation(SurfaceValidationError),
    SurfaceRecovery(SurfaceRecoveryError),
    ProtectedBoundaryComplex(PlcBuildError),
    TetrahedronGeneration(TetrahedronGenerationError),
    TetrahedronRecovery(TetrahedronRecoveryError),
    Validation(AnalysisMeshValidationError),
    MissingTetrahedronNode { node_id: String },
}

impl std::fmt::Display for SolidMeshError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(err) => write!(formatter, "source topology extraction failed: {err}"),
            Self::CadTopology(err) => write!(formatter, "CAD topology normalization failed: {err}"),
            Self::CadEvaluation(err) => write!(formatter, "CAD evaluation setup failed: {err}"),
            Self::Curve(err) => write!(formatter, "curve discretization failed: {err}"),
            Self::Surface(err) => write!(formatter, "surface discretization failed: {err}"),
            Self::SurfaceValidation(err) => write!(formatter, "surface validation failed: {err}"),
            Self::SurfaceRecovery(err) => write!(formatter, "surface recovery failed: {err}"),
            Self::ProtectedBoundaryComplex(err) => {
                write!(
                    formatter,
                    "protected boundary complex validation failed: {err}"
                )
            }
            Self::TetrahedronGeneration(err) => {
                write!(formatter, "initial Tetrahedron generation failed: {err}")
            }
            Self::TetrahedronRecovery(err) => {
                write!(formatter, "Tetrahedron recovery failed: {err}")
            }
            Self::Validation(err) => {
                write!(formatter, "solid mesh validation failed: {err:?}")
            }
            Self::MissingTetrahedronNode { node_id } => {
                write!(
                    formatter,
                    "solid Tetrahedron mesh references missing node {node_id}"
                )
            }
        }
    }
}

impl std::error::Error for SolidMeshError {}
