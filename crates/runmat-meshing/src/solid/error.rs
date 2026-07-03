use runmat_meshing_cad::{CadEvaluationError, CadTopologyError, SourceTopologyError};
use runmat_meshing_core::{MeshBackendKind, MeshKindRequest, VolumeElementKind};
use runmat_meshing_curve::CurveDiscretizationError;
use runmat_meshing_plc::{build::PlcBuildError, validate::PlcValidationError};
use runmat_meshing_surface::SurfaceDiscretizationError;
use runmat_meshing_tetrahedron::{generate::TetrahedronGenerationError, structured_grid};

#[derive(Debug)]
pub enum SolidMeshingError {
    UnsupportedBackend(MeshBackendKind),
    UnsupportedMeshKind(MeshKindRequest),
    UnsupportedElementKind(VolumeElementKind),
    InvalidElementBudget,
    InvalidTargetSize,
    SourceTopology(SourceTopologyError),
    CadTopology(CadTopologyError),
    CadEvaluation(CadEvaluationError),
    Curve(CurveDiscretizationError),
    Surface(SurfaceDiscretizationError),
    ProtectedBoundaryComplex(PlcBuildError),
    ProtectedBoundaryValidation(PlcValidationError),
    Tetrahedron(TetrahedronGenerationError),
    StructuredGrid(structured_grid::MeshingError),
}

impl std::fmt::Display for SolidMeshingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBackend(backend) => {
                write!(formatter, "unsupported solid meshing backend: {backend:?}")
            }
            Self::UnsupportedMeshKind(kind) => {
                write!(formatter, "unsupported analysis mesh kind: {kind:?}")
            }
            Self::UnsupportedElementKind(kind) => {
                write!(formatter, "unsupported volume element kind: {kind:?}")
            }
            Self::InvalidElementBudget => write!(formatter, "max_elements must be greater than 0"),
            Self::InvalidTargetSize => {
                write!(
                    formatter,
                    "target_size must be auto or a finite positive length"
                )
            }
            Self::SourceTopology(err) => write!(formatter, "source topology failed: {err}"),
            Self::CadTopology(err) => write!(formatter, "CAD topology failed: {err}"),
            Self::CadEvaluation(err) => write!(formatter, "CAD evaluation failed: {err}"),
            Self::Curve(err) => write!(formatter, "curve meshing failed: {err}"),
            Self::Surface(err) => write!(formatter, "surface meshing failed: {err}"),
            Self::ProtectedBoundaryComplex(err) => write!(formatter, "PLC build failed: {err}"),
            Self::ProtectedBoundaryValidation(err) => {
                write!(formatter, "PLC validation failed: {err}")
            }
            Self::Tetrahedron(err) => write!(formatter, "Tetrahedron generation failed: {err}"),
            Self::StructuredGrid(err) => {
                write!(
                    formatter,
                    "structured-grid Tetrahedron meshing failed: {err}"
                )
            }
        }
    }
}

impl std::error::Error for SolidMeshingError {}
