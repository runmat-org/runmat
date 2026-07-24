use serde::{Deserialize, Serialize};

use crate::contracts::{
    CadModel, CurveMesh, MeshingStage, ProtectedBoundaryComplex, SizingFieldContract,
    SolveReadinessReport, SurfaceMesh, TetrahedronMesh,
};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshingStageArtifacts {
    #[serde(default)]
    pub cad_model: Option<CadModel>,
    #[serde(default)]
    pub sizing_field: Option<SizingFieldContract>,
    #[serde(default)]
    pub curve_mesh: Option<CurveMesh>,
    #[serde(default)]
    pub surface_mesh: Option<SurfaceMesh>,
    #[serde(default)]
    pub protected_boundary_complex: Option<ProtectedBoundaryComplex>,
    #[serde(default)]
    pub initial_tetrahedron_mesh: Option<TetrahedronMesh>,
    #[serde(default)]
    pub recovered_tetrahedron_mesh: Option<TetrahedronMesh>,
    #[serde(default)]
    pub optimized_tetrahedron_mesh: Option<TetrahedronMesh>,
    #[serde(default)]
    pub solve_readiness: Option<SolveReadinessReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshingStageContractError {
    MissingPrerequisite {
        stage: MeshingStage,
        prerequisite: MeshingStage,
    },
    InvalidProtectedBoundaryComplex,
    UnrecoveredTetrahedronMesh,
    UnoptimizedTetrahedronMesh,
    SolveReadinessFailed,
}

impl std::fmt::Display for MeshingStageContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingPrerequisite {
                stage,
                prerequisite,
            } => write!(
                formatter,
                "{stage:?} requires completed {prerequisite:?} before it can run"
            ),
            Self::InvalidProtectedBoundaryComplex => {
                write!(formatter, "Tetrahedron generation requires a validated PLC")
            }
            Self::UnrecoveredTetrahedronMesh => {
                write!(
                    formatter,
                    "optimization requires recovered Tetrahedron constraints"
                )
            }
            Self::UnoptimizedTetrahedronMesh => {
                write!(
                    formatter,
                    "solve readiness requires optimized Tetrahedron topology"
                )
            }
            Self::SolveReadinessFailed => {
                write!(
                    formatter,
                    "analysis artifact requires passing solve-readiness gates"
                )
            }
        }
    }
}

impl std::error::Error for MeshingStageContractError {}
