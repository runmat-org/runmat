use serde::{Deserialize, Serialize};

use super::{
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

pub fn validate_meshing_stage_order(
    artifacts: &MeshingStageArtifacts,
) -> Result<(), MeshingStageContractError> {
    if artifacts.sizing_field.is_some() {
        require(
            artifacts.cad_model.is_some(),
            MeshingStage::Sizing,
            MeshingStage::CadTopology,
        )?;
    }
    if artifacts.curve_mesh.is_some() {
        require(
            artifacts.cad_model.is_some(),
            MeshingStage::CurveMesh,
            MeshingStage::CadTopology,
        )?;
        require(
            artifacts.sizing_field.is_some(),
            MeshingStage::CurveMesh,
            MeshingStage::Sizing,
        )?;
    }
    if artifacts.surface_mesh.is_some() {
        require(
            artifacts.curve_mesh.is_some(),
            MeshingStage::SurfaceMesh,
            MeshingStage::CurveMesh,
        )?;
    }
    if artifacts.protected_boundary_complex.is_some() {
        require(
            artifacts.surface_mesh.is_some(),
            MeshingStage::ProtectedBoundaryComplex,
            MeshingStage::SurfaceMesh,
        )?;
    }
    if artifacts.initial_tetrahedron_mesh.is_some() {
        require(
            artifacts.protected_boundary_complex.is_some(),
            MeshingStage::TetrahedronMesh,
            MeshingStage::ProtectedBoundaryComplex,
        )?;
        let plc = artifacts
            .protected_boundary_complex
            .as_ref()
            .expect("PLC presence was checked");
        if !plc.validation.valid_for_volume_meshing() {
            return Err(MeshingStageContractError::InvalidProtectedBoundaryComplex);
        }
    }
    if artifacts.recovered_tetrahedron_mesh.is_some() {
        require(
            artifacts.initial_tetrahedron_mesh.is_some(),
            MeshingStage::ConstraintRecovery,
            MeshingStage::TetrahedronMesh,
        )?;
    }
    if let Some(mesh) = &artifacts.optimized_tetrahedron_mesh {
        require(
            artifacts.recovered_tetrahedron_mesh.is_some(),
            MeshingStage::Optimization,
            MeshingStage::ConstraintRecovery,
        )?;
        if !mesh.recovery_complete {
            return Err(MeshingStageContractError::UnrecoveredTetrahedronMesh);
        }
    }
    if let Some(report) = &artifacts.solve_readiness {
        require(
            artifacts.optimized_tetrahedron_mesh.is_some(),
            MeshingStage::SolveReadiness,
            MeshingStage::Optimization,
        )?;
        let mesh = artifacts
            .optimized_tetrahedron_mesh
            .as_ref()
            .expect("optimized mesh presence was checked");
        if !mesh.quality_optimized {
            return Err(MeshingStageContractError::UnoptimizedTetrahedronMesh);
        }
        if !report.ready {
            return Err(MeshingStageContractError::SolveReadinessFailed);
        }
    }
    Ok(())
}

fn require(
    condition: bool,
    stage: MeshingStage,
    prerequisite: MeshingStage,
) -> Result<(), MeshingStageContractError> {
    if condition {
        Ok(())
    } else {
        Err(MeshingStageContractError::MissingPrerequisite {
            stage,
            prerequisite,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::contracts::{
        CadModel, CurveMesh, MeshingStage, PlcValidationSummary, ProtectedBoundaryComplex,
        SizingFieldContract, StageEvidence, SurfaceMesh, TetrahedronMesh,
    };

    #[test]
    fn stage_order_rejects_surface_mesh_without_curves() {
        let artifacts = MeshingStageArtifacts {
            surface_mesh: Some(surface_mesh()),
            ..MeshingStageArtifacts::default()
        };

        assert_eq!(
            validate_meshing_stage_order(&artifacts),
            Err(MeshingStageContractError::MissingPrerequisite {
                stage: MeshingStage::SurfaceMesh,
                prerequisite: MeshingStage::CurveMesh,
            })
        );
    }

    #[test]
    fn stage_order_rejects_tetrahedron_mesh_without_valid_plc() {
        let mut artifacts = complete_prefix_through_plc();
        artifacts
            .protected_boundary_complex
            .as_mut()
            .expect("PLC exists")
            .validation
            .watertight = false;
        artifacts.initial_tetrahedron_mesh = Some(tetrahedron_mesh(false, false));

        assert_eq!(
            validate_meshing_stage_order(&artifacts),
            Err(MeshingStageContractError::InvalidProtectedBoundaryComplex)
        );
    }

    #[test]
    fn stage_order_rejects_solve_readiness_before_optimization() {
        let mut artifacts = complete_prefix_through_plc();
        artifacts.initial_tetrahedron_mesh = Some(tetrahedron_mesh(false, false));
        artifacts.recovered_tetrahedron_mesh = Some(tetrahedron_mesh(true, false));
        artifacts.solve_readiness = Some(SolveReadinessReport {
            ready: true,
            evidence: vec![],
            failure_counts: BTreeMap::new(),
        });

        assert_eq!(
            validate_meshing_stage_order(&artifacts),
            Err(MeshingStageContractError::MissingPrerequisite {
                stage: MeshingStage::SolveReadiness,
                prerequisite: MeshingStage::Optimization,
            })
        );
    }

    #[test]
    fn stage_order_accepts_complete_meshing_sequence() {
        let mut artifacts = complete_prefix_through_plc();
        artifacts.initial_tetrahedron_mesh = Some(tetrahedron_mesh(false, false));
        artifacts.recovered_tetrahedron_mesh = Some(tetrahedron_mesh(true, false));
        artifacts.optimized_tetrahedron_mesh = Some(tetrahedron_mesh(true, true));
        artifacts.solve_readiness = Some(SolveReadinessReport {
            ready: true,
            evidence: vec![StageEvidence::complete(MeshingStage::SolveReadiness)],
            failure_counts: BTreeMap::new(),
        });

        validate_meshing_stage_order(&artifacts)
            .expect("complete topology-first sequence should validate");
    }

    #[test]
    fn contract_artifacts_round_trip_with_stage_evidence() {
        let mut artifacts = complete_prefix_through_plc();
        artifacts
            .protected_boundary_complex
            .as_mut()
            .expect("PLC exists")
            .evidence
            .entity_counts
            .insert("facets".to_string(), 12);

        let encoded = serde_json::to_string(&artifacts).expect("contracts should serialize");
        let decoded: MeshingStageArtifacts =
            serde_json::from_str(&encoded).expect("contracts should deserialize");

        assert_eq!(decoded, artifacts);
    }

    fn complete_prefix_through_plc() -> MeshingStageArtifacts {
        MeshingStageArtifacts {
            cad_model: Some(cad_model()),
            sizing_field: Some(sizing_field()),
            curve_mesh: Some(curve_mesh()),
            surface_mesh: Some(surface_mesh()),
            protected_boundary_complex: Some(protected_boundary_complex()),
            ..MeshingStageArtifacts::default()
        }
    }

    fn cad_model() -> CadModel {
        CadModel {
            model_id: "generic_cube".to_string(),
            unit_scale_to_m: 1.0,
            vertices: vec![],
            edges: vec![],
            faces: vec![],
            shells: vec![],
            volumes: vec![],
            evidence: StageEvidence::complete(MeshingStage::CadTopology),
        }
    }

    fn sizing_field() -> SizingFieldContract {
        SizingFieldContract {
            field_id: "sizing".to_string(),
            global_target_size_m: 0.1,
            min_size_m: None,
            max_size_m: None,
            growth_rate: Some(1.4),
            local_source_count: 0,
            anisotropic_metric_count: 0,
            evidence: StageEvidence::complete(MeshingStage::Sizing),
        }
    }

    fn curve_mesh() -> CurveMesh {
        CurveMesh {
            mesh_id: "curve".to_string(),
            nodes: vec![],
            elements: vec![],
            evidence: StageEvidence::complete(MeshingStage::CurveMesh),
        }
    }

    fn surface_mesh() -> SurfaceMesh {
        SurfaceMesh {
            mesh_id: "surface".to_string(),
            nodes: vec![],
            triangles: vec![],
            evidence: StageEvidence::complete(MeshingStage::SurfaceMesh),
        }
    }

    fn protected_boundary_complex() -> ProtectedBoundaryComplex {
        ProtectedBoundaryComplex {
            complex_id: "plc".to_string(),
            nodes: vec![],
            facets: vec![],
            protected_edges: vec![],
            validation: PlcValidationSummary {
                watertight: true,
                manifold: true,
                shell_nesting_classified: true,
                material_interfaces_classified: true,
            },
            evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
        }
    }

    fn tetrahedron_mesh(recovery_complete: bool, quality_optimized: bool) -> TetrahedronMesh {
        TetrahedronMesh {
            mesh_id: "tetrahedron".to_string(),
            nodes: vec![],
            elements: vec![],
            boundary_faces: vec![],
            recovery_complete,
            quality_optimized,
            evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
        }
    }
}
