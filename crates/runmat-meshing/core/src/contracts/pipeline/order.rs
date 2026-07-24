use crate::contracts::MeshingStage;

use super::{MeshingStageArtifacts, MeshingStageContractError};

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
    if let Some(mesh) = &artifacts.recovered_tetrahedron_mesh {
        require(
            artifacts.initial_tetrahedron_mesh.is_some(),
            MeshingStage::ConstraintRecovery,
            MeshingStage::TetrahedronMesh,
        )?;
        if !mesh.recovery_complete {
            return Err(MeshingStageContractError::UnrecoveredTetrahedronMesh);
        }
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
