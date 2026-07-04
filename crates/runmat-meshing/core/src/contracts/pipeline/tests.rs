use std::collections::BTreeMap;

use super::*;
use crate::contracts::{
    CadModel, CurveMesh, MeshingStage, PlcValidationSummary, ProtectedBoundaryComplex,
    SizingFieldContract, SolveReadinessReport, StageEvidence, SurfaceMesh, TetrahedronMesh,
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
fn stage_order_rejects_unrecovered_tetrahedron_recovery_stage() {
    let mut artifacts = complete_prefix_through_plc();
    artifacts.initial_tetrahedron_mesh = Some(tetrahedron_mesh(false, false));
    artifacts.recovered_tetrahedron_mesh = Some(tetrahedron_mesh(false, false));

    assert_eq!(
        validate_meshing_stage_order(&artifacts),
        Err(MeshingStageContractError::UnrecoveredTetrahedronMesh)
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
