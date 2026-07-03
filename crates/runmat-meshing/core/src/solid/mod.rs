use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::{
        AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
        AnalysisVolumeElement, MeshBackendSummary, ANALYSIS_MESH_SCHEMA_VERSION,
    },
    cad::eval::{
        build_cad_evaluation_model_with_provider, summarize_cad_evaluation, CadEvaluationModel,
        CadEvaluationReport, CadEvaluationSource, CadFaceEvaluatorProvider,
        NoopCadFaceEvaluatorProvider,
    },
    cad::topology::{build_cad_topology, CadTopologyModel, CadTopologySource},
    contracts::TopologyEntityId,
    curve::{discretize_topology_curves, CurveDiscretization},
    options::{MeshTargetSize, RefinementFocusLevel, VolumeMeshingOptions},
    plc::build::{build_protected_boundary_complex, ProtectedBoundaryComplex},
    predicate::{
        distance_squared, dot, point_in_closed_triangle_surface, point_triangle_distance,
        tetrahedron_centroid, tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian,
        tetrahedron_volume, triangle_centroid, PointInClosedSurface, Triangle3,
    },
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind},
    quality::{AnalysisMeshQualityReport, ElementQuality, QualityThresholds},
    size::field::{
        AnisotropicSizingSample, MeshSizingField, SizingSample, SizingSampleApplication,
        SizingSampleRejection,
    },
    source_topology::{extract_source_topology, SourceTopologyModel},
    surface::recovery::{validate_surface_recovery, SurfaceRecoveryOptions, SurfaceRecoveryReport},
    surface::validate::{
        validate_surface_discretization, SurfaceValidationOptions, SurfaceValidationReport,
    },
    surface::{discretize_cad_surfaces_with_curves, SurfaceDiscretization},
    tetrahedron::generate::{
        generate_initial_tetrahedron_mesh_from_plc,
        generate_structured_box_tetrahedron_mesh_from_plc, TetrahedronMesh,
    },
    tetrahedron::recover::{build_recovery_queue_from_plc, TetrahedronRecoveryQueue},
    tolerance::MeshingTolerance,
    topology::{BoundaryElementKind, VolumeElementKind},
    validation::{validate_analysis_mesh_with_options, AnalysisMeshValidationOptions},
};

#[cfg(test)]
use crate::tetrahedron::generate::TetrahedronGenerationError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidMeshPreparation {
    pub cad_topology: CadTopologyModel,
    pub cad_evaluation: CadEvaluationModel,
    pub cad_evaluation_report: CadEvaluationReport,
    pub topology: SourceTopologyModel,
    pub curves: CurveDiscretization,
    pub surface: SurfaceDiscretization,
    pub surface_validation: SurfaceValidationReport,
    pub surface_recovery: SurfaceRecoveryReport,
    pub protected_boundary_complex: ProtectedBoundaryComplex,
    pub initial_tetrahedron_mesh: TetrahedronMesh,
    pub recovery_queue: TetrahedronRecoveryQueue,
    pub solver_tetrahedron_mesh: TetrahedronMesh,
    pub tetrahedron_stage: SolidTetrahedronStageEvidence,
    pub effective_sizing: Option<MeshSizingField>,
}

mod error;
pub use error::SolidMeshError;

mod artifact;
use artifact::analysis_mesh_from_preparation;
#[cfg(test)]
use artifact::{
    quality_report, solid_validation_options, surface_cad_face_count,
    surface_max_cad_projection_error_m,
};

mod sizing;
#[cfg(test)]
use sizing::{constrained_recovery_topology, requested_sizing_sample_ids};
use sizing::{
    requested_refinement_selection, solid_effective_sizing, solid_mesh_sizing,
    solid_sizing_target_size, target_size_for_mesh, thin_low_face_topology, topology_min_span,
};

mod stage_options;
use stage_options::{curve_options_for_mesh, surface_options_for_mesh};

mod tetrahedron_stage;
use tetrahedron_stage::build_solid_tetrahedron_stage_evidence;
use tetrahedron_stage::tetrahedron_mesh_has_node_near;
pub use tetrahedron_stage::SolidTetrahedronStageEvidence;

pub fn prepare_solid_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<SolidMeshPreparation, SolidMeshError> {
    prepare_solid_mesh_internal(geometry, options, None, &NoopCadFaceEvaluatorProvider)
}

pub fn prepare_solid_mesh_with_cad_evaluator(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<SolidMeshPreparation, SolidMeshError> {
    prepare_solid_mesh_internal(geometry, options, None, evaluator_provider)
}

fn prepare_solid_mesh_internal(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<SolidMeshPreparation, SolidMeshError> {
    let topology = extract_source_topology(geometry).map_err(SolidMeshError::Topology)?;
    let cad_topology =
        build_cad_topology(geometry, &topology).map_err(SolidMeshError::CadTopology)?;
    let cad_evaluation =
        build_cad_evaluation_model_with_provider(&cad_topology, &topology, evaluator_provider)
            .map_err(SolidMeshError::CadEvaluation)?;
    let cad_evaluation_report = summarize_cad_evaluation(&cad_evaluation, &topology)
        .map_err(SolidMeshError::CadEvaluation)?;
    let effective_sizing = solid_effective_sizing(&topology, &cad_evaluation, options, sizing);
    let curves = discretize_topology_curves(&topology, curve_options_for_mesh(&topology, options))
        .map_err(SolidMeshError::Curve)?;
    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        surface_options_for_mesh(&topology),
    )
    .map_err(SolidMeshError::Surface)?;
    let surface_validation =
        validate_surface_discretization(&topology, &surface, SurfaceValidationOptions::default())
            .map_err(SolidMeshError::SurfaceValidation)?;
    let surface_recovery =
        validate_surface_recovery(&topology, &surface, SurfaceRecoveryOptions::default())
            .map_err(SolidMeshError::SurfaceRecovery)?;
    let protected_boundary_complex = build_protected_boundary_complex(&surface)
        .map_err(SolidMeshError::ProtectedBoundaryComplex)?;
    let initial_tetrahedron_mesh =
        generate_initial_tetrahedron_mesh_from_plc(&protected_boundary_complex)
            .map_err(SolidMeshError::TetrahedronGeneration)?;
    let solver_tetrahedron_mesh =
        generate_structured_box_tetrahedron_mesh_from_plc(&protected_boundary_complex)
            .map_err(SolidMeshError::TetrahedronGeneration)?;
    let recovery_queue =
        build_recovery_queue_from_plc(&protected_boundary_complex, &solver_tetrahedron_mesh)
            .map_err(SolidMeshError::TetrahedronRecovery)?;
    let tetrahedron_stage = build_solid_tetrahedron_stage_evidence(
        &topology,
        &surface,
        &solver_tetrahedron_mesh,
        effective_sizing.as_ref(),
    );
    Ok(SolidMeshPreparation {
        cad_topology,
        cad_evaluation,
        cad_evaluation_report,
        topology,
        curves,
        surface,
        surface_validation,
        surface_recovery,
        protected_boundary_complex,
        initial_tetrahedron_mesh,
        recovery_queue,
        solver_tetrahedron_mesh,
        tetrahedron_stage,
        effective_sizing,
    })
}

pub fn generate_solid_analysis_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, SolidMeshError> {
    let preparation = prepare_solid_mesh(geometry, options)?;
    analysis_mesh_from_preparation(&preparation, options, None)
}

pub fn generate_solid_analysis_mesh_with_cad_evaluator(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<AnalysisMeshArtifact, SolidMeshError> {
    let preparation = prepare_solid_mesh_with_cad_evaluator(geometry, options, evaluator_provider)?;
    analysis_mesh_from_preparation(&preparation, options, None)
}

pub fn generate_solid_analysis_mesh_with_sizing(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, SolidMeshError> {
    let topology = extract_source_topology(geometry).map_err(SolidMeshError::Topology)?;
    let base_target_size_m = target_size_for_mesh(&topology, options);
    let effective_target_size_m =
        solid_sizing_target_size(base_target_size_m, sizing, options, Some(&topology));
    let mut effective_options = options.clone();
    if effective_target_size_m < base_target_size_m {
        effective_options.target_size = MeshTargetSize::LengthM(effective_target_size_m);
    }
    let preparation = prepare_solid_mesh_internal(
        geometry,
        &effective_options,
        Some(sizing),
        &NoopCadFaceEvaluatorProvider,
    )?;
    analysis_mesh_from_preparation(&preparation, &effective_options, Some(sizing))
}

pub fn generate_solid_analysis_mesh_with_sizing_and_cad_evaluator(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: &MeshSizingField,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<AnalysisMeshArtifact, SolidMeshError> {
    let topology = extract_source_topology(geometry).map_err(SolidMeshError::Topology)?;
    let base_target_size_m = target_size_for_mesh(&topology, options);
    let effective_target_size_m =
        solid_sizing_target_size(base_target_size_m, sizing, options, Some(&topology));
    let mut effective_options = options.clone();
    if effective_target_size_m < base_target_size_m {
        effective_options.target_size = MeshTargetSize::LengthM(effective_target_size_m);
    }
    let preparation = prepare_solid_mesh_internal(
        geometry,
        &effective_options,
        Some(sizing),
        evaluator_provider,
    )?;
    analysis_mesh_from_preparation(&preparation, &effective_options, Some(sizing))
}

#[cfg(test)]
mod tests;
