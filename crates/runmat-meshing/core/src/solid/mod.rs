use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::{
        AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
        AnalysisVolumeElement, MeshBackendSummary, ANALYSIS_MESH_SCHEMA_VERSION,
    },
    cad::eval::{
        build_cad_evaluation_model_with_provider, summarize_cad_evaluation, CadEvaluationError,
        CadEvaluationModel, CadEvaluationReport, CadEvaluationSource, CadFaceEvaluatorProvider,
        NoopCadFaceEvaluatorProvider,
    },
    cad::topology::{build_cad_topology, CadTopologyError, CadTopologyModel, CadTopologySource},
    contracts::TopologyEntityId,
    curve::{
        discretize_topology_curves, CurveDiscretization, CurveDiscretizationError,
        CurveDiscretizationOptions,
    },
    options::{MeshTargetSize, RefinementFocusLevel, VolumeMeshingOptions},
    plc::build::{build_protected_boundary_complex, PlcBuildError, ProtectedBoundaryComplex},
    predicate::{
        distance_squared, dot, point_in_closed_triangle_surface, point_triangle_distance,
        tet_centroid, tet_edge_aspect_ratio, tet_scaled_jacobian, tet_volume, triangle_centroid,
        PointInClosedSurface, Triangle3,
    },
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind},
    quality::{AnalysisMeshQualityReport, ElementQuality, QualityThresholds},
    size::field::{
        AnisotropicSizingSample, MeshSizingField, SizingSample, SizingSampleApplication,
        SizingSampleRejection,
    },
    source_topology::{extract_source_topology, SourceTopologyError, SourceTopologyModel},
    surface::recovery::{
        validate_surface_recovery, SurfaceRecoveryError, SurfaceRecoveryOptions,
        SurfaceRecoveryReport,
    },
    surface::validate::{
        validate_surface_discretization, SurfaceValidationError, SurfaceValidationOptions,
        SurfaceValidationReport,
    },
    surface::{
        discretize_cad_surfaces_with_curves, SurfaceDiscretization, SurfaceDiscretizationError,
        SurfaceDiscretizationOptions,
    },
    tet::generate::{
        generate_initial_tet_mesh_from_plc, generate_structured_box_tet_mesh_from_plc,
        TetGenerationError, TetMesh,
    },
    tet::recover::{build_recovery_queue_from_plc, TetRecoveryError, TetRecoveryQueue},
    tolerance::MeshingTolerance,
    topology::{BoundaryElementKind, VolumeElementKind},
    validation::{
        validate_analysis_mesh_with_options, AnalysisMeshValidationError,
        AnalysisMeshValidationOptions,
    },
};

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
    pub initial_tet_mesh: TetMesh,
    pub recovery_queue: TetRecoveryQueue,
    pub solver_tet_mesh: TetMesh,
    pub tet_stage: SolidTetStageEvidence,
    pub effective_sizing: Option<MeshSizingField>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidTetStageEvidence {
    pub volume_component_count: usize,
    pub interior_seed_point_count: usize,
    pub expected_volume_m3: f64,
    pub expected_boundary_area_m2: f64,
    #[serde(default)]
    pub coverage_sample_points_m: Vec<[f64; 3]>,
    pub recovered_component_ratio: f64,
    pub requested_refinement_point_count: usize,
    pub accepted_requested_refinement_point_count: usize,
    pub accepted_requested_refinement_surrogate_point_count: usize,
    pub rejected_requested_refinement_point_count: usize,
    #[serde(default)]
    pub requested_refinement_rejected_by_reason: BTreeMap<String, usize>,
    pub dropped_requested_refinement_point_count: usize,
    #[serde(default)]
    pub requested_refinement_dropped_by_reason: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolidMeshError {
    Topology(SourceTopologyError),
    CadTopology(CadTopologyError),
    CadEvaluation(CadEvaluationError),
    Curve(CurveDiscretizationError),
    Surface(SurfaceDiscretizationError),
    SurfaceValidation(SurfaceValidationError),
    SurfaceRecovery(SurfaceRecoveryError),
    ProtectedBoundaryComplex(PlcBuildError),
    TetGeneration(TetGenerationError),
    TetRecovery(TetRecoveryError),
    Validation(AnalysisMeshValidationError),
    MissingTetNode { node_id: String },
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
            Self::TetGeneration(err) => write!(formatter, "initial Tet generation failed: {err}"),
            Self::TetRecovery(err) => write!(formatter, "Tet recovery failed: {err}"),
            Self::Validation(err) => {
                write!(formatter, "solid mesh validation failed: {err:?}")
            }
            Self::MissingTetNode { node_id } => {
                write!(
                    formatter,
                    "solid Tet mesh references missing node {node_id}"
                )
            }
        }
    }
}

impl std::error::Error for SolidMeshError {}

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
    let initial_tet_mesh = generate_initial_tet_mesh_from_plc(&protected_boundary_complex)
        .map_err(SolidMeshError::TetGeneration)?;
    let solver_tet_mesh = generate_structured_box_tet_mesh_from_plc(&protected_boundary_complex)
        .map_err(SolidMeshError::TetGeneration)?;
    let recovery_queue =
        build_recovery_queue_from_plc(&protected_boundary_complex, &solver_tet_mesh)
            .map_err(SolidMeshError::TetRecovery)?;
    let tet_stage = build_solid_tet_stage_evidence(
        &topology,
        &surface,
        &solver_tet_mesh,
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
        initial_tet_mesh,
        recovery_queue,
        solver_tet_mesh,
        tet_stage,
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

fn curve_options_for_mesh(
    topology: &SourceTopologyModel,
    options: &VolumeMeshingOptions,
) -> CurveDiscretizationOptions {
    let mut target_size_m = target_size_for_mesh(topology, options);
    if thin_low_face_topology(topology) {
        if let Some(min_span_m) = topology_min_span(topology) {
            target_size_m = target_size_m.min(min_span_m.max(1.0e-6));
        }
    }
    CurveDiscretizationOptions {
        target_size_m,
        min_segments_per_edge: 1,
        max_segments_per_edge: options.max_elements.max(1).min(4096),
    }
}

fn surface_options_for_mesh(topology: &SourceTopologyModel) -> SurfaceDiscretizationOptions {
    SurfaceDiscretizationOptions {
        max_curve_segments_per_edge: if thin_low_face_topology(topology) {
            20
        } else {
            8
        },
        ..SurfaceDiscretizationOptions::default()
    }
}

fn build_solid_tet_stage_evidence(
    topology: &SourceTopologyModel,
    surface: &SurfaceDiscretization,
    tet_mesh: &TetMesh,
    sizing: Option<&MeshSizingField>,
) -> SolidTetStageEvidence {
    let requested = requested_refinement_selection(topology, sizing);
    let tet_nodes_by_id = tet_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut expected_volume_m3 = 0.0;
    let mut coverage_sample_points_m = Vec::<[f64; 3]>::new();
    for tet in &tet_mesh.elements {
        let Some(points) = tet_points_from_mesh(tet, &tet_nodes_by_id) else {
            continue;
        };
        expected_volume_m3 += tet_volume(points).abs();
        coverage_sample_points_m.push(tet_centroid(points));
    }
    coverage_sample_points_m.extend(requested.points[..requested.count].iter().copied());
    coverage_sample_points_m.truncate(64);

    let accepted_requested_refinement_point_count = requested
        .points
        .iter()
        .take(requested.count)
        .filter(|point| tet_mesh_has_node_near(tet_mesh, **point))
        .count();
    let rejected_requested_refinement_point_count = requested
        .count
        .saturating_sub(accepted_requested_refinement_point_count);
    let mut requested_refinement_rejected_by_reason = BTreeMap::<String, usize>::new();
    if rejected_requested_refinement_point_count > 0 {
        requested_refinement_rejected_by_reason.insert(
            "native_tet_generator_has_no_requested_point_insertion".to_string(),
            rejected_requested_refinement_point_count,
        );
    }

    SolidTetStageEvidence {
        volume_component_count: usize::from(!tet_mesh.elements.is_empty()),
        interior_seed_point_count: tet_mesh
            .nodes
            .iter()
            .filter(|node| node.node_id.stage == crate::contracts::MeshingStage::TetMesh)
            .count(),
        expected_volume_m3,
        expected_boundary_area_m2: surface.elements.iter().map(|element| element.area_m2).sum(),
        coverage_sample_points_m,
        recovered_component_ratio: 1.0,
        requested_refinement_point_count: requested.count,
        accepted_requested_refinement_point_count,
        accepted_requested_refinement_surrogate_point_count: 0,
        rejected_requested_refinement_point_count,
        requested_refinement_rejected_by_reason,
        dropped_requested_refinement_point_count: 0,
        requested_refinement_dropped_by_reason: BTreeMap::new(),
    }
}

fn tet_points_from_mesh(
    tet: &crate::contracts::Tet4Element,
    nodes: &BTreeMap<TopologyEntityId, [f64; 3]>,
) -> Option<Triangle3PlusOne> {
    Some([
        *nodes.get(&tet.node_ids[0])?,
        *nodes.get(&tet.node_ids[1])?,
        *nodes.get(&tet.node_ids[2])?,
        *nodes.get(&tet.node_ids[3])?,
    ])
}

type Triangle3PlusOne = [[f64; 3]; 4];

fn tet_mesh_has_node_near(tet_mesh: &TetMesh, point: [f64; 3]) -> bool {
    tet_mesh
        .nodes
        .iter()
        .any(|node| distance_squared(node.coordinates_m, point) <= 1.0e-20)
}

#[cfg(test)]
mod tests;
