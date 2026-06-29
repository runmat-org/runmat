use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::{
        AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
        AnalysisVolumeElement, MeshBackendSummary, ANALYSIS_MESH_SCHEMA_VERSION,
    },
    cad_eval::{
        build_cad_evaluation_model_with_provider, summarize_cad_evaluation, CadEvaluationError,
        CadEvaluationModel, CadEvaluationReport, CadEvaluationSource, CadFaceEvaluatorProvider,
        NoopCadFaceEvaluatorProvider,
    },
    cad_topology::{build_cad_topology, CadTopologyError, CadTopologyModel, CadTopologySource},
    curve::{
        discretize_topology_curves, CurveDiscretization, CurveDiscretizationError,
        CurveDiscretizationOptions,
    },
    options::{MeshTargetSize, RefinementFocusLevel, VolumeMeshingOptions},
    predicate::{distance_squared, dot, tet_centroid, tet_scaled_jacobian, triangle_centroid},
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind},
    quality::{AnalysisMeshQualityReport, ElementQuality, QualityThresholds},
    sizing::{
        AnisotropicSizingSample, MeshSizingField, SizingSample, SizingSampleApplication,
        SizingSampleRejection,
    },
    source_topology::{extract_source_topology, SourceTopologyError, SourceTopologyModel},
    surface::{
        discretize_cad_surfaces_with_curves, SurfaceDiscretization, SurfaceDiscretizationError,
        SurfaceDiscretizationOptions,
    },
    surface_recovery::{
        validate_surface_recovery, SurfaceRecoveryError, SurfaceRecoveryOptions,
        SurfaceRecoveryReport,
    },
    surface_validate::{
        validate_surface_discretization, SurfaceValidationError, SurfaceValidationOptions,
        SurfaceValidationReport,
    },
    tet_candidate::{
        form_tet_candidates, TetCandidateError, TetCandidateNodeSource, TetCandidateOptions,
        TetCandidateSet,
    },
    topology::{BoundaryElementKind, VolumeElementKind},
    validation::{
        validate_analysis_mesh_with_options, AnalysisMeshValidationError,
        AnalysisMeshValidationOptions,
    },
    volume_candidate::{
        prepare_volume_candidates, VolumeCandidateError, VolumeCandidateOptions, VolumeCandidateSet,
    },
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductionMeshPreparation {
    pub cad_topology: CadTopologyModel,
    pub cad_evaluation: CadEvaluationModel,
    pub cad_evaluation_report: CadEvaluationReport,
    pub topology: SourceTopologyModel,
    pub curves: CurveDiscretization,
    pub surface: SurfaceDiscretization,
    pub surface_validation: SurfaceValidationReport,
    pub surface_recovery: SurfaceRecoveryReport,
    pub volume_candidates: VolumeCandidateSet,
    pub tet_candidates: TetCandidateSet,
    pub effective_sizing: Option<MeshSizingField>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProductionMeshError {
    Topology(SourceTopologyError),
    CadTopology(CadTopologyError),
    CadEvaluation(CadEvaluationError),
    Curve(CurveDiscretizationError),
    Surface(SurfaceDiscretizationError),
    SurfaceValidation(SurfaceValidationError),
    SurfaceRecovery(SurfaceRecoveryError),
    VolumeCandidate(VolumeCandidateError),
    TetCandidate(TetCandidateError),
    Validation(AnalysisMeshValidationError),
    MissingCandidateNode { node_id: u32 },
}

impl std::fmt::Display for ProductionMeshError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(err) => write!(formatter, "source topology extraction failed: {err}"),
            Self::CadTopology(err) => write!(formatter, "CAD topology normalization failed: {err}"),
            Self::CadEvaluation(err) => write!(formatter, "CAD evaluation setup failed: {err}"),
            Self::Curve(err) => write!(formatter, "curve discretization failed: {err}"),
            Self::Surface(err) => write!(formatter, "surface discretization failed: {err}"),
            Self::SurfaceValidation(err) => write!(formatter, "surface validation failed: {err}"),
            Self::SurfaceRecovery(err) => write!(formatter, "surface recovery failed: {err}"),
            Self::VolumeCandidate(err) => {
                write!(formatter, "volume candidate preparation failed: {err}")
            }
            Self::TetCandidate(err) => write!(formatter, "Tet candidate formation failed: {err}"),
            Self::Validation(err) => {
                write!(formatter, "production mesh validation failed: {err:?}")
            }
            Self::MissingCandidateNode { node_id } => {
                write!(formatter, "candidate Tet references missing node {node_id}")
            }
        }
    }
}

impl std::error::Error for ProductionMeshError {}

pub fn prepare_production_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<ProductionMeshPreparation, ProductionMeshError> {
    prepare_production_mesh_internal(geometry, options, None, &NoopCadFaceEvaluatorProvider)
}

pub fn prepare_production_mesh_with_cad_evaluator(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<ProductionMeshPreparation, ProductionMeshError> {
    prepare_production_mesh_internal(geometry, options, None, evaluator_provider)
}

fn prepare_production_mesh_internal(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<ProductionMeshPreparation, ProductionMeshError> {
    let topology = extract_source_topology(geometry).map_err(ProductionMeshError::Topology)?;
    let cad_topology =
        build_cad_topology(geometry, &topology).map_err(ProductionMeshError::CadTopology)?;
    let cad_evaluation =
        build_cad_evaluation_model_with_provider(&cad_topology, &topology, evaluator_provider)
            .map_err(ProductionMeshError::CadEvaluation)?;
    let cad_evaluation_report = summarize_cad_evaluation(&cad_evaluation, &topology)
        .map_err(ProductionMeshError::CadEvaluation)?;
    let effective_sizing = production_effective_sizing(&topology, &cad_evaluation, options, sizing);
    let curves = discretize_topology_curves(&topology, curve_options_for_mesh(&topology, options))
        .map_err(ProductionMeshError::Curve)?;
    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 8,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .map_err(ProductionMeshError::Surface)?;
    let surface_validation =
        validate_surface_discretization(&topology, &surface, SurfaceValidationOptions::default())
            .map_err(ProductionMeshError::SurfaceValidation)?;
    let surface_recovery =
        validate_surface_recovery(&topology, &surface, SurfaceRecoveryOptions::default())
            .map_err(ProductionMeshError::SurfaceRecovery)?;
    let volume_candidates = prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
        .map_err(ProductionMeshError::VolumeCandidate)?;
    let tet_candidates = form_tet_candidates(
        &surface,
        &volume_candidates,
        tet_candidate_options_for_mesh(&topology, options, effective_sizing.as_ref()),
    )
    .map_err(ProductionMeshError::TetCandidate)?;

    Ok(ProductionMeshPreparation {
        cad_topology,
        cad_evaluation,
        cad_evaluation_report,
        topology,
        curves,
        surface,
        surface_validation,
        surface_recovery,
        volume_candidates,
        tet_candidates,
        effective_sizing,
    })
}

pub fn generate_production_analysis_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, ProductionMeshError> {
    let preparation = prepare_production_mesh(geometry, options)?;
    analysis_mesh_from_preparation(&preparation, options, None)
}

pub fn generate_production_analysis_mesh_with_cad_evaluator(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<AnalysisMeshArtifact, ProductionMeshError> {
    let preparation =
        prepare_production_mesh_with_cad_evaluator(geometry, options, evaluator_provider)?;
    analysis_mesh_from_preparation(&preparation, options, None)
}

pub fn generate_production_analysis_mesh_with_sizing(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, ProductionMeshError> {
    let topology = extract_source_topology(geometry).map_err(ProductionMeshError::Topology)?;
    let base_target_size_m = target_size_for_mesh(&topology, options);
    let effective_target_size_m =
        production_sizing_target_size(base_target_size_m, sizing, options);
    let mut effective_options = options.clone();
    if effective_target_size_m < base_target_size_m {
        effective_options.target_size = MeshTargetSize::LengthM(effective_target_size_m);
    }
    let preparation = prepare_production_mesh_internal(
        geometry,
        &effective_options,
        Some(sizing),
        &NoopCadFaceEvaluatorProvider,
    )?;
    analysis_mesh_from_preparation(&preparation, &effective_options, Some(sizing))
}

pub fn generate_production_analysis_mesh_with_sizing_and_cad_evaluator(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: &MeshSizingField,
    evaluator_provider: &dyn CadFaceEvaluatorProvider,
) -> Result<AnalysisMeshArtifact, ProductionMeshError> {
    let topology = extract_source_topology(geometry).map_err(ProductionMeshError::Topology)?;
    let base_target_size_m = target_size_for_mesh(&topology, options);
    let effective_target_size_m =
        production_sizing_target_size(base_target_size_m, sizing, options);
    let mut effective_options = options.clone();
    if effective_target_size_m < base_target_size_m {
        effective_options.target_size = MeshTargetSize::LengthM(effective_target_size_m);
    }
    let preparation = prepare_production_mesh_internal(
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
    let target_size_m = target_size_for_mesh(topology, options);
    CurveDiscretizationOptions {
        target_size_m,
        min_segments_per_edge: 1,
        max_segments_per_edge: options.max_elements.max(1).min(4096),
    }
}

fn tet_candidate_options_for_mesh(
    topology: &SourceTopologyModel,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
) -> TetCandidateOptions {
    let quality = QualityThresholds::default();
    let requested_refinement = requested_refinement_points(sizing);
    TetCandidateOptions {
        interior_target_size_m: Some(target_size_for_mesh(topology, options)),
        requested_refinement_points: requested_refinement.0,
        requested_refinement_point_count: requested_refinement.1,
        max_interior_seed_points: options.max_elements.max(1).min(512),
        max_global_insertion_points: options.max_elements.max(1).min(4096),
        allow_fan_fallback: false,
        dense_recovery_layer_count: 8,
        max_dense_recovery_nodes: options.max_elements.max(1).min(250_000),
        max_refinement_passes: match options.refinement.strategy {
            crate::options::RefinementStrategy::None => 0,
            _ => 3,
        },
        max_radius_edge_ratio: 2.5,
        sizing_compliance_tolerance: 0.35,
        min_scaled_jacobian: quality.min_scaled_jacobian,
        max_optimization_passes: 2,
        smoothing_relaxation: 0.30,
        sliver_aspect_ratio: 1.0 / quality.min_scaled_jacobian,
        ..TetCandidateOptions::default()
    }
}

fn requested_refinement_points(sizing: Option<&MeshSizingField>) -> ([[f64; 3]; 16], usize) {
    let Some(sizing) = sizing else {
        return ([[0.0; 3]; 16], 0);
    };
    let mut points = [[0.0; 3]; 16];
    let mut count = 0_usize;
    for sample in &sizing.samples {
        if !sample.target_size_m.is_finite()
            || sample.target_size_m <= 0.0
            || sample.position_m.iter().any(|value| !value.is_finite())
            || points[..count]
                .iter()
                .any(|point| distance_squared(*point, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        points[count] = sample.position_m;
        count += 1;
        if count >= points.len() {
            break;
        }
    }
    for sample in &sizing.anisotropic_samples {
        if !sample.is_valid_metric()
            || points[..count]
                .iter()
                .any(|point| distance_squared(*point, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        points[count] = sample.position_m;
        count += 1;
        if count >= points.len() {
            break;
        }
    }
    (points, count)
}

fn production_effective_sizing(
    topology: &SourceTopologyModel,
    cad_evaluation: &CadEvaluationModel,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
) -> Option<MeshSizingField> {
    let mut effective = sizing.cloned().unwrap_or_default();
    for sample in anisotropic_equivalent_sizing_samples(&effective.anisotropic_samples) {
        if effective
            .samples
            .iter()
            .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        effective.samples.push(sample);
    }
    if options.refinement.focus.curvature {
        let base_target_size_m = effective
            .global_target_size_m
            .filter(|value| value.is_finite() && *value > 0.0)
            .or(match options.target_size {
                MeshTargetSize::LengthM(length_m) if length_m.is_finite() && length_m > 0.0 => {
                    Some(clamp_mesh_target_size(length_m, options))
                }
                MeshTargetSize::LengthM(_) | MeshTargetSize::Auto => None,
            });
        for sample in cad_curvature_sizing_samples(cad_evaluation, base_target_size_m) {
            if effective
                .samples
                .iter()
                .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
            {
                continue;
            }
            effective.samples.push(sample);
        }
    }
    if options.refinement.focus.small_features {
        for sample in cad_feature_edge_sizing_samples(topology) {
            if effective
                .samples
                .iter()
                .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
            {
                continue;
            }
            effective.samples.push(sample);
        }
        for sample in cad_proximity_sizing_samples(topology) {
            if effective
                .samples
                .iter()
                .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
            {
                continue;
            }
            effective.samples.push(sample);
        }
    }
    for sample in cad_interface_sizing_samples(topology, options.refinement.focus.interfaces) {
        if effective
            .samples
            .iter()
            .any(|existing| distance_squared(existing.position_m, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        effective.samples.push(sample);
    }
    if sizing.is_some() || !effective.samples.is_empty() {
        Some(effective)
    } else {
        None
    }
}

fn anisotropic_equivalent_sizing_samples(samples: &[AnisotropicSizingSample]) -> Vec<SizingSample> {
    samples
        .iter()
        .filter(|sample| sample.is_valid_metric())
        .filter_map(|sample| {
            let target_size_m = sample
                .target_sizes_m
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            (target_size_m.is_finite() && target_size_m > 0.0).then_some(SizingSample {
                position_m: sample.position_m,
                target_size_m,
                reason: sample
                    .reason
                    .clone()
                    .or_else(|| Some("anisotropic.metric".to_string())),
            })
        })
        .collect()
}

fn cad_curvature_sizing_samples(
    cad_evaluation: &CadEvaluationModel,
    base_target_size_m: Option<f64>,
) -> Vec<SizingSample> {
    cad_evaluation
        .face_frames
        .iter()
        .filter_map(|frame| {
            let curvature = frame.max_curvature_estimate_1_per_m?;
            if !curvature.is_finite() || curvature <= 0.0 {
                return None;
            }
            let radius_m = 1.0 / curvature;
            if !radius_m.is_finite() || radius_m <= 0.0 {
                return None;
            }
            let mut target_size_m = radius_m * 0.25;
            if let Some(base_target_size_m) =
                base_target_size_m.filter(|value| value.is_finite() && *value > 0.0)
            {
                target_size_m = target_size_m.min(base_target_size_m);
            }
            (target_size_m.is_finite() && target_size_m > 0.0).then_some(SizingSample {
                position_m: frame.origin_m,
                target_size_m,
                reason: Some("cad.curvature".to_string()),
            })
        })
        .collect()
}

fn cad_feature_edge_sizing_samples(topology: &SourceTopologyModel) -> Vec<SizingSample> {
    let max_span = (0..3)
        .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    if !max_span.is_finite() || max_span <= 0.0 {
        return Vec::new();
    }
    let threshold_m = max_span * 0.35;
    topology
        .edges
        .iter()
        .filter_map(|edge| {
            if !edge.length_m.is_finite() || edge.length_m <= 0.0 || edge.length_m > threshold_m {
                return None;
            }
            let left = topology
                .vertices
                .get(edge.node_ids[0] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[0])?;
            let right = topology
                .vertices
                .get(edge.node_ids[1] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[1])?;
            Some(SizingSample {
                position_m: [
                    (left.coordinates_m[0] + right.coordinates_m[0]) * 0.5,
                    (left.coordinates_m[1] + right.coordinates_m[1]) * 0.5,
                    (left.coordinates_m[2] + right.coordinates_m[2]) * 0.5,
                ],
                target_size_m: edge.length_m * 0.5,
                reason: Some("cad.feature_edge".to_string()),
            })
        })
        .collect()
}

fn cad_interface_sizing_samples(
    topology: &SourceTopologyModel,
    focus: RefinementFocusLevel,
) -> Vec<SizingSample> {
    let target_fraction = match focus {
        RefinementFocusLevel::Off => return Vec::new(),
        RefinementFocusLevel::Normal => 0.5,
        RefinementFocusLevel::Fine => 0.25,
    };
    topology
        .edges
        .iter()
        .filter_map(|edge| {
            if edge.region_ids.len() < 2 || !edge.length_m.is_finite() || edge.length_m <= 0.0 {
                return None;
            }
            let left = topology
                .vertices
                .get(edge.node_ids[0] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[0])?;
            let right = topology
                .vertices
                .get(edge.node_ids[1] as usize)
                .filter(|vertex| vertex.vertex_id == edge.node_ids[1])?;
            Some(SizingSample {
                position_m: [
                    (left.coordinates_m[0] + right.coordinates_m[0]) * 0.5,
                    (left.coordinates_m[1] + right.coordinates_m[1]) * 0.5,
                    (left.coordinates_m[2] + right.coordinates_m[2]) * 0.5,
                ],
                target_size_m: edge.length_m * target_fraction,
                reason: Some("cad.interface".to_string()),
            })
        })
        .collect()
}

fn cad_proximity_sizing_samples(topology: &SourceTopologyModel) -> Vec<SizingSample> {
    let max_span = (0..3)
        .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    if !max_span.is_finite() || max_span <= 0.0 {
        return Vec::new();
    }
    let threshold_m = max_span * 0.35;
    let mut samples = Vec::<SizingSample>::new();
    for left_index in 0..topology.faces.len() {
        let left = &topology.faces[left_index];
        let Some(left_centroid) = topology_face_centroid(topology, left.node_ids) else {
            continue;
        };
        for right in topology.faces.iter().skip(left_index + 1) {
            if left
                .node_ids
                .iter()
                .any(|node_id| right.node_ids.contains(node_id))
            {
                continue;
            }
            if dot(left.unit_normal, right.unit_normal) > -0.75 {
                continue;
            }
            let Some(right_centroid) = topology_face_centroid(topology, right.node_ids) else {
                continue;
            };
            let gap_m = distance_squared(left_centroid, right_centroid).sqrt();
            if !gap_m.is_finite() || gap_m <= 0.0 || gap_m > threshold_m {
                continue;
            }
            samples.push(SizingSample {
                position_m: [
                    (left_centroid[0] + right_centroid[0]) * 0.5,
                    (left_centroid[1] + right_centroid[1]) * 0.5,
                    (left_centroid[2] + right_centroid[2]) * 0.5,
                ],
                target_size_m: gap_m * 0.5,
                reason: Some("cad.proximity".to_string()),
            });
        }
    }
    samples
}

fn topology_face_centroid(topology: &SourceTopologyModel, node_ids: [u32; 3]) -> Option<[f64; 3]> {
    let a = topology
        .vertices
        .get(node_ids[0] as usize)
        .filter(|vertex| vertex.vertex_id == node_ids[0])?
        .coordinates_m;
    let b = topology
        .vertices
        .get(node_ids[1] as usize)
        .filter(|vertex| vertex.vertex_id == node_ids[1])?
        .coordinates_m;
    let c = topology
        .vertices
        .get(node_ids[2] as usize)
        .filter(|vertex| vertex.vertex_id == node_ids[2])?
        .coordinates_m;
    Some([
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ])
}

fn target_size_for_mesh(topology: &SourceTopologyModel, options: &VolumeMeshingOptions) -> f64 {
    let target_size_m = match options.target_size {
        MeshTargetSize::LengthM(length_m) if length_m.is_finite() && length_m > 0.0 => length_m,
        MeshTargetSize::LengthM(_) | MeshTargetSize::Auto => {
            let span = (0..3)
                .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            (span / 20.0).max(1.0e-6)
        }
    };
    clamp_mesh_target_size(target_size_m, options)
}

fn production_sizing_target_size(
    base_target_size_m: f64,
    sizing: &MeshSizingField,
    options: &VolumeMeshingOptions,
) -> f64 {
    let mut target_size_m = base_target_size_m;
    let mut global_target_size_m = base_target_size_m;
    if let Some(candidate) = sizing.global_target_size_m {
        if candidate.is_finite() && candidate > 0.0 {
            target_size_m = candidate;
            global_target_size_m = candidate;
        }
    }
    for sample in &sizing.samples {
        if sample.target_size_m.is_finite() && sample.target_size_m > 0.0 {
            let sample_target_size_m = production_sample_target_size(
                sample.target_size_m,
                Some(global_target_size_m),
                sizing.growth_rate.or(options.growth_rate),
                sizing.min_size_m.or(options.min_size_m),
                sizing.max_size_m.or(options.max_size_m),
            );
            target_size_m = target_size_m.min(sample_target_size_m);
        }
    }
    for sample in anisotropic_equivalent_sizing_samples(&sizing.anisotropic_samples) {
        let sample_target_size_m = production_sample_target_size(
            sample.target_size_m,
            Some(global_target_size_m),
            sizing.growth_rate.or(options.growth_rate),
            sizing.min_size_m.or(options.min_size_m),
            sizing.max_size_m.or(options.max_size_m),
        );
        target_size_m = target_size_m.min(sample_target_size_m);
    }
    if let Some(min_size_m) = sizing.min_size_m.or(options.min_size_m) {
        if min_size_m.is_finite() && min_size_m > 0.0 {
            target_size_m = target_size_m.max(min_size_m);
        }
    }
    if let Some(max_size_m) = sizing.max_size_m.or(options.max_size_m) {
        if max_size_m.is_finite() && max_size_m > 0.0 {
            target_size_m = target_size_m.min(max_size_m);
        }
    }
    target_size_m.max(1.0e-9)
}

fn clamp_mesh_target_size(mut value: f64, options: &VolumeMeshingOptions) -> f64 {
    if let Some(min_size_m) = options.min_size_m {
        value = value.max(min_size_m);
    }
    if let Some(max_size_m) = options.max_size_m {
        value = value.min(max_size_m);
    }
    value
}

fn analysis_mesh_from_preparation(
    preparation: &ProductionMeshPreparation,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
) -> Result<AnalysisMeshArtifact, ProductionMeshError> {
    let mut node_id_map = BTreeMap::<u32, u32>::new();
    let referenced_candidate_node_ids = referenced_candidate_node_ids(preparation);
    let nodes = preparation
        .tet_candidates
        .nodes
        .iter()
        .filter(|node| referenced_candidate_node_ids.contains(&node.node_id))
        .enumerate()
        .map(|(index, node)| {
            let node_id = index as u32 + 1;
            node_id_map.insert(node.node_id, node_id);
            AnalysisMeshNode {
                node_id,
                coordinates_m: node.coordinates_m,
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: preparation.topology.source_geometry_id.clone(),
                    source_geometry_revision: preparation.topology.source_geometry_revision,
                    source_entity_kind: match node.source {
                        TetCandidateNodeSource::Surface => SourceEntityKind::Mesh,
                        TetCandidateNodeSource::InteriorSeed => SourceEntityKind::Body,
                    },
                    source_entity_id: node.node_id.to_string(),
                    region_ids: Vec::new(),
                }],
            }
        })
        .collect::<Vec<_>>();

    let mut source_surface_to_tet = BTreeMap::<u32, Vec<String>>::new();
    let mut volume_elements = Vec::<AnalysisVolumeElement>::new();
    let mut quality_elements = Vec::<ElementQuality>::new();
    let mut tet_centroids = Vec::<(String, [f64; 3])>::new();
    let candidate_nodes_by_id = preparation
        .tet_candidates
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tet in &preparation.tet_candidates.tets {
        let element_id = format!("prod_tet_{}", tet.tet_id + 1);
        let node_ids = tet
            .node_ids
            .iter()
            .map(|node_id| {
                node_id_map
                    .get(node_id)
                    .copied()
                    .ok_or(ProductionMeshError::MissingCandidateNode { node_id: *node_id })
            })
            .collect::<Result<Vec<_>, _>>()?;
        source_surface_to_tet
            .entry(tet.source_surface_element_id)
            .or_default()
            .push(element_id.clone());
        volume_elements.push(AnalysisVolumeElement {
            element_id: element_id.clone(),
            kind: VolumeElementKind::Tet4,
            node_ids,
            material_region_id: tet
                .region_ids
                .first()
                .cloned()
                .unwrap_or_else(|| "region_default".to_string()),
            provenance: vec![MeshEntityProvenance {
                source_geometry_id: preparation.topology.source_geometry_id.clone(),
                source_geometry_revision: preparation.topology.source_geometry_revision,
                source_entity_kind: SourceEntityKind::Face,
                source_entity_id: tet.source_surface_element_id.to_string(),
                region_ids: tet.region_ids.clone(),
            }],
        });
        let tet_points = [
            *candidate_nodes_by_id.get(&tet.node_ids[0]).ok_or(
                ProductionMeshError::MissingCandidateNode {
                    node_id: tet.node_ids[0],
                },
            )?,
            *candidate_nodes_by_id.get(&tet.node_ids[1]).ok_or(
                ProductionMeshError::MissingCandidateNode {
                    node_id: tet.node_ids[1],
                },
            )?,
            *candidate_nodes_by_id.get(&tet.node_ids[2]).ok_or(
                ProductionMeshError::MissingCandidateNode {
                    node_id: tet.node_ids[2],
                },
            )?,
            *candidate_nodes_by_id.get(&tet.node_ids[3]).ok_or(
                ProductionMeshError::MissingCandidateNode {
                    node_id: tet.node_ids[3],
                },
            )?,
        ];
        tet_centroids.push((element_id.clone(), tet_centroid(tet_points)));
        quality_elements.push(ElementQuality {
            element_id,
            scaled_jacobian: (1.0 / tet.aspect_ratio.max(1.0)).min(1.0),
            exact_scaled_jacobian: tet_scaled_jacobian(tet_points),
            aspect_ratio: tet.aspect_ratio,
            volume_m3: tet.volume_m3,
        });
    }

    let boundary_faces = preparation
        .surface
        .elements
        .iter()
        .map(|element| {
            let node_ids = element
                .node_ids
                .iter()
                .map(|node_id| {
                    node_id_map
                        .get(node_id)
                        .copied()
                        .ok_or(ProductionMeshError::MissingCandidateNode { node_id: *node_id })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut adjacent_volume_element_ids = source_surface_to_tet
                .get(&element.element_id)
                .cloned()
                .unwrap_or_default();
            if adjacent_volume_element_ids.is_empty() {
                if let Some(nearest_tet_id) =
                    nearest_tet_for_boundary_element(element, &preparation.surface, &tet_centroids)
                {
                    adjacent_volume_element_ids.push(nearest_tet_id);
                }
            }
            Ok(AnalysisBoundaryFace {
                face_id: format!("prod_boundary_{}", element.element_id + 1),
                kind: BoundaryElementKind::Tri3,
                node_ids,
                adjacent_volume_element_ids,
                region_ids: element.region_ids.clone(),
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: preparation.topology.source_geometry_id.clone(),
                    source_geometry_revision: preparation.topology.source_geometry_revision,
                    source_entity_kind: SourceEntityKind::Face,
                    source_entity_id: element.source_face_id.to_string(),
                    region_ids: element.region_ids.clone(),
                }],
            })
        })
        .collect::<Result<Vec<_>, ProductionMeshError>>()?;

    let boundary_edges = boundary_edges_from_faces(&boundary_faces, preparation);
    let backend = production_backend_summary(preparation, &boundary_faces, &boundary_edges);

    let mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: format!("production_{}", preparation.topology.mesh_id),
        nodes,
        volume_elements,
        boundary_faces,
        boundary_edges,
        quality: quality_report(
            quality_elements,
            preparation
                .surface
                .elements
                .iter()
                .map(|element| element.max_projection_error_m),
        ),
        sizing: production_mesh_sizing(
            options,
            preparation.effective_sizing.as_ref().or(sizing),
            preparation,
        ),
        backend,
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "production_topology_tet_candidate/v1".to_string(),
            source_geometry_id: preparation.topology.source_geometry_id.clone(),
            source_geometry_revision: preparation.topology.source_geometry_revision,
            source_geometry_sha256: preparation.topology.source_geometry_sha256.clone(),
        },
    };
    validate_analysis_mesh_with_options(&mesh, production_validation_options(preparation, options))
        .map_err(ProductionMeshError::Validation)?;
    Ok(mesh)
}

fn production_mesh_sizing(
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
    preparation: &ProductionMeshPreparation,
) -> MeshSizingField {
    let mut mesh_sizing = sizing.cloned().unwrap_or_default();
    if mesh_sizing.global_target_size_m.is_none() {
        mesh_sizing.global_target_size_m = match options.target_size {
            MeshTargetSize::LengthM(length) => Some(clamp_mesh_target_size(length, options)),
            MeshTargetSize::Auto => None,
        };
    }
    if mesh_sizing.min_size_m.is_none() {
        mesh_sizing.min_size_m = options.min_size_m;
    }
    if mesh_sizing.max_size_m.is_none() {
        mesh_sizing.max_size_m = options.max_size_m;
    }
    if mesh_sizing.growth_rate.is_none() {
        mesh_sizing.growth_rate = options.growth_rate;
    }
    if let Some(sizing) = sizing {
        let mut seen_positions = Vec::<[f64; 3]>::new();
        let requested_sample_ids = requested_sizing_sample_ids(sizing);
        let accepted_requested_ids = preparation
            .tet_candidates
            .accepted_requested_refinement_sample_indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let dropped_requested_ids = preparation
            .tet_candidates
            .dropped_requested_refinement_sample_indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        mesh_sizing.applied_samples.clear();
        mesh_sizing.rejected_samples.clear();
        for (sample_index, sample) in sizing.samples.iter().enumerate() {
            let valid_position = sample.position_m.iter().all(|value| value.is_finite());
            let valid_size = sample.target_size_m.is_finite() && sample.target_size_m > 0.0;
            if !valid_position || !valid_size {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m: sample.target_size_m,
                    status: "skipped_invalid".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some("sample position and target size must be finite".to_string()),
                });
                continue;
            }
            let target_size_m = production_sample_target_size(
                sample.target_size_m,
                mesh_sizing.global_target_size_m,
                mesh_sizing.growth_rate,
                mesh_sizing.min_size_m,
                mesh_sizing.max_size_m,
            );
            if seen_positions
                .iter()
                .any(|position| distance_squared(*position, sample.position_m) <= 1.0e-24)
            {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m: sample.target_size_m,
                    status: "skipped_duplicate".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some("sample position was already represented".to_string()),
                });
                continue;
            }
            seen_positions.push(sample.position_m);
            let requested_id = requested_sample_ids.get(&sample_index).copied();
            let inserted_breakpoint_count = usize::from(requested_id.is_some_and(|requested_id| {
                accepted_requested_ids.contains(&requested_id)
                    || preparation
                        .tet_candidates
                        .accepted_requested_refinement_points
                        .iter()
                        .any(|point| distance_squared(*point, sample.position_m) <= 1.0e-24)
            }));
            if inserted_breakpoint_count > 0 {
                mesh_sizing.applied_samples.push(SizingSampleApplication {
                    position_m: sample.position_m,
                    target_size_m,
                    inserted_breakpoint_count,
                    reason: sample.reason.clone(),
                    detail: Some("production_requested_tet_seed_accepted".to_string()),
                });
            } else if requested_id
                .is_some_and(|requested_id| dropped_requested_ids.contains(&requested_id))
            {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m,
                    status: "dropped_after_repair".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some("production requested Tet seed was removed by repair".to_string()),
                });
            } else if requested_id.is_some() {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m,
                    status: "rejected_by_recovery".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some(
                        "production requested Tet seed was not accepted by recovery".to_string(),
                    ),
                });
            } else {
                mesh_sizing.rejected_samples.push(SizingSampleRejection {
                    position_m: sample.position_m,
                    target_size_m,
                    status: "skipped_budget".to_string(),
                    reason: sample.reason.clone(),
                    detail: Some("requested Tet seed budget was exhausted".to_string()),
                });
            }
        }
    }
    mesh_sizing
}

fn requested_sizing_sample_ids(sizing: &MeshSizingField) -> BTreeMap<usize, usize> {
    let mut requested_sample_ids = BTreeMap::<usize, usize>::new();
    let mut requested_points = [[0.0; 3]; 16];
    let mut requested_count = 0_usize;
    for (sample_index, sample) in sizing.samples.iter().enumerate() {
        if !sample.target_size_m.is_finite()
            || sample.target_size_m <= 0.0
            || sample.position_m.iter().any(|value| !value.is_finite())
            || requested_points[..requested_count]
                .iter()
                .any(|point| distance_squared(*point, sample.position_m) <= 1.0e-24)
        {
            continue;
        }
        requested_sample_ids.insert(sample_index, requested_count);
        requested_points[requested_count] = sample.position_m;
        requested_count += 1;
        if requested_count >= requested_points.len() {
            break;
        }
    }
    requested_sample_ids
}

fn production_sample_target_size(
    mut target_size_m: f64,
    global_target_size_m: Option<f64>,
    growth_rate: Option<f64>,
    min_size_m: Option<f64>,
    max_size_m: Option<f64>,
) -> f64 {
    if let (Some(global_target_size_m), Some(growth_rate)) = (
        global_target_size_m.filter(|value| value.is_finite() && *value > 0.0),
        growth_rate.filter(|value| value.is_finite() && *value >= 1.0),
    ) {
        target_size_m = target_size_m.max(global_target_size_m / growth_rate);
    }
    if let Some(min_size_m) = min_size_m.filter(|value| value.is_finite() && *value > 0.0) {
        target_size_m = target_size_m.max(min_size_m);
    }
    if let Some(max_size_m) = max_size_m.filter(|value| value.is_finite() && *value > 0.0) {
        target_size_m = target_size_m.min(max_size_m);
    }
    target_size_m
}

fn production_backend_summary(
    preparation: &ProductionMeshPreparation,
    boundary_faces: &[AnalysisBoundaryFace],
    boundary_edges: &[AnalysisBoundaryEdge],
) -> MeshBackendSummary {
    MeshBackendSummary {
        backend: "production".to_string(),
        algorithm: "production_topology_tet_candidate/v1".to_string(),
        source_topology_vertex_count: preparation.topology.vertices.len(),
        source_topology_edge_count: preparation.topology.edges.len(),
        source_topology_face_count: preparation.topology.faces.len(),
        cad_topology_source: cad_topology_source_label(preparation.cad_topology.source).to_string(),
        cad_vertex_count: preparation.cad_topology.report.vertex_count,
        cad_edge_count: preparation.cad_topology.report.edge_count,
        cad_face_count: preparation.cad_topology.report.face_count,
        cad_shell_count: preparation.cad_topology.report.shell_count,
        cad_volume_count: preparation.cad_topology.report.volume_count,
        cad_semantic_face_count: preparation.cad_topology.report.semantic_face_count,
        cad_imported_face_count: preparation.cad_topology.report.imported_face_count,
        cad_evaluator_face_count: preparation.cad_topology.report.evaluator_face_count,
        cad_generic_face_count: preparation.cad_topology.report.generic_face_count,
        cad_closed_shell_count: preparation.cad_topology.report.closed_shell_count,
        cad_evaluation_source: cad_evaluation_source_label(preparation.cad_evaluation.source)
            .to_string(),
        cad_face_frame_count: preparation.cad_evaluation_report.face_frame_count,
        cad_evaluation_evaluator_face_count: preparation.cad_evaluation_report.evaluator_face_count,
        cad_evaluation_live_query_face_count: preparation
            .cad_evaluation_report
            .live_query_face_count,
        cad_evaluation_exact_query_face_count: preparation
            .cad_evaluation_report
            .exact_query_face_count,
        cad_evaluation_point_supported_face_count: preparation
            .cad_evaluation_report
            .point_evaluation_supported_face_count,
        cad_evaluation_projection_supported_face_count: preparation
            .cad_evaluation_report
            .projection_supported_face_count,
        cad_evaluation_normal_supported_face_count: preparation
            .cad_evaluation_report
            .normal_supported_face_count,
        cad_evaluation_derivative_supported_face_count: preparation
            .cad_evaluation_report
            .derivative_supported_face_count,
        cad_evaluation_curvature_supported_face_count: preparation
            .cad_evaluation_report
            .curvature_supported_face_count,
        cad_evaluation_missing_exact_query_face_count: preparation
            .cad_evaluation_report
            .missing_exact_query_face_count,
        cad_evaluation_sample_count: preparation.cad_evaluation_report.evaluator_sample_count,
        cad_evaluation_rejected_sample_count: preparation
            .cad_evaluation_report
            .evaluator_rejected_sample_count,
        cad_projection_query_count: preparation.cad_evaluation_report.projection_query_count,
        cad_derivative_query_count: preparation.cad_evaluation_report.derivative_query_count,
        cad_curvature_query_count: preparation.cad_evaluation_report.curvature_query_count,
        cad_uv_domain_face_count: preparation.cad_evaluation_report.uv_domain_face_count,
        cad_uv_projection_out_of_bounds_count: preparation
            .cad_evaluation_report
            .uv_projection_out_of_bounds_count,
        cad_max_projection_error_m: preparation.cad_evaluation_report.max_projection_error_m,
        cad_max_normal_deviation: preparation.cad_evaluation_report.max_normal_deviation,
        cad_max_curvature_estimate_1_per_m: preparation
            .cad_evaluation_report
            .max_curvature_estimate_1_per_m
            .unwrap_or(0.0),
        curve_element_count: preparation.curves.elements.len(),
        surface_element_count: preparation.surface.elements.len(),
        surface_source_edge_loop_count: preparation.surface_validation.source_edge_loop_count,
        surface_closed_edge_loop_count: preparation
            .surface_validation
            .closed_source_edge_loop_count,
        surface_projection_error_m: preparation.surface_validation.max_projection_error_m,
        surface_face_coverage_ratio: preparation.surface_validation.face_coverage_ratio,
        surface_cad_face_count: surface_cad_face_count(&preparation.surface),
        surface_exact_cad_sample_node_count: preparation.surface.exact_cad_sample_node_count,
        surface_rejected_exact_cad_sample_count: preparation
            .surface
            .rejected_exact_cad_sample_count,
        surface_max_cad_projection_error_m: surface_max_cad_projection_error_m(
            &preparation.surface,
        ),
        volume_candidate_count: preparation.volume_candidates.components.len(),
        interior_seed_point_count: preparation.tet_candidates.interior_seed_points.len(),
        tet_candidate_count: preparation.tet_candidates.tets.len(),
        tet_recovered_component_ratio: preparation
            .tet_candidates
            .recovery
            .recovered_component_ratio,
        tet_fan_fallback_component_count: preparation
            .tet_candidates
            .recovery
            .fan_fallback_component_count,
        tet_candidate_volume_ratio: preparation
            .tet_candidates
            .recovery
            .total_candidate_volume_ratio,
        tet_refinement_pass_count: preparation.tet_candidates.recovery.refinement_pass_count,
        tet_refinement_point_count: preparation.tet_candidates.recovery.refinement_point_count,
        tet_requested_refinement_point_count: preparation
            .tet_candidates
            .recovery
            .requested_refinement_point_count,
        tet_accepted_requested_refinement_candidate_count: preparation
            .tet_candidates
            .recovery
            .accepted_requested_refinement_candidate_count,
        tet_accepted_requested_refinement_point_count: preparation
            .tet_candidates
            .recovery
            .accepted_requested_refinement_point_count,
        tet_accepted_requested_refinement_surrogate_point_count: preparation
            .tet_candidates
            .recovery
            .accepted_requested_refinement_surrogate_point_count,
        tet_rejected_requested_refinement_point_count: preparation
            .tet_candidates
            .recovery
            .rejected_requested_refinement_point_count,
        tet_requested_refinement_rejected_by_reason: preparation
            .tet_candidates
            .recovery
            .requested_refinement_rejected_by_reason
            .clone(),
        tet_dropped_requested_refinement_point_count: preparation
            .tet_candidates
            .recovery
            .dropped_requested_refinement_point_count,
        tet_requested_refinement_dropped_by_reason: preparation
            .tet_candidates
            .recovery
            .requested_refinement_dropped_by_reason
            .clone(),
        tet_max_radius_edge_ratio: preparation.tet_candidates.recovery.max_radius_edge_ratio,
        tet_sizing_violation_count: preparation.tet_candidates.recovery.sizing_violation_count,
        tet_min_exact_scaled_jacobian: preparation
            .tet_candidates
            .recovery
            .min_exact_scaled_jacobian,
        tet_exact_scaled_jacobian_below_threshold_count: preparation
            .tet_candidates
            .recovery
            .exact_scaled_jacobian_below_threshold_count,
        tet_exact_scaled_jacobian_bins: preparation
            .tet_candidates
            .recovery
            .exact_scaled_jacobian_bins
            .clone(),
        tet_optimization_pass_count: preparation.tet_candidates.recovery.optimization_pass_count,
        tet_smoothed_point_count: preparation.tet_candidates.recovery.smoothed_point_count,
        tet_sliver_candidate_count: preparation.tet_candidates.recovery.sliver_candidate_count,
        tet_sliver_removed_count: preparation.tet_candidates.recovery.sliver_removed_count,
        tet_optimization_target_seed_count: preparation
            .tet_candidates
            .recovery
            .optimization_target_seed_count,
        tet_optimization_skipped_target_seed_count: preparation
            .tet_candidates
            .recovery
            .optimization_skipped_target_seed_count,
        tet_optimization_rejected_edit_count: preparation
            .tet_candidates
            .recovery
            .optimization_rejected_edit_count,
        tet_optimization_initial_max_aspect_ratio: preparation
            .tet_candidates
            .recovery
            .optimization_initial_max_aspect_ratio,
        tet_optimization_final_max_aspect_ratio: preparation
            .tet_candidates
            .recovery
            .optimization_final_max_aspect_ratio,
        tet_optimization_initial_min_exact_scaled_jacobian: preparation
            .tet_candidates
            .recovery
            .optimization_initial_min_exact_scaled_jacobian,
        tet_optimization_final_min_exact_scaled_jacobian: preparation
            .tet_candidates
            .recovery
            .optimization_final_min_exact_scaled_jacobian,
        tet_untangling_pass_count: preparation.tet_candidates.recovery.untangling_pass_count,
        tet_untangling_initial_near_singular_count: preparation
            .tet_candidates
            .recovery
            .untangling_initial_near_singular_count,
        tet_untangling_final_near_singular_count: preparation
            .tet_candidates
            .recovery
            .untangling_final_near_singular_count,
        tet_untangling_relocated_seed_count: preparation
            .tet_candidates
            .recovery
            .untangling_relocated_seed_count,
        tet_untangling_reconnected_edge_star_count: preparation
            .tet_candidates
            .recovery
            .untangling_reconnected_edge_star_count,
        tet_untangling_reconnected_boundary_adjacent_cavity_count: preparation
            .tet_candidates
            .recovery
            .untangling_reconnected_boundary_adjacent_cavity_count,
        tet_exact_quality_repair_pass_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_repair_pass_count,
        tet_exact_quality_reconnected_cavity_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_reconnected_cavity_count,
        tet_exact_quality_reconnection_quality_gain_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_reconnection_quality_gain_count,
        tet_exact_quality_face_neighbor_reconnected_cavity_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_face_neighbor_reconnected_cavity_count,
        tet_exact_quality_connected_reconnected_cavity_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_connected_reconnected_cavity_count,
        tet_exact_quality_boundary_adjacent_reconnected_cavity_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_boundary_adjacent_reconnected_cavity_count,
        tet_exact_quality_expanded_connected_reconnected_cavity_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_expanded_connected_reconnected_cavity_count,
        tet_exact_quality_split_cavity_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_split_cavity_count,
        tet_exact_quality_seed_star_collapse_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_seed_star_collapse_count,
        tet_exact_quality_seed_star_relocation_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_seed_star_relocation_count,
        tet_exact_quality_unrepaired_total_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_unrepaired_total_count,
        tet_exact_quality_unrepaired_general_cavity_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_unrepaired_general_cavity_count,
        tet_exact_quality_unrepaired_boundary_adjacent_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_unrepaired_boundary_adjacent_count,
        tet_exact_quality_unrepaired_interior_seed_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_unrepaired_interior_seed_count,
        tet_exact_quality_unrepaired_edge_star_count: preparation
            .tet_candidates
            .recovery
            .exact_quality_unrepaired_edge_star_count,
        boundary_face_recovery_ratio: boundary_face_recovery_ratio(boundary_faces),
        boundary_edge_recovery_ratio: boundary_edge_recovery_ratio(boundary_faces, boundary_edges),
    }
}

fn cad_topology_source_label(source: CadTopologySource) -> &'static str {
    match source {
        CadTopologySource::SemanticCad => "semantic_cad",
        CadTopologySource::GenericCadMesh => "generic_cad_mesh",
        CadTopologySource::MeshFallback => "mesh_fallback",
    }
}

fn cad_evaluation_source_label(source: CadEvaluationSource) -> &'static str {
    match source {
        CadEvaluationSource::ParametricCad => "parametric_cad",
        CadEvaluationSource::ImportedEvaluatorSamples => "imported_evaluator_samples",
        CadEvaluationSource::PlanarFacetApproximation => "planar_facet_approximation",
    }
}

fn surface_cad_face_count(surface: &SurfaceDiscretization) -> usize {
    surface
        .elements
        .iter()
        .filter_map(|element| element.cad_face_id.as_ref())
        .collect::<BTreeSet<_>>()
        .len()
}

fn surface_max_cad_projection_error_m(surface: &SurfaceDiscretization) -> f64 {
    surface
        .elements
        .iter()
        .map(|element| element.max_projection_error_m)
        .fold(0.0_f64, f64::max)
}

fn boundary_face_recovery_ratio(boundary_faces: &[AnalysisBoundaryFace]) -> f64 {
    if boundary_faces.is_empty() {
        return 1.0;
    }
    let recovered_count = boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .count();
    recovered_count as f64 / boundary_faces.len() as f64
}

fn nearest_tet_for_boundary_element(
    element: &crate::surface::SurfaceElement,
    surface: &SurfaceDiscretization,
    tet_centroids: &[(String, [f64; 3])],
) -> Option<String> {
    let points = [
        surface.nodes[element.node_ids[0] as usize].coordinates_m,
        surface.nodes[element.node_ids[1] as usize].coordinates_m,
        surface.nodes[element.node_ids[2] as usize].coordinates_m,
    ];
    let centroid = triangle_centroid(points);
    tet_centroids
        .iter()
        .min_by(|(_, left), (_, right)| {
            distance_squared(*left, centroid).total_cmp(&distance_squared(*right, centroid))
        })
        .map(|(element_id, _)| element_id.clone())
}

fn boundary_edge_recovery_ratio(
    boundary_faces: &[AnalysisBoundaryFace],
    boundary_edges: &[AnalysisBoundaryEdge],
) -> f64 {
    let expected_edges = boundary_face_edge_set(boundary_faces);
    if expected_edges.is_empty() {
        return 1.0;
    }
    let recovered_edges = boundary_edges
        .iter()
        .filter(|edge| !edge.adjacent_boundary_face_ids.is_empty())
        .map(|edge| sorted_edge(edge.node_ids[0], edge.node_ids[1]))
        .collect::<BTreeSet<_>>();
    expected_edges
        .iter()
        .filter(|edge| recovered_edges.contains(*edge))
        .count() as f64
        / expected_edges.len() as f64
}

fn boundary_face_edge_set(boundary_faces: &[AnalysisBoundaryFace]) -> BTreeSet<[u32; 2]> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in boundary_faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        edges.extend(triangle_edges([
            face.node_ids[0],
            face.node_ids[1],
            face.node_ids[2],
        ]));
    }
    edges
}

fn boundary_edges_from_faces(
    boundary_faces: &[AnalysisBoundaryFace],
    preparation: &ProductionMeshPreparation,
) -> Vec<AnalysisBoundaryEdge> {
    let mut edges = BTreeMap::<[u32; 2], BoundaryEdgeAccumulator>::new();
    for face in boundary_faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        for edge in triangle_edges([face.node_ids[0], face.node_ids[1], face.node_ids[2]]) {
            let accumulator = edges
                .entry(edge)
                .or_insert_with(|| BoundaryEdgeAccumulator {
                    node_ids: edge,
                    adjacent_boundary_face_ids: Vec::new(),
                    region_ids: BTreeSet::new(),
                });
            accumulator
                .adjacent_boundary_face_ids
                .push(face.face_id.clone());
            accumulator
                .region_ids
                .extend(face.region_ids.iter().cloned());
        }
    }
    edges
        .into_values()
        .enumerate()
        .map(|(index, edge)| {
            let region_ids = edge.region_ids.into_iter().collect::<Vec<_>>();
            AnalysisBoundaryEdge {
                edge_id: format!("prod_boundary_edge_{}", index + 1),
                node_ids: edge.node_ids,
                adjacent_boundary_face_ids: edge.adjacent_boundary_face_ids,
                region_ids: region_ids.clone(),
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: preparation.topology.source_geometry_id.clone(),
                    source_geometry_revision: preparation.topology.source_geometry_revision,
                    source_entity_kind: SourceEntityKind::Edge,
                    source_entity_id: format!("boundary_edge_{}", index + 1),
                    region_ids,
                }],
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundaryEdgeAccumulator {
    node_ids: [u32; 2],
    adjacent_boundary_face_ids: Vec<String>,
    region_ids: BTreeSet<String>,
}

fn triangle_edges(node_ids: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(node_ids[0], node_ids[1]),
        sorted_edge(node_ids[1], node_ids[2]),
        sorted_edge(node_ids[2], node_ids[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn referenced_candidate_node_ids(preparation: &ProductionMeshPreparation) -> BTreeSet<u32> {
    let mut node_ids = BTreeSet::<u32>::new();
    for tet in &preparation.tet_candidates.tets {
        node_ids.extend(tet.node_ids);
    }
    for element in &preparation.surface.elements {
        node_ids.extend(element.node_ids);
    }
    node_ids
}

fn production_validation_options(
    preparation: &ProductionMeshPreparation,
    options: &VolumeMeshingOptions,
) -> AnalysisMeshValidationOptions {
    AnalysisMeshValidationOptions {
        quality: options.validation.quality,
        max_volume_element_count: Some(options.max_elements),
        max_volume_component_count: options
            .validation
            .max_volume_component_count
            .or(Some(preparation.volume_candidates.components.len())),
        coverage_sample_points_m: preparation
            .tet_candidates
            .interior_seed_points
            .iter()
            .copied()
            .take(64)
            .collect(),
        min_coverage_sample_ratio: 1.0,
        expected_bounds_m: Some([
            preparation.topology.bounds_min_m,
            preparation.topology.bounds_max_m,
        ]),
        min_bounds_coverage_ratio: options.validation.min_bounds_coverage_ratio,
        expected_volume_m3: Some(preparation.volume_candidates.total_volume_m3),
        min_volume_coverage_ratio: options.validation.min_volume_coverage_ratio,
        expected_boundary_area_m2: Some(preparation.volume_candidates.total_surface_area_m2),
        min_boundary_area_ratio: options.validation.min_boundary_area_ratio,
        min_boundary_face_recovery_ratio: options.validation.min_boundary_face_recovery_ratio,
        min_boundary_edge_recovery_ratio: options.validation.min_boundary_edge_recovery_ratio,
        require_no_fan_fallback: true,
        require_no_unrepaired_exact_quality: true,
        ..AnalysisMeshValidationOptions::default()
    }
}

fn quality_report(
    elements: Vec<ElementQuality>,
    boundary_projection_errors_m: impl IntoIterator<Item = f64>,
) -> AnalysisMeshQualityReport {
    if elements.is_empty() {
        return AnalysisMeshQualityReport::default();
    }
    let mut boundary_projection_error_count = 0_usize;
    let mut boundary_projection_error_sum_m = 0.0_f64;
    let mut max_boundary_projection_error_m = 0.0_f64;
    for error_m in boundary_projection_errors_m {
        if !error_m.is_finite() {
            continue;
        }
        boundary_projection_error_count += 1;
        boundary_projection_error_sum_m += error_m;
        max_boundary_projection_error_m = max_boundary_projection_error_m.max(error_m);
    }
    let mean_boundary_projection_error_m = if boundary_projection_error_count == 0 {
        0.0
    } else {
        boundary_projection_error_sum_m / boundary_projection_error_count as f64
    };
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let min_exact_scaled_jacobian = elements
        .iter()
        .map(|element| element.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .sum::<f64>()
        / elements.len() as f64;
    AnalysisMeshQualityReport {
        min_scaled_jacobian,
        min_exact_scaled_jacobian,
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: 0,
        mean_boundary_projection_error_m,
        max_boundary_projection_error_m,
        elements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
        CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, GeometryAsset,
        GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping, SourceGeometry,
        SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
    };

    #[test]
    fn preparation_runs_topology_curve_surface_and_volume_candidate_stages() {
        let preparation =
            prepare_production_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
                .expect("production preparation should run");

        assert_eq!(preparation.topology.faces.len(), 12);
        assert_eq!(preparation.cad_topology.report.vertex_count, 8);
        assert_eq!(preparation.cad_topology.report.edge_count, 18);
        assert_eq!(preparation.cad_topology.report.face_count, 12);
        assert_eq!(preparation.cad_topology.report.closed_shell_count, 1);
        assert_eq!(preparation.cad_evaluation.report.face_frame_count, 12);
        assert_eq!(
            preparation
                .cad_evaluation_report
                .missing_exact_query_face_count,
            0
        );
        assert_eq!(preparation.cad_evaluation_report.projection_query_count, 36);
        assert_eq!(
            preparation.cad_evaluation_report.max_projection_error_m,
            0.0
        );
        assert_eq!(preparation.surface.elements.len(), 768);
        assert_eq!(preparation.surface.exact_cad_sample_node_count, 0);
        assert_eq!(preparation.surface.rejected_exact_cad_sample_count, 0);
        assert_eq!(surface_cad_face_count(&preparation.surface), 12);
        assert_eq!(
            surface_max_cad_projection_error_m(&preparation.surface),
            0.0
        );
        assert!(preparation
            .surface
            .elements
            .iter()
            .all(|element| element.cad_face_id.is_some()));
        assert_eq!(preparation.surface_validation.source_edge_loop_count, 1);
        assert_eq!(
            preparation.surface_validation.closed_source_edge_loop_count,
            1
        );
        assert_eq!(preparation.surface_validation.face_coverage_ratio, 1.0);
        assert_eq!(preparation.surface_recovery.surface_element_count, 768);
        assert_eq!(preparation.surface_recovery.open_edge_count, 0);
        assert_eq!(preparation.surface_recovery.nonmanifold_edge_count, 0);
        assert_eq!(preparation.surface_recovery.source_face_coverage_ratio, 1.0);
        assert_eq!(preparation.volume_candidates.components.len(), 1);
        assert!(preparation.tet_candidates.tets.len() > 12);
        assert_eq!(
            preparation
                .tet_candidates
                .recovery
                .fan_fallback_component_count,
            0
        );
        assert_eq!(
            preparation
                .tet_candidates
                .recovery
                .recovered_component_ratio,
            1.0
        );
        assert!((preparation.volume_candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert!((preparation.tet_candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert!(!preparation.curves.elements.is_empty());
    }

    #[test]
    fn production_mesh_generates_analysis_mesh_artifact_from_candidates() {
        let preparation =
            prepare_production_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
                .expect("production preparation should generate");
        let mesh =
            generate_production_analysis_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
                .expect("production mesh should generate from candidates");

        crate::validate_analysis_mesh(&mesh, crate::QualityThresholds::default())
            .expect("production candidate mesh should validate");
        assert!(mesh.nodes.len() > 9);
        assert!(mesh.volume_elements.len() > 12);
        assert_eq!(mesh.boundary_faces.len(), 768);
        assert_eq!(mesh.boundary_edges.len(), 1152);
        assert!(mesh
            .boundary_edges
            .iter()
            .all(|edge| !edge.adjacent_boundary_face_ids.is_empty()));
        assert_eq!(mesh.backend.backend, "production");
        assert_eq!(
            mesh.backend.algorithm,
            "production_topology_tet_candidate/v1"
        );
        assert_eq!(mesh.backend.source_topology_face_count, 12);
        assert_eq!(mesh.backend.cad_topology_source, "generic_cad_mesh");
        assert_eq!(mesh.backend.cad_vertex_count, 8);
        assert_eq!(mesh.backend.cad_edge_count, 18);
        assert_eq!(mesh.backend.cad_face_count, 12);
        assert_eq!(mesh.backend.cad_closed_shell_count, 1);
        assert_eq!(mesh.backend.cad_imported_face_count, 0);
        assert_eq!(mesh.backend.cad_evaluator_face_count, 0);
        assert_eq!(
            mesh.backend.cad_evaluation_source,
            "planar_facet_approximation"
        );
        assert_eq!(mesh.backend.cad_face_frame_count, 12);
        assert_eq!(mesh.backend.cad_evaluation_evaluator_face_count, 0);
        assert_eq!(
            mesh.backend.cad_evaluation_missing_exact_query_face_count,
            0
        );
        assert_eq!(mesh.backend.cad_projection_query_count, 36);
        assert_eq!(mesh.backend.cad_max_projection_error_m, 0.0);
        assert_eq!(mesh.backend.surface_element_count, 768);
        assert_eq!(mesh.backend.surface_source_edge_loop_count, 1);
        assert_eq!(mesh.backend.surface_closed_edge_loop_count, 1);
        assert_eq!(mesh.backend.surface_face_coverage_ratio, 1.0);
        assert_eq!(mesh.backend.surface_cad_face_count, 12);
        assert_eq!(mesh.backend.surface_exact_cad_sample_node_count, 0);
        assert_eq!(mesh.backend.surface_rejected_exact_cad_sample_count, 0);
        assert_eq!(mesh.backend.surface_max_cad_projection_error_m, 0.0);
        assert_eq!(mesh.backend.volume_candidate_count, 1);
        assert!(mesh.backend.tet_candidate_count > 12);
        assert_eq!(mesh.backend.tet_fan_fallback_component_count, 0);
        assert_eq!(mesh.backend.tet_recovered_component_ratio, 1.0);
        assert!((mesh.backend.tet_candidate_volume_ratio - 1.0).abs() < 1.0e-12);
        let validation_options =
            production_validation_options(&preparation, &VolumeMeshingOptions::default());
        assert!(!validation_options.coverage_sample_points_m.is_empty());
        assert_eq!(validation_options.min_coverage_sample_ratio, 1.0);
        assert!(validation_options.require_no_unrepaired_exact_quality);
        assert!(validation_options
            .coverage_sample_points_m
            .iter()
            .all(|point| point.iter().all(|value| value.is_finite())));
        assert!(mesh.backend.tet_max_radius_edge_ratio.is_finite());
        assert!(mesh.backend.tet_min_exact_scaled_jacobian.is_finite());
        assert!(
            (mesh.backend.tet_min_exact_scaled_jacobian - mesh.quality.min_exact_scaled_jacobian)
                .abs()
                < 1.0e-12
        );
        assert_eq!(
            mesh.backend.tet_exact_scaled_jacobian_below_threshold_count,
            mesh.quality
                .elements
                .iter()
                .filter(|element| element.exact_scaled_jacobian
                    < QualityThresholds::default().min_scaled_jacobian)
                .count()
        );
        assert_eq!(
            mesh.backend.tet_exact_scaled_jacobian_below_threshold_count,
            0
        );
        assert_eq!(
            mesh.backend
                .tet_exact_scaled_jacobian_bins
                .values()
                .sum::<usize>(),
            mesh.backend.tet_candidate_count
        );
        assert!(!mesh.backend.tet_exact_scaled_jacobian_bins.is_empty());
        assert!(mesh.backend.tet_optimization_pass_count <= 2);
        assert!(
            mesh.backend.tet_smoothed_point_count <= mesh.backend.interior_seed_point_count * 2
        );
        assert!(mesh
            .backend
            .tet_optimization_initial_max_aspect_ratio
            .is_finite());
        assert!(mesh
            .backend
            .tet_optimization_final_max_aspect_ratio
            .is_finite());
        assert!(
            mesh.backend.tet_optimization_final_max_aspect_ratio
                <= mesh.backend.tet_optimization_initial_max_aspect_ratio + 1.0e-12
        );
        assert!(mesh
            .backend
            .tet_optimization_initial_min_exact_scaled_jacobian
            .is_finite());
        assert!(
            mesh.backend
                .tet_optimization_final_min_exact_scaled_jacobian
                + 1.0e-12
                >= mesh
                    .backend
                    .tet_optimization_initial_min_exact_scaled_jacobian
        );
        assert!(mesh.quality.min_exact_scaled_jacobian.is_finite());
        assert!(mesh
            .quality
            .elements
            .iter()
            .all(|element| element.exact_scaled_jacobian.is_finite()));
        assert_eq!(
            mesh.quality.max_boundary_projection_error_m,
            mesh.backend.surface_max_cad_projection_error_m
        );
        assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
        assert_eq!(mesh.backend.boundary_edge_recovery_ratio, 1.0);
        assert_eq!(
            mesh.provenance.algorithm,
            "production_topology_tet_candidate/v1"
        );
    }

    #[test]
    fn production_mesh_evidence_preserves_recovery_and_untangling_summary() {
        let options = VolumeMeshingOptions::default();
        let mut preparation = prepare_production_mesh(&cube_geometry(), &options)
            .expect("production preparation should generate");
        preparation.tet_candidates.recovery.untangling_pass_count = 2;
        preparation
            .tet_candidates
            .recovery
            .untangling_initial_near_singular_count = 6;
        preparation
            .tet_candidates
            .recovery
            .untangling_final_near_singular_count = 1;
        preparation
            .tet_candidates
            .recovery
            .untangling_relocated_seed_count = 3;
        preparation
            .tet_candidates
            .recovery
            .untangling_reconnected_edge_star_count = 4;
        preparation
            .tet_candidates
            .recovery
            .untangling_reconnected_boundary_adjacent_cavity_count = 5;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_repair_pass_count = 1;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_reconnected_cavity_count = 2;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_reconnection_quality_gain_count = 1;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_boundary_adjacent_reconnected_cavity_count = 3;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_face_neighbor_reconnected_cavity_count = 4;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_connected_reconnected_cavity_count = 5;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_expanded_connected_reconnected_cavity_count = 6;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_split_cavity_count = 7;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_seed_star_collapse_count = 8;
        preparation
            .tet_candidates
            .recovery
            .exact_quality_seed_star_relocation_count = 9;

        let mesh = analysis_mesh_from_preparation(&preparation, &options, None)
            .expect("production mesh should generate with recovery summary evidence");
        let evidence = crate::build_mesh_evidence_artifact(
            &mesh,
            &production_validation_options(&preparation, &options),
        );

        assert_eq!(mesh.backend.tet_untangling_pass_count, 2);
        assert_eq!(mesh.backend.tet_untangling_initial_near_singular_count, 6);
        assert_eq!(mesh.backend.tet_untangling_final_near_singular_count, 1);
        assert_eq!(mesh.backend.tet_untangling_relocated_seed_count, 3);
        assert_eq!(mesh.backend.tet_untangling_reconnected_edge_star_count, 4);
        assert_eq!(
            mesh.backend
                .tet_untangling_reconnected_boundary_adjacent_cavity_count,
            5
        );
        assert_eq!(mesh.backend.tet_exact_quality_repair_pass_count, 1);
        assert_eq!(mesh.backend.tet_exact_quality_reconnected_cavity_count, 2);
        assert_eq!(
            mesh.backend
                .tet_exact_quality_reconnection_quality_gain_count,
            1
        );
        assert_eq!(
            mesh.backend
                .tet_exact_quality_boundary_adjacent_reconnected_cavity_count,
            3
        );
        assert_eq!(
            mesh.backend
                .tet_exact_quality_face_neighbor_reconnected_cavity_count,
            4
        );
        assert_eq!(
            mesh.backend
                .tet_exact_quality_connected_reconnected_cavity_count,
            5
        );
        assert_eq!(
            mesh.backend
                .tet_exact_quality_expanded_connected_reconnected_cavity_count,
            6
        );
        assert_eq!(mesh.backend.tet_exact_quality_split_cavity_count, 7);
        assert_eq!(mesh.backend.tet_exact_quality_seed_star_collapse_count, 8);
        assert_eq!(mesh.backend.tet_exact_quality_seed_star_relocation_count, 9);
        assert_eq!(evidence.tet_recovery.untangling_pass_count, 2);
        assert_eq!(
            evidence.tet_recovery.untangling_initial_near_singular_count,
            6
        );
        assert_eq!(
            evidence.tet_recovery.untangling_final_near_singular_count,
            1
        );
        assert_eq!(evidence.tet_recovery.untangling_relocated_seed_count, 3);
        assert_eq!(
            evidence.tet_recovery.untangling_reconnected_edge_star_count,
            4
        );
        assert_eq!(
            evidence
                .tet_recovery
                .untangling_reconnected_boundary_adjacent_cavity_count,
            5
        );
        assert_eq!(evidence.tet_recovery.exact_quality_repair_pass_count, 1);
        assert_eq!(
            evidence.tet_recovery.exact_quality_reconnected_cavity_count,
            2
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_reconnection_quality_gain_count,
            1
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_boundary_adjacent_reconnected_cavity_count,
            3
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_face_neighbor_reconnected_cavity_count,
            4
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_connected_reconnected_cavity_count,
            5
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_expanded_connected_reconnected_cavity_count,
            6
        );
        assert_eq!(evidence.tet_recovery.exact_quality_split_cavity_count, 7);
        assert_eq!(
            evidence.tet_recovery.exact_quality_seed_star_collapse_count,
            8
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_seed_star_relocation_count,
            9
        );
    }

    #[test]
    fn production_mesh_preserves_boundary_and_material_region_provenance() {
        let options = VolumeMeshingOptions::default();
        let preparation = prepare_production_mesh(&cube_geometry(), &options)
            .expect("production preparation should generate");
        let mesh = analysis_mesh_from_preparation(&preparation, &options, None)
            .expect("production mesh should generate from multi-region fixture");
        let evidence = crate::build_mesh_evidence_artifact(
            &mesh,
            &production_validation_options(&preparation, &options),
        );

        assert!(mesh
            .boundary_faces
            .iter()
            .any(|face| face.region_ids.iter().any(|region| region == "root")));
        assert!(mesh
            .boundary_faces
            .iter()
            .any(|face| face.region_ids.iter().any(|region| region == "tip")));
        assert!(mesh
            .boundary_edges
            .iter()
            .any(|edge| edge.region_ids.iter().any(|region| region == "root")));
        assert!(mesh
            .boundary_edges
            .iter()
            .any(|edge| edge.region_ids.iter().any(|region| region == "tip")));
        assert!(mesh
            .volume_elements
            .iter()
            .any(|element| element.material_region_id == "root"));
        assert!(mesh
            .volume_elements
            .iter()
            .any(|element| element.material_region_id == "tip"));
        assert!(mesh.boundary_faces.iter().all(|face| {
            face.provenance.iter().any(|provenance| {
                provenance.source_entity_kind == SourceEntityKind::Face
                    && provenance.region_ids == face.region_ids
            })
        }));
        assert!(mesh.boundary_edges.iter().all(|edge| {
            edge.provenance.iter().any(|provenance| {
                provenance.source_entity_kind == SourceEntityKind::Edge
                    && provenance.region_ids == edge.region_ids
            })
        }));
        assert!(mesh.volume_elements.iter().all(|element| {
            element.provenance.iter().any(|provenance| {
                provenance.source_entity_kind == SourceEntityKind::Face
                    && provenance
                        .region_ids
                        .iter()
                        .any(|region| region == &element.material_region_id)
            })
        }));
        assert!(
            evidence
                .regions
                .boundary_region_face_counts
                .get("root")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .boundary_region_face_counts
                .get("tip")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .boundary_region_edge_counts
                .get("root")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .boundary_region_edge_counts
                .get("tip")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .material_region_element_counts
                .get("root")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .material_region_element_counts
                .get("tip")
                .copied()
                .unwrap_or_default()
                > 0
        );
    }

    #[test]
    fn production_mesh_preserves_regions_through_requested_refinement() {
        let mut options = VolumeMeshingOptions::default();
        options.target_size = MeshTargetSize::LengthM(0.75);
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        options.refinement.focus.interfaces = RefinementFocusLevel::Off;
        let sizing = MeshSizingField {
            samples: vec![
                SizingSample {
                    position_m: [0.25, 0.25, 0.25],
                    target_size_m: 0.25,
                    reason: Some("structural.load_regions".to_string()),
                },
                SizingSample {
                    position_m: [0.75, 0.75, 0.75],
                    target_size_m: 0.25,
                    reason: Some("structural.constraint_regions".to_string()),
                },
            ],
            ..MeshSizingField::default()
        };

        let coarse = generate_production_analysis_mesh(&cube_geometry(), &options)
            .expect("baseline production mesh should generate");
        let mesh =
            generate_production_analysis_mesh_with_sizing(&cube_geometry(), &options, &sizing)
                .expect("production mesh should generate with requested refinement");
        let validation = AnalysisMeshValidationOptions {
            required_boundary_region_ids: vec!["root".to_string(), "tip".to_string()],
            required_material_region_ids: vec!["root".to_string(), "tip".to_string()],
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            require_no_fan_fallback: true,
            require_no_unrepaired_exact_quality: true,
            ..AnalysisMeshValidationOptions::default()
        };
        crate::validate_analysis_mesh_with_options(&mesh, validation.clone())
            .expect("required regions should remain selectable after requested refinement");
        let evidence = crate::build_mesh_evidence_artifact(&mesh, &validation);

        assert_eq!(mesh.backend.tet_requested_refinement_point_count, 2);
        assert!(mesh.backend.tet_accepted_requested_refinement_point_count > 0);
        for sample in &sizing.samples {
            let reason = sample.reason.as_deref().expect("reasoned sizing sample");
            let coarse_near_count =
                volume_element_centroid_count_within(&coarse, sample.position_m, 0.30);
            let refined_near_count =
                volume_element_centroid_count_within(&mesh, sample.position_m, 0.30);
            assert!(
                refined_near_count > coarse_near_count,
                "{reason} should create denser retained Tet topology near the requested patch; coarse={coarse_near_count}, refined={refined_near_count}"
            );
            assert!(
                mesh.sizing.applied_samples.iter().any(|applied| {
                    applied.reason.as_deref() == Some(reason)
                        && applied.inserted_breakpoint_count > 0
                }),
                "{reason} should have an accepted requested Tet seed"
            );
        }
        assert!(
            mesh.sizing.applied_samples.iter().any(|sample| {
                sample.reason.as_deref() == Some("structural.load_regions")
                    && sample.inserted_breakpoint_count > 0
            }) || mesh.sizing.applied_samples.iter().any(|sample| {
                sample.reason.as_deref() == Some("structural.constraint_regions")
                    && sample.inserted_breakpoint_count > 0
            })
        );
        assert_eq!(
            evidence.sizing.requested_tet_refinement_point_count,
            mesh.backend.tet_requested_refinement_point_count
        );
        assert_eq!(
            evidence
                .sizing
                .accepted_requested_tet_refinement_point_count,
            mesh.backend.tet_accepted_requested_refinement_point_count
        );
        assert!(evidence
            .sizing
            .applied_by_reason
            .contains_key("structural.load_regions"));
        assert!(evidence
            .sizing
            .applied_by_reason
            .contains_key("structural.constraint_regions"));
        assert!(
            evidence
                .regions
                .boundary_region_recovered_face_counts
                .get("root")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .boundary_region_recovered_face_counts
                .get("tip")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .material_region_element_counts
                .get("root")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            evidence
                .regions
                .material_region_element_counts
                .get("tip")
                .copied()
                .unwrap_or_default()
                > 0
        );
    }

    fn volume_element_centroid_count_within(
        mesh: &AnalysisMeshArtifact,
        point_m: [f64; 3],
        radius_m: f64,
    ) -> usize {
        let radius_squared_m = radius_m * radius_m;
        mesh.volume_elements
            .iter()
            .filter(|element| {
                let Some(centroid) = analysis_volume_element_centroid(mesh, element) else {
                    return false;
                };
                distance_squared(centroid, point_m) <= radius_squared_m
            })
            .count()
    }

    fn analysis_volume_element_centroid(
        mesh: &AnalysisMeshArtifact,
        element: &AnalysisVolumeElement,
    ) -> Option<[f64; 3]> {
        if element.node_ids.is_empty() {
            return None;
        }
        let mut centroid = [0.0; 3];
        for node_id in &element.node_ids {
            let node = mesh.nodes.iter().find(|node| node.node_id == *node_id)?;
            centroid[0] += node.coordinates_m[0];
            centroid[1] += node.coordinates_m[1];
            centroid[2] += node.coordinates_m[2];
        }
        let scale = 1.0 / element.node_ids.len() as f64;
        Some([
            centroid[0] * scale,
            centroid[1] * scale,
            centroid[2] * scale,
        ])
    }

    #[test]
    fn production_sizing_maps_requested_ids_after_skipped_samples() {
        let mut options = VolumeMeshingOptions::default();
        options.target_size = MeshTargetSize::LengthM(0.75);
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        options.refinement.focus.interfaces = RefinementFocusLevel::Off;
        let sizing = MeshSizingField {
            samples: vec![
                SizingSample {
                    position_m: [0.10, 0.10, 0.10],
                    target_size_m: f64::NAN,
                    reason: Some("invalid".to_string()),
                },
                SizingSample {
                    position_m: [0.25, 0.25, 0.25],
                    target_size_m: 0.25,
                    reason: Some("structural.load_regions".to_string()),
                },
            ],
            ..MeshSizingField::default()
        };

        let mesh =
            generate_production_analysis_mesh_with_sizing(&cube_geometry(), &options, &sizing)
                .expect("production mesh should generate with a skipped sample");

        assert!(mesh.sizing.applied_samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("structural.load_regions")
                && sample.inserted_breakpoint_count > 0
        }));
        assert!(mesh.sizing.rejected_samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("invalid") && sample.status == "skipped_invalid"
        }));
    }

    #[test]
    fn production_sizing_reports_unaccepted_requested_samples_as_rejections() {
        let mut options = VolumeMeshingOptions::default();
        options.target_size = MeshTargetSize::LengthM(0.75);
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        options.refinement.focus.interfaces = RefinementFocusLevel::Off;
        let sizing = MeshSizingField {
            samples: vec![
                SizingSample {
                    position_m: [0.25, 0.25, 0.25],
                    target_size_m: 0.25,
                    reason: Some("structural.load_regions".to_string()),
                },
                SizingSample {
                    position_m: [2.0, 2.0, 2.0],
                    target_size_m: 0.25,
                    reason: Some("structural.constraint_regions".to_string()),
                },
            ],
            ..MeshSizingField::default()
        };

        let mesh =
            generate_production_analysis_mesh_with_sizing(&cube_geometry(), &options, &sizing)
                .expect("production mesh should generate with a rejected requested sample");
        let evidence =
            crate::build_mesh_evidence_artifact(&mesh, &AnalysisMeshValidationOptions::default());

        assert!(mesh.sizing.applied_samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("structural.load_regions")
                && sample.inserted_breakpoint_count > 0
        }));
        assert!(mesh.sizing.rejected_samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("structural.constraint_regions")
                && sample.status == "rejected_by_recovery"
        }));
        assert_eq!(
            evidence
                .sizing
                .rejected_by_reason
                .get("structural.constraint_regions"),
            Some(&1)
        );
        assert_eq!(
            evidence
                .sizing
                .rejected_by_status
                .get("rejected_by_recovery"),
            Some(&1)
        );
        assert_eq!(
            evidence
                .sizing
                .requested_tet_refinement_rejected_by_reason
                .get("outside_volume"),
            Some(&1)
        );
    }

    #[test]
    fn production_sizing_bounds_requested_sample_ids_to_seed_budget() {
        let mut samples = Vec::<SizingSample>::new();
        for index in 0..17 {
            samples.push(SizingSample {
                position_m: [
                    0.10 + 0.02 * index as f64,
                    0.25 + 0.01 * index as f64,
                    0.25 + 0.005 * index as f64,
                ],
                target_size_m: 0.25,
                reason: Some(format!("requested.marker.{index}")),
            });
        }
        let sizing = MeshSizingField {
            samples,
            ..MeshSizingField::default()
        };

        let requested_ids = requested_sizing_sample_ids(&sizing);

        assert_eq!(requested_ids.len(), 16);
        assert_eq!(requested_ids.get(&0), Some(&0));
        assert_eq!(requested_ids.get(&15), Some(&15));
        assert_eq!(requested_ids.get(&16), None);
    }

    #[test]
    fn production_sizing_includes_cad_curvature_samples() {
        let mut options = VolumeMeshingOptions::default();
        options.target_size = MeshTargetSize::LengthM(0.5);
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = true;

        let mesh =
            generate_production_analysis_mesh(&cube_geometry_with_curvature_evaluator(), &options)
                .expect("production mesh should generate with CAD curvature sizing");

        assert!(mesh.backend.cad_curvature_query_count > 0);
        assert!(mesh.backend.cad_max_curvature_estimate_1_per_m > 0.0);
        assert!(mesh
            .sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("cad.curvature")
                && sample.target_size_m < 0.5));
        assert!(mesh.sizing.applied_samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("cad.curvature")
                && sample
                    .detail
                    .as_deref()
                    .is_some_and(|detail| detail.starts_with("production_requested_tet_seed_"))
        }));
        let evidence = crate::evidence::build_mesh_evidence_artifact(
            &mesh,
            &AnalysisMeshValidationOptions::default(),
        );
        assert!(evidence.sizing.generated_cad_sample_count > 0);
        assert_eq!(
            evidence.sizing.generated_cad_by_reason.get("cad.curvature"),
            Some(
                &mesh
                    .sizing
                    .samples
                    .iter()
                    .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
                    .count()
            )
        );
        assert_eq!(
            evidence.sizing.applied_by_reason.get("cad.curvature"),
            Some(
                &mesh
                    .sizing
                    .applied_samples
                    .iter()
                    .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
                    .count()
            )
        );
        let curvature_inserted = evidence
            .sizing
            .inserted_breakpoint_by_reason
            .get("cad.curvature")
            .copied()
            .unwrap_or_default();
        let curvature_uninserted = evidence
            .sizing
            .uninserted_sample_by_reason
            .get("cad.curvature")
            .copied()
            .unwrap_or_default();
        assert_eq!(
            curvature_inserted + curvature_uninserted,
            mesh.sizing
                .applied_samples
                .iter()
                .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
                .count()
        );
        assert!(mesh.backend.tet_requested_refinement_point_count > 0);
    }

    #[test]
    fn production_preparation_consumes_live_cad_provider_for_surface_and_sizing() {
        #[derive(Debug)]
        struct LiveProvider;

        impl CadFaceEvaluatorProvider for LiveProvider {
            fn evaluate_face(
                &self,
                request: &crate::cad_eval::CadFaceEvaluationRequest<'_>,
            ) -> Vec<CadFaceEvaluationSample> {
                assert_eq!(request.imported_face_id, Some(1));
                assert_eq!(request.evaluator_id, Some("cad_face_1"));
                assert!(request.supports_projection);
                assert!(request.supports_derivatives);
                assert!(request.supports_curvature);
                vec![
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::BackendQuery,
                        point_m: [0.0, 0.0, 1.0],
                        uv: Some([0.0, 0.0]),
                        projected_point_m: Some([0.0, 0.0, 1.0]),
                        unit_normal: Some([0.0, 0.0, 1.0]),
                        projection_error_m: Some(0.0),
                    },
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::TessellationEstimate,
                        point_m: [0.75, 0.25, 1.0],
                        uv: Some([0.75, 0.25]),
                        projected_point_m: Some([0.75, 0.25, 1.0]),
                        unit_normal: Some([0.0, 0.8, 0.6]),
                        projection_error_m: Some(0.0),
                    },
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::TessellationEstimate,
                        point_m: [0.90, 0.70, 1.0],
                        uv: Some([0.90, 0.70]),
                        projected_point_m: Some([0.90, 0.70, 1.0]),
                        unit_normal: Some([0.8, 0.0, 0.6]),
                        projection_error_m: Some(0.0),
                    },
                ]
            }
        }

        let mut options = VolumeMeshingOptions::default();
        options.target_size = MeshTargetSize::LengthM(0.5);
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = true;
        options.refinement.focus.small_features = false;
        options.refinement.focus.interfaces = RefinementFocusLevel::Off;
        let mut geometry = cube_geometry_with_curvature_evaluator();
        geometry.source_geometry.cad_evaluators[0].faces[0]
            .evaluation_samples
            .clear();

        let preparation =
            prepare_production_mesh_with_cad_evaluator(&geometry, &options, &LiveProvider)
                .expect("production preparation should consume live CAD provider");
        let mesh = analysis_mesh_from_preparation(&preparation, &options, None)
            .expect("production mesh should preserve live CAD evidence");
        let evidence = crate::build_mesh_evidence_artifact(
            &mesh,
            &production_validation_options(&preparation, &options),
        );
        let curvature_samples = preparation
            .effective_sizing
            .as_ref()
            .expect("live curvature should create sizing")
            .samples
            .iter()
            .filter(|sample| sample.reason.as_deref() == Some("cad.curvature"))
            .collect::<Vec<_>>();

        assert_eq!(
            preparation.cad_evaluation.source,
            CadEvaluationSource::ParametricCad
        );
        assert!(preparation.cad_evaluation_report.live_query_face_count > 0);
        assert_eq!(
            preparation.cad_evaluation_report.live_query_face_count,
            preparation.cad_evaluation_report.exact_query_face_count
        );
        assert_eq!(
            preparation
                .cad_evaluation_report
                .projection_supported_face_count,
            preparation.cad_evaluation_report.live_query_face_count
        );
        assert_eq!(
            preparation
                .cad_evaluation_report
                .normal_supported_face_count,
            preparation.cad_evaluation_report.live_query_face_count
        );
        assert_eq!(
            preparation
                .cad_evaluation_report
                .derivative_supported_face_count,
            preparation.cad_evaluation_report.live_query_face_count
        );
        assert_eq!(
            preparation
                .cad_evaluation_report
                .curvature_supported_face_count,
            preparation.cad_evaluation_report.live_query_face_count
        );
        assert_eq!(
            preparation
                .cad_evaluation_report
                .missing_exact_query_face_count,
            0
        );
        assert!(preparation.cad_evaluation_report.derivative_query_count > 0);
        assert!(preparation.cad_evaluation_report.curvature_query_count > 0);
        assert!(preparation.cad_evaluation_report.uv_domain_face_count > 0);
        assert!(preparation
            .surface
            .elements
            .iter()
            .any(|element| element.cad_face_id.is_some()));
        assert!(!curvature_samples.is_empty());
        assert!(curvature_samples
            .iter()
            .all(|sample| sample.target_size_m < 0.5));
        assert_eq!(mesh.backend.cad_evaluation_source, "parametric_cad");
        assert_eq!(
            mesh.backend.cad_evaluation_live_query_face_count,
            preparation.cad_evaluation_report.live_query_face_count
        );
        assert_eq!(
            mesh.backend.cad_evaluation_projection_supported_face_count,
            preparation
                .cad_evaluation_report
                .projection_supported_face_count
        );
        assert_eq!(
            mesh.backend.cad_evaluation_curvature_supported_face_count,
            preparation
                .cad_evaluation_report
                .curvature_supported_face_count
        );
        assert_eq!(
            mesh.backend.cad_evaluation_missing_exact_query_face_count,
            0
        );
        assert_eq!(
            mesh.backend.cad_uv_domain_face_count,
            preparation.cad_evaluation_report.uv_domain_face_count
        );
        assert_eq!(
            mesh.backend.cad_uv_projection_out_of_bounds_count,
            preparation
                .cad_evaluation_report
                .uv_projection_out_of_bounds_count
        );
        assert_eq!(
            mesh.backend.surface_cad_face_count,
            surface_cad_face_count(&preparation.surface)
        );
        assert_eq!(
            mesh.backend.surface_exact_cad_sample_node_count,
            preparation.surface.exact_cad_sample_node_count
        );
        assert_eq!(
            mesh.backend.surface_rejected_exact_cad_sample_count,
            preparation.surface.rejected_exact_cad_sample_count
        );
        assert_eq!(
            mesh.backend.surface_max_cad_projection_error_m,
            surface_max_cad_projection_error_m(&preparation.surface)
        );
        assert_eq!(
            evidence.cad.evaluation_source,
            mesh.backend.cad_evaluation_source
        );
        assert_eq!(
            evidence.cad.live_query_face_count,
            preparation.cad_evaluation_report.live_query_face_count
        );
        assert_eq!(
            evidence.cad.exact_query_face_count,
            preparation.cad_evaluation_report.exact_query_face_count
        );
        assert_eq!(
            evidence.cad.projection_supported_face_count,
            preparation
                .cad_evaluation_report
                .projection_supported_face_count
        );
        assert_eq!(
            evidence.cad.curvature_supported_face_count,
            preparation
                .cad_evaluation_report
                .curvature_supported_face_count
        );
        assert_eq!(
            evidence.cad.surface_exact_cad_sample_node_count,
            preparation.surface.exact_cad_sample_node_count
        );
        assert_eq!(
            evidence.cad.surface_rejected_exact_cad_sample_count,
            preparation.surface.rejected_exact_cad_sample_count
        );
        assert_eq!(
            evidence.cad.surface_max_projection_error_m,
            surface_max_cad_projection_error_m(&preparation.surface)
        );
        assert!(mesh
            .sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("cad.curvature")));
    }

    #[test]
    fn production_sizing_includes_small_feature_edge_and_proximity_samples() {
        let mut options = VolumeMeshingOptions::default();
        options.target_size = MeshTargetSize::LengthM(0.5);
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = true;
        options.refinement.focus.interfaces = RefinementFocusLevel::Off;

        let geometry = thin_box_geometry();
        let topology = extract_source_topology(&geometry).expect("topology");
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
        let cad_evaluation = crate::cad_eval::build_cad_evaluation_model(&cad_topology, &topology)
            .expect("cad evaluation");
        let sizing = production_effective_sizing(&topology, &cad_evaluation, &options, None)
            .expect("feature-edge sizing");

        assert!(sizing.samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("cad.feature_edge") && sample.target_size_m < 0.1
        }));
        assert!(sizing.samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("cad.proximity") && sample.target_size_m < 0.1
        }));
        assert!(requested_refinement_points(Some(&sizing)).1 > 4);
    }

    #[test]
    fn production_sizing_consumes_valid_anisotropic_samples_conservatively() {
        let mut options = VolumeMeshingOptions::default();
        options.target_size = MeshTargetSize::LengthM(0.5);
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        options.refinement.focus.interfaces = RefinementFocusLevel::Off;
        let sizing = MeshSizingField {
            anisotropic_samples: vec![
                AnisotropicSizingSample {
                    position_m: [0.25, 0.25, 0.25],
                    target_sizes_m: [0.05, 0.2, 0.4],
                    directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    reason: Some("boundary_layer".to_string()),
                },
                AnisotropicSizingSample {
                    position_m: [0.75, 0.75, 0.75],
                    target_sizes_m: [0.05, -0.2, 0.4],
                    directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    reason: Some("cad.proximity".to_string()),
                },
            ],
            ..MeshSizingField::default()
        };
        let geometry = cube_geometry();
        let topology = extract_source_topology(&geometry).expect("topology");
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
        let cad_evaluation = crate::cad_eval::build_cad_evaluation_model(&cad_topology, &topology)
            .expect("cad evaluation");

        let effective =
            production_effective_sizing(&topology, &cad_evaluation, &options, Some(&sizing))
                .expect("anisotropic sizing should produce an effective sizing field");

        assert_eq!(effective.anisotropic_samples.len(), 2);
        assert!(effective.samples.iter().any(|sample| {
            sample.reason.as_deref() == Some("boundary_layer")
                && sample.position_m == [0.25, 0.25, 0.25]
                && (sample.target_size_m - 0.05).abs() <= 1.0e-12
        }));
        assert!(!effective
            .samples
            .iter()
            .any(|sample| sample.position_m == [0.75, 0.75, 0.75]));
        assert_eq!(requested_refinement_points(Some(&effective)).1, 1);
        assert_eq!(production_sizing_target_size(0.5, &sizing, &options), 0.05);
    }

    #[test]
    fn production_sizing_includes_interface_samples_by_focus_level() {
        let mut options = VolumeMeshingOptions::default();
        options.refinement.strategy = crate::options::RefinementStrategy::Adaptive;
        options.refinement.focus.curvature = false;
        options.refinement.focus.small_features = false;
        options.refinement.focus.interfaces = RefinementFocusLevel::Normal;
        let geometry = cube_geometry();
        let topology = extract_source_topology(&geometry).expect("topology");
        let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
        let cad_evaluation = crate::cad_eval::build_cad_evaluation_model(&cad_topology, &topology)
            .expect("cad evaluation");

        let normal = production_effective_sizing(&topology, &cad_evaluation, &options, None)
            .expect("normal interface sizing");
        let normal_targets = normal
            .samples
            .iter()
            .filter(|sample| sample.reason.as_deref() == Some("cad.interface"))
            .map(|sample| sample.target_size_m)
            .collect::<Vec<_>>();
        assert!(!normal_targets.is_empty());
        assert!(normal_targets
            .iter()
            .all(|target| (*target - 0.5).abs() < 1.0e-12));

        options.refinement.focus.interfaces = RefinementFocusLevel::Fine;
        let fine = production_effective_sizing(&topology, &cad_evaluation, &options, None)
            .expect("fine interface sizing");
        let fine_targets = fine
            .samples
            .iter()
            .filter(|sample| sample.reason.as_deref() == Some("cad.interface"))
            .map(|sample| sample.target_size_m)
            .collect::<Vec<_>>();
        assert_eq!(fine_targets.len(), normal_targets.len());
        assert!(fine_targets
            .iter()
            .all(|target| (*target - 0.25).abs() < 1.0e-12));

        options.refinement.focus.interfaces = RefinementFocusLevel::Off;
        assert!(production_effective_sizing(&topology, &cad_evaluation, &options, None).is_none());
    }

    #[test]
    fn quality_report_summarizes_boundary_projection_error() {
        let quality = quality_report(
            vec![ElementQuality {
                element_id: "tet_1".to_string(),
                scaled_jacobian: 0.5,
                exact_scaled_jacobian: 0.45,
                aspect_ratio: 2.0,
                volume_m3: 1.0,
            }],
            [2.0e-6, f64::NAN, 4.0e-6],
        );

        assert_eq!(quality.mean_boundary_projection_error_m, 3.0e-6);
        assert_eq!(quality.max_boundary_projection_error_m, 4.0e-6);
    }

    #[test]
    fn production_validation_rejects_surface_projection_error() {
        let mut preparation =
            prepare_production_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
                .expect("production preparation should run");
        preparation.surface.elements[0].max_projection_error_m = 2.0e-6;

        let mut options = VolumeMeshingOptions::default();
        options.validation.quality.max_boundary_projection_error_m = 1.0e-6;
        let err = analysis_mesh_from_preparation(&preparation, &options, None)
            .expect_err("projection error should fail production validation");

        assert!(matches!(
            err,
            ProductionMeshError::Validation(
                crate::AnalysisMeshValidationError::QualityThresholdFailed { reason }
            ) if reason == "max_boundary_projection_error_m"
        ));
    }

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_production_cube".to_string(),
            source: GeometrySource {
                path: "/fixtures/generic_cube.step".to_string(),
                sha256: "generic-cube".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 1,
            meshes: vec![MeshDescriptor {
                mesh_id: "cube_surface".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 8,
                element_count: 12,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "cube_surface",
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                vec![
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
            )],
            regions: vec![
                Region {
                    region_id: "root".to_string(),
                    name: "root".to_string(),
                    tag: None,
                    cad_ownership: None,
                },
                Region {
                    region_id: "tip".to_string(),
                    name: "tip".to_string(),
                    tag: None,
                    cad_ownership: None,
                },
            ],
            region_entity_mappings: vec![
                RegionEntityMapping::new(
                    "root",
                    "cube_surface",
                    EntityKind::Face,
                    vec![EntityIdRange::new(0, 6)],
                ),
                RegionEntityMapping::new(
                    "tip",
                    "cube_surface",
                    EntityKind::Face,
                    vec![EntityIdRange::new(6, 6)],
                ),
            ],
            diagnostics: Vec::new(),
        }
    }

    fn cube_geometry_with_curvature_evaluator() -> GeometryAsset {
        let mut geometry = cube_geometry();
        geometry.regions.push(Region {
            region_id: "curved_face".to_string(),
            name: "curved_face".to_string(),
            tag: Some("cad_face".to_string()),
            cad_ownership: Some(CadRegionOwnership {
                face_id: Some(1),
                label: Some(CadLabelRef {
                    label_entry: "0:1:1".to_string(),
                    name: "curved_face".to_string(),
                    kind: CadSemanticKind::Face,
                }),
                owner_path: Vec::new(),
                layers: Vec::new(),
                color: None,
                material: None,
            }),
        });
        geometry
            .region_entity_mappings
            .push(RegionEntityMapping::new(
                "curved_face",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(2, 2)],
            ));
        geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
            evaluator_id: "cad_evaluator_test".to_string(),
            backend: "test".to_string(),
            format_name: "step".to_string(),
            requires_source_geometry: true,
            faces: vec![CadFaceEvaluator {
                evaluator_id: "cad_face_1".to_string(),
                imported_face_id: 1,
                name: "curved_face".to_string(),
                supports_point_evaluation: true,
                supports_projection: true,
                supports_normal: true,
                supports_derivatives: true,
                supports_curvature: true,
                reference_point_m: Some([0.5, 0.5, 1.0]),
                reference_unit_normal: Some([0.0, 0.0, 1.0]),
                evaluation_samples: vec![
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::BackendQuery,
                        point_m: [0.0, 0.0, 1.0],
                        uv: Some([0.0, 0.0]),
                        projected_point_m: Some([0.0, 0.0, 1.0]),
                        unit_normal: Some([0.0, 0.0, 1.0]),
                        projection_error_m: Some(0.0),
                    },
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::BackendQuery,
                        point_m: [1.0, 0.0, 1.0],
                        uv: Some([1.0, 0.0]),
                        projected_point_m: Some([1.0, 0.0, 1.0]),
                        unit_normal: Some([0.0, 0.9, 0.4358898943540673]),
                        projection_error_m: Some(0.0),
                    },
                    CadFaceEvaluationSample {
                        source: CadFaceEvaluationSampleSource::BackendQuery,
                        point_m: [0.0, 1.0, 1.0],
                        uv: Some([0.0, 1.0]),
                        projected_point_m: Some([0.0, 1.0, 1.0]),
                        unit_normal: Some([0.9, 0.0, 0.4358898943540673]),
                        projection_error_m: Some(0.0),
                    },
                ],
            }],
            curves: Vec::new(),
        }];
        geometry
    }

    fn thin_box_geometry() -> GeometryAsset {
        let mut geometry = cube_geometry();
        geometry.geometry_id = "geo_production_thin_box".to_string();
        geometry.source.sha256 = "generic-thin-box".to_string();
        if let Some(surface) = geometry.surface_meshes.first_mut() {
            for vertex in &mut surface.vertices {
                if vertex[2] > 0.0 {
                    vertex[2] = 0.1;
                }
            }
        }
        geometry
    }
}
