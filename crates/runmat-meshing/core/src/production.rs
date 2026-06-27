use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::{
        AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
        AnalysisVolumeElement, MeshBackendSummary, ANALYSIS_MESH_SCHEMA_VERSION,
    },
    cad_eval::{
        build_cad_evaluation_model, summarize_cad_evaluation, CadEvaluationError,
        CadEvaluationModel, CadEvaluationReport, CadEvaluationSource,
    },
    cad_topology::{build_cad_topology, CadTopologyError, CadTopologyModel, CadTopologySource},
    curve::{
        discretize_topology_curves, CurveDiscretization, CurveDiscretizationError,
        CurveDiscretizationOptions,
    },
    options::{MeshTargetSize, VolumeMeshingOptions},
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind},
    quality::{AnalysisMeshQualityReport, ElementQuality},
    sizing::MeshSizingField,
    source_topology::{extract_source_topology, SourceTopologyError, SourceTopologyModel},
    surface::{
        discretize_cad_surfaces, SurfaceDiscretization, SurfaceDiscretizationError,
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
    let topology = extract_source_topology(geometry).map_err(ProductionMeshError::Topology)?;
    let cad_topology =
        build_cad_topology(geometry, &topology).map_err(ProductionMeshError::CadTopology)?;
    let cad_evaluation = build_cad_evaluation_model(&cad_topology, &topology)
        .map_err(ProductionMeshError::CadEvaluation)?;
    let cad_evaluation_report = summarize_cad_evaluation(&cad_evaluation, &topology)
        .map_err(ProductionMeshError::CadEvaluation)?;
    let curves = discretize_topology_curves(&topology, curve_options_for_mesh(&topology, options))
        .map_err(ProductionMeshError::Curve)?;
    let surface = discretize_cad_surfaces(
        &topology,
        &cad_evaluation,
        SurfaceDiscretizationOptions::default(),
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
        tet_candidate_options_for_mesh(&topology, options),
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
    })
}

pub fn generate_production_analysis_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, ProductionMeshError> {
    let preparation = prepare_production_mesh(geometry, options)?;
    analysis_mesh_from_preparation(&preparation, options)
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
) -> TetCandidateOptions {
    TetCandidateOptions {
        interior_target_size_m: Some(target_size_for_mesh(topology, options)),
        max_interior_seed_points: options.max_elements.max(1).min(128),
        allow_fan_fallback: false,
        max_refinement_passes: match options.refinement.strategy {
            crate::options::RefinementStrategy::None => 0,
            _ => 2,
        },
        max_radius_edge_ratio: 2.5,
        sizing_compliance_tolerance: 0.35,
        max_optimization_passes: match options.refinement.strategy {
            crate::options::RefinementStrategy::None => 0,
            _ => 2,
        },
        smoothing_relaxation: 0.30,
        sliver_aspect_ratio: 20.0,
        ..TetCandidateOptions::default()
    }
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
    target_size_m
}

fn analysis_mesh_from_preparation(
    preparation: &ProductionMeshPreparation,
    options: &VolumeMeshingOptions,
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
        quality_elements.push(ElementQuality {
            element_id,
            scaled_jacobian: (1.0 / tet.aspect_ratio.max(1.0)).min(1.0),
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
            Ok(AnalysisBoundaryFace {
                face_id: format!("prod_boundary_{}", element.element_id + 1),
                kind: BoundaryElementKind::Tri3,
                node_ids,
                adjacent_volume_element_ids: source_surface_to_tet
                    .get(&element.element_id)
                    .cloned()
                    .unwrap_or_default(),
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
        quality: quality_report(quality_elements),
        sizing: MeshSizingField {
            global_target_size_m: match options.target_size {
                MeshTargetSize::LengthM(length) => Some(length),
                MeshTargetSize::Auto => None,
            },
            ..MeshSizingField::default()
        },
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
        cad_projection_query_count: preparation.cad_evaluation_report.projection_query_count,
        cad_max_projection_error_m: preparation.cad_evaluation_report.max_projection_error_m,
        cad_max_normal_deviation: preparation.cad_evaluation_report.max_normal_deviation,
        curve_element_count: preparation.curves.elements.len(),
        surface_element_count: preparation.surface.elements.len(),
        surface_source_edge_loop_count: preparation.surface_validation.source_edge_loop_count,
        surface_closed_edge_loop_count: preparation
            .surface_validation
            .closed_source_edge_loop_count,
        surface_projection_error_m: preparation.surface_validation.max_projection_error_m,
        surface_face_coverage_ratio: preparation.surface_validation.face_coverage_ratio,
        surface_cad_face_count: surface_cad_face_count(&preparation.surface),
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
        tet_max_radius_edge_ratio: preparation.tet_candidates.recovery.max_radius_edge_ratio,
        tet_sizing_violation_count: preparation.tet_candidates.recovery.sizing_violation_count,
        tet_optimization_pass_count: preparation.tet_candidates.recovery.optimization_pass_count,
        tet_smoothed_point_count: preparation.tet_candidates.recovery.smoothed_point_count,
        tet_sliver_candidate_count: preparation.tet_candidates.recovery.sliver_candidate_count,
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
        ..AnalysisMeshValidationOptions::default()
    }
}

fn quality_report(elements: Vec<ElementQuality>) -> AnalysisMeshQualityReport {
    if elements.is_empty() {
        return AnalysisMeshQualityReport::default();
    }
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
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
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: 0,
        mean_boundary_projection_error_m: 0.0,
        max_boundary_projection_error_m: 0.0,
        elements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
        RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
        UnitSystem,
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
        assert_eq!(preparation.cad_evaluation_report.projection_query_count, 36);
        assert_eq!(
            preparation.cad_evaluation_report.max_projection_error_m,
            0.0
        );
        assert_eq!(preparation.surface.elements.len(), 12);
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
        assert_eq!(preparation.surface_validation.source_edge_loop_count, 18);
        assert_eq!(
            preparation.surface_validation.closed_source_edge_loop_count,
            18
        );
        assert_eq!(preparation.surface_validation.face_coverage_ratio, 1.0);
        assert_eq!(preparation.surface_recovery.surface_element_count, 12);
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
        let mesh =
            generate_production_analysis_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
                .expect("production mesh should generate from candidates");

        crate::validate_analysis_mesh(&mesh, crate::QualityThresholds::default())
            .expect("production candidate mesh should validate");
        assert!(mesh.nodes.len() > 9);
        assert!(mesh.volume_elements.len() > 12);
        assert_eq!(mesh.boundary_faces.len(), 12);
        assert_eq!(mesh.boundary_edges.len(), 18);
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
        assert_eq!(mesh.backend.cad_projection_query_count, 36);
        assert_eq!(mesh.backend.cad_max_projection_error_m, 0.0);
        assert_eq!(mesh.backend.surface_element_count, 12);
        assert_eq!(mesh.backend.surface_source_edge_loop_count, 18);
        assert_eq!(mesh.backend.surface_closed_edge_loop_count, 18);
        assert_eq!(mesh.backend.surface_face_coverage_ratio, 1.0);
        assert_eq!(mesh.backend.surface_cad_face_count, 12);
        assert_eq!(mesh.backend.surface_max_cad_projection_error_m, 0.0);
        assert_eq!(mesh.backend.volume_candidate_count, 1);
        assert!(mesh.backend.tet_candidate_count > 12);
        assert_eq!(mesh.backend.tet_fan_fallback_component_count, 0);
        assert_eq!(mesh.backend.tet_recovered_component_ratio, 1.0);
        assert!((mesh.backend.tet_candidate_volume_ratio - 1.0).abs() < 1.0e-12);
        assert!(mesh.backend.tet_max_radius_edge_ratio.is_finite());
        assert!(mesh.backend.tet_optimization_pass_count <= 2);
        assert!(
            mesh.backend.tet_smoothed_point_count <= mesh.backend.interior_seed_point_count * 2
        );
        assert_eq!(mesh.backend.boundary_face_recovery_ratio, 1.0);
        assert_eq!(mesh.backend.boundary_edge_recovery_ratio, 1.0);
        assert_eq!(
            mesh.provenance.algorithm,
            "production_topology_tet_candidate/v1"
        );
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
}
