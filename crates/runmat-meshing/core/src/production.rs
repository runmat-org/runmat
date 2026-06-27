use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    artifact::{
        AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode, AnalysisVolumeElement,
        ANALYSIS_MESH_SCHEMA_VERSION,
    },
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
        discretize_topology_surfaces, SurfaceDiscretization, SurfaceDiscretizationError,
        SurfaceDiscretizationOptions,
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
    pub topology: SourceTopologyModel,
    pub curves: CurveDiscretization,
    pub surface: SurfaceDiscretization,
    pub volume_candidates: VolumeCandidateSet,
    pub tet_candidates: TetCandidateSet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionMeshError {
    Topology(SourceTopologyError),
    Curve(CurveDiscretizationError),
    Surface(SurfaceDiscretizationError),
    VolumeCandidate(VolumeCandidateError),
    TetCandidate(TetCandidateError),
    Validation(AnalysisMeshValidationError),
    MissingCandidateNode { node_id: u32 },
}

impl std::fmt::Display for ProductionMeshError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(err) => write!(formatter, "source topology extraction failed: {err}"),
            Self::Curve(err) => write!(formatter, "curve discretization failed: {err}"),
            Self::Surface(err) => write!(formatter, "surface discretization failed: {err}"),
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
    let curves = discretize_topology_curves(&topology, curve_options_for_mesh(&topology, options))
        .map_err(ProductionMeshError::Curve)?;
    let surface = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
        .map_err(ProductionMeshError::Surface)?;
    let volume_candidates = prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
        .map_err(ProductionMeshError::VolumeCandidate)?;
    let tet_candidates = form_tet_candidates(
        &surface,
        &volume_candidates,
        tet_candidate_options_for_mesh(&topology, options),
    )
    .map_err(ProductionMeshError::TetCandidate)?;

    Ok(ProductionMeshPreparation {
        topology,
        curves,
        surface,
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
        max_interior_seed_points: options.max_elements.max(1).min(4096),
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
    let nodes = preparation
        .tet_candidates
        .nodes
        .iter()
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

    let mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: format!("production_{}", preparation.topology.mesh_id),
        nodes,
        volume_elements,
        boundary_faces,
        boundary_edges: Vec::new(),
        quality: quality_report(quality_elements),
        sizing: MeshSizingField {
            global_target_size_m: match options.target_size {
                MeshTargetSize::LengthM(length) => Some(length),
                MeshTargetSize::Auto => None,
            },
            ..MeshSizingField::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "production_topology_tet_candidate/v1".to_string(),
            source_geometry_id: preparation.topology.source_geometry_id.clone(),
            source_geometry_revision: preparation.topology.source_geometry_revision,
            source_geometry_sha256: preparation.topology.source_geometry_sha256.clone(),
        },
    };
    validate_analysis_mesh_with_options(&mesh, production_validation_options(preparation))
        .map_err(ProductionMeshError::Validation)?;
    Ok(mesh)
}

fn production_validation_options(
    preparation: &ProductionMeshPreparation,
) -> AnalysisMeshValidationOptions {
    AnalysisMeshValidationOptions {
        expected_bounds_m: Some([
            preparation.topology.bounds_min_m,
            preparation.topology.bounds_max_m,
        ]),
        expected_volume_m3: Some(preparation.volume_candidates.total_volume_m3),
        expected_boundary_area_m2: Some(preparation.volume_candidates.total_surface_area_m2),
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
        assert_eq!(preparation.surface.elements.len(), 12);
        assert_eq!(preparation.volume_candidates.components.len(), 1);
        assert_eq!(preparation.tet_candidates.tets.len(), 12);
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
        assert_eq!(mesh.nodes.len(), 9);
        assert_eq!(mesh.volume_elements.len(), 12);
        assert_eq!(mesh.boundary_faces.len(), 12);
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
