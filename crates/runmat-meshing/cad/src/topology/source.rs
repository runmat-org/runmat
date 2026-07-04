use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{CadFaceEvaluationSample, GeometryAsset};
use serde::{Deserialize, Serialize};

use super::SourceTopologyModel;

mod semantics;

use semantics::{
    cad_topology_source, evaluator_faces_by_imported_id, face_regions_by_source_face,
    imported_face_id_for_regions, imported_face_ids_by_region, semantic_face_regions,
    stable_face_id,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadTopologySource {
    SemanticCad,
    GenericCadMesh,
    MeshFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CadEntityKind {
    Vertex,
    Edge,
    Face,
    Shell,
    Volume,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CadEntityId {
    pub kind: CadEntityKind,
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadVertex {
    pub entity_id: CadEntityId,
    pub source_vertex_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadEdge {
    pub entity_id: CadEntityId,
    pub source_edge_id: u32,
    pub vertex_ids: [String; 2],
    pub adjacent_face_ids: Vec<String>,
    pub length_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadFace {
    pub entity_id: CadEntityId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_face_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_id: Option<String>,
    #[serde(default)]
    pub evaluator_supports_point_evaluation: bool,
    #[serde(default)]
    pub evaluator_supports_projection: bool,
    #[serde(default)]
    pub evaluator_supports_normal: bool,
    #[serde(default)]
    pub evaluator_supports_derivatives: bool,
    #[serde(default)]
    pub evaluator_supports_curvature: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_reference_point_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_unit_normal: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluator_samples: Vec<CadFaceEvaluationSample>,
    pub source_face_ids: Vec<u32>,
    pub source_edge_ids: Vec<u32>,
    pub loop_edge_ids: Vec<String>,
    pub region_ids: Vec<String>,
    pub area_m2: f64,
    pub unit_normal: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadShell {
    pub entity_id: CadEntityId,
    pub face_ids: Vec<String>,
    pub closed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadVolume {
    pub entity_id: CadEntityId,
    pub shell_ids: Vec<String>,
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadTopologyReport {
    pub source: CadTopologySource,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub shell_count: usize,
    pub volume_count: usize,
    pub semantic_face_count: usize,
    pub imported_face_count: usize,
    pub evaluator_face_count: usize,
    pub generic_face_count: usize,
    pub closed_shell_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadTopologyModel {
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_geometry_sha256: Option<String>,
    pub source: CadTopologySource,
    pub vertices: Vec<CadVertex>,
    pub edges: Vec<CadEdge>,
    pub faces: Vec<CadFace>,
    pub shells: Vec<CadShell>,
    pub volumes: Vec<CadVolume>,
    pub report: CadTopologyReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CadTopologyError {
    EmptyTopology,
}

impl std::fmt::Display for CadTopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTopology => write!(formatter, "source topology has no vertices or faces"),
        }
    }
}

impl std::error::Error for CadTopologyError {}

pub fn build_cad_topology(
    geometry: &GeometryAsset,
    topology: &SourceTopologyModel,
) -> Result<CadTopologyModel, CadTopologyError> {
    if topology.vertices.is_empty() || topology.faces.is_empty() {
        return Err(CadTopologyError::EmptyTopology);
    }

    let source = cad_topology_source(geometry);
    let semantic_face_regions = semantic_face_regions(geometry);
    let imported_face_ids_by_region = imported_face_ids_by_region(geometry);
    let evaluator_faces = evaluator_faces_by_imported_id(geometry);
    let face_region_by_source_face = face_regions_by_source_face(geometry);

    let vertices = topology
        .vertices
        .iter()
        .map(|vertex| CadVertex {
            entity_id: CadEntityId {
                kind: CadEntityKind::Vertex,
                id: format!("cad_vertex_{}", vertex.vertex_id),
            },
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();

    let face_seeds = topology
        .faces
        .iter()
        .map(|face| {
            let mapped_region_ids = face_region_by_source_face
                .get(&face.source_triangle_id)
                .cloned();
            let identity_region_ids = mapped_region_ids.clone().unwrap_or_default();
            let region_ids = mapped_region_ids.unwrap_or_else(|| face.region_ids.clone());
            let imported_face_id =
                imported_face_id_for_regions(&imported_face_ids_by_region, &identity_region_ids);
            let evaluator_face = imported_face_id.and_then(|face_id| evaluator_faces.get(&face_id));
            CadFace {
                entity_id: CadEntityId {
                    kind: CadEntityKind::Face,
                    id: stable_face_id(
                        &semantic_face_regions,
                        identity_region_ids.as_slice(),
                        face.face_id,
                    ),
                },
                imported_face_id,
                evaluator_id: evaluator_face.map(|face| face.evaluator_id.clone()),
                evaluator_supports_point_evaluation: evaluator_face
                    .is_some_and(|face| face.supports_point_evaluation),
                evaluator_supports_projection: evaluator_face
                    .is_some_and(|face| face.supports_projection),
                evaluator_supports_normal: evaluator_face.is_some_and(|face| face.supports_normal),
                evaluator_supports_derivatives: evaluator_face
                    .is_some_and(|face| face.supports_derivatives),
                evaluator_supports_curvature: evaluator_face
                    .is_some_and(|face| face.supports_curvature),
                evaluator_reference_point_m: evaluator_face.and_then(|face| face.reference_point_m),
                evaluator_unit_normal: evaluator_face.and_then(|face| face.reference_unit_normal),
                evaluator_samples: evaluator_face
                    .map(|face| face.evaluation_samples.clone())
                    .unwrap_or_default(),
                source_face_ids: vec![face.face_id],
                source_edge_ids: face.edge_ids.to_vec(),
                loop_edge_ids: face
                    .edge_ids
                    .iter()
                    .map(|edge_id| format!("cad_edge_{edge_id}"))
                    .collect(),
                region_ids,
                area_m2: face.area_m2,
                unit_normal: face.unit_normal,
            }
        })
        .collect::<Vec<_>>();
    let faces = merge_stable_cad_faces(face_seeds);

    let face_ids_by_source_face = faces
        .iter()
        .flat_map(|face| {
            face.source_face_ids
                .iter()
                .map(|source_face_id| (*source_face_id, face.entity_id.id.clone()))
                .collect::<Vec<_>>()
        })
        .collect::<BTreeMap<_, _>>();
    let edges = topology
        .edges
        .iter()
        .map(|edge| CadEdge {
            entity_id: CadEntityId {
                kind: CadEntityKind::Edge,
                id: format!("cad_edge_{}", edge.edge_id),
            },
            source_edge_id: edge.edge_id,
            vertex_ids: [
                format!("cad_vertex_{}", edge.node_ids[0]),
                format!("cad_vertex_{}", edge.node_ids[1]),
            ],
            adjacent_face_ids: edge
                .adjacent_face_ids
                .iter()
                .filter_map(|face_id| face_ids_by_source_face.get(face_id).cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            length_m: edge.length_m,
        })
        .collect::<Vec<_>>();

    let closed = topology
        .edges
        .iter()
        .all(|edge| edge.adjacent_face_ids.len() == 2);
    let shell = CadShell {
        entity_id: CadEntityId {
            kind: CadEntityKind::Shell,
            id: "cad_shell_0".to_string(),
        },
        face_ids: faces.iter().map(|face| face.entity_id.id.clone()).collect(),
        closed,
    };
    let volume = CadVolume {
        entity_id: CadEntityId {
            kind: CadEntityKind::Volume,
            id: "cad_volume_0".to_string(),
        },
        shell_ids: vec![shell.entity_id.id.clone()],
        region_ids: topology.region_ids.clone(),
    };

    let semantic_face_count = faces
        .iter()
        .filter(|face| semantic_face_regions.contains(&face.entity_id.id))
        .count();
    let imported_face_count = faces
        .iter()
        .filter(|face| face.imported_face_id.is_some())
        .count();
    let evaluator_face_count = faces
        .iter()
        .filter(|face| {
            face.imported_face_id
                .is_some_and(|face_id| evaluator_faces.contains_key(&face_id))
        })
        .count();
    let generic_face_count = faces.len().saturating_sub(semantic_face_count);
    let report = CadTopologyReport {
        source,
        vertex_count: vertices.len(),
        edge_count: edges.len(),
        face_count: faces.len(),
        shell_count: 1,
        volume_count: usize::from(closed),
        semantic_face_count,
        imported_face_count,
        evaluator_face_count,
        generic_face_count,
        closed_shell_count: usize::from(closed),
    };

    Ok(CadTopologyModel {
        source_geometry_id: topology.source_geometry_id.clone(),
        source_geometry_revision: topology.source_geometry_revision,
        source_geometry_sha256: topology.source_geometry_sha256.clone(),
        source,
        vertices,
        edges,
        faces,
        shells: vec![shell],
        volumes: closed.then_some(volume).into_iter().collect(),
        report,
    })
}

fn merge_stable_cad_faces(face_seeds: Vec<CadFace>) -> Vec<CadFace> {
    let mut merged = BTreeMap::<String, (CadFace, BTreeMap<u32, usize>)>::new();
    for face in face_seeds {
        let key = face.entity_id.id.clone();
        let source_edge_counts = face
            .source_edge_ids
            .iter()
            .map(|edge_id| (*edge_id, 1_usize))
            .collect::<BTreeMap<_, _>>();
        match merged.get_mut(&key) {
            Some((existing, edge_counts)) => {
                existing
                    .source_face_ids
                    .extend(face.source_face_ids.iter().copied());
                existing
                    .source_edge_ids
                    .extend(face.source_edge_ids.iter().copied());
                existing.region_ids.extend(face.region_ids.iter().cloned());
                existing.unit_normal = area_weighted_normal(
                    existing.unit_normal,
                    existing.area_m2,
                    face.unit_normal,
                    face.area_m2,
                );
                existing.area_m2 += face.area_m2;
                for (source_edge_id, count) in source_edge_counts {
                    *edge_counts.entry(source_edge_id).or_default() += count;
                }
            }
            None => {
                merged.insert(key, (face, source_edge_counts));
            }
        }
    }
    merged
        .into_values()
        .map(|(mut face, edge_counts)| {
            face.source_face_ids.sort_unstable();
            face.source_face_ids.dedup();
            face.source_edge_ids.sort_unstable();
            face.source_edge_ids.dedup();
            face.region_ids.sort();
            face.region_ids.dedup();
            face.loop_edge_ids = edge_counts
                .into_iter()
                .filter_map(|(edge_id, count)| (count == 1).then(|| format!("cad_edge_{edge_id}")))
                .collect();
            face
        })
        .collect()
}

fn area_weighted_normal(
    left_normal: [f64; 3],
    left_area_m2: f64,
    right_normal: [f64; 3],
    right_area_m2: f64,
) -> [f64; 3] {
    let combined = [
        left_normal[0] * left_area_m2 + right_normal[0] * right_area_m2,
        left_normal[1] * left_area_m2 + right_normal[1] * right_area_m2,
        left_normal[2] * left_area_m2 + right_normal[2] * right_area_m2,
    ];
    let length =
        (combined[0] * combined[0] + combined[1] * combined[1] + combined[2] * combined[2]).sqrt();
    if !length.is_finite() || length <= f64::EPSILON {
        left_normal
    } else {
        [
            combined[0] / length,
            combined[1] / length,
            combined[2] / length,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract_source_topology;
    use runmat_geometry_core::{
        CadEvaluatorSet, CadFaceEvaluator, CadLabelRef, CadRegionOwnership, CadSemanticKind,
        EntityIdRange, EntityKind, GeometrySource, MeshDescriptor, MeshKind, Region,
        RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
        UnitSystem,
    };

    #[test]
    fn builds_generic_cad_topology_from_source_triangles() {
        let geometry = cube_geometry(false);
        let topology = extract_source_topology(&geometry).expect("topology should extract");

        let cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");

        assert_eq!(cad.source, CadTopologySource::GenericCadMesh);
        assert_eq!(cad.report.vertex_count, 8);
        assert_eq!(cad.report.edge_count, 18);
        assert_eq!(cad.report.face_count, 12);
        assert_eq!(cad.report.closed_shell_count, 1);
        assert_eq!(cad.report.volume_count, 1);
        assert_eq!(cad.report.semantic_face_count, 0);
        assert_eq!(cad.report.imported_face_count, 0);
        assert_eq!(cad.report.evaluator_face_count, 0);
        assert_eq!(cad.report.generic_face_count, 12);
    }

    #[test]
    fn preserves_semantic_cad_face_regions() {
        let geometry = cube_geometry(true);
        let topology = extract_source_topology(&geometry).expect("topology should extract");

        let cad = build_cad_topology(&geometry, &topology).expect("cad topology should build");

        assert_eq!(cad.source, CadTopologySource::SemanticCad);
        assert_eq!(cad.report.face_count, 11);
        assert_eq!(cad.report.semantic_face_count, 1);
        assert_eq!(cad.report.imported_face_count, 1);
        assert_eq!(cad.report.evaluator_face_count, 1);
        let semantic_face = cad
            .faces
            .iter()
            .find(|face| face.entity_id.id == "face_000001")
            .expect("semantic face should be merged");
        assert_eq!(semantic_face.imported_face_id, Some(1));
        assert_eq!(
            semantic_face.evaluator_reference_point_m,
            Some([0.5, 0.5, 0.0])
        );
        assert_eq!(semantic_face.source_face_ids, vec![0, 1]);
        assert_eq!(semantic_face.source_edge_ids.len(), 5);
        assert_eq!(semantic_face.loop_edge_ids.len(), 4);
        assert!((semantic_face.area_m2 - 1.0).abs() <= 1.0e-12);
        assert_eq!(semantic_face.unit_normal, [0.0, 0.0, -1.0]);
    }

    fn cube_geometry(with_semantic_face: bool) -> runmat_geometry_core::GeometryAsset {
        let face_region = Region {
            region_id: "face_000001".to_string(),
            name: "face".to_string(),
            tag: Some("cad_face".to_string()),
            cad_ownership: with_semantic_face.then(|| CadRegionOwnership {
                face_id: Some(1),
                label: Some(CadLabelRef {
                    label_entry: "0:1:1".to_string(),
                    name: "face".to_string(),
                    kind: CadSemanticKind::Face,
                }),
                owner_path: Vec::new(),
                layers: Vec::new(),
                color: None,
                material: None,
            }),
        };
        runmat_geometry_core::GeometryAsset {
            geometry_id: "geo_cad_topology_cube".to_string(),
            source: GeometrySource {
                path: "/fixtures/generic_cube.step".to_string(),
                sha256: "generic-cube".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
                cad_evaluators: if with_semantic_face {
                    vec![CadEvaluatorSet {
                        evaluator_id: "cad_evaluator_test".to_string(),
                        backend: "test".to_string(),
                        format_name: "step".to_string(),
                        requires_source_geometry: true,
                        faces: vec![CadFaceEvaluator {
                            evaluator_id: "cad_face_1".to_string(),
                            imported_face_id: 1,
                            name: "face".to_string(),
                            supports_point_evaluation: true,
                            supports_projection: true,
                            supports_normal: true,
                            supports_derivatives: true,
                            supports_curvature: true,
                            reference_point_m: Some([0.5, 0.5, 0.0]),
                            reference_unit_normal: Some([0.0, 0.0, 1.0]),
                            evaluation_samples: Vec::new(),
                        }],
                        curves: Vec::new(),
                    }]
                } else {
                    Vec::new()
                },
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
            regions: vec![face_region],
            region_entity_mappings: vec![RegionEntityMapping::new(
                "face_000001",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(0, 2)],
            )],
            diagnostics: Vec::new(),
        }
    }
}
