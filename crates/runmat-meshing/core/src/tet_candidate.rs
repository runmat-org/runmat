use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    surface::{SurfaceDiscretization, SurfaceElement},
    volume_candidate::{VolumeCandidateComponent, VolumeCandidateSet},
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateOptions {
    pub min_volume_m3: f64,
    pub max_aspect_ratio: f64,
}

impl Default for TetCandidateOptions {
    fn default() -> Self {
        Self {
            min_volume_m3: 1.0e-18,
            max_aspect_ratio: 1.0e6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateNode {
    pub node_id: u32,
    pub coordinates_m: [f64; 3],
    pub source: TetCandidateNodeSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetCandidateNodeSource {
    Surface,
    InteriorSeed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetCandidate {
    pub tet_id: u32,
    pub component_id: u32,
    pub node_ids: [u32; 4],
    pub source_surface_element_id: u32,
    pub region_ids: Vec<String>,
    pub volume_m3: f64,
    pub aspect_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetCandidateSet {
    pub nodes: Vec<TetCandidateNode>,
    pub tets: Vec<TetCandidate>,
    pub total_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetCandidateError {
    MissingSurfaceNode { node_id: u32 },
    MissingSurfaceElement { element_id: u32 },
    InvalidOptions,
    EmptyCandidateSet,
}

impl std::fmt::Display for TetCandidateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSurfaceNode { node_id } => {
                write!(formatter, "surface node {node_id} is missing")
            }
            Self::MissingSurfaceElement { element_id } => {
                write!(formatter, "surface element {element_id} is missing")
            }
            Self::InvalidOptions => write!(
                formatter,
                "Tet candidate options must use finite positive volume and aspect ratio limits"
            ),
            Self::EmptyCandidateSet => write!(formatter, "no valid Tet candidates were generated"),
        }
    }
}

impl std::error::Error for TetCandidateError {}

pub fn form_tet_candidates(
    surface: &SurfaceDiscretization,
    volume_candidates: &VolumeCandidateSet,
    options: TetCandidateOptions,
) -> Result<TetCandidateSet, TetCandidateError> {
    validate_options(options)?;
    let mut nodes = surface
        .nodes
        .iter()
        .map(|node| TetCandidateNode {
            node_id: node.node_id,
            coordinates_m: node.coordinates_m,
            source: TetCandidateNodeSource::Surface,
        })
        .collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.node_id);

    let surface_nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let surface_elements = surface
        .elements
        .iter()
        .map(|element| (element.element_id, element))
        .collect::<BTreeMap<_, _>>();

    let mut tets = Vec::<TetCandidate>::new();
    let mut next_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    for component in &volume_candidates.components {
        let interior_node_id = next_node_id;
        next_node_id = next_node_id.saturating_add(1);
        nodes.push(TetCandidateNode {
            node_id: interior_node_id,
            coordinates_m: component_interior_point(component),
            source: TetCandidateNodeSource::InteriorSeed,
        });
        append_component_tets(
            component,
            interior_node_id,
            &surface_nodes,
            &surface_elements,
            options,
            &mut tets,
        )?;
    }

    if tets.is_empty() {
        return Err(TetCandidateError::EmptyCandidateSet);
    }
    let total_volume_m3 = tets.iter().map(|tet| tet.volume_m3).sum();
    Ok(TetCandidateSet {
        nodes,
        tets,
        total_volume_m3,
    })
}

fn validate_options(options: TetCandidateOptions) -> Result<(), TetCandidateError> {
    if !options.min_volume_m3.is_finite()
        || options.min_volume_m3 <= 0.0
        || !options.max_aspect_ratio.is_finite()
        || options.max_aspect_ratio <= 0.0
    {
        return Err(TetCandidateError::InvalidOptions);
    }
    Ok(())
}

fn append_component_tets(
    component: &VolumeCandidateComponent,
    interior_node_id: u32,
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
    surface_elements: &BTreeMap<u32, &SurfaceElement>,
    options: TetCandidateOptions,
    tets: &mut Vec<TetCandidate>,
) -> Result<(), TetCandidateError> {
    let interior = component_interior_point(component);
    for element_id in &component.surface_element_ids {
        let element =
            surface_elements
                .get(element_id)
                .ok_or(TetCandidateError::MissingSurfaceElement {
                    element_id: *element_id,
                })?;
        let mut node_ids = [
            element.node_ids[0],
            element.node_ids[1],
            element.node_ids[2],
            interior_node_id,
        ];
        let points = tet_points(node_ids, interior, surface_nodes)?;
        let mut signed_volume_m3 = tet_signed_volume(points);
        if signed_volume_m3 < 0.0 {
            node_ids.swap(1, 2);
            signed_volume_m3 = -signed_volume_m3;
        }
        let volume_m3 = signed_volume_m3.abs();
        if volume_m3 < options.min_volume_m3 {
            continue;
        }
        let aspect_ratio = tet_aspect_ratio(points);
        if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
            continue;
        }
        tets.push(TetCandidate {
            tet_id: tets.len() as u32,
            component_id: component.component_id,
            node_ids,
            source_surface_element_id: element.element_id,
            region_ids: element.region_ids.clone(),
            volume_m3,
            aspect_ratio,
        });
    }
    Ok(())
}

fn component_interior_point(component: &VolumeCandidateComponent) -> [f64; 3] {
    [
        (component.bounds_min_m[0] + component.bounds_max_m[0]) * 0.5,
        (component.bounds_min_m[1] + component.bounds_max_m[1]) * 0.5,
        (component.bounds_min_m[2] + component.bounds_max_m[2]) * 0.5,
    ]
}

fn tet_points(
    node_ids: [u32; 4],
    interior: [f64; 3],
    surface_nodes: &BTreeMap<u32, [f64; 3]>,
) -> Result<[[f64; 3]; 4], TetCandidateError> {
    Ok([
        *surface_nodes
            .get(&node_ids[0])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[0],
            })?,
        *surface_nodes
            .get(&node_ids[1])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[1],
            })?,
        *surface_nodes
            .get(&node_ids[2])
            .ok_or(TetCandidateError::MissingSurfaceNode {
                node_id: node_ids[2],
            })?,
        interior,
    ])
}

fn tet_signed_volume(points: [[f64; 3]; 4]) -> f64 {
    dot(
        sub(points[1], points[0]),
        cross(sub(points[2], points[0]), sub(points[3], points[0])),
    ) / 6.0
}

fn tet_aspect_ratio(points: [[f64; 3]; 4]) -> f64 {
    let mut min_edge = f64::INFINITY;
    let mut max_edge = 0.0_f64;
    for left_index in 0..4 {
        for right_index in (left_index + 1)..4 {
            let length = distance(points[left_index], points[right_index]);
            min_edge = min_edge.min(length);
            max_edge = max_edge.max(length);
        }
    }
    max_edge / min_edge.max(f64::EPSILON)
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        discretize_topology_surfaces, extract_source_topology, prepare_volume_candidates,
        SurfaceDiscretizationOptions, VolumeCandidateOptions,
    };
    use runmat_geometry_core::{
        EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
        RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
        UnitSystem,
    };

    #[test]
    fn forms_positive_tet_candidates_from_closed_cube_surface() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();

        let candidates =
            form_tet_candidates(&surface, &volume_candidates, TetCandidateOptions::default())
                .expect("Tet candidates should form");

        assert_eq!(candidates.nodes.len(), 9);
        assert_eq!(candidates.tets.len(), 12);
        assert!(candidates
            .tets
            .iter()
            .all(|tet| tet.volume_m3 > 0.0 && tet.aspect_ratio.is_finite()));
        assert!((candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert!(candidates
            .nodes
            .iter()
            .any(|node| matches!(node.source, TetCandidateNodeSource::InteriorSeed)));
    }

    #[test]
    fn rejects_invalid_candidate_options() {
        let (surface, volume_candidates) = cube_surface_and_volume_candidates();
        let err = form_tet_candidates(
            &surface,
            &volume_candidates,
            TetCandidateOptions {
                min_volume_m3: 0.0,
                ..TetCandidateOptions::default()
            },
        )
        .expect_err("invalid options should fail");

        assert_eq!(err, TetCandidateError::InvalidOptions);
    }

    fn cube_surface_and_volume_candidates(
    ) -> (crate::SurfaceDiscretization, crate::VolumeCandidateSet) {
        let topology = extract_source_topology(&cube_geometry()).expect("topology should extract");
        let surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        let volume_candidates =
            prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
                .expect("volume candidates should prepare");
        (surface, volume_candidates)
    }

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_tet_candidate_cube".to_string(),
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
