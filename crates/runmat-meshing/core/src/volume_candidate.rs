use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::surface::{SurfaceDiscretization, SurfaceElement};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VolumeCandidateOptions {
    pub require_closed: bool,
    pub min_component_volume_m3: f64,
}

impl Default for VolumeCandidateOptions {
    fn default() -> Self {
        Self {
            require_closed: true,
            min_component_volume_m3: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeCandidateComponent {
    pub component_id: u32,
    pub surface_element_ids: Vec<u32>,
    pub source_face_ids: Vec<u32>,
    pub node_ids: Vec<u32>,
    pub region_ids: Vec<String>,
    pub bounds_min_m: [f64; 3],
    pub bounds_max_m: [f64; 3],
    pub surface_area_m2: f64,
    pub signed_volume_m3: f64,
    pub volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeCandidateSet {
    pub components: Vec<VolumeCandidateComponent>,
    pub total_surface_area_m2: f64,
    pub total_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VolumeCandidateError {
    EmptySurface,
    MissingSurfaceNode { element_id: u32, node_id: u32 },
    OpenBoundaryEdge { edge: [u32; 2], count: usize },
    NonManifoldBoundaryEdge { edge: [u32; 2], count: usize },
    NonFiniteComponentVolume { component_id: u32 },
    ComponentVolumeBelowMinimum { component_id: u32 },
}

impl std::fmt::Display for VolumeCandidateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySurface => write!(formatter, "surface discretization has no elements"),
            Self::MissingSurfaceNode {
                element_id,
                node_id,
            } => write!(
                formatter,
                "surface element {element_id} references missing surface node {node_id}"
            ),
            Self::OpenBoundaryEdge { edge, count } => write!(
                formatter,
                "surface edge {}-{} has incidence {count}, expected 2",
                edge[0], edge[1]
            ),
            Self::NonManifoldBoundaryEdge { edge, count } => write!(
                formatter,
                "surface edge {}-{} has non-manifold incidence {count}, expected 2",
                edge[0], edge[1]
            ),
            Self::NonFiniteComponentVolume { component_id } => write!(
                formatter,
                "volume candidate component {component_id} has non-finite volume"
            ),
            Self::ComponentVolumeBelowMinimum { component_id } => write!(
                formatter,
                "volume candidate component {component_id} is below the minimum volume"
            ),
        }
    }
}

impl std::error::Error for VolumeCandidateError {}

pub fn prepare_volume_candidates(
    surface: &SurfaceDiscretization,
    options: VolumeCandidateOptions,
) -> Result<VolumeCandidateSet, VolumeCandidateError> {
    if surface.elements.is_empty() {
        return Err(VolumeCandidateError::EmptySurface);
    }
    let edge_to_elements = surface_edge_incidence(surface)?;
    if options.require_closed {
        validate_closed_surface_edges(&edge_to_elements)?;
    }

    let components = surface_components(surface, &edge_to_elements);
    let mut candidates = Vec::<VolumeCandidateComponent>::with_capacity(components.len());
    for (component_id, element_indices) in components.into_iter().enumerate() {
        let candidate = build_component(surface, component_id as u32, &element_indices)?;
        if !candidate.volume_m3.is_finite() {
            return Err(VolumeCandidateError::NonFiniteComponentVolume {
                component_id: candidate.component_id,
            });
        }
        if candidate.volume_m3 < options.min_component_volume_m3 {
            return Err(VolumeCandidateError::ComponentVolumeBelowMinimum {
                component_id: candidate.component_id,
            });
        }
        candidates.push(candidate);
    }

    let total_surface_area_m2 = candidates
        .iter()
        .map(|component| component.surface_area_m2)
        .sum();
    let total_volume_m3 = candidates.iter().map(|component| component.volume_m3).sum();
    Ok(VolumeCandidateSet {
        components: candidates,
        total_surface_area_m2,
        total_volume_m3,
    })
}

fn surface_edge_incidence(
    surface: &SurfaceDiscretization,
) -> Result<BTreeMap<[u32; 2], Vec<usize>>, VolumeCandidateError> {
    let mut edge_to_elements = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (element_index, element) in surface.elements.iter().enumerate() {
        for node_id in element.node_ids {
            if surface
                .nodes
                .get(node_id as usize)
                .is_none_or(|node| node.node_id != node_id)
            {
                return Err(VolumeCandidateError::MissingSurfaceNode {
                    element_id: element.element_id,
                    node_id,
                });
            }
        }
        for edge in triangle_edges(element.node_ids) {
            edge_to_elements
                .entry(edge)
                .or_default()
                .push(element_index);
        }
    }
    Ok(edge_to_elements)
}

fn validate_closed_surface_edges(
    edge_to_elements: &BTreeMap<[u32; 2], Vec<usize>>,
) -> Result<(), VolumeCandidateError> {
    for (edge, elements) in edge_to_elements {
        if elements.len() == 1 {
            return Err(VolumeCandidateError::OpenBoundaryEdge {
                edge: *edge,
                count: elements.len(),
            });
        }
        if elements.len() > 2 {
            return Err(VolumeCandidateError::NonManifoldBoundaryEdge {
                edge: *edge,
                count: elements.len(),
            });
        }
    }
    Ok(())
}

fn surface_components(
    surface: &SurfaceDiscretization,
    edge_to_elements: &BTreeMap<[u32; 2], Vec<usize>>,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::<usize>::new(); surface.elements.len()];
    for elements in edge_to_elements.values() {
        for left_index in 0..elements.len() {
            for right_index in (left_index + 1)..elements.len() {
                let left = elements[left_index];
                let right = elements[right_index];
                adjacency[left].push(right);
                adjacency[right].push(left);
            }
        }
    }

    let mut visited = vec![false; surface.elements.len()];
    let mut components = Vec::<Vec<usize>>::new();
    for start in 0..surface.elements.len() {
        if visited[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        visited[start] = true;
        let mut component = Vec::<usize>::new();
        while let Some(element_index) = queue.pop_front() {
            component.push(element_index);
            for neighbor in &adjacency[element_index] {
                if !visited[*neighbor] {
                    visited[*neighbor] = true;
                    queue.push_back(*neighbor);
                }
            }
        }
        components.push(component);
    }
    components
}

fn build_component(
    surface: &SurfaceDiscretization,
    component_id: u32,
    element_indices: &[usize],
) -> Result<VolumeCandidateComponent, VolumeCandidateError> {
    let first_element = &surface.elements[element_indices[0]];
    let first_node = surface_node(surface, first_element, first_element.node_ids[0])?;
    let mut bounds_min_m = first_node;
    let mut bounds_max_m = first_node;
    let mut node_ids = BTreeSet::<u32>::new();
    let mut source_face_ids = BTreeSet::<u32>::new();
    let mut region_ids = BTreeSet::<String>::new();
    let mut surface_element_ids = Vec::<u32>::with_capacity(element_indices.len());
    let mut surface_area_m2 = 0.0_f64;
    let mut signed_volume_m3 = 0.0_f64;

    for element_index in element_indices {
        let element = &surface.elements[*element_index];
        surface_element_ids.push(element.element_id);
        source_face_ids.insert(element.source_face_id);
        region_ids.extend(element.region_ids.iter().cloned());
        surface_area_m2 += element.area_m2;
        let points = element_points(surface, element)?;
        signed_volume_m3 += signed_tet_volume_from_origin(points);
        for (node_id, point) in element.node_ids.into_iter().zip(points) {
            node_ids.insert(node_id);
            for axis in 0..3 {
                bounds_min_m[axis] = bounds_min_m[axis].min(point[axis]);
                bounds_max_m[axis] = bounds_max_m[axis].max(point[axis]);
            }
        }
    }

    Ok(VolumeCandidateComponent {
        component_id,
        surface_element_ids,
        source_face_ids: source_face_ids.into_iter().collect(),
        node_ids: node_ids.into_iter().collect(),
        region_ids: region_ids.into_iter().collect(),
        bounds_min_m,
        bounds_max_m,
        surface_area_m2,
        signed_volume_m3,
        volume_m3: signed_volume_m3.abs(),
    })
}

fn element_points(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
) -> Result<[[f64; 3]; 3], VolumeCandidateError> {
    Ok([
        surface_node(surface, element, element.node_ids[0])?,
        surface_node(surface, element, element.node_ids[1])?,
        surface_node(surface, element, element.node_ids[2])?,
    ])
}

fn surface_node(
    surface: &SurfaceDiscretization,
    element: &SurfaceElement,
    node_id: u32,
) -> Result<[f64; 3], VolumeCandidateError> {
    surface
        .nodes
        .get(node_id as usize)
        .filter(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
        .ok_or(VolumeCandidateError::MissingSurfaceNode {
            element_id: element.element_id,
            node_id,
        })
}

fn signed_tet_volume_from_origin(points: [[f64; 3]; 3]) -> f64 {
    dot(points[0], cross(points[1], points[2])) / 6.0
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(triangle[0], triangle[1]),
        sorted_edge(triangle[1], triangle[2]),
        sorted_edge(triangle[2], triangle[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        discretize_topology_surfaces, extract_source_topology, SurfaceDiscretizationOptions,
    };
    use runmat_geometry_core::{
        EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
        RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
        UnitSystem,
    };

    #[test]
    fn prepares_closed_cube_volume_candidate() {
        let topology = extract_source_topology(&cube_geometry()).expect("topology should extract");
        let surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");

        let candidates = prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
            .expect("volume candidates should prepare");

        assert_eq!(candidates.components.len(), 1);
        let component = &candidates.components[0];
        assert_eq!(component.node_ids.len(), 8);
        assert_eq!(component.surface_element_ids.len(), 12);
        assert_eq!(component.source_face_ids.len(), 12);
        assert_eq!(
            component.region_ids,
            vec!["root".to_string(), "tip".to_string()]
        );
        assert!((component.surface_area_m2 - 6.0).abs() < 1.0e-12);
        assert!((component.volume_m3 - 1.0).abs() < 1.0e-12);
        assert_eq!(component.bounds_min_m, [0.0, 0.0, 0.0]);
        assert_eq!(component.bounds_max_m, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_open_surface_when_closure_is_required() {
        let topology = extract_source_topology(&cube_geometry()).expect("topology should extract");
        let mut surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        surface.elements.pop();

        let err = prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
            .expect_err("open surface should fail");

        assert!(matches!(err, VolumeCandidateError::OpenBoundaryEdge { .. }));
    }

    #[test]
    fn separates_disconnected_surface_components() {
        let topology = extract_source_topology(&cube_geometry()).expect("topology should extract");
        let mut surface =
            discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
                .expect("surface should discretize");
        let original_node_count = surface.nodes.len() as u32;
        for node in surface.nodes.clone() {
            let mut shifted = node;
            shifted.node_id += original_node_count;
            shifted.source_vertex_id += original_node_count;
            shifted.coordinates_m[0] += 2.0;
            surface.nodes.push(shifted);
        }
        for element in surface.elements.clone() {
            let mut shifted = element;
            shifted.element_id += 12;
            shifted.node_ids = shifted
                .node_ids
                .map(|node_id| node_id + original_node_count);
            shifted.source_face_id += 12;
            surface.elements.push(shifted);
        }

        let candidates = prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
            .expect("two closed components should prepare");

        assert_eq!(candidates.components.len(), 2);
        assert!((candidates.total_volume_m3 - 2.0).abs() < 1.0e-12);
    }

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_volume_candidate_cube".to_string(),
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
