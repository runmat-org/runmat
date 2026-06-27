use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};

use crate::boundary::{BoundaryMeshInput, BoundaryMeshInputError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyVertex {
    pub vertex_id: u32,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyEdge {
    pub edge_id: u32,
    pub node_ids: [u32; 2],
    pub adjacent_face_ids: Vec<u32>,
    pub region_ids: Vec<String>,
    pub length_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyFace {
    pub face_id: u32,
    pub source_triangle_id: u32,
    pub node_ids: [u32; 3],
    pub edge_ids: [u32; 3],
    pub region_ids: Vec<String>,
    pub area_m2: f64,
    pub unit_normal: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTopologyModel {
    pub mesh_id: String,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_geometry_sha256: Option<String>,
    pub vertices: Vec<SourceTopologyVertex>,
    pub edges: Vec<SourceTopologyEdge>,
    pub faces: Vec<SourceTopologyFace>,
    pub bounds_min_m: [f64; 3],
    pub bounds_max_m: [f64; 3],
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceTopologyError {
    BoundaryInput(BoundaryMeshInputError),
}

impl std::fmt::Display for SourceTopologyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundaryInput(err) => write!(formatter, "{err}"),
        }
    }
}

impl std::error::Error for SourceTopologyError {}

impl From<BoundaryMeshInputError> for SourceTopologyError {
    fn from(value: BoundaryMeshInputError) -> Self {
        Self::BoundaryInput(value)
    }
}

pub fn extract_source_topology(
    geometry: &GeometryAsset,
) -> Result<SourceTopologyModel, SourceTopologyError> {
    let input = BoundaryMeshInput::from_geometry(geometry)?;
    Ok(source_topology_from_boundary_input(&input))
}

pub fn source_topology_from_boundary_input(input: &BoundaryMeshInput) -> SourceTopologyModel {
    let vertices = input
        .vertices
        .iter()
        .copied()
        .enumerate()
        .map(|(vertex_id, coordinates_m)| SourceTopologyVertex {
            vertex_id: vertex_id as u32,
            coordinates_m,
        })
        .collect::<Vec<_>>();

    let mut edge_ids = BTreeMap::<[u32; 2], u32>::new();
    let mut edge_regions = BTreeMap::<[u32; 2], BTreeSet<String>>::new();
    let mut edge_faces = BTreeMap::<[u32; 2], Vec<u32>>::new();
    let mut faces = Vec::<SourceTopologyFace>::with_capacity(input.triangles.len());

    for (face_index, triangle) in input.triangles.iter().enumerate() {
        let face_id = face_index as u32;
        let mut face_edge_ids = [0_u32; 3];
        for (local_edge_index, edge) in triangle_edges(triangle.node_ids).into_iter().enumerate() {
            let next_edge_id = edge_ids.len() as u32;
            let edge_id = *edge_ids.entry(edge).or_insert(next_edge_id);
            face_edge_ids[local_edge_index] = edge_id;
            edge_faces.entry(edge).or_default().push(face_id);
            edge_regions
                .entry(edge)
                .or_default()
                .extend(triangle.region_ids.iter().cloned());
        }
        let vertices = triangle_vertices(input, triangle.node_ids).unwrap_or([[0.0; 3]; 3]);
        faces.push(SourceTopologyFace {
            face_id,
            source_triangle_id: triangle.triangle_id,
            node_ids: triangle.node_ids,
            edge_ids: face_edge_ids,
            region_ids: triangle.region_ids.clone(),
            area_m2: triangle_area(vertices),
            unit_normal: triangle_unit_normal(vertices),
        });
    }

    let mut edges = edge_ids
        .iter()
        .map(|(node_ids, edge_id)| {
            let left = input.vertices[node_ids[0] as usize];
            let right = input.vertices[node_ids[1] as usize];
            SourceTopologyEdge {
                edge_id: *edge_id,
                node_ids: *node_ids,
                adjacent_face_ids: edge_faces.remove(node_ids).unwrap_or_default(),
                region_ids: edge_regions
                    .remove(node_ids)
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                length_m: distance(left, right),
            }
        })
        .collect::<Vec<_>>();
    edges.sort_by_key(|edge| edge.edge_id);

    SourceTopologyModel {
        mesh_id: input.mesh_id.clone(),
        source_geometry_id: input.source_geometry_id.clone(),
        source_geometry_revision: input.source_geometry_revision,
        source_geometry_sha256: input.source_geometry_sha256.clone(),
        vertices,
        edges,
        faces,
        bounds_min_m: input.bounds_min_m,
        bounds_max_m: input.bounds_max_m,
        region_ids: input.region_ids.clone(),
    }
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

fn triangle_vertices(input: &BoundaryMeshInput, node_ids: [u32; 3]) -> Option<[[f64; 3]; 3]> {
    Some([
        *input.vertices.get(node_ids[0] as usize)?,
        *input.vertices.get(node_ids[1] as usize)?,
        *input.vertices.get(node_ids[2] as usize)?,
    ])
}

fn triangle_area(vertices: [[f64; 3]; 3]) -> f64 {
    0.5 * norm(cross(
        sub(vertices[1], vertices[0]),
        sub(vertices[2], vertices[0]),
    ))
}

fn triangle_unit_normal(vertices: [[f64; 3]; 3]) -> [f64; 3] {
    let normal = cross(sub(vertices[1], vertices[0]), sub(vertices[2], vertices[0]));
    let length = norm(normal);
    if !length.is_finite() || length <= f64::EPSILON {
        return [0.0, 0.0, 0.0];
    }
    [normal[0] / length, normal[1] / length, normal[2] / length]
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

fn norm(value: [f64; 3]) -> f64 {
    distance([0.0, 0.0, 0.0], value)
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
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
    fn extracts_deterministic_closed_shell_topology() {
        let topology = extract_source_topology(&cube_geometry()).expect("topology should extract");

        assert_eq!(topology.vertices.len(), 8);
        assert_eq!(topology.faces.len(), 12);
        assert_eq!(topology.edges.len(), 18);
        assert_eq!(
            topology.region_ids,
            vec!["root".to_string(), "tip".to_string()]
        );
        assert!(topology
            .edges
            .iter()
            .all(|edge| edge.adjacent_face_ids.len() == 2));
        assert!(topology.faces.iter().all(|face| face.area_m2 > 0.0));
        assert!(topology
            .faces
            .iter()
            .all(|face| norm(face.unit_normal) > 0.999999));
    }

    #[test]
    fn topology_converts_geometry_units_to_meters() {
        let mut geometry = cube_geometry();
        geometry.units = UnitSystem::Millimeter;

        let topology = extract_source_topology(&geometry).expect("topology should extract");

        assert_eq!(topology.bounds_max_m, [0.001, 0.001, 0.001]);
        assert!(topology.edges.iter().any(|edge| {
            (edge.length_m - 0.001).abs() < 1.0e-12
                && edge.region_ids.iter().any(|region| region == "root")
        }));
    }

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_topology_cube".to_string(),
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
