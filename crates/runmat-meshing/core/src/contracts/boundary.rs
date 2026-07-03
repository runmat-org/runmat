use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{EntityKind, GeometryAsset, UnitSystem};
use serde::{Deserialize, Serialize};

use super::provenance::{MeshEntityProvenance, SourceEntityKind};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryMeshTriangle {
    pub triangle_id: u32,
    pub node_ids: [u32; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryMeshInput {
    pub mesh_id: String,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_geometry_sha256: Option<String>,
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<BoundaryMeshTriangle>,
    pub bounds_min_m: [f64; 3],
    pub bounds_max_m: [f64; 3],
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryMeshInputError {
    NoSurfaceMeshes,
    EmptySurfaceMesh {
        mesh_id: String,
    },
    NonFiniteVertex {
        mesh_id: String,
        vertex_index: usize,
    },
    TriangleIndexOutOfBounds {
        mesh_id: String,
        triangle_id: u32,
    },
    DegenerateBounds {
        mesh_id: String,
    },
    DegenerateTriangle {
        mesh_id: String,
        triangle_id: u32,
    },
    OpenBoundaryEdge {
        mesh_id: String,
        edge: [u32; 2],
        count: u32,
    },
}

impl std::fmt::Display for BoundaryMeshInputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSurfaceMeshes => write!(formatter, "geometry has no surface mesh input"),
            Self::EmptySurfaceMesh { mesh_id } => {
                write!(
                    formatter,
                    "surface mesh {mesh_id} has no vertices or triangles"
                )
            }
            Self::NonFiniteVertex {
                mesh_id,
                vertex_index,
            } => write!(
                formatter,
                "surface mesh {mesh_id} has non-finite vertex {vertex_index}"
            ),
            Self::TriangleIndexOutOfBounds {
                mesh_id,
                triangle_id,
            } => write!(
                formatter,
                "surface mesh {mesh_id} triangle {triangle_id} references an unknown vertex"
            ),
            Self::DegenerateBounds { mesh_id } => {
                write!(
                    formatter,
                    "surface mesh {mesh_id} does not span a 3D volume"
                )
            }
            Self::DegenerateTriangle {
                mesh_id,
                triangle_id,
            } => write!(
                formatter,
                "surface mesh {mesh_id} triangle {triangle_id} collapses after vertex welding"
            ),
            Self::OpenBoundaryEdge {
                mesh_id,
                edge,
                count,
            } => write!(
                formatter,
                "surface mesh {mesh_id} boundary edge {}-{} has incidence {count}, expected 2",
                edge[0], edge[1]
            ),
        }
    }
}

impl std::error::Error for BoundaryMeshInputError {}

impl BoundaryMeshInput {
    pub fn from_geometry(geometry: &GeometryAsset) -> Result<Self, BoundaryMeshInputError> {
        let surface = geometry
            .surface_meshes
            .iter()
            .min_by(|left, right| left.mesh_id.cmp(&right.mesh_id))
            .ok_or(BoundaryMeshInputError::NoSurfaceMeshes)?;
        if surface.vertices.is_empty() || surface.triangles.is_empty() {
            return Err(BoundaryMeshInputError::EmptySurfaceMesh {
                mesh_id: surface.mesh_id.clone(),
            });
        }

        let coordinate_scale = geometry_unit_scale_to_meters(geometry.units);
        let vertices_m = surface
            .vertices
            .iter()
            .map(|vertex| {
                [
                    vertex[0] * coordinate_scale,
                    vertex[1] * coordinate_scale,
                    vertex[2] * coordinate_scale,
                ]
            })
            .collect::<Vec<_>>();

        let mut min = vertices_m[0];
        let mut max = vertices_m[0];
        for (vertex_index, vertex) in vertices_m.iter().copied().enumerate() {
            if vertex.iter().any(|coordinate| !coordinate.is_finite()) {
                return Err(BoundaryMeshInputError::NonFiniteVertex {
                    mesh_id: surface.mesh_id.clone(),
                    vertex_index,
                });
            }
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        if (0..3).any(|axis| max[axis] <= min[axis]) {
            return Err(BoundaryMeshInputError::DegenerateBounds {
                mesh_id: surface.mesh_id.clone(),
            });
        }

        let (vertices, vertex_map) = weld_surface_vertices(&vertices_m, min, max);
        let mut welded_triangles = Vec::with_capacity(surface.triangles.len());
        let mut edge_incidence = BTreeMap::<[u32; 2], u32>::new();
        for (triangle_id, triangle) in surface.triangles.iter().copied().enumerate() {
            if triangle
                .iter()
                .any(|node_id| *node_id as usize >= surface.vertices.len())
            {
                return Err(BoundaryMeshInputError::TriangleIndexOutOfBounds {
                    mesh_id: surface.mesh_id.clone(),
                    triangle_id: triangle_id as u32,
                });
            }
            let triangle = [
                vertex_map[triangle[0] as usize],
                vertex_map[triangle[1] as usize],
                vertex_map[triangle[2] as usize],
            ];
            if triangle[0] == triangle[1]
                || triangle[1] == triangle[2]
                || triangle[2] == triangle[0]
            {
                continue;
            }
            for edge in triangle_edges(triangle) {
                *edge_incidence.entry(edge).or_insert(0) += 1;
            }
            welded_triangles.push((triangle_id, triangle));
        }
        if welded_triangles.is_empty() {
            return Err(BoundaryMeshInputError::EmptySurfaceMesh {
                mesh_id: surface.mesh_id.clone(),
            });
        }
        for (edge, count) in edge_incidence {
            if count != 2 {
                return Err(BoundaryMeshInputError::OpenBoundaryEdge {
                    mesh_id: surface.mesh_id.clone(),
                    edge,
                    count,
                });
            }
        }

        let mut all_region_ids = geometry
            .regions
            .iter()
            .map(|region| region.region_id.clone())
            .collect::<Vec<_>>();
        all_region_ids.sort();
        all_region_ids.dedup();

        let mut triangles = Vec::with_capacity(welded_triangles.len());
        for (triangle_id, node_ids) in welded_triangles {
            let mut region_ids = geometry
                .region_entity_mappings
                .iter()
                .filter(|mapping| {
                    mapping.mesh_id == surface.mesh_id
                        && matches!(mapping.entity_kind, EntityKind::Face | EntityKind::Element)
                        && mapping.contains_entity(triangle_id as u64)
                })
                .map(|mapping| mapping.region_id.clone())
                .collect::<Vec<_>>();
            region_ids.sort();
            region_ids.dedup();
            if region_ids.is_empty() {
                region_ids = all_region_ids.clone();
            }
            triangles.push(BoundaryMeshTriangle {
                triangle_id: triangle_id as u32,
                node_ids,
                region_ids: region_ids.clone(),
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: geometry.geometry_id.clone(),
                    source_geometry_revision: geometry.revision,
                    source_entity_kind: SourceEntityKind::Face,
                    source_entity_id: triangle_id.to_string(),
                    region_ids,
                }],
            });
        }

        let region_ids = triangles
            .iter()
            .flat_map(|triangle| triangle.region_ids.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        Ok(Self {
            mesh_id: surface.mesh_id.clone(),
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
            source_geometry_sha256: Some(geometry.source.sha256.clone()),
            vertices,
            triangles,
            bounds_min_m: min,
            bounds_max_m: max,
            region_ids,
        })
    }
}

fn geometry_unit_scale_to_meters(units: UnitSystem) -> f64 {
    match units {
        UnitSystem::Meter | UnitSystem::Unspecified => 1.0,
        UnitSystem::Millimeter => 0.001,
        UnitSystem::Inch => 0.0254,
    }
}

fn weld_surface_vertices(
    vertices: &[[f64; 3]],
    bounds_min_m: [f64; 3],
    bounds_max_m: [f64; 3],
) -> (Vec<[f64; 3]>, Vec<u32>) {
    let tolerance = weld_tolerance_m(bounds_min_m, bounds_max_m);
    let mut buckets = BTreeMap::<[i64; 3], Vec<u32>>::new();
    let mut welded_vertices = Vec::<[f64; 3]>::new();
    let mut vertex_map = Vec::<u32>::with_capacity(vertices.len());

    for vertex in vertices {
        let key = weld_key(*vertex, tolerance);
        let mut welded_id = None;
        for neighbor_key in neighboring_weld_keys(key) {
            let Some(candidates) = buckets.get(&neighbor_key) else {
                continue;
            };
            for candidate_id in candidates {
                let candidate = welded_vertices[*candidate_id as usize];
                if distance(candidate, *vertex) <= tolerance {
                    welded_id = Some(*candidate_id);
                    break;
                }
            }
            if welded_id.is_some() {
                break;
            }
        }

        let welded_id = match welded_id {
            Some(welded_id) => welded_id,
            None => {
                let welded_id = welded_vertices.len() as u32;
                welded_vertices.push(*vertex);
                buckets.entry(key).or_default().push(welded_id);
                welded_id
            }
        };
        vertex_map.push(welded_id);
    }

    (welded_vertices, vertex_map)
}

fn weld_tolerance_m(bounds_min_m: [f64; 3], bounds_max_m: [f64; 3]) -> f64 {
    let span = (0..3)
        .map(|axis| bounds_max_m[axis] - bounds_min_m[axis])
        .fold(0.0_f64, f64::max);
    (span * 1.0e-8).max(1.0e-9)
}

fn weld_key(vertex: [f64; 3], tolerance: f64) -> [i64; 3] {
    [
        (vertex[0] / tolerance).round() as i64,
        (vertex[1] / tolerance).round() as i64,
        (vertex[2] / tolerance).round() as i64,
    ]
}

fn neighboring_weld_keys(key: [i64; 3]) -> impl Iterator<Item = [i64; 3]> {
    (-1..=1).flat_map(move |dx| {
        (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| [key[0] + dx, key[1] + dy, key[2] + dz]))
    })
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(triangle[0], triangle[1]),
        sorted_edge(triangle[1], triangle[2]),
        sorted_edge(triangle[2], triangle[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    [left.min(right), left.max(right)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
        SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
    };

    #[test]
    fn boundary_input_welds_face_local_duplicate_vertices() {
        let mut geometry = cube_geometry_with_shared_vertices();
        geometry.surface_meshes[0] = SurfaceMesh::new(
            "cube_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![
                [0, 2, 1],
                [4, 6, 5],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
                [22, 23, 24],
                [25, 26, 27],
                [28, 29, 30],
                [31, 32, 33],
                [34, 35, 36],
            ],
        );

        let input = BoundaryMeshInput::from_geometry(&geometry)
            .expect("closed cube with face-local vertices should weld");

        assert_eq!(input.vertices.len(), 8);
        assert_eq!(input.triangles.len(), 12);
    }

    #[test]
    fn boundary_input_rejects_open_shell_after_welding() {
        let mut geometry = cube_geometry_with_shared_vertices();
        geometry.surface_meshes[0].triangles.pop();

        let err = BoundaryMeshInput::from_geometry(&geometry).expect_err("open shell should fail");

        assert!(matches!(
            err,
            BoundaryMeshInputError::OpenBoundaryEdge { .. }
        ));
    }

    #[test]
    fn boundary_input_converts_millimeter_vertices_to_meters() {
        let mut geometry = cube_geometry_with_shared_vertices();
        geometry.units = UnitSystem::Millimeter;
        for vertex in &mut geometry.surface_meshes[0].vertices {
            for coordinate in vertex {
                *coordinate *= 1000.0;
            }
        }

        let input = BoundaryMeshInput::from_geometry(&geometry)
            .expect("millimeter cube should convert to meter boundary input");

        assert_eq!(input.bounds_min_m, [0.0, 0.0, 0.0]);
        assert_eq!(input.bounds_max_m, [1.0, 1.0, 1.0]);
        assert!(input
            .vertices
            .iter()
            .any(|vertex| *vertex == [1.0, 1.0, 1.0]));
    }

    fn cube_geometry_with_shared_vertices() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_boundary_cube".to_string(),
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
            regions: vec![Region {
                region_id: "region_all".to_string(),
                name: "all".to_string(),
                tag: None,
                cad_ownership: None,
            }],
            region_entity_mappings: vec![RegionEntityMapping::all_faces(
                "region_all",
                "cube_surface",
                12,
            )],
            diagnostics: Vec::new(),
        }
    }
}
