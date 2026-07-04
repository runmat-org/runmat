use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{EntityKind, GeometryAsset, UnitSystem};

mod types;

pub use types::{
    SourceTopologyEdge, SourceTopologyError, SourceTopologyFace, SourceTopologyModel,
    SourceTopologyVertex,
};
pub(crate) use types::{SourceTopologyInput, SourceTopologyTriangle};

impl SourceTopologyInput {
    fn from_geometry(geometry: &GeometryAsset) -> Result<Self, SourceTopologyError> {
        let surface = geometry
            .surface_meshes
            .iter()
            .min_by(|left, right| left.mesh_id.cmp(&right.mesh_id))
            .ok_or(SourceTopologyError::NoSurfaceMeshes)?;
        if surface.vertices.is_empty() || surface.triangles.is_empty() {
            return Err(SourceTopologyError::EmptySurfaceMesh {
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
                return Err(SourceTopologyError::NonFiniteVertex {
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
            return Err(SourceTopologyError::DegenerateBounds {
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
                return Err(SourceTopologyError::TriangleIndexOutOfBounds {
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
            return Err(SourceTopologyError::EmptySurfaceMesh {
                mesh_id: surface.mesh_id.clone(),
            });
        }
        for (edge, count) in edge_incidence {
            if count != 2 {
                return Err(SourceTopologyError::OpenBoundaryEdge {
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
            triangles.push(SourceTopologyTriangle {
                triangle_id: triangle_id as u32,
                node_ids,
                region_ids,
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

pub fn extract_source_topology(
    geometry: &GeometryAsset,
) -> Result<SourceTopologyModel, SourceTopologyError> {
    let input = SourceTopologyInput::from_geometry(geometry)?;
    Ok(source_topology_from_boundary_input(&input))
}

pub(crate) fn source_topology_from_boundary_input(
    input: &SourceTopologyInput,
) -> SourceTopologyModel {
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

fn triangle_vertices(input: &SourceTopologyInput, node_ids: [u32; 3]) -> Option<[[f64; 3]; 3]> {
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

#[cfg(test)]
mod tests;
