use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::GeometryAsset;

mod geometry;
mod input;
mod types;
mod weld;

use geometry::{distance, triangle_area, triangle_unit_normal, triangle_vertices};

pub use types::{
    SourceTopologyEdge, SourceTopologyError, SourceTopologyFace, SourceTopologyModel,
    SourceTopologyVertex,
};
pub(crate) use types::{SourceTopologyInput, SourceTopologyTriangle};

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

#[cfg(test)]
mod tests;
