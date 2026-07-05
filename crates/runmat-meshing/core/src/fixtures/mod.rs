use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_cad::{
    SourceTopologyEdge, SourceTopologyFace, SourceTopologyModel, SourceTopologyVertex,
};

pub const MODULE_PURPOSE: &str = "generic topology-first fixture builders and expected evidence";

pub fn generic_line_source_topology(length_m: f64) -> SourceTopologyModel {
    SourceTopologyModel {
        mesh_id: "generic_line".to_string(),
        source_geometry_id: "generic_line_geometry".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices: vec![
            SourceTopologyVertex {
                vertex_id: 0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            SourceTopologyVertex {
                vertex_id: 1,
                coordinates_m: [length_m, 0.0, 0.0],
            },
        ],
        edges: vec![SourceTopologyEdge {
            edge_id: 0,
            node_ids: [0, 1],
            adjacent_face_ids: Vec::new(),
            region_ids: vec!["generic_line_edge".to_string()],
            length_m,
        }],
        faces: Vec::new(),
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [length_m, 0.0, 0.0],
        region_ids: vec!["generic_line_edge".to_string()],
        material_region_ids: Vec::new(),
    }
}

pub fn generic_triangle_source_topology() -> SourceTopologyModel {
    source_topology_from_triangles(
        "generic_triangle",
        "generic_triangle_geometry",
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![([0, 1, 2], vec!["generic_triangle_face".to_string()])],
        vec!["generic_triangle_face".to_string()],
    )
}

pub fn generic_cube_source_topology() -> SourceTopologyModel {
    source_topology_from_triangles(
        "generic_cube_surface",
        "generic_cube_geometry",
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
            ([0, 2, 1], vec!["generic_cube_boundary".to_string()]),
            ([0, 3, 2], vec!["generic_cube_boundary".to_string()]),
            ([4, 5, 6], vec!["generic_cube_boundary".to_string()]),
            ([4, 6, 7], vec!["generic_cube_boundary".to_string()]),
            ([0, 1, 5], vec!["generic_cube_boundary".to_string()]),
            ([0, 5, 4], vec!["generic_cube_boundary".to_string()]),
            ([1, 2, 6], vec!["generic_cube_boundary".to_string()]),
            ([1, 6, 5], vec!["generic_cube_boundary".to_string()]),
            ([2, 3, 7], vec!["generic_cube_boundary".to_string()]),
            ([2, 7, 6], vec!["generic_cube_boundary".to_string()]),
            ([3, 0, 4], vec!["generic_cube_boundary".to_string()]),
            ([3, 4, 7], vec!["generic_cube_boundary".to_string()]),
        ],
        vec!["generic_cube_boundary".to_string()],
    )
}

fn source_topology_from_triangles(
    mesh_id: &str,
    source_geometry_id: &str,
    vertices_m: Vec<[f64; 3]>,
    triangles: Vec<([u32; 3], Vec<String>)>,
    region_ids: Vec<String>,
) -> SourceTopologyModel {
    let vertices = vertices_m
        .iter()
        .copied()
        .enumerate()
        .map(|(vertex_id, coordinates_m)| SourceTopologyVertex {
            vertex_id: vertex_id as u32,
            coordinates_m,
        })
        .collect::<Vec<_>>();
    let mut edge_ids = BTreeMap::<[u32; 2], u32>::new();
    let mut edge_faces = BTreeMap::<[u32; 2], Vec<u32>>::new();
    let mut edge_regions = BTreeMap::<[u32; 2], BTreeSet<String>>::new();
    let mut faces = Vec::<SourceTopologyFace>::with_capacity(triangles.len());

    for (face_id, (node_ids, face_region_ids)) in triangles.into_iter().enumerate() {
        let face_id = face_id as u32;
        let mut face_edge_ids = [0_u32; 3];
        for (edge_index, edge) in triangle_edges(node_ids).into_iter().enumerate() {
            let next_edge_id = edge_ids.len() as u32;
            let edge_id = *edge_ids.entry(edge).or_insert(next_edge_id);
            face_edge_ids[edge_index] = edge_id;
            edge_faces.entry(edge).or_default().push(face_id);
            edge_regions
                .entry(edge)
                .or_default()
                .extend(face_region_ids.iter().cloned());
        }
        let triangle_vertices = [
            vertices_m[node_ids[0] as usize],
            vertices_m[node_ids[1] as usize],
            vertices_m[node_ids[2] as usize],
        ];
        faces.push(SourceTopologyFace {
            face_id,
            source_triangle_id: face_id,
            node_ids,
            edge_ids: face_edge_ids,
            region_ids: face_region_ids,
            material_region_ids: Vec::new(),
            area_m2: triangle_area(triangle_vertices),
            unit_normal: triangle_unit_normal(triangle_vertices),
        });
    }

    let mut edges = edge_ids
        .iter()
        .map(|(node_ids, edge_id)| SourceTopologyEdge {
            edge_id: *edge_id,
            node_ids: *node_ids,
            adjacent_face_ids: edge_faces.remove(node_ids).unwrap_or_default(),
            region_ids: edge_regions
                .remove(node_ids)
                .unwrap_or_default()
                .into_iter()
                .collect(),
            length_m: distance(
                vertices_m[node_ids[0] as usize],
                vertices_m[node_ids[1] as usize],
            ),
        })
        .collect::<Vec<_>>();
    edges.sort_by_key(|edge| edge.edge_id);

    SourceTopologyModel {
        mesh_id: mesh_id.to_string(),
        source_geometry_id: source_geometry_id.to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices,
        edges,
        faces,
        bounds_min_m: bounds_min(&vertices_m),
        bounds_max_m: bounds_max(&vertices_m),
        region_ids,
        material_region_ids: Vec::new(),
    }
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

fn triangle_area(vertices: [[f64; 3]; 3]) -> f64 {
    let cross = cross(sub(vertices[1], vertices[0]), sub(vertices[2], vertices[0]));
    0.5 * norm(cross)
}

fn triangle_unit_normal(vertices: [[f64; 3]; 3]) -> [f64; 3] {
    let cross = cross(sub(vertices[1], vertices[0]), sub(vertices[2], vertices[0]));
    let length = norm(cross);
    if length <= f64::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [cross[0] / length, cross[1] / length, cross[2] / length]
    }
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

fn norm(vector: [f64; 3]) -> f64 {
    (vector[0].powi(2) + vector[1].powi(2) + vector[2].powi(2)).sqrt()
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    norm(sub(left, right))
}

fn bounds_min(vertices_m: &[[f64; 3]]) -> [f64; 3] {
    vertices_m
        .iter()
        .fold([f64::INFINITY; 3], |mut bounds, vertex| {
            for axis in 0..3 {
                bounds[axis] = bounds[axis].min(vertex[axis]);
            }
            bounds
        })
}

fn bounds_max(vertices_m: &[[f64; 3]]) -> [f64; 3] {
    vertices_m
        .iter()
        .fold([f64::NEG_INFINITY; 3], |mut bounds, vertex| {
            for axis in 0..3 {
                bounds[axis] = bounds[axis].max(vertex[axis]);
            }
            bounds
        })
}

#[cfg(test)]
mod tests {
    use super::{
        generic_cube_source_topology, generic_line_source_topology,
        generic_triangle_source_topology,
    };

    #[test]
    fn line_fixture_preserves_source_edge_endpoint_topology() {
        let topology = generic_line_source_topology(2.5);

        assert_eq!(topology.vertices.len(), 2);
        assert_eq!(topology.edges.len(), 1);
        assert!(topology.faces.is_empty());
        assert_eq!(topology.edges[0].node_ids, [0, 1]);
        assert_eq!(topology.edges[0].length_m, 2.5);
        assert!(topology.edges[0].adjacent_face_ids.is_empty());
    }

    #[test]
    fn triangle_fixture_exposes_face_edges_and_normal() {
        let topology = generic_triangle_source_topology();

        assert_eq!(topology.vertices.len(), 3);
        assert_eq!(topology.edges.len(), 3);
        assert_eq!(topology.faces.len(), 1);
        assert_eq!(topology.faces[0].edge_ids.len(), 3);
        assert_eq!(topology.faces[0].area_m2, 0.5);
        assert_eq!(topology.faces[0].unit_normal, [0.0, 0.0, 1.0]);
        assert!(topology
            .edges
            .iter()
            .all(|edge| edge.adjacent_face_ids == vec![0]));
    }

    #[test]
    fn cube_fixture_exposes_closed_boundary_topology() {
        let topology = generic_cube_source_topology();

        assert_eq!(topology.vertices.len(), 8);
        assert_eq!(topology.faces.len(), 12);
        assert_eq!(topology.edges.len(), 18);
        assert_eq!(topology.bounds_min_m, [0.0, 0.0, 0.0]);
        assert_eq!(topology.bounds_max_m, [1.0, 1.0, 1.0]);
        assert!(topology
            .edges
            .iter()
            .all(|edge| edge.adjacent_face_ids.len() == 2));
    }
}
