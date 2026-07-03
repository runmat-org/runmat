use runmat_geometry_core::GeometryAsset;
use std::collections::{BTreeMap, BTreeSet};

use super::types::{default_coordinate_characteristic_length_m, default_coordinate_span_m};

#[derive(Debug, Clone, Default)]
pub(super) struct ElementGeometryMetrics {
    pub(super) node_count: u64,
    pub(super) edge_count: u64,
    pub(super) mean_edge_length_m: f64,
    pub(super) mean_area_m2: f64,
    pub(super) coverage_ratio: f64,
    pub(super) reference_coordinates_m: [[f64; 3]; 3],
    pub(super) reference_area_m2: f64,
    pub(super) control_volume_cell_count: u64,
    pub(super) control_volume_face_count: u64,
    pub(super) control_volume_internal_face_count: u64,
    pub(super) control_volume_boundary_face_count: u64,
    pub(super) element_topology_sample: ElementTopologySample,
    pub(super) element_topology_node_coordinates_m: Vec<[f64; 3]>,
    pub(super) element_topology_edge_nodes: Vec<[u32; 2]>,
    pub(super) element_topology_element_edges: Vec<[u32; 3]>,
    pub(super) element_topology_element_orientations: Vec<[i8; 3]>,
    pub(super) element_topology_element_areas_m2: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ElementTopologySample {
    pub(super) element_count: u64,
    pub(super) edge_count: u64,
    pub(super) edge_nodes: [[u32; 2]; 8],
    pub(super) node_coordinates_m: [[f64; 3]; 8],
    pub(super) element_edges: [[u32; 3]; 4],
    pub(super) element_orientations: [[i8; 3]; 4],
    pub(super) element_areas_m2: [f64; 4],
}

pub(super) fn mesh_element_geometry_metrics(
    geometry: &GeometryAsset,
    mesh_id: &str,
) -> ElementGeometryMetrics {
    let Some(surface) = geometry
        .surface_meshes
        .iter()
        .find(|surface| surface.mesh_id == mesh_id)
    else {
        return ElementGeometryMetrics::default();
    };
    let Some(descriptor) = geometry.meshes.iter().find(|mesh| mesh.mesh_id == mesh_id) else {
        return ElementGeometryMetrics::default();
    };

    let mut referenced_nodes = BTreeSet::<u32>::new();
    let mut unique_edges = BTreeSet::<(u32, u32)>::new();
    let mut edge_length_sum = 0.0_f64;
    let mut edge_length_count = 0_u64;
    let mut area_sum = 0.0_f64;
    let mut valid_triangle_count = 0_u64;
    let mut edge_incidence = BTreeMap::<(u32, u32), u64>::new();
    let mut edge_indices = BTreeMap::<(u32, u32), u32>::new();
    let mut element_topology_sample = ElementTopologySample::default();
    let element_topology_node_coordinates_m = surface.vertices.clone();
    let mut element_topology_edge_nodes = Vec::<[u32; 2]>::new();
    let mut element_topology_element_edges = Vec::<[u32; 3]>::new();
    let mut element_topology_element_orientations = Vec::<[i8; 3]>::new();
    let mut element_topology_element_areas_m2 = Vec::<f64>::new();
    let mut reference_coordinates_m = [[0.0_f64; 3]; 3];
    let mut reference_area_m2 = 0.0_f64;
    for triangle in &surface.triangles {
        let indices = [triangle[0], triangle[1], triangle[2]];
        let Some(vertices) = triangle_vertices(&surface.vertices, indices) else {
            continue;
        };
        valid_triangle_count += 1;
        let triangle_area = triangle_area_m2(vertices);
        if reference_area_m2 == 0.0 && triangle_area.is_finite() && triangle_area > 0.0 {
            reference_coordinates_m = vertices;
            reference_area_m2 = triangle_area;
        }
        for index in indices {
            referenced_nodes.insert(index);
        }
        for (index, vertex) in indices.into_iter().zip(vertices) {
            if (index as usize) < element_topology_sample.node_coordinates_m.len() {
                element_topology_sample.node_coordinates_m[index as usize] = vertex;
            }
        }
        for (left, right) in [
            (indices[0], indices[1]),
            (indices[1], indices[2]),
            (indices[2], indices[0]),
        ] {
            let edge = (left.min(right), left.max(right));
            unique_edges.insert(edge);
            *edge_incidence.entry(edge).or_insert(0) += 1;
            if !edge_indices.contains_key(&edge) {
                let edge_index = edge_indices.len() as u32;
                edge_indices.insert(edge, edge_index);
                element_topology_edge_nodes.push([edge.0, edge.1]);
                if (edge_index as usize) < element_topology_sample.edge_nodes.len() {
                    element_topology_sample.edge_nodes[edge_index as usize] = [edge.0, edge.1];
                    element_topology_sample.edge_count = (edge_index as u64 + 1)
                        .min(element_topology_sample.edge_nodes.len() as u64);
                }
            }
        }
        let mut full_element_edges = [0_u32; 3];
        let mut full_element_orientations = [0_i8; 3];
        for (local_index, (left, right)) in [
            (indices[0], indices[1]),
            (indices[1], indices[2]),
            (indices[2], indices[0]),
        ]
        .into_iter()
        .enumerate()
        {
            let edge = (left.min(right), left.max(right));
            full_element_edges[local_index] = *edge_indices.get(&edge).unwrap_or(&0);
            full_element_orientations[local_index] = if left <= right { 1 } else { -1 };
        }
        element_topology_element_edges.push(full_element_edges);
        element_topology_element_orientations.push(full_element_orientations);
        element_topology_element_areas_m2.push(triangle_area);
        if (element_topology_sample.element_count as usize) < 4 {
            let element_index = element_topology_sample.element_count as usize;
            for (local_index, (left, right)) in [
                (indices[0], indices[1]),
                (indices[1], indices[2]),
                (indices[2], indices[0]),
            ]
            .into_iter()
            .enumerate()
            {
                let edge = (left.min(right), left.max(right));
                element_topology_sample.element_edges[element_index][local_index] =
                    *edge_indices.get(&edge).unwrap_or(&0);
                element_topology_sample.element_orientations[element_index][local_index] =
                    if left <= right { 1 } else { -1 };
            }
            element_topology_sample.element_areas_m2[element_index] = triangle_area;
            element_topology_sample.element_count += 1;
        }
        for (left, right) in [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ] {
            edge_length_sum += distance_m(left, right);
            edge_length_count += 1;
        }
        area_sum += triangle_area;
    }

    let control_volume_internal_face_count =
        edge_incidence.values().filter(|count| **count > 1).count() as u64;
    let control_volume_boundary_face_count =
        edge_incidence.values().filter(|count| **count == 1).count() as u64;

    ElementGeometryMetrics {
        node_count: referenced_nodes.len() as u64,
        edge_count: unique_edges.len() as u64,
        mean_edge_length_m: if edge_length_count == 0 {
            0.0
        } else {
            edge_length_sum / edge_length_count as f64
        },
        mean_area_m2: if valid_triangle_count == 0 {
            0.0
        } else {
            area_sum / valid_triangle_count as f64
        },
        coverage_ratio: if descriptor.element_count == 0 {
            0.0
        } else {
            (valid_triangle_count as f64 / descriptor.element_count as f64).clamp(0.0, 1.0)
        },
        reference_coordinates_m,
        reference_area_m2,
        control_volume_cell_count: valid_triangle_count,
        control_volume_face_count: unique_edges.len() as u64,
        control_volume_internal_face_count,
        control_volume_boundary_face_count,
        element_topology_sample,
        element_topology_node_coordinates_m,
        element_topology_edge_nodes,
        element_topology_element_edges,
        element_topology_element_orientations,
        element_topology_element_areas_m2,
    }
}

pub(super) fn mesh_coordinate_span_m(geometry: &GeometryAsset, mesh_id: &str) -> [f64; 3] {
    let Some(surface) = geometry
        .surface_meshes
        .iter()
        .find(|surface| surface.mesh_id == mesh_id)
    else {
        return default_coordinate_span_m();
    };
    let Some(first) = surface.vertices.first().copied() else {
        return default_coordinate_span_m();
    };
    let mut min = first;
    let mut max = first;
    for vertex in &surface.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    [
        finite_positive_or_default(max[0] - min[0], 0.0),
        finite_positive_or_default(max[1] - min[1], 0.0),
        finite_positive_or_default(max[2] - min[2], 0.0),
    ]
}

pub(super) fn coordinate_active_dimension_count(span_m: [f64; 3]) -> u8 {
    span_m
        .iter()
        .filter(|span| span.is_finite() && **span > 1.0e-12)
        .count()
        .max(1) as u8
}

pub(super) fn coordinate_characteristic_length_m(
    span_m: [f64; 3],
    active_dimension_count: u8,
    node_count: u64,
) -> f64 {
    let active_spans = span_m
        .into_iter()
        .filter(|span| span.is_finite() && *span > 1.0e-12)
        .collect::<Vec<_>>();
    if active_spans.is_empty() {
        return default_coordinate_characteristic_length_m();
    }
    let domain_measure = active_spans.iter().product::<f64>();
    let node_scale = (node_count.max(2) as f64).powf(1.0 / active_dimension_count.max(1) as f64);
    finite_positive_or_default(
        domain_measure.powf(1.0 / active_dimension_count as f64) / node_scale,
        1.0,
    )
}

fn triangle_vertices(vertices: &[[f64; 3]], indices: [u32; 3]) -> Option<[[f64; 3]; 3]> {
    let a = *vertices.get(indices[0] as usize)?;
    let b = *vertices.get(indices[1] as usize)?;
    let c = *vertices.get(indices[2] as usize)?;
    Some([a, b, c])
}

fn distance_m(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((right[0] - left[0]).powi(2) + (right[1] - left[1]).powi(2) + (right[2] - left[2]).powi(2))
        .sqrt()
}

fn triangle_area_m2(vertices: [[f64; 3]; 3]) -> f64 {
    let ab = [
        vertices[1][0] - vertices[0][0],
        vertices[1][1] - vertices[0][1],
        vertices[1][2] - vertices[0][2],
    ];
    let ac = [
        vertices[2][0] - vertices[0][0],
        vertices[2][1] - vertices[0][1],
        vertices[2][2] - vertices[0][2],
    ];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

fn finite_positive_or_default(value: f64, default: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        default
    }
}
