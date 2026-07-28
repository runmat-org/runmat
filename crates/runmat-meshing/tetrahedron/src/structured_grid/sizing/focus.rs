use std::collections::BTreeMap;

use super::super::*;

pub(crate) fn append_geometry_focus_sizing_samples(
    input: &BoundaryMeshInput,
    options: &VolumeMeshingOptions,
    sizing: &mut MeshSizingField,
) {
    if options.refinement.focus.curvature {
        sizing.samples.extend(curvature_sizing_samples(input));
    }
    if options.refinement.focus.small_features {
        sizing.samples.extend(small_feature_sizing_samples(input));
    }
}

fn curvature_sizing_samples(input: &BoundaryMeshInput) -> Vec<SizingSample> {
    let mut triangles_by_edge = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (triangle_index, triangle) in input.triangles.iter().enumerate() {
        for edge in triangle_edges(triangle.node_ids) {
            triangles_by_edge
                .entry(edge)
                .or_default()
                .push(triangle_index);
        }
    }

    triangles_by_edge
        .into_iter()
        .filter_map(|(edge, triangle_indices)| {
            if triangle_indices.len() != 2 {
                return None;
            }
            let left = input.triangles.get(triangle_indices[0])?;
            let right = input.triangles.get(triangle_indices[1])?;
            let left_normal = triangle_unit_normal(input, left.node_ids)?;
            let right_normal = triangle_unit_normal(input, right.node_ids)?;
            let normal_dot = dot(left_normal, right_normal).clamp(-1.0, 1.0);
            if 1.0 - normal_dot.abs() <= 0.05 {
                return None;
            }
            let left_vertex = *input.vertices.get(edge[0] as usize)?;
            let right_vertex = *input.vertices.get(edge[1] as usize)?;
            let edge_length = distance(left_vertex, right_vertex);
            (edge_length.is_finite() && edge_length > 0.0).then_some(SizingSample {
                position_m: midpoint(left_vertex, right_vertex),
                target_size_m: edge_length * 0.5,
                reason: Some("geometry.curvature".to_string()),
            })
        })
        .collect()
}

fn small_feature_sizing_samples(input: &BoundaryMeshInput) -> Vec<SizingSample> {
    let max_span = boundary_max_span(input);
    if !max_span.is_finite() || max_span <= 0.0 {
        return Vec::new();
    }
    let threshold = max_span * 0.35;
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            let min_edge = triangle_min_edge(vertices);
            if !min_edge.is_finite() || min_edge <= 0.0 || min_edge > threshold {
                return None;
            }
            Some(SizingSample {
                position_m: triangle_centroid(vertices),
                target_size_m: min_edge * 0.5,
                reason: Some("geometry.small_features".to_string()),
            })
        })
        .collect()
}
