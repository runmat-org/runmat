use super::*;

pub(super) fn quality_report(
    elements: Vec<ElementQuality>,
    boundary_projection_errors_m: Vec<f64>,
) -> AnalysisMeshQualityReport {
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let min_exact_scaled_jacobian = elements
        .iter()
        .map(|element| element.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = if elements.is_empty() {
        0.0
    } else {
        elements
            .iter()
            .map(|element| element.aspect_ratio)
            .sum::<f64>()
            / elements.len() as f64
    };
    let mean_boundary_projection_error_m = if boundary_projection_errors_m.is_empty() {
        0.0
    } else {
        boundary_projection_errors_m.iter().sum::<f64>() / boundary_projection_errors_m.len() as f64
    };
    let max_boundary_projection_error_m = boundary_projection_errors_m
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    AnalysisMeshQualityReport {
        min_scaled_jacobian,
        min_exact_scaled_jacobian,
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: 0,
        mean_boundary_projection_error_m,
        max_boundary_projection_error_m,
        elements,
    }
}

pub(super) fn boundary_projection_errors(
    input: &BoundaryMeshInput,
    boundary_faces: &[AnalysisBoundaryFace],
    nodes: &[AnalysisMeshNode],
) -> Vec<f64> {
    boundary_faces
        .iter()
        .filter_map(|face| {
            let centroid = element_centroid(nodes, &face.node_ids)?;
            nearest_boundary_triangle_distance(input, centroid)
        })
        .filter(|distance_m| distance_m.is_finite())
        .collect()
}

pub(super) fn project_boundary_nodes_if_quality_improves(
    input: &BoundaryMeshInput,
    nodes: Vec<AnalysisMeshNode>,
    volume_elements: &[AnalysisVolumeElement],
    boundary_faces: &[AnalysisBoundaryFace],
    original_quality: AnalysisMeshQualityReport,
) -> (Vec<AnalysisMeshNode>, AnalysisMeshQualityReport) {
    let mut projection_targets = Vec::<(u32, [f64; 3], [f64; 3])>::new();
    for node_id in boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let Some(node) = nodes.get(node_id.saturating_sub(1) as usize) else {
            continue;
        };
        let Some(projected) = nearest_boundary_triangle_point(input, node.coordinates_m) else {
            continue;
        };
        if distance(node.coordinates_m, projected) > 0.0 {
            projection_targets.push((node_id, node.coordinates_m, projected));
        }
    }
    if projection_targets.is_empty() {
        return (nodes, original_quality);
    }

    let thresholds = QualityThresholds::default();
    let mut best_nodes = nodes.clone();
    let mut best_quality = original_quality.clone();
    for relaxation in [1.0_f64, 0.75, 0.5, 0.25, 0.125] {
        let mut candidate_nodes = nodes.clone();
        for (node_id, original, projected) in &projection_targets {
            let Some(node) = candidate_nodes.get_mut(node_id.saturating_sub(1) as usize) else {
                continue;
            };
            node.coordinates_m = [
                original[0] + relaxation * (projected[0] - original[0]),
                original[1] + relaxation * (projected[1] - original[1]),
                original[2] + relaxation * (projected[2] - original[2]),
            ];
        }

        let Some(candidate_element_quality) =
            element_quality_for_nodes(volume_elements, &candidate_nodes)
        else {
            continue;
        };
        let candidate_quality = quality_report(
            candidate_element_quality.clone(),
            boundary_projection_errors(input, boundary_faces, &candidate_nodes),
        );
        let improved_projection = candidate_quality.max_boundary_projection_error_m
            < best_quality.max_boundary_projection_error_m;
        let quality_ok = candidate_quality.min_scaled_jacobian.is_finite()
            && candidate_quality.min_scaled_jacobian >= thresholds.min_scaled_jacobian
            && candidate_quality.max_aspect_ratio.is_finite()
            && candidate_quality.max_aspect_ratio <= thresholds.max_aspect_ratio
            && candidate_quality.inverted_element_count == 0
            && evaluate_boundary_quality_candidate(
                &best_quality.elements,
                &candidate_element_quality,
                BoundaryQualityCandidateConstraints {
                    boundary_recovery_preserved: true,
                    target_volume_preserved: true,
                    source_provenance_preserved: true,
                },
                BoundaryQualityCandidateOptions {
                    allow_min_exact_quality_regression_above_threshold: true,
                    require_exact_quality_improvement: false,
                    ..BoundaryQualityCandidateOptions::default()
                },
            )
            .is_ok_and(|evaluation| evaluation.accepted);
        if improved_projection && quality_ok {
            best_nodes = candidate_nodes;
            best_quality = candidate_quality;
        }
    }
    (best_nodes, best_quality)
}

pub(super) fn element_quality_for_nodes(
    volume_elements: &[AnalysisVolumeElement],
    nodes: &[AnalysisMeshNode],
) -> Option<Vec<ElementQuality>> {
    let mut qualities = Vec::with_capacity(volume_elements.len());
    for element in volume_elements {
        let node_ids: [u32; 4] = element.node_ids.as_slice().try_into().ok()?;
        let volume_m3 = tet_volume(node_ids, nodes);
        if !volume_m3.is_finite() || volume_m3 <= 0.0 {
            return None;
        }
        let aspect_ratio = tet_aspect_ratio(node_ids, nodes);
        let exact_scaled_jacobian = tet_points(node_ids, nodes)
            .map(tet_scaled_jacobian)
            .unwrap_or(0.0);
        qualities.push(ElementQuality {
            element_id: element.element_id.clone(),
            scaled_jacobian: 1.0 / aspect_ratio.max(1.0),
            exact_scaled_jacobian,
            aspect_ratio,
            volume_m3,
        });
    }
    Some(qualities)
}

fn nearest_boundary_triangle_distance(input: &BoundaryMeshInput, point: [f64; 3]) -> Option<f64> {
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            Some(point_triangle_distance(point, vertices))
        })
        .filter(|distance_m| distance_m.is_finite())
        .min_by(f64::total_cmp)
}

fn nearest_boundary_triangle_point(input: &BoundaryMeshInput, point: [f64; 3]) -> Option<[f64; 3]> {
    input
        .triangles
        .iter()
        .filter_map(|triangle| {
            let vertices = triangle_vertices(input, triangle.node_ids)?;
            let closest = closest_point_on_triangle(point, vertices);
            Some((distance(point, closest), closest))
        })
        .filter(|(distance_m, _)| distance_m.is_finite())
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, closest)| closest)
}

fn point_triangle_distance(point: [f64; 3], vertices: [[f64; 3]; 3]) -> f64 {
    distance(point, closest_point_on_triangle(point, vertices))
}

fn closest_point_on_triangle(point: [f64; 3], vertices: [[f64; 3]; 3]) -> [f64; 3] {
    let [a, b, c] = vertices;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(point, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }

    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let bc = sub(c, b);
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(bc, w));
    }

    let normal = cross(ab, ac);
    let normal_dot = dot(normal, normal);
    if normal_dot <= f64::EPSILON {
        return [a, b, c]
            .into_iter()
            .min_by(|left, right| distance(point, *left).total_cmp(&distance(point, *right)))
            .unwrap_or(a);
    }
    let signed_distance_scale = dot(ap, normal) / normal_dot;
    sub(point, scale(normal, signed_distance_scale))
}

pub(super) fn orient_tet(mut node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> [u32; 4] {
    if tet_volume(node_ids, nodes) < 0.0 {
        node_ids.swap(0, 1);
    }
    node_ids
}

pub(super) fn tet_volume(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> f64 {
    let Some([a, b, c, d]) = tet_points(node_ids, nodes) else {
        return 0.0;
    };
    dot(sub(b, a), cross(sub(c, a), sub(d, a))) / 6.0
}

pub(super) fn tet_aspect_ratio(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> f64 {
    let Some(points) = tet_points(node_ids, nodes) else {
        return f64::INFINITY;
    };
    let mut min_edge = f64::INFINITY;
    let mut max_edge = 0.0_f64;
    for (left, right) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let length = norm(sub(points[left], points[right]));
        min_edge = min_edge.min(length);
        max_edge = max_edge.max(length);
    }
    max_edge / min_edge.max(f64::EPSILON)
}

pub(super) fn tet_points(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> Option<[[f64; 3]; 4]> {
    Some([
        nodes
            .get(node_ids[0].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[1].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[2].checked_sub(1)? as usize)?
            .coordinates_m,
        nodes
            .get(node_ids[3].checked_sub(1)? as usize)?
            .coordinates_m,
    ])
}

fn element_centroid(nodes: &[AnalysisMeshNode], node_ids: &[u32]) -> Option<[f64; 3]> {
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0; 3];
    for node_id in node_ids {
        let coordinates = nodes.get(node_id.checked_sub(1)? as usize)?.coordinates_m;
        centroid[0] += coordinates[0];
        centroid[1] += coordinates[1];
        centroid[2] += coordinates[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}
