use std::collections::BTreeMap;

use runmat_meshing_cad::{project_to_face, CadFaceEvaluationFrame, SourceTopologyFace};

use crate::math::{cross, dot, sub, triangle_area};

use super::{
    boundary::FaceCurveSegment,
    geometry::{
        boundary_loop_polygons, boundary_triangulation_points, sorted_node_pair, triangle_edges_2d,
    },
    sampling::{
        append_exact_face_domain_sample_points, append_face_lattice_points, face_area_is_recovered,
        face_edges_are_recovered, has_exact_face_domain_samples, ExactCadSampleSurfaceReport,
    },
    triangulation::{triangulate_face_points, triangulate_triangle_points_by_insertion},
    SurfaceElement, SurfaceNode, INTERNAL_SOURCE_EDGE_ID,
};

fn face_centroid_from_segments(nodes: &[SurfaceNode], segments: &[FaceCurveSegment]) -> [f64; 3] {
    let mut sum = [0.0_f64; 3];
    let mut count = 0.0_f64;
    for segment in segments {
        let point = nodes[segment.node_ids[0] as usize].coordinates_m;
        sum[0] += point[0];
        sum[1] += point[1];
        sum[2] += point[2];
        count += 1.0;
    }
    if count <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [sum[0] / count, sum[1] / count, sum[2] / count]
}

pub(super) fn append_curve_driven_face_elements(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    segment_loops: &[Vec<FaceCurveSegment>],
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) -> ExactCadSampleSurfaceReport {
    let segments = segment_loops
        .iter()
        .flat_map(|loop_segments| loop_segments.iter().copied())
        .collect::<Vec<_>>();
    if segments.len() <= 3 && !has_exact_face_domain_samples(frame) {
        if segments.len() == 3 {
            append_curve_triangle_face_element(face, frame, &segments, nodes, elements);
        } else {
            append_curve_fan_face_elements(face, frame, &segments, nodes, elements);
        }
        return ExactCadSampleSurfaceReport::default();
    }

    let node_start = nodes.len();
    let element_start = elements.len();
    let mut boundary_edge_ids = BTreeMap::<[u32; 2], u32>::new();
    for segment in &segments {
        boundary_edge_ids.insert(
            sorted_node_pair(segment.node_ids[0], segment.node_ids[1]),
            segment.source_edge_id,
        );
    }

    let mut points = boundary_triangulation_points(frame, &segments, nodes);
    let boundary_point_count = points.len();
    let boundary_polygons = boundary_loop_polygons(frame, segment_loops, nodes);
    let sample_report =
        append_exact_face_domain_sample_points(face, frame, &boundary_polygons, nodes, &mut points);
    append_face_lattice_points(
        face,
        frame,
        &boundary_polygons,
        &segments,
        nodes,
        &mut points,
    );
    let triangles = if segment_loops.len() == 1 && boundary_point_count == 3 {
        triangulate_triangle_points_by_insertion(&points, boundary_point_count)
    } else {
        triangulate_face_points(&points, &boundary_polygons)
    };
    if triangles.is_empty() {
        append_curve_fan_face_elements(face, frame, &segments, nodes, elements);
        return sample_report;
    }

    for triangle in triangles {
        let mut node_ids = triangle.point_indices.map(|index| points[index].node_id);
        if node_ids[0] == node_ids[1] || node_ids[1] == node_ids[2] || node_ids[2] == node_ids[0] {
            continue;
        }
        let mut parametric_node_uv = triangle.point_indices.map(|index| points[index].uv);
        let mut coordinates = node_ids.map(|node_id| nodes[node_id as usize].coordinates_m);
        if triangle_area(coordinates) <= f64::EPSILON {
            continue;
        }
        if dot(
            cross(
                sub(coordinates[1], coordinates[0]),
                sub(coordinates[2], coordinates[0]),
            ),
            frame.unit_normal,
        ) < 0.0
        {
            node_ids.swap(1, 2);
            parametric_node_uv.swap(1, 2);
            coordinates.swap(1, 2);
        }
        let source_edge_ids = [
            boundary_edge_ids
                .get(&sorted_node_pair(node_ids[0], node_ids[1]))
                .copied()
                .unwrap_or(INTERNAL_SOURCE_EDGE_ID),
            boundary_edge_ids
                .get(&sorted_node_pair(node_ids[1], node_ids[2]))
                .copied()
                .unwrap_or(INTERNAL_SOURCE_EDGE_ID),
            boundary_edge_ids
                .get(&sorted_node_pair(node_ids[2], node_ids[0]))
                .copied()
                .unwrap_or(INTERNAL_SOURCE_EDGE_ID),
        ];
        let max_projection_error_m = node_ids
            .iter()
            .map(|node_id| {
                project_to_face(frame, nodes[*node_id as usize].coordinates_m).distance_m
            })
            .fold(0.0_f64, f64::max);
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: Some(frame.face_id.clone()),
            source_edge_ids,
            node_ids,
            parametric_node_uv,
            max_projection_error_m,
            region_ids: face.region_ids.clone(),
            material_region_ids: face.material_region_ids.clone(),
            area_m2: triangle_area(coordinates),
            unit_normal: frame.unit_normal,
        });
    }
    if !face_area_is_recovered(face, &elements[element_start..])
        || !face_edges_are_recovered(&elements[element_start..], &boundary_edge_ids)
    {
        nodes.truncate(node_start);
        elements.truncate(element_start);
        append_curve_fan_face_elements(face, frame, &segments, nodes, elements);
        return sample_report.rejected_after_area_guard();
    }
    sample_report
}

pub(super) fn append_curve_fan_face_elements(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    segments: &[FaceCurveSegment],
    nodes: &mut Vec<SurfaceNode>,
    elements: &mut Vec<SurfaceElement>,
) {
    let centroid = face_centroid_from_segments(nodes, segments);
    let centroid_projection = project_to_face(frame, centroid);
    let centroid_node_id = nodes.len() as u32;
    nodes.push(SurfaceNode {
        node_id: centroid_node_id,
        source_vertex_id: u32::MAX,
        coordinates_m: centroid,
    });

    for segment in segments {
        let mut node_ids = [segment.node_ids[0], segment.node_ids[1], centroid_node_id];
        let mut points = [
            nodes[segment.node_ids[0] as usize].coordinates_m,
            nodes[segment.node_ids[1] as usize].coordinates_m,
            centroid,
        ];
        let mut parametric_node_uv = [
            project_to_face(frame, points[0]).uv,
            project_to_face(frame, points[1]).uv,
            centroid_projection.uv,
        ];
        if dot(
            cross(sub(points[1], points[0]), sub(points[2], points[0])),
            frame.unit_normal,
        ) < 0.0
        {
            node_ids.swap(1, 2);
            points.swap(1, 2);
            parametric_node_uv.swap(1, 2);
        }
        let left_projection = project_to_face(frame, points[0]);
        let right_projection = project_to_face(frame, points[1]);
        let max_projection_error_m = left_projection
            .distance_m
            .max(right_projection.distance_m)
            .max(centroid_projection.distance_m);
        let segment_edge = sorted_node_pair(segment.node_ids[0], segment.node_ids[1]);
        let source_edge_ids = triangle_edges_2d([0, 1, 2]).map(|edge| {
            if sorted_node_pair(node_ids[edge[0]], node_ids[edge[1]]) == segment_edge {
                segment.source_edge_id
            } else {
                INTERNAL_SOURCE_EDGE_ID
            }
        });
        elements.push(SurfaceElement {
            element_id: elements.len() as u32,
            source_face_id: face.face_id,
            cad_face_id: Some(frame.face_id.clone()),
            source_edge_ids,
            node_ids,
            parametric_node_uv,
            max_projection_error_m,
            region_ids: face.region_ids.clone(),
            material_region_ids: face.material_region_ids.clone(),
            area_m2: triangle_area(points),
            unit_normal: frame.unit_normal,
        });
    }
}

fn append_curve_triangle_face_element(
    face: &SourceTopologyFace,
    frame: &CadFaceEvaluationFrame,
    segments: &[FaceCurveSegment],
    nodes: &[SurfaceNode],
    elements: &mut Vec<SurfaceElement>,
) {
    let boundary_edge_ids = segments
        .iter()
        .map(|segment| {
            (
                sorted_node_pair(segment.node_ids[0], segment.node_ids[1]),
                segment.source_edge_id,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut node_ids = [
        segments[0].node_ids[0],
        segments[0].node_ids[1],
        segments[1].node_ids[1],
    ];
    let mut points = node_ids.map(|node_id| nodes[node_id as usize].coordinates_m);
    let mut parametric_node_uv =
        node_ids.map(|node_id| project_to_face(frame, nodes[node_id as usize].coordinates_m).uv);
    if dot(
        cross(sub(points[1], points[0]), sub(points[2], points[0])),
        frame.unit_normal,
    ) < 0.0
    {
        node_ids.swap(1, 2);
        points.swap(1, 2);
        parametric_node_uv.swap(1, 2);
    }
    let source_edge_ids = triangle_edges_2d([0, 1, 2]).map(|edge| {
        boundary_edge_ids
            .get(&sorted_node_pair(node_ids[edge[0]], node_ids[edge[1]]))
            .copied()
            .unwrap_or(INTERNAL_SOURCE_EDGE_ID)
    });
    let max_projection_error_m = node_ids
        .iter()
        .map(|node_id| project_to_face(frame, nodes[*node_id as usize].coordinates_m).distance_m)
        .fold(0.0_f64, f64::max);
    elements.push(SurfaceElement {
        element_id: elements.len() as u32,
        source_face_id: face.face_id,
        cad_face_id: Some(frame.face_id.clone()),
        source_edge_ids,
        node_ids,
        parametric_node_uv,
        max_projection_error_m,
        region_ids: face.region_ids.clone(),
        material_region_ids: face.material_region_ids.clone(),
        area_m2: triangle_area(points),
        unit_normal: frame.unit_normal,
    });
}
