use std::collections::{BTreeMap, BTreeSet};

use super::{
    geometry::{
        circumcircle_contains, point_in_triangle_2d, point_in_trimmed_domain_2d,
        point_on_segment_2d, sorted_index_pair, super_triangle_points, triangle_area_2d,
        triangle_centroid_2d, triangle_edges_2d, TriangulationPoint, TriangulationTriangle,
    },
    FaceTriangulationPoint,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FaceTriangle {
    pub(super) point_indices: [usize; 3],
}

pub(super) fn triangulate_face_points(
    points: &[FaceTriangulationPoint],
    boundary_polygons: &[Vec<[f64; 2]>],
) -> Vec<FaceTriangle> {
    if points.len() < 3 {
        return Vec::new();
    }
    let mut work_points = points
        .iter()
        .map(|point| TriangulationPoint {
            uv: point.uv,
            original_index: Some(0),
            is_super: false,
        })
        .collect::<Vec<_>>();
    for (index, point) in work_points.iter_mut().enumerate() {
        point.original_index = Some(index);
    }
    let super_start = work_points.len();
    work_points.extend(super_triangle_points(points));
    let mut triangles = vec![TriangulationTriangle {
        point_indices: [super_start, super_start + 1, super_start + 2],
    }];

    for point_index in 0..points.len() {
        let point = work_points[point_index].uv;
        let mut bad_indices = Vec::<usize>::new();
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            if circumcircle_contains(
                triangle.point_indices.map(|index| work_points[index].uv),
                point,
            ) {
                bad_indices.push(triangle_index);
            }
        }
        if bad_indices.is_empty() {
            continue;
        }
        let bad_set = bad_indices.iter().copied().collect::<BTreeSet<_>>();
        let mut edge_counts = BTreeMap::<[usize; 2], usize>::new();
        for triangle_index in &bad_indices {
            for edge in triangle_edges_2d(triangles[*triangle_index].point_indices) {
                *edge_counts
                    .entry(sorted_index_pair(edge[0], edge[1]))
                    .or_default() += 1;
            }
        }
        let cavity_edges = edge_counts
            .into_iter()
            .filter_map(|(edge, count)| (count == 1).then_some(edge))
            .collect::<Vec<_>>();
        triangles = triangles
            .into_iter()
            .enumerate()
            .filter_map(|(index, triangle)| (!bad_set.contains(&index)).then_some(triangle))
            .collect();
        for edge in cavity_edges {
            let point_indices = [edge[0], edge[1], point_index];
            if triangle_area_2d(point_indices.map(|index| work_points[index].uv)).abs()
                > f64::EPSILON
            {
                triangles.push(TriangulationTriangle { point_indices });
            }
        }
    }

    triangles
        .into_iter()
        .filter(|triangle| {
            !triangle
                .point_indices
                .iter()
                .any(|index| work_points[*index].is_super)
        })
        .filter_map(|triangle| {
            let point_indices = triangle
                .point_indices
                .map(|index| work_points[index].original_index);
            Some(FaceTriangle {
                point_indices: [point_indices[0]?, point_indices[1]?, point_indices[2]?],
            })
        })
        .filter(|triangle| {
            let centroid =
                triangle_centroid_2d(triangle.point_indices.map(|index| points[index].uv));
            point_in_trimmed_domain_2d(centroid, boundary_polygons)
        })
        .collect()
}

pub(super) fn triangulate_triangle_points_by_insertion(
    points: &[FaceTriangulationPoint],
    boundary_point_count: usize,
) -> Vec<FaceTriangle> {
    if boundary_point_count != 3 || points.len() < 3 {
        return Vec::new();
    }
    let mut triangles = vec![FaceTriangle {
        point_indices: [0, 1, 2],
    }];
    for point_index in boundary_point_count..points.len() {
        let edge_hits = triangles
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(triangle_index, triangle)| {
                triangle_edge_containing_point(point_index, triangle, points)
                    .map(|edge| (triangle_index, triangle, edge))
            })
            .collect::<Vec<_>>();
        if !edge_hits.is_empty() {
            for (triangle_index, triangle, edge) in edge_hits.into_iter().rev() {
                triangles.swap_remove(triangle_index);
                let opposite = triangle
                    .point_indices
                    .into_iter()
                    .find(|index| *index != edge[0] && *index != edge[1])
                    .expect("triangle edge should have an opposite point");
                push_non_degenerate_face_triangle(
                    &mut triangles,
                    [edge[0], point_index, opposite],
                    points,
                );
                push_non_degenerate_face_triangle(
                    &mut triangles,
                    [point_index, edge[1], opposite],
                    points,
                );
            }
            continue;
        }
        let Some((triangle_index, triangle)) =
            triangles.iter().copied().enumerate().find(|(_, triangle)| {
                point_in_triangle_2d(
                    points[point_index].uv,
                    triangle.point_indices.map(|index| points[index].uv),
                )
            })
        else {
            continue;
        };
        triangles.swap_remove(triangle_index);
        for point_indices in [
            [
                triangle.point_indices[0],
                triangle.point_indices[1],
                point_index,
            ],
            [
                triangle.point_indices[1],
                triangle.point_indices[2],
                point_index,
            ],
            [
                triangle.point_indices[2],
                triangle.point_indices[0],
                point_index,
            ],
        ] {
            push_non_degenerate_face_triangle(&mut triangles, point_indices, points);
        }
    }
    triangles
}

fn triangle_edge_containing_point(
    point_index: usize,
    triangle: FaceTriangle,
    points: &[FaceTriangulationPoint],
) -> Option<[usize; 2]> {
    let point = points[point_index].uv;
    triangle_edges_2d(triangle.point_indices)
        .into_iter()
        .find(|&edge| point_on_segment_2d(point, points[edge[0]].uv, points[edge[1]].uv))
}

fn push_non_degenerate_face_triangle(
    triangles: &mut Vec<FaceTriangle>,
    point_indices: [usize; 3],
    points: &[FaceTriangulationPoint],
) {
    if triangle_area_2d(point_indices.map(|index| points[index].uv)).abs() > f64::EPSILON {
        triangles.push(FaceTriangle { point_indices });
    }
}
