use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    predicate::{
        distance_squared, point_in_closed_triangle_surface, tetrahedron_centroid,
        tetrahedron_circumsphere, tetrahedron_scaled_jacobian, triangle_area, Point3,
        PointInClosedSurface,
    },
    tolerance::MeshingTolerance,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_centroid, cavity_boundary_triangles,
    },
    topology::{tetrahedron_edges, tetrahedron_faces},
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError,
};

pub fn generate_constrained_cavity_component_steiner_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    component_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
    max_nodes: usize,
) -> Result<Vec<ConstrainedCavityNode>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    if max_nodes == 0 || component_tetrahedra.is_empty() {
        return Ok(Vec::new());
    }
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let cavity_centroid = cavity_boundary_node_centroid(cavity, &boundary_node_map).ok_or(
        ConstrainedCavityRefillError::Validation(
            ConstrainedCavityValidationError::TooFewBoundaryFaces {
                boundary_face_count: cavity.boundary_faces.len(),
            },
        ),
    )?;
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in component_tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !node_map.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillError::MissingBoundaryNode { node_id });
            }
        }
    }
    let mut scored_candidates = Vec::<(f64, Point3)>::new();
    let mut sorted_tetrahedra = component_tetrahedra.to_vec();
    sorted_tetrahedra.sort_by(|left, right| {
        left.exact_scaled_jacobian
            .total_cmp(&right.exact_scaled_jacobian)
            .then_with(|| left.aspect_ratio.total_cmp(&right.aspect_ratio).reverse())
    });
    for tetrahedron in sorted_tetrahedra {
        let tetrahedron_points = tetrahedron.node_ids.map(|node_id| node_map[&node_id]);
        let tetrahedron_centroid = tetrahedron_centroid(tetrahedron_points);
        component_steiner_candidate_points(
            tetrahedron.node_ids,
            tetrahedron_points,
            tetrahedron_centroid,
            cavity_centroid,
        )
        .into_iter()
        .filter(|point| {
            candidate_respects_protected_boundary_distance(
                cavity,
                &boundary_node_map,
                *point,
                options,
            )
        })
        .filter(|point| {
            point_in_closed_triangle_surface(
                *point,
                &boundary_triangles,
                MeshingTolerance::default(),
            ) == PointInClosedSurface::Inside
        })
        .for_each(|point| {
            let score = component_steiner_candidate_quality_score(
                tetrahedron.node_ids,
                tetrahedron_points,
                point,
            );
            if score.is_finite() && score > 0.0 {
                scored_candidates.push((score, point));
            }
        });
    }
    scored_candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1[0].total_cmp(&right.1[0]))
            .then_with(|| left.1[1].total_cmp(&right.1[1]))
            .then_with(|| left.1[2].total_cmp(&right.1[2]))
    });
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let mut selected_points = Vec::<Point3>::new();
    for (_, point) in scored_candidates {
        if existing_points
            .iter()
            .chain(selected_points.iter())
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        selected_points.push(point);
        if selected_points.len() >= max_nodes {
            break;
        }
    }
    let mut next_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    Ok(selected_points
        .into_iter()
        .map(|coordinates_m| {
            while node_ids.contains(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            let node = ConstrainedCavityNode {
                node_id: next_node_id,
                coordinates_m,
            };
            next_node_id = next_node_id.saturating_add(1);
            node
        })
        .collect())
}

pub(super) fn component_steiner_candidate_points(
    node_ids: [u32; 4],
    points: [Point3; 4],
    tetrahedron_centroid: Point3,
    cavity_centroid: Point3,
) -> Vec<Point3> {
    let mut candidates = Vec::<Point3>::new();
    candidates.push(tetrahedron_centroid);
    candidates.push(tetrahedron_incenter(points));
    if let Some((center, _)) = tetrahedron_circumsphere(points, MeshingTolerance::default()) {
        candidates.push(center);
        for fraction in [0.25, 0.5, 0.75] {
            candidates.push(interpolate(tetrahedron_centroid, center, fraction));
        }
    }
    for face in tetrahedron_faces(node_ids) {
        let face_points = face.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate| *candidate == node_id)
                .expect("tetrahedron face node should be in tetrahedron");
            points[index]
        });
        let face_centroid = [
            (face_points[0][0] + face_points[1][0] + face_points[2][0]) / 3.0,
            (face_points[0][1] + face_points[1][1] + face_points[2][1]) / 3.0,
            (face_points[0][2] + face_points[1][2] + face_points[2][2]) / 3.0,
        ];
        for fraction in [0.18, 0.33, 0.5, 0.67] {
            candidates.push(interpolate(face_centroid, tetrahedron_centroid, fraction));
        }
        for fraction in [0.12, 0.25, 0.4] {
            candidates.push(interpolate(face_centroid, cavity_centroid, fraction));
        }
    }
    for edge in tetrahedron_edges(node_ids) {
        let edge_points = edge.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate| *candidate == node_id)
                .expect("tetrahedron edge node should be in tetrahedron");
            points[index]
        });
        let midpoint = [
            (edge_points[0][0] + edge_points[1][0]) * 0.5,
            (edge_points[0][1] + edge_points[1][1]) * 0.5,
            (edge_points[0][2] + edge_points[1][2]) * 0.5,
        ];
        for fraction in [0.2, 0.4, 0.6] {
            candidates.push(interpolate(midpoint, tetrahedron_centroid, fraction));
        }
    }
    candidates
}

pub(super) fn component_steiner_candidate_quality_score(
    node_ids: [u32; 4],
    points: [Point3; 4],
    candidate: Point3,
) -> f64 {
    let mut min_quality = f64::INFINITY;
    for face in tetrahedron_faces(node_ids) {
        let face_points = face.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate_id| *candidate_id == node_id)
                .expect("tetrahedron face node should be in tetrahedron");
            points[index]
        });
        let quality = tetrahedron_scaled_jacobian([
            face_points[0],
            face_points[1],
            face_points[2],
            candidate,
        ]);
        min_quality = min_quality.min(quality);
    }
    min_quality
}

fn tetrahedron_incenter(points: [Point3; 4]) -> Point3 {
    let weights = [
        triangle_area([points[1], points[2], points[3]]),
        triangle_area([points[0], points[3], points[2]]),
        triangle_area([points[0], points[1], points[3]]),
        triangle_area([points[0], points[2], points[1]]),
    ];
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= f64::EPSILON {
        return tetrahedron_centroid(points);
    }
    [
        (points[0][0] * weights[0]
            + points[1][0] * weights[1]
            + points[2][0] * weights[2]
            + points[3][0] * weights[3])
            / total,
        (points[0][1] * weights[0]
            + points[1][1] * weights[1]
            + points[2][1] * weights[2]
            + points[3][1] * weights[3])
            / total,
        (points[0][2] * weights[0]
            + points[1][2] * weights[1]
            + points[2][2] * weights[2]
            + points[3][2] * weights[3])
            / total,
    ]
}

fn interpolate(left: Point3, right: Point3, fraction: f64) -> Point3 {
    [
        left[0] * (1.0 - fraction) + right[0] * fraction,
        left[1] * (1.0 - fraction) + right[1] * fraction,
        left[2] * (1.0 - fraction) + right[2] * fraction,
    ]
}
