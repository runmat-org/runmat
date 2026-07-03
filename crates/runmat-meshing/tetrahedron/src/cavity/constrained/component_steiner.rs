use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    predicate::{
        distance_squared, point_in_closed_triangle_surface, tetrahedron_centroid, Point3,
        PointInClosedSurface,
    },
    tolerance::MeshingTolerance,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_centroid, cavity_boundary_triangles,
    },
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError,
};

mod candidates;

use candidates::{component_steiner_candidate_points, component_steiner_candidate_quality_score};

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
