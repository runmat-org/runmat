use std::collections::BTreeSet;

use runmat_meshing_core::{
    predicate::{
        distance_squared, point_in_closed_triangle_surface, tetrahedron_centroid, Point3,
        PointInClosedSurface,
    },
    tolerance::MeshingTolerance,
};

use super::super::{
    boundary_nodes::{
        boundary_node_coordinates, candidate_respects_protected_boundary_distance,
        cavity_boundary_node_centroid, cavity_boundary_triangles,
    },
    geometry::face_centroid,
    refill_tetrahedra::raw_refill_tetrahedron_with_rejection_reason,
    solid_empty::solid_empty_boundary_faces,
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions,
};
use super::local_cap_apex_candidates;

pub fn generate_constrained_cavity_boundary_cap_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    max_nodes: usize,
) -> Result<Vec<ConstrainedCavityNode>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    if max_nodes == 0 {
        return Ok(Vec::new());
    }
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        return Ok(Vec::new());
    };
    let solid_empty_faces =
        solid_empty_boundary_faces(cavity, &boundary_node_map, &boundary_triangles, options);
    if solid_empty_faces.is_empty() {
        return Ok(Vec::new());
    }
    let cap_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let mut scored_candidates = Vec::<(f64, [u32; 3], Point3, &'static str)>::new();
    for face in solid_empty_faces {
        let Some(surface_point) = face_centroid(face, &boundary_node_map) else {
            continue;
        };
        for candidate in
            local_cap_apex_candidates(face, surface_point, cavity_centroid, &boundary_node_map)
        {
            if existing_points
                .iter()
                .any(|existing| distance_squared(*existing, candidate.coordinates_m) <= 1.0e-24)
            {
                continue;
            }
            if !candidate_respects_protected_boundary_distance(
                cavity,
                &boundary_node_map,
                candidate.coordinates_m,
                options,
            ) {
                continue;
            }
            if point_in_closed_triangle_surface(
                candidate.coordinates_m,
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                continue;
            }
            let tetrahedron_points = [
                boundary_node_map[&face[0]],
                boundary_node_map[&face[1]],
                boundary_node_map[&face[2]],
                candidate.coordinates_m,
            ];
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(tetrahedron_points),
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                continue;
            }
            let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], cap_node_id],
                tetrahedron_points,
                options,
            ) else {
                continue;
            };
            scored_candidates.push((
                tetrahedron.exact_scaled_jacobian,
                face,
                candidate.coordinates_m,
                candidate.source,
            ));
        }
    }
    scored_candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2[0].total_cmp(&right.2[0]))
            .then_with(|| left.2[1].total_cmp(&right.2[1]))
            .then_with(|| left.2[2].total_cmp(&right.2[2]))
            .then_with(|| left.3.cmp(right.3))
    });
    let mut selected_faces = BTreeSet::<[u32; 3]>::new();
    let mut selected_points = Vec::<Point3>::new();
    for (_, face, point, _) in scored_candidates {
        if selected_faces.contains(&face) {
            continue;
        }
        if existing_points
            .iter()
            .chain(selected_points.iter())
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        selected_faces.insert(face);
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
