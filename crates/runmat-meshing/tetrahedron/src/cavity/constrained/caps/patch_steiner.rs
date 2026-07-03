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
        cavity_boundary_node_centroid, cavity_boundary_node_ids, cavity_boundary_triangles,
    },
    connectivity::tetrahedralize_points,
    geometry::{centroid_of_node_set, face_centroid},
    missing_faces::{missing_face_components, MissingFaceLink},
    refill_faces::missing_refill_boundary_faces,
    refill_tetrahedra::raw_refill_tetrahedron_with_rejection_reason,
    validation::{validate_constrained_cavity, validate_refill_options},
    ConnectivityPoint, ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError,
};
use super::patch_steiner_candidate_points;

pub fn generate_constrained_cavity_patch_steiner_nodes(
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
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if boundary_node_ids.len() < 4 {
        return Ok(Vec::new());
    }
    let boundary_points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&boundary_points) {
        let node_ids = tetrahedron
            .vertices
            .map(|index| boundary_points[index].node_id);
        let points = tetrahedron
            .vertices
            .map(|index| boundary_points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    if missing_faces.is_empty() {
        return Ok(Vec::new());
    }
    let cavity_centroid = cavity_boundary_node_centroid(cavity, &boundary_node_map).ok_or(
        ConstrainedCavityRefillError::Validation(
            ConstrainedCavityValidationError::TooFewBoundaryFaces {
                boundary_face_count: cavity.boundary_faces.len(),
            },
        ),
    )?;
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let mut scored_candidates = Vec::<(usize, f64, Point3)>::new();
    for component in missing_face_components(&missing_faces, MissingFaceLink::Node) {
        let mut patch_node_ids = BTreeSet::<u32>::new();
        for face_index in &component {
            patch_node_ids.extend(missing_faces[*face_index]);
        }
        let mut surface_points = Vec::<Point3>::new();
        if let Some(point) = centroid_of_node_set(&patch_node_ids, &boundary_node_map) {
            surface_points.push(point);
        }
        for face_index in &component {
            if let Some(point) = face_centroid(missing_faces[*face_index], &boundary_node_map) {
                surface_points.push(point);
            }
        }
        for surface_point in surface_points {
            for point in
                patch_steiner_candidate_points(surface_point, cavity_centroid, &boundary_triangles)
            {
                if !candidate_respects_protected_boundary_distance(
                    cavity,
                    &boundary_node_map,
                    point,
                    options,
                ) {
                    continue;
                }
                let nearest_distance = existing_points
                    .iter()
                    .map(|existing| distance_squared(*existing, point))
                    .fold(f64::INFINITY, f64::min);
                if nearest_distance.is_finite() && nearest_distance > 1.0e-24 {
                    scored_candidates.push((component.len(), nearest_distance, point));
                }
            }
        }
    }
    scored_candidates.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.total_cmp(&left.1))
            .then_with(|| left.2[0].total_cmp(&right.2[0]))
            .then_with(|| left.2[1].total_cmp(&right.2[1]))
            .then_with(|| left.2[2].total_cmp(&right.2[2]))
    });
    let mut selected_points = Vec::<Point3>::new();
    for (_, _, point) in scored_candidates {
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
