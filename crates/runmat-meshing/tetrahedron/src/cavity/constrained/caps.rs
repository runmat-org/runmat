#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use runmat_meshing_core::{
    predicate::{
        distance_squared, point_in_closed_triangle_surface, tetrahedron_centroid, Point3,
        PointInClosedSurface, Triangle3,
    },
    tolerance::MeshingTolerance,
};

use super::{
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

mod apex_candidates;
pub(super) use apex_candidates::local_cap_apex_candidates;
#[cfg(test)]
pub(super) use apex_candidates::LocalCapApexCandidate;

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

#[cfg(test)]
pub(super) fn best_local_cap_for_face(
    face: [u32; 3],
    surface_point: [f64; 3],
    cavity_centroid: [f64; 3],
    apex_node_id: u32,
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Option<([f64; 3], ConstrainedCavityRefillTetrahedron)> {
    local_cap_apex_candidates(face, surface_point, cavity_centroid, node_coordinates)
        .into_iter()
        .filter_map(|apex| {
            let tetrahedron_points = [
                node_coordinates[&face[0]],
                node_coordinates[&face[1]],
                node_coordinates[&face[2]],
                apex.coordinates_m,
            ];
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(tetrahedron_points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                return None;
            }
            let tetrahedron = raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], apex_node_id],
                tetrahedron_points,
                options,
            )
            .ok()?;
            Some((apex.coordinates_m, tetrahedron))
        })
        .max_by(|left, right| {
            left.1
                .exact_scaled_jacobian
                .total_cmp(&right.1.exact_scaled_jacobian)
                .then_with(|| right.1.aspect_ratio.total_cmp(&left.1.aspect_ratio))
        })
}

#[cfg(test)]
pub(super) fn best_shared_patch_cap_for_faces(
    faces: &[[u32; 3]],
    cavity_centroid: [f64; 3],
    apex_node_id: u32,
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Option<([f64; 3], Vec<ConstrainedCavityRefillTetrahedron>)> {
    let mut candidate_points = Vec::<[f64; 3]>::new();
    let mut patch_node_ids = BTreeSet::<u32>::new();
    for face in faces {
        patch_node_ids.extend(*face);
        let Some(surface_point) = face_centroid(*face, node_coordinates) else {
            continue;
        };
        candidate_points.extend(
            local_cap_apex_candidates(*face, surface_point, cavity_centroid, node_coordinates)
                .into_iter()
                .map(|candidate| candidate.coordinates_m),
        );
    }
    if let Some(surface_point) = centroid_of_node_set(&patch_node_ids, node_coordinates) {
        if let Some(point) =
            patch_steiner_point_inside_cavity(surface_point, cavity_centroid, boundary_triangles)
        {
            candidate_points.push(point);
        }
    }

    let mut best = None::<([f64; 3], Vec<ConstrainedCavityRefillTetrahedron>, f64)>;
    for point in candidate_points {
        let mut patch_tetrahedra =
            Vec::<ConstrainedCavityRefillTetrahedron>::with_capacity(faces.len());
        for face in faces {
            let tetrahedron_points = [
                node_coordinates[&face[0]],
                node_coordinates[&face[1]],
                node_coordinates[&face[2]],
                point,
            ];
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(tetrahedron_points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                patch_tetrahedra.clear();
                break;
            }
            let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], apex_node_id],
                tetrahedron_points,
                options,
            ) else {
                patch_tetrahedra.clear();
                break;
            };
            patch_tetrahedra.push(tetrahedron);
        }
        if patch_tetrahedra.len() != faces.len() {
            continue;
        }
        let min_quality = patch_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if best
            .as_ref()
            .is_none_or(|(_, _, best_quality)| min_quality > *best_quality)
        {
            best = Some((point, patch_tetrahedra, min_quality));
        }
    }
    best.map(|(point, patch_tetrahedra, _)| (point, patch_tetrahedra))
}

#[cfg(test)]
pub(super) fn patch_steiner_point_inside_cavity(
    surface_point: [f64; 3],
    cavity_centroid: [f64; 3],
    boundary_triangles: &[Triangle3],
) -> Option<[f64; 3]> {
    patch_steiner_candidate_points(surface_point, cavity_centroid, boundary_triangles)
        .into_iter()
        .next()
}

pub(super) fn patch_steiner_candidate_points(
    surface_point: [f64; 3],
    cavity_centroid: [f64; 3],
    boundary_triangles: &[Triangle3],
) -> Vec<[f64; 3]> {
    let mut candidates = Vec::<[f64; 3]>::new();
    if point_in_closed_triangle_surface(
        surface_point,
        boundary_triangles,
        MeshingTolerance::default(),
    ) == PointInClosedSurface::Inside
    {
        candidates.push(surface_point);
    }
    for point in [0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.4, 0.58, 0.78]
        .into_iter()
        .map(|fraction| {
            [
                surface_point[0] + (cavity_centroid[0] - surface_point[0]) * fraction,
                surface_point[1] + (cavity_centroid[1] - surface_point[1]) * fraction,
                surface_point[2] + (cavity_centroid[2] - surface_point[2]) * fraction,
            ]
        })
        .filter(|point| {
            point_in_closed_triangle_surface(
                *point,
                boundary_triangles,
                MeshingTolerance::default(),
            ) == PointInClosedSurface::Inside
        })
    {
        if candidates
            .iter()
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        candidates.push(point);
    }
    candidates
}
