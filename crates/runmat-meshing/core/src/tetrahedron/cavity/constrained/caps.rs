#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;

#[cfg(test)]
use crate::predicate::tetrahedron_centroid;
use crate::{
    predicate::{
        distance_squared, point_in_closed_triangle_surface, PointInClosedSurface, Triangle3,
    },
    tolerance::MeshingTolerance,
};

#[cfg(test)]
use super::{
    geometry::{centroid_of_node_set, face_centroid},
    raw_refill_tetrahedron_with_rejection_reason, ConstrainedCavityRefillOptions,
    ConstrainedCavityRefillTetrahedron,
};

mod apex_candidates;
pub(super) use apex_candidates::local_cap_apex_candidates;
#[cfg(test)]
pub(super) use apex_candidates::LocalCapApexCandidate;

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
