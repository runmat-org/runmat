use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, PointInClosedSurface,
    },
    quality::tolerance::MeshingTolerance,
};

use super::{
    boundary_nodes::{
        boundary_node_coordinates, cavity_boundary_node_ids, cavity_boundary_triangles,
    },
    refill_tetrahedra::raw_refill_tetrahedron_with_rejection_reason,
    topology::{sorted_face, tetrahedron_faces},
    validation::{validate_constrained_cavity, validate_refill_options},
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions,
};

pub fn constrained_cavity_refill_pressure_boundary_faces(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<[u32; 3]>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    if node_ids.len() < 4 || boundary_faces.is_empty() {
        return Ok(Vec::new());
    }

    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut face_candidate_counts = boundary_faces
        .iter()
        .map(|face| (*face, 0_usize))
        .collect::<BTreeMap<_, _>>();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    let tetrahedron_boundary_faces =
                        tetrahedron_faces(tetrahedron_node_ids).map(sorted_face);
                    if !tetrahedron_boundary_faces
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    )
                    .is_err()
                    {
                        continue;
                    }
                    for face in tetrahedron_boundary_faces {
                        if let Some(count) = face_candidate_counts.get_mut(&face) {
                            *count += 1;
                        }
                    }
                }
            }
        }
    }
    let min_count = face_candidate_counts
        .values()
        .copied()
        .min()
        .unwrap_or_default();
    Ok(face_candidate_counts
        .into_iter()
        .filter_map(|(face, count)| (count == min_count).then_some(face))
        .collect())
}
