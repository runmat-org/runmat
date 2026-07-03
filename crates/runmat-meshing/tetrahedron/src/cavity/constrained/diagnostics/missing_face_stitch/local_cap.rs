use super::*;

use super::exact_cover::finish_cap_stitch_exact_cover_diagnostic;
use super::setup::{missing_face_stitch_setup, MissingFaceStitchSetup};

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_local_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    let MissingFaceStitchSetup {
        boundary_node_map,
        boundary_triangles,
        boundary_node_ids,
        boundary_refill_tetrahedra,
        missing_faces,
        missing_face_patches,
        mut diagnostic,
    } = missing_face_stitch_setup(cavity, boundary_nodes, options, MissingFaceLink::Node)?;
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_tetrahedra = boundary_refill_tetrahedra;
    let mut inserted_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut next_node_id = next_cavity_node_id(cavity);
    let cap_tetrahedron_start = candidate_tetrahedra.len();
    let mut capped_missing_face_indices = BTreeSet::<usize>::new();
    for (face_index, face) in missing_faces.iter().enumerate() {
        let Some(surface_point) = face_centroid(*face, &boundary_node_map) else {
            continue;
        };
        let Some((coordinates_m, cap_tetrahedron)) = best_local_cap_for_face(
            *face,
            surface_point,
            cavity_centroid,
            next_node_id,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) else {
            continue;
        };
        while node_points.contains_key(&next_node_id) {
            next_node_id = next_node_id.saturating_add(1);
        }
        node_points.insert(next_node_id, coordinates_m);
        inserted_nodes.push(ConstrainedCavityNode {
            node_id: next_node_id,
            coordinates_m,
        });
        candidate_tetrahedra.push(cap_tetrahedron);
        diagnostic.capped_face_count += 1;
        capped_missing_face_indices.insert(face_index);
        next_node_id = next_node_id.saturating_add(1);
    }
    for patch in &missing_face_patches {
        let capped_count = patch
            .iter()
            .filter(|face_index| capped_missing_face_indices.contains(face_index))
            .count();
        *diagnostic
            .patch_capped_face_count_histogram
            .entry(capped_count)
            .or_default() += 1;
        if capped_count < patch.len() {
            diagnostic.uncapped_faces.extend(
                patch
                    .iter()
                    .filter(|face_index| !capped_missing_face_indices.contains(face_index))
                    .map(|face_index| missing_faces[*face_index]),
            );
            *diagnostic
                .incomplete_patch_size_histogram
                .entry(patch.len())
                .or_default() += 1;
        }
    }
    diagnostic.inserted_node_count = inserted_nodes.len();
    if diagnostic.capped_face_count < diagnostic.missing_face_count {
        diagnostic.reason = "incomplete_local_caps";
        diagnostic.candidate_tetrahedron_count = candidate_tetrahedra.len();
        return Ok(diagnostic);
    }
    let cap_tetrahedron_count = candidate_tetrahedra.len() - cap_tetrahedron_start;

    let connector_points = node_points
        .iter()
        .map(|(node_id, coordinates_m)| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: *coordinates_m,
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut seen_tetrahedra = candidate_tetrahedra
        .iter()
        .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
        .collect::<BTreeSet<_>>();
    for tetrahedron in tetrahedralize_points(&connector_points) {
        let node_ids = tetrahedron
            .vertices
            .map(|index| connector_points[index].node_id);
        if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(node_ids)) {
            continue;
        }
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| connector_points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            candidate_tetrahedra.push(tetrahedron);
        }
    }
    diagnostic.side_connector_candidate_count = append_cap_side_connector_tetrahedra(
        cap_tetrahedron_start,
        cap_tetrahedron_count,
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &node_points,
        &inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<BTreeSet<_>>(),
        &boundary_triangles,
        options,
    );
    diagnostic.candidate_tetrahedron_count = candidate_tetrahedra.len();
    let cap_side_mate_counts = cap_side_face_mate_counts(
        &candidate_tetrahedra[cap_tetrahedron_start..cap_tetrahedron_start + cap_tetrahedron_count],
        &candidate_tetrahedra,
        &inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<BTreeSet<_>>(),
    );
    diagnostic.cap_side_face_count = cap_side_mate_counts.len();
    diagnostic.zero_mate_cap_side_face_count = cap_side_mate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().max().unwrap_or(0);
    let open_interior_faces = open_interior_refill_faces(cavity, &candidate_tetrahedra);
    diagnostic.open_interior_face_count = open_interior_faces.len();
    diagnostic.open_interior_component_count =
        missing_face_components(&open_interior_faces, MissingFaceLink::Node).len();
    diagnostic.open_interior_component_size_histogram = component_size_histogram(
        missing_face_component_sizes(&open_interior_faces, MissingFaceLink::Node),
    );
    let (with_orphan, without_orphan) =
        candidate_orphan_interior_face_counts(cavity, &candidate_tetrahedra);
    diagnostic.candidate_with_orphan_interior_face_count = with_orphan;
    diagnostic.candidate_without_orphan_interior_face_count = without_orphan;
    if candidate_tetrahedra.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    Ok(finish_cap_stitch_exact_cover_diagnostic(
        cavity,
        &candidate_tetrahedra,
        options,
        diagnostic,
    ))
}
