use super::*;

use super::finish::finish_missing_face_stitch_candidates;
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
    finish_missing_face_stitch_candidates(
        cavity,
        candidate_tetrahedra,
        cap_tetrahedron_start,
        &node_points,
        &inserted_nodes,
        &boundary_triangles,
        options,
        diagnostic,
    )
}
