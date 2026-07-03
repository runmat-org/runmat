use super::*;

use super::finish::finish_missing_face_stitch_candidates;
use super::setup::{missing_face_stitch_setup, MissingFaceStitchSetup};

pub(super) fn diagnostic_missing_face_shared_cap_stitch_with_link(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    patch_link: MissingFaceLink,
    incomplete_reason: &'static str,
    fallback_to_face_caps: bool,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    let MissingFaceStitchSetup {
        boundary_node_map,
        boundary_triangles,
        boundary_node_ids,
        boundary_refill_tetrahedra,
        missing_faces,
        missing_face_patches,
        mut diagnostic,
    } = missing_face_stitch_setup(cavity, boundary_nodes, options, patch_link)?;
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
    for patch in &missing_face_patches {
        let faces = patch
            .iter()
            .map(|face_index| missing_faces[*face_index])
            .collect::<Vec<_>>();
        if let Some((coordinates_m, mut cap_tetrahedra)) = best_shared_patch_cap_for_faces(
            &faces,
            cavity_centroid,
            next_node_id,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) {
            while node_points.contains_key(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            node_points.insert(next_node_id, coordinates_m);
            inserted_nodes.push(ConstrainedCavityNode {
                node_id: next_node_id,
                coordinates_m,
            });
            diagnostic.capped_face_count += cap_tetrahedra.len();
            *diagnostic
                .patch_capped_face_count_histogram
                .entry(cap_tetrahedra.len())
                .or_default() += 1;
            candidate_tetrahedra.append(&mut cap_tetrahedra);
            next_node_id = next_node_id.saturating_add(1);
            continue;
        }

        let mut capped_count = 0_usize;
        if fallback_to_face_caps {
            for face in &faces {
                let Some(surface_point) = face_centroid(*face, &boundary_node_map) else {
                    continue;
                };
                while node_points.contains_key(&next_node_id) {
                    next_node_id = next_node_id.saturating_add(1);
                }
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
                node_points.insert(next_node_id, coordinates_m);
                inserted_nodes.push(ConstrainedCavityNode {
                    node_id: next_node_id,
                    coordinates_m,
                });
                candidate_tetrahedra.push(cap_tetrahedron);
                capped_count += 1;
                next_node_id = next_node_id.saturating_add(1);
            }
            diagnostic.capped_face_count += capped_count;
        }
        *diagnostic
            .patch_capped_face_count_histogram
            .entry(capped_count)
            .or_default() += 1;
        if capped_count < patch.len() {
            diagnostic.uncapped_faces.extend(
                patch
                    .iter()
                    .filter(|face_index| {
                        let face = missing_faces[**face_index];
                        !candidate_tetrahedra[cap_tetrahedron_start..]
                            .iter()
                            .any(|tetrahedron| {
                                tetrahedron_faces(tetrahedron.node_ids)
                                    .map(sorted_face)
                                    .contains(&face)
                            })
                    })
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
        diagnostic.reason = incomplete_reason;
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
