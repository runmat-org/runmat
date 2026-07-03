use super::*;

mod exact_cover;

use exact_cover::finish_shared_cap_exact_cover_diagnostic;

pub(super) fn diagnostic_missing_face_shared_cap_stitch_with_link(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    patch_link: MissingFaceLink,
    incomplete_reason: &'static str,
    fallback_to_face_caps: bool,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
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
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let missing_face_patches = missing_face_components(&missing_faces, patch_link);
    let mut diagnostic = MissingFaceLocalCapStitchDiagnostic {
        missing_face_count: missing_faces.len(),
        missing_faces: missing_faces.clone(),
        patch_count: missing_face_patches.len(),
        patch_size_histogram: component_size_histogram(
            missing_face_patches
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
        ),
        patch_capped_face_count_histogram: BTreeMap::new(),
        incomplete_patch_size_histogram: BTreeMap::new(),
        uncapped_faces: Vec::new(),
        capped_face_count: 0,
        inserted_node_count: 0,
        side_connector_candidate_count: 0,
        candidate_tetrahedron_count: 0,
        cap_side_face_count: 0,
        zero_mate_cap_side_face_count: 0,
        min_cap_side_face_mate_count: 0,
        max_cap_side_face_mate_count: 0,
        open_interior_face_count: 0,
        open_interior_component_count: 0,
        open_interior_component_size_histogram: BTreeMap::new(),
        candidate_with_orphan_interior_face_count: 0,
        candidate_without_orphan_interior_face_count: 0,
        root_boundary_zero_raw_candidate_face_count: 0,
        root_boundary_zero_addable_candidate_face_count: 0,
        root_boundary_min_raw_candidate_count: 0,
        root_boundary_min_addable_candidate_count: 0,
        root_boundary_max_addable_candidate_count: 0,
        cover_dead_end_reason: "not_evaluated",
        cover_dead_end_depth: 0,
        cover_dead_end_reason_histogram: BTreeMap::new(),
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
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
    let inserted_node_ids = inserted_nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    diagnostic.side_connector_candidate_count = append_cap_side_connector_tetrahedra(
        cap_tetrahedron_start,
        cap_tetrahedron_count,
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &node_points,
        &inserted_node_ids,
        &boundary_triangles,
        options,
    );
    diagnostic.candidate_tetrahedron_count = candidate_tetrahedra.len();
    let cap_side_mate_counts = cap_side_face_mate_counts(
        &candidate_tetrahedra[cap_tetrahedron_start..cap_tetrahedron_start + cap_tetrahedron_count],
        &candidate_tetrahedra,
        &inserted_node_ids,
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
    Ok(finish_shared_cap_exact_cover_diagnostic(
        cavity,
        &candidate_tetrahedra,
        options,
        diagnostic,
    ))
}
