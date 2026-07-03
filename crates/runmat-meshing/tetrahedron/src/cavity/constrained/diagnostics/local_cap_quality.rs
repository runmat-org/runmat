use runmat_meshing_core::quality::predicate::tetrahedron_scaled_jacobian;

use super::*;

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_local_cap_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapQualityDiagnostic, ConstrainedCavityRefillError> {
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
    let mut diagnostic = MissingFaceLocalCapQualityDiagnostic {
        missing_face_count: missing_faces.len(),
        pass_face_count: 0,
        failed_face_count: 0,
        candidate_count: 0,
        candidate_source_bins: BTreeMap::new(),
        max_scaled_jacobian: 0.0,
        max_failed_face_scaled_jacobian: 0.0,
        failed_face_scaled_jacobian_bins: BTreeMap::new(),
        failed_face_source_bins: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    if missing_faces.is_empty() {
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        return Ok(diagnostic);
    };
    let mut next_node_id = next_cavity_node_id(cavity);
    for face in missing_faces {
        let Some(surface_point) = face_centroid(face, &boundary_node_map) else {
            continue;
        };
        let mut face_passed = false;
        let mut best_failed_face_quality = 0.0_f64;
        let mut best_failed_face_source = None::<&'static str>;
        for apex in
            local_cap_apex_candidates(face, surface_point, cavity_centroid, &boundary_node_map)
        {
            let tetrahedron_points = [
                boundary_node_map[&face[0]],
                boundary_node_map[&face[1]],
                boundary_node_map[&face[2]],
                apex.coordinates_m,
            ];
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(tetrahedron_points),
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                *diagnostic
                    .rejected_by_reason
                    .entry("cap_centroid_outside_cavity")
                    .or_default() += 1;
                continue;
            }
            while boundary_node_map.contains_key(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            diagnostic.candidate_count += 1;
            *diagnostic
                .candidate_source_bins
                .entry(apex.source)
                .or_default() += 1;
            let exact_scaled_jacobian = tetrahedron_scaled_jacobian(tetrahedron_points);
            match raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], next_node_id],
                tetrahedron_points,
                options,
            ) {
                Ok(tetrahedron) => {
                    diagnostic.max_scaled_jacobian = diagnostic
                        .max_scaled_jacobian
                        .max(tetrahedron.exact_scaled_jacobian);
                    face_passed = true;
                }
                Err(reason) => {
                    if exact_scaled_jacobian.is_finite() {
                        if exact_scaled_jacobian > best_failed_face_quality {
                            best_failed_face_quality = exact_scaled_jacobian;
                            best_failed_face_source = Some(apex.source);
                        }
                    }
                    *diagnostic.rejected_by_reason.entry(reason).or_default() += 1;
                }
            }
            next_node_id = next_node_id.saturating_add(1);
        }
        diagnostic.pass_face_count += usize::from(face_passed);
        if !face_passed && best_failed_face_quality.is_finite() && best_failed_face_quality > 0.0 {
            diagnostic.failed_face_count += 1;
            diagnostic.max_failed_face_scaled_jacobian = diagnostic
                .max_failed_face_scaled_jacobian
                .max(best_failed_face_quality);
            *diagnostic
                .failed_face_scaled_jacobian_bins
                .entry(diagnostic_scaled_jacobian_bin(best_failed_face_quality))
                .or_default() += 1;
            if let Some(source) = best_failed_face_source {
                *diagnostic
                    .failed_face_source_bins
                    .entry(source)
                    .or_default() += 1;
            }
        }
    }
    Ok(diagnostic)
}
