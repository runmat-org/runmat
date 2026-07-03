use super::*;

#[cfg(test)]
pub(crate) fn diagnostic_boundary_node_completion(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryNodeCompletionDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
            refill_tetrahedra.push(tetrahedron);
        }
    }
    let mut aggregate = BoundaryNodeCompletionDiagnostic {
        reason: "boundary_node_completion_no_missing_faces",
        missing_face_count: 0,
        cap_candidate_count: 0,
        outside_candidate_count: 0,
        duplicate_candidate_count: 0,
        max_rejected_scaled_jacobian: 0.0,
        rejected_scaled_jacobian_bins: BTreeMap::new(),
        max_rejected_cap_height_ratio: 0.0,
        rejected_cap_height_ratio_bins: BTreeMap::new(),
        rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_cap_node_ids: BTreeMap::new(),
        split_cap_candidate_count: 0,
        split_cap_pass_count: 0,
        max_split_cap_scaled_jacobian: 0.0,
        split_cap_scaled_jacobian_bins: BTreeMap::new(),
        split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        split_cap_apex_limited_node_ids: BTreeMap::new(),
        edge_split_cap_candidate_count: 0,
        edge_split_cap_pass_count: 0,
        max_edge_split_cap_scaled_jacobian: 0.0,
        edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        three_edge_split_cap_candidate_count: 0,
        three_edge_split_cap_pass_count: 0,
        max_three_edge_split_cap_scaled_jacobian: 0.0,
        three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    loop {
        let missing_faces = missing_refill_boundary_faces(cavity, &refill_tetrahedra)
            .map_err(ConstrainedCavityRefillError::Validation)?;
        let Some(missing_face) = missing_faces.first().copied() else {
            break;
        };
        aggregate.missing_face_count = missing_faces.len();
        let diagnostic = diagnostic_boundary_face_completion(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tetrahedra,
            &boundary_triangles,
            options,
            missing_faces.len(),
        );
        aggregate.cap_candidate_count += diagnostic.cap_candidate_count;
        aggregate.outside_candidate_count += diagnostic.outside_candidate_count;
        aggregate.duplicate_candidate_count += diagnostic.duplicate_candidate_count;
        aggregate.max_rejected_scaled_jacobian = aggregate
            .max_rejected_scaled_jacobian
            .max(diagnostic.max_rejected_scaled_jacobian);
        aggregate.max_rejected_cap_height_ratio = aggregate
            .max_rejected_cap_height_ratio
            .max(diagnostic.max_rejected_cap_height_ratio);
        for (bin, count) in diagnostic.rejected_scaled_jacobian_bins {
            *aggregate
                .rejected_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_cap_height_ratio_bins {
            *aggregate
                .rejected_cap_height_ratio_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_scaled_jacobian_worst_corner_bins {
            *aggregate
                .rejected_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.rejected_cap_node_ids {
            *aggregate.rejected_cap_node_ids.entry(node_id).or_default() += count;
        }
        aggregate.split_cap_candidate_count += diagnostic.split_cap_candidate_count;
        aggregate.split_cap_pass_count += diagnostic.split_cap_pass_count;
        aggregate.max_split_cap_scaled_jacobian = aggregate
            .max_split_cap_scaled_jacobian
            .max(diagnostic.max_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_bins {
            *aggregate
                .split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.split_cap_apex_limited_node_ids {
            *aggregate
                .split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.edge_split_cap_candidate_count += diagnostic.edge_split_cap_candidate_count;
        aggregate.edge_split_cap_pass_count += diagnostic.edge_split_cap_pass_count;
        aggregate.max_edge_split_cap_scaled_jacobian = aggregate
            .max_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.edge_split_cap_apex_limited_node_ids {
            *aggregate
                .edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.three_edge_split_cap_candidate_count +=
            diagnostic.three_edge_split_cap_candidate_count;
        aggregate.three_edge_split_cap_pass_count += diagnostic.three_edge_split_cap_pass_count;
        aggregate.max_three_edge_split_cap_scaled_jacobian = aggregate
            .max_three_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_three_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.three_edge_split_cap_apex_limited_node_ids {
            *aggregate
                .three_edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        for (reason, count) in diagnostic.rejected_by_reason {
            *aggregate.rejected_by_reason.entry(reason).or_default() += count;
        }
        let Some(tetrahedron) = best_boundary_face_completion_tetrahedron(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tetrahedra,
            &boundary_triangles,
            options,
        ) else {
            aggregate.reason = "boundary_node_completion_no_candidate";
            return Ok(aggregate);
        };
        refill_tetrahedra.push(tetrahedron);
    }
    if aggregate.missing_face_count == 0 {
        return Ok(BoundaryNodeCompletionDiagnostic {
            reason: "boundary_node_completion_no_missing_faces",
            missing_face_count: 0,
            cap_candidate_count: 0,
            outside_candidate_count: 0,
            duplicate_candidate_count: 0,
            max_rejected_scaled_jacobian: 0.0,
            rejected_scaled_jacobian_bins: BTreeMap::new(),
            max_rejected_cap_height_ratio: 0.0,
            rejected_cap_height_ratio_bins: BTreeMap::new(),
            rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            rejected_cap_node_ids: BTreeMap::new(),
            split_cap_candidate_count: 0,
            split_cap_pass_count: 0,
            max_split_cap_scaled_jacobian: 0.0,
            split_cap_scaled_jacobian_bins: BTreeMap::new(),
            split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            split_cap_apex_limited_node_ids: BTreeMap::new(),
            edge_split_cap_candidate_count: 0,
            edge_split_cap_pass_count: 0,
            max_edge_split_cap_scaled_jacobian: 0.0,
            edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            three_edge_split_cap_candidate_count: 0,
            three_edge_split_cap_pass_count: 0,
            max_three_edge_split_cap_scaled_jacobian: 0.0,
            three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            rejected_by_reason: BTreeMap::new(),
        });
    }
    aggregate.reason = "boundary_node_completion_completed";
    Ok(aggregate)
}

#[cfg(test)]
pub(crate) fn diagnostic_interior_star_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<InteriorStarQualityDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut diagnostic = InteriorStarQualityDiagnostic {
        candidate_count: 0,
        pass_count: 0,
        scaled_worst_face_candidate_count: 0,
        scaled_worst_face_pass_count: 0,
        max_min_scaled_jacobian: 0.0,
        max_scaled_worst_face_min_scaled_jacobian: 0.0,
        min_scaled_jacobian_bins: BTreeMap::new(),
        min_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    let mut seen_interior_nodes = BTreeSet::<u32>::new();
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    for node in interior_candidates {
        if !seen_interior_nodes.insert(node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("duplicate_interior_node")
                .or_default() += 1;
            continue;
        }
        if boundary_node_ids.contains(&node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("interior_node_reuses_boundary_node")
                .or_default() += 1;
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            *diagnostic
                .rejected_by_reason
                .entry("protected_boundary_distance")
                .or_default() += 1;
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            *diagnostic
                .rejected_by_reason
                .entry("interior_point_outside_cavity")
                .or_default() += 1;
            continue;
        }
        diagnostic.candidate_count += 1;
        match star_refill_candidate_with_rejection_reason(
            cavity,
            &boundary_node_map,
            node.clone(),
            diagnostic_options,
        ) {
            Ok(Ok(refill)) => {
                let min_quality = refill
                    .tetrahedra
                    .iter()
                    .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
                    .fold(f64::INFINITY, f64::min);
                if min_quality.is_finite() {
                    diagnostic.max_min_scaled_jacobian =
                        diagnostic.max_min_scaled_jacobian.max(min_quality);
                    *diagnostic
                        .min_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(min_quality))
                        .or_default() += 1;
                    if let Some(worst_tetrahedron) =
                        refill.tetrahedra.iter().min_by(|left, right| {
                            left.exact_scaled_jacobian
                                .total_cmp(&right.exact_scaled_jacobian)
                        })
                    {
                        let points = worst_tetrahedron.node_ids.map(|node_id| {
                            if node_id == node.node_id {
                                node.coordinates_m
                            } else {
                                boundary_node_map[&node_id]
                            }
                        });
                        *diagnostic
                            .min_scaled_jacobian_worst_corner_bins
                            .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                            .or_default() += 1;
                    }
                    if min_quality >= options.min_scaled_jacobian {
                        diagnostic.pass_count += 1;
                    }
                    if let Some((scaled_count, scaled_quality)) = scaled_worst_face_star_quality(
                        cavity,
                        &boundary_node_map,
                        &boundary_triangles,
                        node,
                        &refill,
                        diagnostic_options,
                    ) {
                        diagnostic.scaled_worst_face_candidate_count += scaled_count;
                        diagnostic.max_scaled_worst_face_min_scaled_jacobian = diagnostic
                            .max_scaled_worst_face_min_scaled_jacobian
                            .max(scaled_quality);
                        diagnostic.scaled_worst_face_pass_count +=
                            usize::from(scaled_quality >= options.min_scaled_jacobian);
                    }
                }
            }
            Ok(Err(reason)) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
            Err(err) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_validation_reason(&err))
                    .or_default() += 1;
            }
        }
    }
    Ok(diagnostic)
}

#[cfg(test)]
fn scaled_worst_face_star_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    node: &ConstrainedCavityNode,
    refill: &ConstrainedCavityRefill,
    options: ConstrainedCavityRefillOptions,
) -> Option<(usize, f64)> {
    let worst_tetrahedron = refill.tetrahedra.iter().min_by(|left, right| {
        left.exact_scaled_jacobian
            .total_cmp(&right.exact_scaled_jacobian)
    })?;
    let face_nodes = worst_tetrahedron
        .node_ids
        .into_iter()
        .filter(|node_id| *node_id != node.node_id)
        .collect::<Vec<_>>();
    if face_nodes.len() != 3 {
        return None;
    }
    let face_points = face_nodes
        .iter()
        .map(|node_id| boundary_nodes.get(node_id).copied())
        .collect::<Option<Vec<_>>>()?;
    let face_centroid = [
        (face_points[0][0] + face_points[1][0] + face_points[2][0]) / 3.0,
        (face_points[0][1] + face_points[1][1] + face_points[2][1]) / 3.0,
        (face_points[0][2] + face_points[1][2] + face_points[2][2]) / 3.0,
    ];
    let direction = [
        node.coordinates_m[0] - face_centroid[0],
        node.coordinates_m[1] - face_centroid[1],
        node.coordinates_m[2] - face_centroid[2],
    ];
    let distance_squared =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    if !distance_squared.is_finite()
        || distance_squared <= MeshingTolerance::default().absolute_m.powi(2)
    {
        return None;
    }

    let mut candidate_count = 0_usize;
    let mut best_quality = 0.0_f64;
    for scale in [0.5, 0.7, 0.85, 1.15, 1.35, 1.6, 2.0] {
        let coordinates_m = [
            face_centroid[0] + direction[0] * scale,
            face_centroid[1] + direction[1] * scale,
            face_centroid[2] + direction[2] * scale,
        ];
        if point_in_closed_triangle_surface(
            coordinates_m,
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        candidate_count += 1;
        let adjusted = ConstrainedCavityNode {
            node_id: node.node_id,
            coordinates_m,
        };
        let Ok(Ok(refill)) =
            star_refill_candidate_with_rejection_reason(cavity, boundary_nodes, adjusted, options)
        else {
            continue;
        };
        let min_quality = refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if min_quality.is_finite() {
            best_quality = best_quality.max(min_quality);
        }
    }
    (candidate_count > 0).then_some((candidate_count, best_quality))
}

#[cfg(test)]
fn diagnostic_boundary_face_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
    missing_face_count: usize,
) -> BoundaryNodeCompletionDiagnostic {
    let mut cap_candidate_count = 0_usize;
    let mut outside_candidate_count = 0_usize;
    let mut duplicate_candidate_count = 0_usize;
    let mut max_rejected_scaled_jacobian = 0.0_f64;
    let mut rejected_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut max_rejected_cap_height_ratio = 0.0_f64;
    let mut rejected_cap_height_ratio_bins = BTreeMap::<String, usize>::new();
    let mut rejected_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut rejected_cap_node_ids = BTreeMap::<u32, usize>::new();
    let mut split_cap_candidate_count = 0_usize;
    let mut split_cap_pass_count = 0_usize;
    let mut max_split_cap_scaled_jacobian = 0.0_f64;
    let mut split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut split_cap_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut edge_split_cap_candidate_count = 0_usize;
    let mut edge_split_cap_pass_count = 0_usize;
    let mut max_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut three_edge_split_cap_candidate_count = 0_usize;
    let mut three_edge_split_cap_pass_count = 0_usize;
    let mut max_three_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut three_edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut three_edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut three_edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut rejected_by_reason = BTreeMap::<&'static str, usize>::new();
    let mut saw_non_duplicate = false;
    for node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&node_id) {
            continue;
        }
        let node_ids = [face[0], face[1], face[2], node_id];
        let points = node_ids.map(|id| boundary_nodes[&id]);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            outside_candidate_count += 1;
            continue;
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
            Ok(tetrahedron) => {
                cap_candidate_count += 1;
                if refill_tetrahedra.iter().any(|existing| {
                    sorted_tetrahedron_nodes(existing.node_ids)
                        == sorted_tetrahedron_nodes(tetrahedron.node_ids)
                }) {
                    duplicate_candidate_count += 1;
                } else {
                    saw_non_duplicate = true;
                }
            }
            Err(reason) => {
                *rejected_cap_node_ids.entry(node_id).or_default() += 1;
                let exact_scaled_jacobian = tetrahedron_scaled_jacobian(points);
                if exact_scaled_jacobian.is_finite() {
                    max_rejected_scaled_jacobian =
                        max_rejected_scaled_jacobian.max(exact_scaled_jacobian);
                    *rejected_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(exact_scaled_jacobian))
                        .or_default() += 1;
                    *rejected_scaled_jacobian_worst_corner_bins
                        .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                        .or_default() += 1;
                }
                let cap_height_ratio =
                    diagnostic_face_apex_height_ratio(face, node_id, boundary_nodes);
                if cap_height_ratio.is_finite() {
                    max_rejected_cap_height_ratio =
                        max_rejected_cap_height_ratio.max(cap_height_ratio);
                    *rejected_cap_height_ratio_bins
                        .entry(diagnostic_height_ratio_bin(cap_height_ratio))
                        .or_default() += 1;
                }
                if let Some((split_min_quality, split_worst_corner)) =
                    diagnostic_split_cap_min_scaled_jacobian(face, node_id, boundary_nodes, options)
                {
                    split_cap_candidate_count += 1;
                    max_split_cap_scaled_jacobian =
                        max_split_cap_scaled_jacobian.max(split_min_quality);
                    *split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(split_min_quality))
                        .or_default() += 1;
                    *split_cap_scaled_jacobian_worst_corner_bins
                        .entry(split_worst_corner)
                        .or_default() += 1;
                    if split_worst_corner == "apex" {
                        *split_cap_apex_limited_node_ids.entry(node_id).or_default() += 1;
                    }
                    if split_min_quality >= options.min_scaled_jacobian {
                        split_cap_pass_count += 1;
                    }
                }
                if let Some((edge_split_min_quality, edge_split_worst_corner)) =
                    diagnostic_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    edge_split_cap_candidate_count += 1;
                    max_edge_split_cap_scaled_jacobian =
                        max_edge_split_cap_scaled_jacobian.max(edge_split_min_quality);
                    *edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(edge_split_min_quality))
                        .or_default() += 1;
                    *edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(edge_split_worst_corner)
                        .or_default() += 1;
                    if edge_split_worst_corner == "apex" {
                        *edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if edge_split_min_quality >= options.min_scaled_jacobian {
                        edge_split_cap_pass_count += 1;
                    }
                }
                if let Some((three_edge_split_min_quality, three_edge_split_worst_corner)) =
                    diagnostic_three_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    three_edge_split_cap_candidate_count += 1;
                    max_three_edge_split_cap_scaled_jacobian =
                        max_three_edge_split_cap_scaled_jacobian.max(three_edge_split_min_quality);
                    *three_edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(three_edge_split_min_quality))
                        .or_default() += 1;
                    *three_edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(three_edge_split_worst_corner)
                        .or_default() += 1;
                    if three_edge_split_worst_corner == "apex" {
                        *three_edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if three_edge_split_min_quality >= options.min_scaled_jacobian {
                        three_edge_split_cap_pass_count += 1;
                    }
                }
                *rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
        }
    }
    let reason = if saw_non_duplicate {
        "boundary_node_completion_has_candidate"
    } else if duplicate_candidate_count > 0 {
        "boundary_node_completion_duplicate_tetrahedron"
    } else {
        "boundary_node_completion_no_candidate"
    };
    BoundaryNodeCompletionDiagnostic {
        reason,
        missing_face_count,
        cap_candidate_count,
        outside_candidate_count,
        duplicate_candidate_count,
        max_rejected_scaled_jacobian,
        rejected_scaled_jacobian_bins,
        max_rejected_cap_height_ratio,
        rejected_cap_height_ratio_bins,
        rejected_scaled_jacobian_worst_corner_bins,
        rejected_cap_node_ids,
        split_cap_candidate_count,
        split_cap_pass_count,
        max_split_cap_scaled_jacobian,
        split_cap_scaled_jacobian_bins,
        split_cap_scaled_jacobian_worst_corner_bins,
        split_cap_apex_limited_node_ids,
        edge_split_cap_candidate_count,
        edge_split_cap_pass_count,
        max_edge_split_cap_scaled_jacobian,
        edge_split_cap_scaled_jacobian_bins,
        edge_split_cap_scaled_jacobian_worst_corner_bins,
        edge_split_cap_apex_limited_node_ids,
        three_edge_split_cap_candidate_count,
        three_edge_split_cap_pass_count,
        max_three_edge_split_cap_scaled_jacobian,
        three_edge_split_cap_scaled_jacobian_bins,
        three_edge_split_cap_scaled_jacobian_worst_corner_bins,
        three_edge_split_cap_apex_limited_node_ids,
        rejected_by_reason,
    }
}

#[cfg(test)]
fn diagnostic_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|split_node| {
            split_completion_tetrahedra_for_node(
                face,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tetrahedra| {
                tetrahedra
                    .iter()
                    .map(|tetrahedron| {
                        let points = tetrahedron.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tetrahedron.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

#[cfg(test)]
fn diagnostic_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_edge_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|(edge, split_node)| {
            edge_split_completion_tetrahedra_for_node(
                face,
                edge,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tetrahedra| {
                tetrahedra
                    .iter()
                    .map(|tetrahedron| {
                        let points = tetrahedron.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tetrahedron.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

#[cfg(test)]
fn diagnostic_three_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let split_nodes = boundary_face_mid_edge_split_nodes(face, boundary_nodes);
    let split_node_by_edge = face_edges(face)
        .into_iter()
        .zip(split_nodes.iter())
        .map(|(edge, node)| (sorted_edge(edge), node.node_id))
        .collect::<BTreeMap<_, _>>();
    let split_node_coordinates = split_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    three_edge_split_completion_tetrahedra_for_node(
        face,
        cap_node_id,
        &split_node_by_edge,
        &split_node_coordinates,
        boundary_nodes,
        diagnostic_options,
    )
    .map(|tetrahedra| {
        tetrahedra
            .iter()
            .map(|tetrahedron| {
                let points = tetrahedron.node_ids.map(|node_id| {
                    split_node_coordinates
                        .get(&node_id)
                        .copied()
                        .unwrap_or_else(|| boundary_nodes[&node_id])
                });
                (
                    tetrahedron.exact_scaled_jacobian,
                    diagnostic_scaled_jacobian_worst_corner_label(points),
                )
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap_or((f64::INFINITY, "face_vertex"))
    })
}

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

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_local_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
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
    let missing_face_patches = missing_face_components(&missing_faces, MissingFaceLink::Node);
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
    if candidate_tetrahedra.len() > 4_096 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        &candidate_tetrahedra,
        options.volume_relative_tolerance,
        25_000,
    );
    let root_availability = search.root_boundary_availability();
    diagnostic.root_boundary_zero_raw_candidate_face_count =
        root_availability.zero_raw_candidate_face_count;
    diagnostic.root_boundary_zero_addable_candidate_face_count =
        root_availability.zero_addable_candidate_face_count;
    diagnostic.root_boundary_min_raw_candidate_count = root_availability.min_raw_candidate_count;
    diagnostic.root_boundary_min_addable_candidate_count =
        root_availability.min_addable_candidate_count;
    diagnostic.root_boundary_max_addable_candidate_count =
        root_availability.max_addable_candidate_count;
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.cover_dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.cover_dead_end_reason = dead_end.reason;
        diagnostic.cover_dead_end_depth = dead_end.depth;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 25_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidate_tetrahedra[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_shared_patch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Node,
        "incomplete_shared_patch_caps",
        false,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_edge_subpatch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Edge,
        "incomplete_edge_subpatch_caps",
        false,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_hybrid_subpatch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Edge,
        "incomplete_hybrid_subpatch_caps",
        true,
    )
}

#[cfg(test)]
fn diagnostic_missing_face_shared_cap_stitch_with_link(
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
    if candidate_tetrahedra.len() > 4_096 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        &candidate_tetrahedra,
        options.volume_relative_tolerance,
        25_000,
    );
    let root_availability = search.root_boundary_availability();
    diagnostic.root_boundary_zero_raw_candidate_face_count =
        root_availability.zero_raw_candidate_face_count;
    diagnostic.root_boundary_zero_addable_candidate_face_count =
        root_availability.zero_addable_candidate_face_count;
    diagnostic.root_boundary_min_raw_candidate_count = root_availability.min_raw_candidate_count;
    diagnostic.root_boundary_min_addable_candidate_count =
        root_availability.min_addable_candidate_count;
    diagnostic.root_boundary_max_addable_candidate_count =
        root_availability.max_addable_candidate_count;
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.cover_dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.cover_dead_end_reason = dead_end.reason;
        diagnostic.cover_dead_end_depth = dead_end.depth;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 25_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidate_tetrahedra[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_missing_face_clusters(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryMissingFaceClusterDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
            refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let edge_component_sizes = missing_face_component_sizes(&missing_faces, MissingFaceLink::Edge);
    let node_components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    let node_component_sizes = node_components.iter().map(Vec::len).collect::<Vec<_>>();
    let mut node_component_common_node_count_histogram = BTreeMap::<usize, usize>::new();
    let mut node_component_common_node_ids = BTreeMap::<u32, usize>::new();
    for component in &node_components {
        let common_node_ids = missing_face_component_common_node_ids(&missing_faces, component);
        *node_component_common_node_count_histogram
            .entry(common_node_ids.len())
            .or_default() += 1;
        for node_id in common_node_ids {
            *node_component_common_node_ids.entry(node_id).or_default() += 1;
        }
    }
    Ok(BoundaryMissingFaceClusterDiagnostic {
        missing_face_count: missing_faces.len(),
        edge_component_count: edge_component_sizes.len(),
        edge_component_size_histogram: component_size_histogram(edge_component_sizes),
        node_component_count: node_component_sizes.len(),
        node_component_size_histogram: component_size_histogram(node_component_sizes),
        node_component_common_node_count_histogram,
        node_component_common_node_ids,
    })
}
