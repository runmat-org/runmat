use super::*;

#[test]
fn missing_face_local_cap_quality_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_missing_face_local_cap_quality(&cavity, &nodes, refill_options())
        .expect("local cap diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.pass_face_count, 0);
    assert_eq!(diagnostic.failed_face_count, 0);
    assert_eq!(diagnostic.candidate_count, 0);
    assert!(diagnostic.candidate_source_bins.is_empty());
    assert_eq!(diagnostic.max_scaled_jacobian, 0.0);
    assert_eq!(diagnostic.max_failed_face_scaled_jacobian, 0.0);
    assert!(diagnostic.failed_face_scaled_jacobian_bins.is_empty());
    assert!(diagnostic.failed_face_source_bins.is_empty());
    assert!(diagnostic.rejected_by_reason.is_empty());
}

#[test]
fn local_cap_apex_candidates_include_optimized_normal_offsets() {
    let face = [0, 1, 2];
    let nodes = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.18, 0.72, 0.0]),
    ]);
    let surface_point = face_centroid(face, &nodes).expect("face should have a centroid");
    let candidates = local_cap_apex_candidates(face, surface_point, [0.3, 0.2, 0.8], &nodes);

    let quality_for = |candidate: &LocalCapApexCandidate| {
        tetrahedron_scaled_jacobian([
            nodes[&face[0]],
            nodes[&face[1]],
            nodes[&face[2]],
            candidate.coordinates_m,
        ])
    };
    let best_discrete_positive = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_positive")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_discrete_negative = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_negative")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_optimized_positive = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_optimized_positive")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_optimized_negative = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_optimized_negative")
        .map(quality_for)
        .fold(0.0_f64, f64::max);

    assert!(best_optimized_positive >= best_discrete_positive);
    assert!(best_optimized_negative >= best_discrete_negative);
}

#[test]
fn local_cap_apex_candidates_include_inplane_inward_offsets() {
    let face = [0, 1, 2];
    let nodes = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.2, 0.8, 0.0]),
    ]);
    let surface_point = face_centroid(face, &nodes).expect("face should have a centroid");
    let candidates = local_cap_apex_candidates(face, surface_point, [0.3, 0.2, 0.8], &nodes);
    let inplane_candidates = candidates
        .iter()
        .filter(|candidate| candidate.source == "inplane_inward")
        .collect::<Vec<_>>();
    let optimized_candidates = candidates
        .iter()
        .filter(|candidate| candidate.source == "inplane_inward_optimized")
        .collect::<Vec<_>>();

    assert!(!inplane_candidates.is_empty());
    assert!(!optimized_candidates.is_empty());
    assert!(inplane_candidates.iter().any(|candidate| {
        candidate.coordinates_m[2] > surface_point[2]
            && (candidate.coordinates_m[0] - surface_point[0]).abs() > 1.0e-6
    }));
    assert!(optimized_candidates.iter().any(|candidate| {
        candidate.coordinates_m[2] > surface_point[2]
            && (candidate.coordinates_m[0] - surface_point[0]).abs() > 1.0e-6
    }));
}

#[test]
fn missing_face_local_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_missing_face_local_cap_stitch(&cavity, &nodes, refill_options())
        .expect("local cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.side_connector_candidate_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert_eq!(diagnostic.cap_side_face_count, 0);
    assert_eq!(diagnostic.zero_mate_cap_side_face_count, 0);
    assert_eq!(diagnostic.min_cap_side_face_mate_count, 0);
    assert_eq!(diagnostic.max_cap_side_face_mate_count, 0);
    assert_eq!(diagnostic.open_interior_face_count, 0);
    assert_eq!(diagnostic.open_interior_component_count, 0);
    assert!(diagnostic.open_interior_component_size_histogram.is_empty());
    assert_eq!(diagnostic.selected_tetrahedron_count, 0);
    assert_eq!(diagnostic.search_attempt_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
}

#[test]
fn missing_face_shared_patch_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic =
        diagnostic_missing_face_shared_patch_cap_stitch(&cavity, &nodes, refill_options())
            .expect("shared patch cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.side_connector_candidate_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}

#[test]
fn missing_face_edge_subpatch_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic =
        diagnostic_missing_face_edge_subpatch_cap_stitch(&cavity, &nodes, refill_options())
            .expect("edge subpatch cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}

#[test]
fn missing_face_hybrid_subpatch_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic =
        diagnostic_missing_face_hybrid_subpatch_cap_stitch(&cavity, &nodes, refill_options())
            .expect("hybrid subpatch cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}

#[test]
fn shared_patch_cap_finds_single_apex_for_simple_patch() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let node_coordinates = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let boundary_triangles = cavity_boundary_triangles(&cavity, &node_coordinates)
        .expect("unit tetrahedron boundary should be valid");
    let faces = [[0, 1, 2], [0, 1, 3]];

    let Some((coordinates_m, cap_tetrahedra)) = best_shared_patch_cap_for_faces(
        &faces,
        [0.25, 0.25, 0.25],
        4,
        &node_coordinates,
        &boundary_triangles,
        refill_options(),
    ) else {
        panic!("simple patch should have a shared cap apex");
    };

    assert_eq!(cap_tetrahedra.len(), faces.len());
    assert!(coordinates_m.iter().all(|value| value.is_finite()));
    assert!(cap_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&4)
            && tetrahedron.exact_scaled_jacobian.is_finite()));
}

#[test]
fn missing_face_components_separate_edge_and_node_connected_patches() {
    let faces = [[0, 1, 2], [2, 1, 3], [3, 4, 5], [3, 6, 7]];

    let edge_histogram =
        component_size_histogram(missing_face_component_sizes(&faces, MissingFaceLink::Edge));
    let node_histogram =
        component_size_histogram(missing_face_component_sizes(&faces, MissingFaceLink::Node));
    let node_components = missing_face_components(&faces, MissingFaceLink::Node);
    let common_node_ids =
        missing_face_component_common_node_ids(&faces, node_components.first().unwrap());

    assert_eq!(edge_histogram, BTreeMap::from([(1, 2), (2, 1)]));
    assert_eq!(node_histogram, BTreeMap::from([(4, 1)]));
    assert_eq!(common_node_ids, Vec::<u32>::new());

    let fan_faces = [[9, 1, 2], [9, 2, 3], [9, 3, 4]];
    let fan_components = missing_face_components(&fan_faces, MissingFaceLink::Node);
    assert_eq!(
        missing_face_component_common_node_ids(&fan_faces, fan_components.first().unwrap()),
        vec![9]
    );
}

#[test]
fn open_interior_refill_faces_reports_unpaired_non_boundary_faces() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let lower = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let upper = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 2, 1, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };

    assert_eq!(
        open_interior_refill_faces(&cavity, &[lower.clone()]),
        vec![[0, 1, 2]]
    );
    assert!(open_interior_refill_faces(&cavity, &[lower, upper]).is_empty());
}

#[test]
fn cap_side_face_mate_counts_report_connector_coverage() {
    let cap_tetrahedron = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 4],
        volume_m3: 1.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    };
    let mate_tetrahedron = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 4, 5],
        volume_m3: 1.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    };

    assert_eq!(
        cap_side_face_mate_counts(
            &[cap_tetrahedron.clone()],
            &[cap_tetrahedron, mate_tetrahedron],
            &BTreeSet::from([4])
        ),
        vec![1, 0, 0]
    );
}

#[test]
fn cap_side_connector_chain_adds_mates_for_open_inserted_faces() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let mut nodes = boundary_node_coordinates(&cavity, &two_tetrahedron_bipyramid_nodes())
        .expect("fixture nodes should cover cavity");
    nodes.insert(5, [0.25, 0.25, 0.0]);
    let boundary_triangles =
        cavity_boundary_triangles(&cavity, &nodes).expect("fixture boundary should evaluate");
    let mut candidate_tetrahedra = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 5],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let mut seen_tetrahedra = candidate_tetrahedra
        .iter()
        .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
        .collect::<BTreeSet<_>>();

    let inserted = append_cap_side_connector_chain_tetrahedra(
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &nodes,
        &BTreeSet::from([5]),
        &boundary_triangles,
        refill_options(),
    );

    assert!(inserted > 0);
    assert!(candidate_tetrahedra.len() > 1);
    assert!(candidate_tetrahedra
        .iter()
        .skip(1)
        .any(|tetrahedron| tetrahedron.node_ids.contains(&5)));
}

#[test]
fn cap_side_connector_chain_recovers_exact_cover_with_inserted_node_mates() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let mut nodes = boundary_node_coordinates(&cavity, &two_tetrahedron_bipyramid_nodes())
        .expect("fixture nodes should cover cavity");
    nodes.insert(5, [0.25, 0.25, 0.0]);
    let boundary_triangles =
        cavity_boundary_triangles(&cavity, &nodes).expect("fixture boundary should evaluate");
    let options = refill_options();
    let mut candidate_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for tetrahedron_node_ids in [[0, 1, 3, 5], [1, 2, 3, 5], [0, 2, 3, 5]] {
        let points = tetrahedron_node_ids.map(|node_id| nodes[&node_id]);
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(tetrahedron_node_ids, points, options)
        {
            seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids));
            candidate_tetrahedra.push(tetrahedron);
        }
    }
    assert!(
        exact_cover_refill_from_candidate_tetrahedra(&cavity, &candidate_tetrahedra, options)
            .expect("initial exact cover should evaluate")
            .is_none()
    );
    let inserted = append_cap_side_connector_chain_tetrahedra(
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &nodes,
        &BTreeSet::from([5]),
        &boundary_triangles,
        options,
    );
    assert_eq!(inserted, 3);
    let refill =
        exact_cover_refill_from_candidate_tetrahedra(&cavity, &candidate_tetrahedra, options)
            .expect("connector exact cover should evaluate")
            .expect("connector mates should close the inserted-node cover");
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("connector cover should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("connector cover should preserve volume");
}
