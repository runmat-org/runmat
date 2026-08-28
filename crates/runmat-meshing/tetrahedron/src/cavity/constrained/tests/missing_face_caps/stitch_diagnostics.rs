use super::super::*;

#[test]
fn missing_face_local_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_missing_face_local_cap_stitch(&cavity, &nodes, refill_options())
        .expect("local cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
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
