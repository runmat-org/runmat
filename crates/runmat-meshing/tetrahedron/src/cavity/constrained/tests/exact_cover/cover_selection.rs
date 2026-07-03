use super::*;

#[test]
fn exact_cover_refill_selects_compatible_subset() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let options = refill_options();
    let candidate_nodes = [[0, 1, 2, 3], [0, 2, 1, 4], [1, 2, 3, 4]];
    let candidates = candidate_nodes
        .map(|node_ids| {
            let points = node_ids.map(|node_id| boundary_nodes[&node_id]);
            raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options)
                .expect("fixture tetrahedron should pass quality gates")
        })
        .to_vec();

    let refill = exact_cover_refill_from_candidate_tetrahedra(&cavity, &candidates, options)
        .expect("exact cover refill should evaluate")
        .expect("exact cover should select the compatible subset");

    assert_eq!(refill.tetrahedra.len(), 2);
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            sorted_tetrahedron_nodes([0, 1, 2, 3]),
            sorted_tetrahedron_nodes([0, 2, 1, 4])
        ])
    );
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("selected subset should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("selected subset should preserve volume");
}

#[test]
fn exact_cover_refill_maximizes_worst_selected_quality() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: [
            [4, 0, 1],
            [4, 1, 2],
            [4, 2, 3],
            [4, 3, 0],
            [5, 1, 0],
            [5, 2, 1],
            [5, 3, 2],
            [5, 0, 3],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let low_worst_cover = [
        ([4, 5, 0, 1], 0.90),
        ([4, 5, 1, 2], 0.20),
        ([4, 5, 2, 3], 0.20),
        ([4, 5, 3, 0], 0.20),
    ];
    let better_worst_cover = [
        ([0, 2, 4, 1], 0.50),
        ([0, 2, 4, 3], 0.50),
        ([0, 2, 5, 1], 0.50),
        ([0, 2, 5, 3], 0.50),
    ];
    let candidates = low_worst_cover
        .into_iter()
        .chain(better_worst_cover)
        .map(
            |(node_ids, exact_scaled_jacobian)| ConstrainedCavityRefillTetrahedron {
                node_ids,
                volume_m3: 0.25,
                aspect_ratio: 1.0,
                exact_scaled_jacobian,
            },
        )
        .collect::<Vec<_>>();

    let refill = exact_cover_refill_from_candidate_tetrahedra(
        &cavity,
        &candidates,
        ConstrainedCavityRefillOptions {
            volume_relative_tolerance: 1.0e-9,
            ..refill_options()
        },
    )
    .expect("exact cover should evaluate")
    .expect("octahedron cavity should have a cover");

    assert_eq!(refill.tetrahedra.len(), 4);
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        better_worst_cover
            .into_iter()
            .map(|(node_ids, _)| sorted_tetrahedron_nodes(node_ids))
            .collect::<BTreeSet<_>>()
    );
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min),
        0.50
    );
}
