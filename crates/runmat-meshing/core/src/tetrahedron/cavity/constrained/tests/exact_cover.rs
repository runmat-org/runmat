use super::*;

mod diagnostics;
mod forced_mate_traces;
mod search_mechanics;

#[test]
fn boundary_node_exact_cover_supports_bounded_multi_ring_bipyramid() {
    let ring_count = 7_u32;
    let top_node_id = ring_count;
    let bottom_node_id = ring_count + 1;
    let mut nodes = (0..ring_count)
        .map(|node_id| {
            let angle = std::f64::consts::TAU * node_id as f64 / ring_count as f64;
            ConstrainedCavityNode {
                node_id,
                coordinates_m: [angle.cos(), angle.sin(), 0.0],
            }
        })
        .collect::<Vec<_>>();
    nodes.push(ConstrainedCavityNode {
        node_id: top_node_id,
        coordinates_m: [0.0, 0.0, 1.0],
    });
    nodes.push(ConstrainedCavityNode {
        node_id: bottom_node_id,
        coordinates_m: [0.0, 0.0, -1.0],
    });

    let options = refill_options();
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut boundary_faces = Vec::<ConstrainedCavityBoundaryFace>::new();
    let mut expected_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for node_id in 0..ring_count {
        let next_node_id = (node_id + 1) % ring_count;
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: [top_node_id, node_id, next_node_id],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: [bottom_node_id, next_node_id, node_id],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
        let tetrahedron_node_ids = [top_node_id, bottom_node_id, node_id, next_node_id];
        expected_tetrahedra.push(
            raw_refill_tetrahedron_with_rejection_reason(
                tetrahedron_node_ids,
                tetrahedron_node_ids.map(|id| node_map[&id]),
                options,
            )
            .expect("ring bipyramid tetrahedron should pass quality gates"),
        );
    }
    let expected_volume_m3 = expected_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.volume_m3)
        .sum::<f64>();
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces,
        protected_node_ids: Vec::new(),
        target_volume_m3: expected_volume_m3,
    };
    validate_constrained_cavity(&cavity).expect("ring bipyramid cavity should validate");
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");

    let refill = boundary_node_exact_cover_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        options,
    )
    .expect("exact cover should evaluate")
    .expect("bounded ring bipyramid should have an exact cover");

    assert_eq!(refill.tetrahedra.len(), ring_count as usize);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("exact cover should preserve the larger cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("exact cover should preserve the larger cavity volume");
}

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
fn exact_cover_on_demand_interior_mates_recovers_forced_mate() {
    let options = refill_options();
    let central = synthetic_refill_tetrahedron([0, 1, 2, 3], 1.0);
    let caps = [
        synthetic_refill_tetrahedron([0, 2, 1, 4], 1.0),
        synthetic_refill_tetrahedron([0, 1, 3, 5], 1.0),
        synthetic_refill_tetrahedron([0, 3, 2, 6], 1.0),
        synthetic_refill_tetrahedron([1, 2, 3, 7], 1.0),
    ];
    let shared_faces = BTreeSet::from([
        sorted_face([0, 1, 2]),
        sorted_face([0, 1, 3]),
        sorted_face([0, 2, 3]),
        sorted_face([1, 2, 3]),
    ]);
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: caps
            .iter()
            .flat_map(|tetrahedron| tetrahedron_faces(tetrahedron.node_ids))
            .map(sorted_face)
            .filter(|face| !shared_faces.contains(face))
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 5.0,
    };
    let refill = exact_cover_refill_from_on_demand_interior_mates(
        &cavity,
        caps.to_vec(),
        caps.into_iter().chain([central]).collect(),
        options,
    )
    .expect("on-demand exact cover should evaluate")
    .expect("on-demand mate injection should recover the cover");

    assert_eq!(refill.tetrahedra.len(), 5);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("on-demand exact cover should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("on-demand exact cover should preserve volume");
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

#[test]
fn exact_cover_search_uses_configured_attempt_limit() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 1, 4],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
    ];
    let mut low_limit_search =
        BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 1);

    assert!(low_limit_search.search().is_none());
    assert!(low_limit_search.attempts > 1);

    let mut sufficient_limit_search =
        BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 2);

    assert_eq!(sufficient_limit_search.search(), Some(vec![0, 1]));
    assert_eq!(sufficient_limit_search.attempts, 2);
}

#[test]
fn exact_cover_trace_reports_volume_overflow_dead_end() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let candidates = [
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 10.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 1, 4],
            volume_m3: 10.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
    ];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-12);

    let (selected, trace) = search.search_with_trace();

    assert!(selected.is_none());
    assert_eq!(
        trace.dead_end,
        Some(BoundaryExactCoverDeadEnd {
            reason: "volume_overflow",
            face: None,
            depth: 1,
            selected_tetrahedra: vec![[0, 1, 2, 3]],
            selected_roles: vec!["branch"],
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.0,
            target_volume_m3: 1.0 / 3.0,
        })
    );
}

#[test]
fn boundary_steiner_exact_cover_diagnostic_reports_centroid_candidate_coverage() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let diagnostic = diagnostic_boundary_steiner_exact_cover(&cavity, &nodes, refill_options())
        .expect("Steiner exact-cover diagnostic should evaluate");

    assert!(diagnostic.candidate_count > 0);
    assert_eq!(diagnostic.zero_candidate_boundary_face_count, 0);
    assert!(diagnostic.search_attempt_count > 0);
    assert_eq!(diagnostic.reason, "cover_found");
    assert!(diagnostic.selected_tetrahedron_count > 0);
}

#[test]
fn boundary_patch_steiner_exact_cover_diagnostic_reports_boundary_complete_fixture() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let diagnostic =
        diagnostic_boundary_patch_steiner_exact_cover(&cavity, &nodes, refill_options())
            .expect("patch Steiner exact-cover diagnostic should evaluate");

    assert_eq!(diagnostic.boundary_node_count, 8);
    assert_eq!(diagnostic.boundary_face_count, 12);
    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert_eq!(diagnostic.steiner_node_count, 0);
    assert_eq!(diagnostic.candidate_count, 0);
    assert_eq!(diagnostic.search_attempt_count, 0);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}
