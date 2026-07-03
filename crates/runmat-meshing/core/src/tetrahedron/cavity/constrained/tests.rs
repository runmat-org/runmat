use super::*;

mod boundary_refinement;
mod exact_cover;
mod extraction_validation;
mod refill_evaluation;
mod solid_empty;

#[test]
fn refill_tetrahedron_component_cavity_preserves_boundary_metadata_and_volume() {
    let source_cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map = boundary_node_coordinates(&source_cavity, &nodes)
        .expect("fixture nodes should cover source boundary");
    let options = refill_options();
    let lower = raw_refill_tetrahedron_with_rejection_reason(
        [0, 1, 2, 3],
        [0, 1, 2, 3].map(|node_id| node_map[&node_id]),
        options,
    )
    .expect("lower bipyramid tetrahedron should pass quality gates");
    let upper = raw_refill_tetrahedron_with_rejection_reason(
        [0, 2, 1, 4],
        [0, 2, 1, 4].map(|node_id| node_map[&node_id]),
        options,
    )
    .expect("upper bipyramid tetrahedron should pass quality gates");

    let component_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &[lower, upper],
        &source_cavity.boundary_faces,
        vec![0],
    )
    .expect("selected component should define a valid cavity");

    assert_eq!(component_cavity.removed_tetrahedron_ids, vec![0, 1]);
    assert_eq!(component_cavity.boundary_faces.len(), 6);
    assert_eq!(component_cavity.protected_node_ids, vec![0]);
    assert!((component_cavity.target_volume_m3 - source_cavity.target_volume_m3).abs() < 1.0e-12);
    assert!(!component_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    let inherited_face = component_cavity
        .boundary_faces
        .iter()
        .find(|face| sorted_face(face.node_ids) == [0, 1, 3])
        .expect("component cavity should preserve inherited source face");
    assert_eq!(inherited_face.region_ids, vec!["body".to_string()]);
}

#[test]
fn boundary_edge_star_recovery_reports_added_tetrahedra() {
    let source_tetrahedra = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]]
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| candidate_tetrahedron(index as u32 + 10, node_ids, 1.0, &["body"]))
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_selected_tetrahedra(&source_tetrahedra, &[0], Vec::new())
        .expect("single-tetrahedron source cavity should extract");

    let recovery = constrained_cavity_recovered_boundary_edge_star_excluding_nodes(
        &cavity,
        &source_tetrahedra,
        [1, 0],
        &[],
    )
    .expect("boundary edge-star recovery should evaluate");

    assert_eq!(recovery.attempted_boundary_faces, Vec::<[u32; 3]>::new());
    assert_eq!(
        recovery.recovered_edge,
        Some(ConstrainedCavityBoundaryEdgeRecoveryStep {
            node_ids: [0, 1],
            added_tetrahedron_ids: vec![11, 12, 13],
            removed_tetrahedron_count_before: 1,
            removed_tetrahedron_count_after: 4,
        })
    );
    assert_eq!(
        recovery.cavity.removed_tetrahedron_ids,
        vec![10, 11, 12, 13]
    );
    validate_constrained_cavity(&recovery.cavity)
        .expect("edge-star recovered cavity should remain valid");
}

#[test]
fn boundary_edge_star_recovery_queue_reports_ordered_steps() {
    let source_tetrahedra = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2], [1, 2, 4, 5]]
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| candidate_tetrahedron(index as u32 + 20, node_ids, 1.0, &["body"]))
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_selected_tetrahedra(&source_tetrahedra, &[0], Vec::new())
        .expect("single-tetrahedron source cavity should extract");

    let recovery = constrained_cavity_recovered_boundary_edge_star_queue_excluding_nodes(
        &cavity,
        &source_tetrahedra,
        &[[0, 1], [2, 4]],
        &[],
    )
    .expect("ordered boundary edge-star queue should evaluate");

    assert_eq!(
        recovery.steps,
        vec![
            ConstrainedCavityBoundaryEdgeRecoveryStep {
                node_ids: [0, 1],
                added_tetrahedron_ids: vec![21, 22],
                removed_tetrahedron_count_before: 1,
                removed_tetrahedron_count_after: 3,
            },
            ConstrainedCavityBoundaryEdgeRecoveryStep {
                node_ids: [2, 4],
                added_tetrahedron_ids: vec![23],
                removed_tetrahedron_count_before: 3,
                removed_tetrahedron_count_after: 4,
            }
        ]
    );
    assert_eq!(
        recovery.cavity.removed_tetrahedron_ids,
        vec![20, 21, 22, 23]
    );
    validate_constrained_cavity(&recovery.cavity)
        .expect("queued edge-star recovered cavity should remain valid");
}

#[test]
fn refill_tetrahedron_component_cavity_round_trips_through_refill_evaluation() {
    let source_cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map = boundary_node_coordinates(&source_cavity, &nodes)
        .expect("fixture nodes should cover source boundary");
    let options = refill_options();
    let component_tetrahedra = [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 2, 1, 4], [0, 2, 1, 4])]
        .into_iter()
        .map(|(node_ids, point_ids)| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                point_ids.map(|node_id| node_map[&node_id]),
                options,
            )
            .expect("component tetrahedron should pass quality gates")
        })
        .collect::<Vec<_>>();
    let component_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &component_tetrahedra,
        &source_cavity.boundary_faces,
        Vec::new(),
    )
    .expect("selected component should define a valid cavity");

    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&component_cavity, &nodes, &[], options)
            .expect("component cavity refill should evaluate");
    let refill = evaluation
        .refill
        .expect("component cavity should be refillable");

    assert_eq!(refill.tetrahedra.len(), component_tetrahedra.len());
    validate_constrained_cavity_boundary_preserved(&component_cavity, &refill.boundary_faces)
        .expect("component refill should preserve derived boundary");
    validate_constrained_cavity_refill_volume(
        component_cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("component refill should preserve derived volume");
}

#[test]
fn component_retriangulation_from_nodes_preserves_boundary_and_volume() {
    let source_cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map = boundary_node_coordinates(&source_cavity, &nodes)
        .expect("fixture nodes should cover source boundary");
    let options = refill_options();
    let component_tetrahedra = [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 2, 1, 4], [0, 2, 1, 4])]
        .into_iter()
        .map(|(node_ids, point_ids)| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                point_ids.map(|node_id| node_map[&node_id]),
                options,
            )
            .expect("component tetrahedron should pass quality gates")
        })
        .collect::<Vec<_>>();
    let component_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &component_tetrahedra,
        &source_cavity.boundary_faces,
        Vec::new(),
    )
    .expect("selected component should define a valid cavity");

    let refill = retriangulate_constrained_cavity_from_nodes(&component_cavity, &nodes, options)
        .expect("component retriangulation should evaluate")
        .expect("component should have an exact cover");

    validate_constrained_cavity_boundary_preserved(&component_cavity, &refill.boundary_faces)
        .expect("component retriangulation should preserve boundary");
    validate_constrained_cavity_refill_volume(
        component_cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("component retriangulation should preserve volume");
}

#[test]
fn component_retriangulation_rejects_duplicate_node_ids() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 3,
        coordinates_m: [0.1, 0.1, 0.1],
    });

    let err = retriangulate_constrained_cavity_from_nodes(&cavity, &nodes, refill_options())
        .expect_err("duplicate node ids should be rejected");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::DuplicateInteriorNode { node_id: 3 }
    );
}

#[test]
fn component_steiner_nodes_are_bounded_inside_and_retriangulatable() {
    let source_cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map = boundary_node_coordinates(&source_cavity, &nodes)
        .expect("fixture nodes should cover source boundary");
    let options = refill_options();
    let component_tetrahedra = [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 2, 1, 4], [0, 2, 1, 4])]
        .into_iter()
        .map(|(node_ids, point_ids)| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                point_ids.map(|node_id| node_map[&node_id]),
                options,
            )
            .expect("component tetrahedron should pass quality gates")
        })
        .collect::<Vec<_>>();
    let component_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &component_tetrahedra,
        &source_cavity.boundary_faces,
        Vec::new(),
    )
    .expect("selected component should define a valid cavity");

    let steiner_nodes = generate_constrained_cavity_component_steiner_nodes(
        &component_cavity,
        &nodes,
        &component_tetrahedra,
        options,
        4,
    )
    .expect("component Steiner generation should evaluate");

    assert_eq!(steiner_nodes.len(), 4);
    assert_eq!(
        steiner_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![5, 6, 7, 8]
    );
    let boundary_node_map = boundary_node_coordinates(&component_cavity, &nodes)
        .expect("fixture nodes should cover component boundary");
    let boundary_triangles = cavity_boundary_triangles(&component_cavity, &boundary_node_map)
        .expect("component boundary should build triangles");
    assert!(steiner_nodes.iter().all(|node| {
        point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
    }));
    let mut nodes_with_steiner = nodes.clone();
    nodes_with_steiner.extend(steiner_nodes);
    let refill = retriangulate_constrained_cavity_from_nodes(
        &component_cavity,
        &nodes_with_steiner,
        options,
    )
    .expect("Steiner component retriangulation should evaluate")
    .expect("component should remain retriangulatable with generated Steiner nodes");
    validate_constrained_cavity_boundary_preserved(&component_cavity, &refill.boundary_faces)
        .expect("Steiner retriangulation should preserve boundary");
    validate_constrained_cavity_refill_volume(
        component_cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("Steiner retriangulation should preserve volume");
}

#[test]
fn patch_steiner_nodes_are_empty_for_boundary_complete_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let steiner_nodes =
        generate_constrained_cavity_patch_steiner_nodes(&cavity, &nodes, refill_options(), 4)
            .expect("patch Steiner generation should evaluate");

    assert!(steiner_nodes.is_empty());
}

#[test]
fn patch_steiner_nodes_are_bounded_inside_and_unique() {
    let cavity = unit_cube_cavity();
    let nodes = unit_cube_nodes();
    let options = refill_options();

    let steiner_nodes =
        generate_constrained_cavity_patch_steiner_nodes(&cavity, &nodes, options, 4)
            .expect("patch Steiner generation should evaluate");

    assert_eq!(steiner_nodes.len(), 4);
    assert_eq!(
        steiner_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![8, 9, 10, 11]
    );
    let boundary_node_map = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_node_map)
        .expect("cavity boundary should build triangles");
    assert!(steiner_nodes.iter().all(|node| {
        point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
    }));
    let mut all_node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    for node in &steiner_nodes {
        assert!(all_node_ids.insert(node.node_id));
    }
}

#[test]
fn shared_face_split_preserves_component_boundary_and_volume() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map =
        boundary_node_coordinates(&cavity, &nodes).expect("fixture nodes should cover cavity");
    let options = refill_options();
    let component_tetrahedra = [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 2, 1, 4], [0, 2, 1, 4])]
        .into_iter()
        .map(|(node_ids, point_ids)| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                point_ids.map(|node_id| node_map[&node_id]),
                options,
            )
            .expect("component tetrahedron should pass quality gates")
        })
        .collect::<Vec<_>>();

    let (split_tetrahedra, split_node) = split_refill_tetrahedra_across_shared_face_at_barycentric(
        &component_tetrahedra,
        &nodes,
        [0, 1, 2],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        options,
    )
    .expect("shared face should split");
    let refill =
        refill_from_tetrahedra(&cavity, split_tetrahedra, options.volume_relative_tolerance)
            .expect("shared-face split should preserve cavity boundary");

    assert_eq!(split_node.node_id, 5);
    assert_eq!(split_node.coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    assert_eq!(refill.tetrahedra.len(), 6);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("shared-face split should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("shared-face split should preserve volume");
}

#[test]
fn shared_face_split_composes_and_preserves_component_boundary_and_volume() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let mut nodes = two_tetrahedron_bipyramid_nodes();
    let node_map =
        boundary_node_coordinates(&cavity, &nodes).expect("fixture nodes should cover cavity");
    let options = refill_options();
    let component_tetrahedra = [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 2, 1, 4], [0, 2, 1, 4])]
        .into_iter()
        .map(|(node_ids, point_ids)| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                point_ids.map(|node_id| node_map[&node_id]),
                options,
            )
            .expect("component tetrahedron should pass quality gates")
        })
        .collect::<Vec<_>>();

    let (first_split_tetrahedra, first_split_node) =
        split_refill_tetrahedra_across_shared_face_at_barycentric(
            &component_tetrahedra,
            &nodes,
            [0, 1, 2],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            options,
        )
        .expect("first shared face should split");
    nodes.push(first_split_node.clone());
    let (second_split_tetrahedra, second_split_node) =
        split_refill_tetrahedra_across_shared_face_at_barycentric(
            &first_split_tetrahedra,
            &nodes,
            [0, 1, first_split_node.node_id],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            options,
        )
        .expect("new shared child face should split");
    let refill = refill_from_tetrahedra(
        &cavity,
        second_split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("composed shared-face split should preserve cavity boundary");

    assert_eq!(first_split_node.node_id, 5);
    assert_eq!(second_split_node.node_id, 6);
    assert_eq!(refill.tetrahedra.len(), 10);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("composed shared-face split should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("composed shared-face split should preserve volume");
}

#[test]
fn shared_face_split_rejects_non_shared_face() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map =
        boundary_node_coordinates(&cavity, &nodes).expect("fixture nodes should cover cavity");
    let options = refill_options();
    let component_tetrahedra = [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 2, 1, 4], [0, 2, 1, 4])]
        .into_iter()
        .map(|(node_ids, point_ids)| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                point_ids.map(|node_id| node_map[&node_id]),
                options,
            )
            .expect("component tetrahedron should pass quality gates")
        })
        .collect::<Vec<_>>();

    let err = split_refill_tetrahedra_across_shared_face_at_barycentric(
        &component_tetrahedra,
        &nodes,
        [0, 1, 3],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        options,
    )
    .expect_err("boundary face should not split as a shared interior face");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronSplitError::FaceIncidenceNotTwo {
            node_ids: [0, 1, 3],
            incident_tetrahedron_count: 1
        }
    );
}

#[test]
fn boundary_node_refill_applies_quality_gated_two_to_three_flip() {
    let nodes = vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.05, 0.55, 0.3],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.55, 0.05, -0.3],
        },
    ];
    let boundary_nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let options = refill_options();
    let baseline_tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| boundary_nodes[&node_id]),
                options,
            )
            .expect("baseline tetrahedron should pass fixture quality gates")
        })
        .collect::<Vec<_>>();
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
            [0, 2, 4],
            [1, 2, 4],
            [0, 1, 4],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: vec!["body".to_string()],
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: baseline_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
    };
    let baseline = refill_from_tetrahedra(
        &cavity,
        baseline_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("baseline should preserve the cavity boundary");

    let evaluation = evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], options)
        .expect("refill evaluation should complete");

    let refill = evaluation.refill.expect("boundary-node refill should pass");
    assert_eq!(refill.tetrahedra.len(), 3);
    assert!(refill_is_better(&refill, &baseline));
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            sorted_tetrahedron_nodes([0, 1, 3, 4]),
            sorted_tetrahedron_nodes([1, 2, 3, 4]),
            sorted_tetrahedron_nodes([0, 2, 3, 4])
        ])
    );
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("flipped refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("flipped refill should preserve volume");
}

#[test]
fn boundary_node_refill_applies_quality_gated_three_to_two_flip() {
    let nodes = vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.45, 0.5, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.5, 0.45, -1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [0.0, 1.0, 0.0],
        },
    ];
    let boundary_nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let options = refill_options();
    let baseline_tetrahedron_node_ids = [[0, 3, 4, 5], [0, 4, 3, 6], [0, 5, 6, 3]];
    let baseline_tetrahedra = baseline_tetrahedron_node_ids
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| boundary_nodes[&node_id]),
                options,
            )
            .expect("baseline tetrahedron should pass fixture quality gates")
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for node_ids in baseline_tetrahedron_node_ids {
        for face in tetrahedron_faces(node_ids) {
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2, 3],
        boundary_faces: face_counts
            .into_iter()
            .filter_map(|(node_ids, count)| {
                (count == 1).then_some(ConstrainedCavityBoundaryFace {
                    node_ids,
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id: None,
                    source_edge_ids: [None, None, None],
                    region_ids: vec!["body".to_string()],
                })
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: baseline_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
    };
    let baseline = refill_from_tetrahedra(
        &cavity,
        baseline_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("baseline should preserve the cavity boundary");

    let evaluation = evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], options)
        .expect("refill evaluation should complete");

    let refill = evaluation.refill.expect("boundary-node refill should pass");
    assert_eq!(refill.tetrahedra.len(), 2);
    assert!(refill_is_better(&refill, &baseline));
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            sorted_tetrahedron_nodes([0, 4, 5, 6]),
            sorted_tetrahedron_nodes([3, 4, 5, 6])
        ])
    );
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("flipped refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("flipped refill should preserve volume");
}

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

#[test]
fn candidate_orphan_interior_face_counts_report_global_orphans() {
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
        candidate_orphan_interior_face_counts(&cavity, &[lower.clone()]),
        (1, 0)
    );
    assert_eq!(
        candidate_orphan_interior_face_counts(&cavity, &[lower, upper]),
        (0, 2)
    );
}

#[test]
fn centroid_interior_refill_candidate_recovers_split_boundary_tetrahedron_cavity() {
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
    validate_constrained_cavity(&cavity).expect("split boundary fixture should be valid");
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

    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let refill = centroid_interior_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        refill_options(),
    )
    .expect("centroid interior refill should evaluate")
    .expect("centroid interior refill should recover the split boundary cavity");

    assert_eq!(refill.inserted_nodes.len(), 1);
    assert_eq!(refill.inserted_nodes[0].node_id, 8);
    assert_eq!(refill.tetrahedra.len(), cavity.boundary_faces.len());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("centroid interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        refill_options().volume_relative_tolerance,
    )
    .expect("centroid interior refill should preserve volume");
}

#[test]
fn interior_star_quality_diagnostic_bins_candidate_quality() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let candidates = vec![
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [3.0, 3.0, 3.0],
        },
    ];

    let diagnostic = diagnostic_interior_star_quality(
        &cavity,
        &nodes,
        &candidates,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.01,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("interior star diagnostic should evaluate");

    assert_eq!(diagnostic.candidate_count, 1);
    assert_eq!(diagnostic.pass_count, 1);
    assert!(diagnostic.max_min_scaled_jacobian >= 0.01);
    assert!(!diagnostic.min_scaled_jacobian_bins.is_empty());
    assert_eq!(
        diagnostic.rejected_by_reason,
        BTreeMap::from([("interior_point_outside_cavity", 1)])
    );
}

#[test]
fn two_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
    ];
    let options = refill_options();

    let refill = two_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("two-interior refill should evaluate")
    .expect("two-interior refill should recover the cavity");

    assert_eq!(refill.inserted_nodes, interior_candidates);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("two-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("two-interior refill should preserve volume");
}

#[test]
fn multi_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
        ConstrainedCavityNode {
            node_id: 12,
            coordinates_m: [0.50, 0.25, 0.0],
        },
    ];
    let options = refill_options();

    let refill = multi_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("multi-interior refill should evaluate")
    .expect("multi-interior refill should recover the cavity");

    assert!(!refill.inserted_nodes.is_empty());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("multi-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("multi-interior refill should preserve volume");
}

#[test]
fn multi_interior_exact_cover_failure_reports_boundary_face_without_addable_candidate() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let options = refill_options();
    let lower_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let lower_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], lower_points, options)
            .expect("fixture tetrahedron should pass quality gates");

    assert_eq!(
        multi_interior_exact_cover_failure_reason(&cavity, &[lower_tetrahedron], options),
        "multi_interior_exact_cover_boundary_face_no_addable_candidate"
    );
}

#[test]
fn exact_cover_trace_reports_boundary_face_without_addable_candidate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: vec![
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 2],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 3],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
        ],
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 3, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
    ];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);

    let (selected, trace) = search.search_with_trace();

    assert!(selected.is_none());
    assert_eq!(
        trace.dead_end,
        Some(BoundaryExactCoverDeadEnd {
            reason: "boundary_face_no_addable_candidate",
            face: Some([0, 1, 2]),
            depth: 0,
            selected_tetrahedra: Vec::new(),
            selected_roles: Vec::new(),
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.0,
            target_volume_m3: 1.0,
        })
    );
    assert_eq!(
        trace.dead_end_reason_counts,
        BTreeMap::from([("boundary_face_no_addable_candidate", 1)])
    );
    assert_eq!(
        trace.dead_end_faces_by_reason,
        BTreeMap::from([(
            "boundary_face_no_addable_candidate",
            BTreeSet::from([[0, 1, 2]])
        )])
    );
}

#[test]
fn boundary_face_completion_skips_duplicate_cap_tetrahedra() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let duplicate_cap = raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, options)
        .expect("fixture cap should pass quality gates");

    let candidate = best_boundary_face_completion_tetrahedron(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &[duplicate_cap],
        &boundary_triangles,
        options,
    );

    assert!(candidate.is_none());
}

#[test]
fn boundary_face_completion_selector_reduces_boundary_delta() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let duplicate_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let duplicate_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], duplicate_points, options)
            .expect("fixture duplicate should pass quality gates");
    let blocked_face = [0, 1, 2];
    let fillable_face = [0, 2, 4];

    let (selected_face, selected_tetrahedron) =
        best_boundary_face_completion_tetrahedron_for_faces(
            &[blocked_face, fillable_face],
            &cavity,
            &boundary_nodes,
            &[duplicate_tetrahedron.clone()],
            &boundary_triangles,
            options,
        )
        .expect("completion search should evaluate")
        .expect("completion search should find a delta-reducing face");

    let initial_delta = refill_boundary_face_delta(&cavity, &[duplicate_tetrahedron.clone()])
        .expect("initial delta should evaluate");
    let next_delta = refill_boundary_face_delta(
        &cavity,
        &[duplicate_tetrahedron, selected_tetrahedron.clone()],
    )
    .expect("next delta should evaluate");
    assert!(
        next_delta.missing.len() + next_delta.unexpected.len()
            < initial_delta.missing.len() + initial_delta.unexpected.len()
    );
    assert!(tetrahedron_faces(selected_tetrahedron.node_ids)
        .map(sorted_face)
        .contains(&sorted_face(selected_face)));
}

#[test]
fn refill_boundary_delta_reports_unexpected_faces() {
    let cavity = unit_tetrahedron_cavity();
    let refill_tetrahedra = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }];

    let delta = refill_boundary_face_delta(&cavity, &refill_tetrahedra)
        .expect("boundary delta should evaluate");

    assert!(delta.missing.contains(&[0, 1, 3]));
    assert!(delta.unexpected.contains(&[0, 1, 4]));
}

#[test]
fn boundary_face_split_completion_reports_inserted_node_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_node, split_tetrahedra) = best_boundary_face_split_completion(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &[],
        options,
    )
    .expect("split completion should evaluate")
    .expect("split completion should generate child cap tetrahedra");

    assert_eq!(inserted_node.node_id, 4);
    assert!(inserted_node.coordinates_m[0] > 0.0);
    assert!(inserted_node.coordinates_m[1] > 0.0);
    assert_eq!(inserted_node.coordinates_m[2], 0.0);
    assert!(inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] < 1.0);
    assert_eq!(split_tetrahedra.len(), 3);
    assert!(split_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&inserted_node.node_id)));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&inserted_node.node_id))
            .count(),
        3
    );
    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("split child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("split completion should preserve the original target volume");
}

#[test]
fn boundary_face_edge_split_completion_reports_inserted_node_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_node, split_tetrahedra) =
        best_boundary_face_edge_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("edge-split completion should evaluate")
        .expect("edge-split completion should generate child cap tetrahedra");

    assert_eq!(inserted_node.node_id, 4);
    assert_eq!(inserted_node.coordinates_m[2], 0.0);
    assert!(
        (inserted_node.coordinates_m[0] == 0.0 && inserted_node.coordinates_m[1] > 0.0)
            || (inserted_node.coordinates_m[1] == 0.0 && inserted_node.coordinates_m[0] > 0.0)
            || (inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] - 1.0).abs()
                <= 1.0e-12
    );
    assert_eq!(split_tetrahedra.len(), 2);
    assert!(split_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&inserted_node.node_id)));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&inserted_node.node_id))
            .count(),
        4
    );
    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("edge-split child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("edge-split completion should preserve the original target volume");
}

#[test]
fn boundary_face_three_edge_split_completion_reports_inserted_nodes_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_nodes, split_tetrahedra) =
        best_boundary_face_three_edge_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("three-edge completion should evaluate")
        .expect("three-edge completion should generate child cap tetrahedra");

    assert_eq!(inserted_nodes.len(), 3);
    assert_eq!(
        inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );
    assert!(inserted_nodes
        .iter()
        .all(|node| node.coordinates_m[2].abs() <= 1.0e-12));
    assert_eq!(split_tetrahedra.len(), 4);
    assert!(split_tetrahedra.iter().all(|tetrahedron| {
        inserted_nodes
            .iter()
            .any(|node| tetrahedron.node_ids.contains(&node.node_id))
    }));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| inserted_nodes
                .iter()
                .any(|node| face.node_ids.contains(&node.node_id)))
            .count(),
        10
    );

    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("three-edge child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("three-edge completion should preserve the original target volume");
}

#[test]
fn boundary_face_split_completion_prefers_higher_quality_split_point() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: tetrahedron_faces([0, 1, 2, 3])
            .into_iter()
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: vec!["body".to_string()],
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 2.0 / 3.0,
    };
    let nodes = vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.649331064611886, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.10383330216927095, 0.5285988568010986, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [1.583996624105325, 0.04591313203731445, 1.25490017426856],
        },
    ];
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let centroid_node = boundary_face_centroid_node([0, 1, 2], &boundary_nodes);
    let centroid_tetrahedra = split_completion_tetrahedra_for_node(
        [0, 1, 2],
        3,
        &centroid_node,
        &boundary_nodes,
        options,
    )
    .expect("centroid split should generate child cap tetrahedra");
    let centroid_min_quality = centroid_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);

    let (_, inserted_node, split_tetrahedra) = best_boundary_face_split_completion(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &[],
        options,
    )
    .expect("split completion should evaluate")
    .expect("split completion should generate child cap tetrahedra");
    let selected_min_quality = split_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);

    assert!(
            selected_min_quality > centroid_min_quality + 1.0e-9,
            "split search should improve on the centroid split: selected={selected_min_quality} centroid={centroid_min_quality}"
        );
    assert_ne!(inserted_node.coordinates_m, centroid_node.coordinates_m);
}

#[test]
fn boundary_face_split_candidates_include_bounded_interior_lattice() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");

    let candidates = boundary_face_split_node_candidates([0, 1, 2], &boundary_nodes);

    assert!(candidates.len() >= 40);
    assert!(candidates.len() <= 64);
    assert!(candidates.iter().all(|node| node.node_id == 4));
    assert!(candidates.iter().all(|node| {
        node.coordinates_m[0] > 0.0
            && node.coordinates_m[1] > 0.0
            && node.coordinates_m[2] == 0.0
            && node.coordinates_m[0] + node.coordinates_m[1] < 1.0
    }));
    assert!(candidates.iter().any(|node| {
        (node.coordinates_m[0] - 0.1).abs() <= 1.0e-12
            && (node.coordinates_m[1] - 0.1).abs() <= 1.0e-12
    }));
}

#[test]
fn boundary_node_completion_diagnostic_classifies_no_cap_candidate() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let diagnostic = diagnostic_boundary_node_completion(
        &cavity,
        &nodes,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("diagnostic should evaluate");

    assert_eq!(diagnostic.reason, "boundary_node_completion_no_candidate");
    assert!(diagnostic.missing_face_count > 0);
    assert_eq!(diagnostic.cap_candidate_count, 0);
    assert!(diagnostic.max_rejected_scaled_jacobian < 0.95);
    assert!(!diagnostic.rejected_scaled_jacobian_bins.is_empty());
    assert!(diagnostic.max_rejected_cap_height_ratio > 0.0);
    assert!(!diagnostic.rejected_cap_height_ratio_bins.is_empty());
    assert!(!diagnostic
        .rejected_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.rejected_cap_node_ids.is_empty());
    assert!(diagnostic.split_cap_candidate_count > 0);
    assert_eq!(diagnostic.split_cap_pass_count, 0);
    assert!(diagnostic.max_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic.split_cap_scaled_jacobian_bins.is_empty());
    assert!(!diagnostic
        .split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.split_cap_apex_limited_node_ids.is_empty());
    assert!(diagnostic.edge_split_cap_candidate_count > 0);
    assert_eq!(diagnostic.edge_split_cap_pass_count, 0);
    assert!(diagnostic.max_edge_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic.edge_split_cap_scaled_jacobian_bins.is_empty());
    assert!(!diagnostic
        .edge_split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.edge_split_cap_apex_limited_node_ids.is_empty());
    assert!(diagnostic.three_edge_split_cap_candidate_count > 0);
    assert_eq!(diagnostic.three_edge_split_cap_pass_count, 0);
    assert!(diagnostic.max_three_edge_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic
        .three_edge_split_cap_scaled_jacobian_bins
        .is_empty());
    assert!(!diagnostic
        .three_edge_split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic
        .three_edge_split_cap_apex_limited_node_ids
        .is_empty());
    assert!(!diagnostic.rejected_by_reason.is_empty());
}

#[test]
fn refill_evaluation_skips_exterior_points_and_accepts_valid_candidate() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [2.0, 2.0, 2.0],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, 0.25],
        },
    ];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        refill_options(),
    )
    .expect("evaluation should complete");

    assert!(evaluation.refill.is_some());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("interior_point_outside_cavity".to_string(), 1)])
    );
}

#[test]
fn refill_evaluation_skips_points_too_close_to_protected_boundary_nodes() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.protected_node_ids = vec![0];
    let nodes = unit_tetrahedron_nodes();
    let candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.01, 0.01, 0.01],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, 0.25],
        },
    ];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        protected_refill_options(),
    )
    .expect("evaluation should continue after protected-distance rejection");

    assert!(evaluation.refill.is_some());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("protected_boundary_distance".to_string(), 1)])
    );
}

#[test]
fn refill_generation_reports_protected_boundary_distance_rejections() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.protected_node_ids = vec![0];
    let nodes = unit_tetrahedron_nodes();
    let candidates = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [0.01, 0.01, 0.01],
    }];

    let err = generate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        protected_refill_options(),
    )
    .expect_err("all candidates too close to protected nodes should fail");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::NoValidCandidate {
            rejected_by_reason: BTreeMap::from([("protected_boundary_distance".to_string(), 1)])
        }
    );
}

#[test]
fn shared_face_flip_preserves_component_boundary_and_volume() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_face_flip_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let flipped_tetrahedra =
        flip_refill_tetrahedra_across_shared_face(&tetrahedra, &nodes, [0, 1, 2], options)
            .expect("shared face should flip");

    assert_eq!(flipped_tetrahedra.len(), 3);
    assert!(flipped_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&3) && tetrahedron.node_ids.contains(&4)));
    let flipped_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &flipped_tetrahedra,
        &cavity.boundary_faces,
        Vec::new(),
    )
    .expect("flipped component should remain a valid cavity");
    validate_constrained_cavity_boundary_preserved(&cavity, &flipped_cavity.boundary_faces)
        .expect("face flip should preserve the component boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        flipped_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
        options.volume_relative_tolerance,
    )
    .expect("face flip should preserve target volume");
}

#[test]
fn shared_face_flip_rejects_boundary_face() {
    let nodes = two_tetrahedron_face_flip_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let err = flip_refill_tetrahedra_across_shared_face(&tetrahedra, &nodes, [0, 1, 3], options)
        .expect_err("boundary face should not have two incident tetrahedra");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronFlipError::FaceIncidenceNotTwo {
            node_ids: [0, 1, 3],
            incident_tetrahedron_count: 1,
        }
    );
}

#[test]
fn shared_edge_flip_preserves_component_boundary_and_volume() {
    let nodes = triangular_edge_ring_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 3, 4, 5], [0, 4, 3, 6], [0, 5, 6, 3]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| {
                    nodes
                        .iter()
                        .find(|node| node.node_id == node_id)
                        .expect("fixture node should exist")
                        .coordinates_m
                }),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_refill_tetrahedron_component(&tetrahedra, &[], Vec::new())
        .expect("edge-ring component should define a valid cavity");

    let flipped_tetrahedra =
        flip_refill_tetrahedra_around_shared_edge(&tetrahedra, &nodes, [0, 3], options)
            .expect("three-tetrahedron edge ring should flip");

    assert_eq!(flipped_tetrahedra.len(), 2);
    assert!(flipped_tetrahedra.iter().all(|tetrahedron| [4, 5, 6]
        .iter()
        .all(|node_id| tetrahedron.node_ids.contains(node_id))));
    let flipped_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &flipped_tetrahedra,
        &cavity.boundary_faces,
        Vec::new(),
    )
    .expect("flipped edge-ring component should remain a valid cavity");
    validate_constrained_cavity_boundary_preserved(&cavity, &flipped_cavity.boundary_faces)
        .expect("edge flip should preserve the component boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        flipped_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
        options.volume_relative_tolerance,
    )
    .expect("edge flip should preserve target volume");
}

#[test]
fn shared_edge_flip_rejects_non_three_tetrahedron_ring() {
    let nodes = two_tetrahedron_bipyramid_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let err = flip_refill_tetrahedra_around_shared_edge(&tetrahedra, &nodes, [0, 1], options)
        .expect_err("two-tetrahedron edge should not be a three-tetrahedron flip ring");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronFlipError::EdgeIncidenceNotThree {
            node_ids: [0, 1],
            incident_tetrahedron_count: 2,
        }
    );
}

#[test]
fn star_refill_candidates_reject_boundary_node_reuse() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let reused = [ConstrainedCavityNode {
        node_id: 0,
        coordinates_m: [0.25, 0.25, 0.25],
    }];

    let err =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &reused, refill_options())
            .expect_err("interior candidate cannot reuse a boundary node");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::InteriorNodeReusesBoundaryNode { node_id: 0 }
    );
}

#[test]
fn validates_closed_tetrahedron_cavity_boundary() {
    let cavity = tetrahedron_cavity();

    let report = validate_constrained_cavity(&cavity).expect("closed cavity should validate");

    assert_eq!(report.boundary_face_count, 4);
    assert_eq!(report.boundary_edge_count, 6);
    assert_eq!(report.boundary_node_count, 4);
    assert_eq!(report.protected_node_count, 2);
    assert_eq!(report.target_volume_m3, 1.0);
}

#[test]
fn rejects_duplicate_boundary_faces() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces[1].node_ids = cavity.boundary_faces[0].node_ids;

    let err =
        validate_constrained_cavity(&cavity).expect_err("duplicate boundary face should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::DuplicateBoundaryFace {
            node_ids: [0, 1, 2]
        }
    );
}

#[test]
fn rejects_open_boundary_edges() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces.pop();

    let err = validate_constrained_cavity(&cavity).expect_err("open boundary should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::TooFewBoundaryFaces {
            boundary_face_count: 3
        }
    );
}

#[test]
fn rejects_protected_nodes_outside_boundary() {
    let mut cavity = tetrahedron_cavity();
    cavity.protected_node_ids.push(99);

    let err = validate_constrained_cavity(&cavity).expect_err("outside protected node should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { node_id: 99 }
    );
}

#[test]
fn rejects_refill_volume_mismatch() {
    let err = validate_constrained_cavity_refill_volume(1.0, 1.2, 1.0e-9)
        .expect_err("volume mismatch should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::InvalidRefillVolume {
            target_volume_m3: 1.0,
            candidate_volume_m3: 1.2,
            tolerance_m3: 1.0e-9
        }
    );
}

#[test]
fn boundary_preservation_rejects_outside_neighbor_loss() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces[0].outside_tetrahedron_ids = vec![99];
    let candidate_faces = cavity
        .boundary_faces
        .iter()
        .cloned()
        .map(|mut face| {
            if sorted_face(face.node_ids) == sorted_face(cavity.boundary_faces[0].node_ids) {
                face.outside_tetrahedron_ids.clear();
            }
            face
        })
        .collect::<Vec<_>>();

    let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect_err("outside neighbor loss should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::BoundaryOutsideTetrahedronMismatch {
            node_ids: sorted_face(cavity.boundary_faces[0].node_ids),
            expected_outside_tetrahedron_ids: vec![99],
            candidate_outside_tetrahedron_ids: Vec::new(),
        }
    );
}

fn tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face([0, 1, 2]),
            face([0, 3, 1]),
            face([1, 3, 2]),
            face([2, 3, 0]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn face(node_ids: [u32; 3]) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: None,
        source_edge_ids: [None, None, None],
        region_ids: Vec::new(),
    }
}

fn provenance_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face_with_provenance(
                [0, 1, 2],
                10,
                [Some(100), Some(101), Some(102)],
                &["loaded", "fixed"],
            ),
            face_with_provenance([0, 3, 1], 11, [Some(103), Some(104), Some(100)], &["fixed"]),
            face_with_provenance([1, 3, 2], 12, [Some(104), Some(105), Some(101)], &["solid"]),
            face_with_provenance([2, 3, 0], 13, [Some(105), Some(103), Some(102)], &["solid"]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn unit_tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: tetrahedron_faces([0, 1, 2, 3])
            .into_iter()
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: vec!["body".to_string()],
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0 / 6.0,
    }
}

fn unit_tetrahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
    ]
}

fn octahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: vec!["body".to_string()],
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 4.0 / 3.0,
    }
}

fn octahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [-1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, -1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

fn unit_cube_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: vec!["body".to_string()],
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    }
}

fn unit_cube_nodes() -> Vec<ConstrainedCavityNode> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(node_id, coordinates_m)| ConstrainedCavityNode {
        node_id: node_id as u32,
        coordinates_m,
    })
    .collect()
}

fn two_tetrahedron_bipyramid_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
            [0, 2, 4],
            [1, 2, 4],
            [0, 1, 4],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: vec!["body".to_string()],
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0 / 3.0,
    }
}

fn two_tetrahedron_bipyramid_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

fn two_tetrahedron_face_flip_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, -1.0],
        },
    ]
}

fn triangular_edge_ring_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, -1.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [-0.5, 0.8660254037844386, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [-0.5, -0.8660254037844386, 0.0],
        },
    ]
}

fn refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn protected_refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        min_protected_node_distance_m: 0.10,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn synthetic_refill_tetrahedron(
    node_ids: [u32; 4],
    volume_m3: f64,
) -> ConstrainedCavityRefillTetrahedron {
    ConstrainedCavityRefillTetrahedron {
        node_ids,
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}

fn face_with_provenance(
    node_ids: [u32; 3],
    source_face_id: u32,
    source_edge_ids: [Option<u32>; 3],
    region_ids: &[&str],
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: Some(source_face_id),
        source_edge_ids,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
    }
}

fn source_edge_for(face: &ConstrainedCavityBoundaryFace, edge: [u32; 2]) -> Option<u32> {
    face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
        .find_map(|(candidate_edge, source_edge_id)| {
            (sorted_edge(candidate_edge) == sorted_edge(edge)).then_some(source_edge_id)
        })
        .flatten()
}

fn candidate_tetrahedron(
    tetrahedron_id: u32,
    node_ids: [u32; 4],
    volume_m3: f64,
    region_ids: &[&str],
) -> CavityTetrahedron {
    CavityTetrahedron {
        tetrahedron_id,
        component_id: 0,
        node_ids,
        source_surface_element_id: 0,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}
