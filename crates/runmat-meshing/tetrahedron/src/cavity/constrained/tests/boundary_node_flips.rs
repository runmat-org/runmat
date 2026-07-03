use super::*;

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
