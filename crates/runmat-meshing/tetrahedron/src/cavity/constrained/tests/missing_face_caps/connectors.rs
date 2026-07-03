use super::super::*;

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
