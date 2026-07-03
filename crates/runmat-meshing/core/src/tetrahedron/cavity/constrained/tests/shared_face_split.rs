use super::*;

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
