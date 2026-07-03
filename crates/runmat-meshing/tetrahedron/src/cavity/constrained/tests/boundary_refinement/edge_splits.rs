use super::super::*;

#[test]
fn boundary_face_edge_split_preserves_source_face_regions_and_split_edge_provenance() {
    let face = face_with_provenance(
        [0, 1, 2],
        10,
        [Some(100), Some(101), Some(102)],
        &["fixed", "loaded"],
    );

    let children = split_constrained_cavity_boundary_face_on_edge(&face, [0, 1], 9)
        .expect("face edge should split");

    assert_eq!(children[0].node_ids, [0, 9, 2]);
    assert_eq!(children[1].node_ids, [9, 1, 2]);
    assert_eq!(children[0].source_edge_ids, [Some(100), None, Some(102)]);
    assert_eq!(children[1].source_edge_ids, [Some(100), Some(101), None]);
    for child in &children {
        assert_eq!(child.source_face_id, Some(10));
        assert_eq!(
            sorted_region_ids(&child.region_ids),
            vec!["fixed".to_string(), "loaded".to_string()]
        );
    }
}

#[test]
fn boundary_face_edge_split_list_replaces_conforming_edge_pair() {
    let cavity = provenance_cavity();

    let split_faces = split_constrained_cavity_boundary_faces_on_edge(
        &cavity.boundary_faces,
        [2, 1, 0],
        [1, 0],
        9,
    )
    .expect("target face edge should split");

    assert_eq!(split_faces.len(), cavity.boundary_faces.len() + 2);
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
    assert_eq!(
        split_faces
            .iter()
            .filter(|face| face.node_ids.contains(&9))
            .count(),
        4
    );
    for untouched in cavity.boundary_faces.iter().skip(2) {
        assert!(split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == sorted_face(untouched.node_ids)));
    }
}

#[test]
fn boundary_face_three_edge_split_refines_target_and_conforming_neighbors() {
    let cavity = provenance_cavity();
    let split_faces = split_constrained_cavity_boundary_faces_on_three_edges(
        &cavity.boundary_faces,
        [2, 1, 0],
        BTreeMap::from([([0, 1], 9), ([1, 2], 10), ([0, 2], 11)]),
    )
    .expect("target face edges should split");

    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
    assert_eq!(
        split_faces
            .iter()
            .filter(|face| face.node_ids.contains(&9)
                || face.node_ids.contains(&10)
                || face.node_ids.contains(&11))
            .count(),
        10
    );
    let target_children = split_faces
        .iter()
        .filter(|face| {
            [9, 10, 11]
                .into_iter()
                .any(|node_id| face.node_ids.contains(&node_id))
                && face.source_face_id == Some(10)
        })
        .collect::<Vec<_>>();
    assert_eq!(target_children.len(), 4);
    assert_eq!(source_edge_for(target_children[0], [0, 9]), Some(100));
    assert_eq!(source_edge_for(target_children[1], [1, 9]), Some(100));
    assert_eq!(source_edge_for(target_children[1], [1, 10]), Some(101));
    assert_eq!(source_edge_for(target_children[2], [2, 10]), Some(101));
    assert_eq!(source_edge_for(target_children[2], [2, 11]), Some(102));
    assert_eq!(source_edge_for(target_children[0], [0, 11]), Some(102));
}
