use super::super::*;

#[test]
fn boundary_face_split_preserves_source_face_regions_and_perimeter_edges() {
    let face = face_with_provenance(
        [0, 1, 2],
        10,
        [Some(100), Some(101), Some(102)],
        &["fixed", "loaded"],
    );

    let children = split_constrained_cavity_boundary_face(&face, 9).expect("face should split");

    assert_eq!(children.len(), 3);
    assert_eq!(children[0].node_ids, [0, 1, 9]);
    assert_eq!(children[1].node_ids, [1, 2, 9]);
    assert_eq!(children[2].node_ids, [2, 0, 9]);
    for child in &children {
        assert_eq!(child.source_face_id, Some(10));
        assert_eq!(
            sorted_region_ids(&child.region_ids),
            vec!["fixed".to_string(), "loaded".to_string()]
        );
    }
    assert_eq!(children[0].source_edge_ids, [Some(100), None, None]);
    assert_eq!(children[1].source_edge_ids, [Some(101), None, None]);
    assert_eq!(children[2].source_edge_ids, [Some(102), None, None]);
}

#[test]
fn boundary_face_list_split_replaces_only_target_face() {
    let cavity = provenance_cavity();

    let split_faces = split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [2, 1, 0], 9)
        .expect("target face should split");

    assert_eq!(split_faces.len(), cavity.boundary_faces.len() + 2);
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        split_faces
            .iter()
            .filter(|face| face.node_ids.contains(&9))
            .count(),
        3
    );
    for untouched in cavity.boundary_faces.iter().skip(1) {
        assert!(split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == sorted_face(untouched.node_ids)));
    }
}

#[test]
fn boundary_face_split_rejects_reused_or_missing_split_targets() {
    let cavity = provenance_cavity();
    let face = &cavity.boundary_faces[0];

    let reused = split_constrained_cavity_boundary_face(face, face.node_ids[0])
        .expect_err("split node cannot reuse an existing face node");
    assert_eq!(
        reused,
        ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
            node_id: face.node_ids[0]
        }
    );

    let missing = split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [10, 11, 12], 9)
        .expect_err("missing target face should fail");
    assert_eq!(
        missing,
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: [10, 11, 12]
        }
    );
}
