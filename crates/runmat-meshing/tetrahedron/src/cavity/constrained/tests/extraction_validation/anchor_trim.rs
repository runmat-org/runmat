use super::super::*;

#[test]
fn anchor_trim_removes_non_manifold_dangling_tetrahedron() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["anchor"]),
        candidate_tetrahedron(9, [0, 1, 4, 5], 0.35, &["dangling"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0, 1],
        0,
        vec![0, 1],
    )
    .expect("trim should evaluate")
    .expect("trim should recover the anchor tetrahedron cavity");

    assert_eq!(cavity.removed_tetrahedron_ids, vec![4]);
    assert_eq!(cavity.target_volume_m3, 0.25);
    assert_eq!(cavity.protected_node_ids, vec![0, 1]);
    assert!(cavity
        .boundary_faces
        .iter()
        .all(|face| face.region_ids == ["anchor"]));
    validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
}

#[test]
fn anchor_trim_preserves_requested_anchor() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["left"]),
        candidate_tetrahedron(9, [0, 1, 4, 5], 0.35, &["right"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0, 1],
        1,
        vec![],
    )
    .expect("trim should evaluate")
    .expect("trim should keep the requested anchor tetrahedron");

    assert_eq!(cavity.removed_tetrahedron_ids, vec![9]);
    assert_eq!(cavity.target_volume_m3, 0.35);
    assert!(cavity
        .boundary_faces
        .iter()
        .all(|face| face.region_ids == ["right"]));
    validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
}

#[test]
fn anchor_trim_searches_past_first_defective_edge() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["anchor"]),
        candidate_tetrahedron(9, [0, 1, 2, 4], 0.35, &["trimmed"]),
        candidate_tetrahedron(11, [0, 1, 2, 5], 0.45, &["kept"]),
        candidate_tetrahedron(13, [0, 1, 4, 5], 0.55, &["kept"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0, 1, 2, 3],
        0,
        vec![],
    )
    .expect("trim should evaluate")
    .expect("trim should find an anchor-containing manifold subset");

    assert_eq!(cavity.removed_tetrahedron_ids, vec![4, 11, 13]);
    assert_eq!(cavity.target_volume_m3, 1.25);
    assert!(cavity
        .boundary_faces
        .iter()
        .all(|face| face.region_ids != ["trimmed"]));
    validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
}

#[test]
fn anchor_trim_returns_none_when_anchor_not_selected() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &[]),
        candidate_tetrahedron(9, [0, 1, 4, 5], 0.35, &[]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0],
        1,
        Vec::new(),
    )
    .expect("trim should evaluate");

    assert!(cavity.is_none());
}
