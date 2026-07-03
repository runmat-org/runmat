use super::super::*;

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
