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

    let candidates = [cap_tetrahedron, mate_tetrahedron];
    assert_eq!(
        cap_side_face_mate_counts(
            std::slice::from_ref(&candidates[0]),
            &candidates,
            &BTreeSet::from([4])
        ),
        vec![1, 0, 0]
    );
}
