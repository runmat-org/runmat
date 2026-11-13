use runmat_meshing_core::NeverCancelled;
#[test]
fn anisotropic_boundary_offcenter_meets_the_original_target_before_the_old_apex() {
    let face = [[0.625, 0.375, 0.], [0.375, 0.25, 0.], [0.5, 0.5, 0.]];
    let interior = [0.4157904457859701, 0.29967863958536134, 0.08589696964873313];
    let metric = runmat_meshing_size::metric::MetricTensor3 {
        xx: 4.005,
        yy: 4.005,
        zz: 0.01,
        xy: 3.995,
        xz: 0.,
        yz: 0.,
    };
    let points = crate::cdt::volume_sliver::offcenter::points_for_face(
        face,
        interior,
        metric,
        0.01,
        &NeverCancelled,
    )
    .unwrap();
    assert!(
        !points.is_empty(),
        "centroid-normal probes miss this admissible corner offcenter"
    );
    for (point, _) in points {
        assert!(point[2] > 0. && point[2] < interior[2]);
        let quality = crate::cdt::volume_quality::evaluate_tetrahedron_geometry(
            [face[0], face[1], face[2], point],
            metric,
        )
        .unwrap()
        .unwrap();
        assert!(quality.metric_scaled_jacobian >= 0.01);
    }
}
