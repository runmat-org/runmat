use super::*;
#[test]
fn local_tensor_support_is_scale_invariant_and_stays_strictly_inside() {
    let metric = MetricTensor3 {
        xx: 4.005,
        yy: 4.005,
        zz: 0.01,
        xy: 3.995,
        xz: 0.,
        yz: 0.,
    };
    for scale in [1e-6, 1., 1e6] {
        let points = [[0.625, 0.375, 0.], [0.5, 0.5, 0.], [0.375, 0.25, 0.]]
            .map(|point| point.map(|value| value * scale));
        let (ratio, fraction) = proposal(points, metric).unwrap();
        assert!(ratio > 1000. && fraction > 0. && fraction < 0.5);
        let midpoint = [0, 1, 2].map(|axis| 0.5 * (points[0][axis] + points[1][axis]));
        let inserted =
            [0, 1, 2].map(|axis| midpoint[axis] + fraction * (points[2][axis] - midpoint[axis]));
        assert!(proposal([points[0], points[1], inserted], metric).is_none());
    }
}
#[test]
fn regular_and_overflowing_geometry_do_not_create_support_proposals() {
    let metric = MetricTensor3::isotropic_length_m(1.).unwrap();
    assert!(proposal([[0., 0., 0.], [1., 0., 0.], [0.5, 0.8, 0.]], metric).is_none());
    assert!(proposal(
        [[f64::MAX, 0., 0.], [-f64::MAX, 0., 0.], [0., 1., 0.]],
        metric
    )
    .is_none());
}
