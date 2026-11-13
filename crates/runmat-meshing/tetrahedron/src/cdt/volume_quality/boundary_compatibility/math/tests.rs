use super::*;
fn directional() -> MetricTensor3 {
    MetricTensor3 {
        xx: 4.005,
        yy: 4.005,
        zz: 0.01,
        xy: 3.995,
        xz: 0.,
        yz: 0.,
    }
}
#[test]
fn edge_star_bound_preserves_scale_and_distinguishes_a_regular_metric() {
    let points = [
        [0.625, 0.375, 0.],
        [0.5, 0.5, 0.],
        [0.375, 0.25, 0.],
        [0.53125, 0.375, 0.09375],
    ];
    for scale in [1e-6, 1., 1e6] {
        let points = points.map(|point| point.map(|value| value * scale));
        let upper = edge_star_upper_bound(points, directional()).unwrap();
        assert!(upper > 0.0022 && upper < 0.0023);
        let isotropic = MetricTensor3 {
            xx: 1.,
            yy: 1.,
            zz: 1.,
            xy: 0.,
            xz: 0.,
            yz: 0.,
        };
        assert!(edge_star_upper_bound(points, isotropic).unwrap() > 0.01);
    }
}
#[test]
fn outward_tetrahedron_bound_contains_the_independent_corner_quality() {
    for metric in [
        directional(),
        MetricTensor3 {
            xx: 1.,
            yy: 1.,
            zz: 1.,
            xy: 0.,
            xz: 0.,
            yz: 0.,
        },
    ] {
        for height in [1e-6, 0.1, 1.] {
            let points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0.2, 0.3, height]];
            let actual = crate::cdt::volume_quality::evaluate_tetrahedron_jacobian(points, metric)
                .unwrap()
                .unwrap();
            let upper = tetrahedron_jacobian_upper_bound(points, metric).unwrap();
            assert!(upper >= actual * (1. - 1e-12));
            assert!(upper <= actual * (1. + 1e-6));
        }
    }
}
#[test]
fn uncertain_or_overflowing_arithmetic_cannot_prove_incompatibility() {
    let same_radius = [[0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]];
    assert!(edge_star_upper_bound(same_radius, directional()).is_none());
    let extreme = [
        [f64::MAX, 0., 0.],
        [-f64::MAX, 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ];
    assert!(tetrahedron_jacobian_upper_bound(extreme, directional()).is_none());
    assert!(I::exact(1.).div(I { lo: -1., hi: 1. }).is_none());
}
