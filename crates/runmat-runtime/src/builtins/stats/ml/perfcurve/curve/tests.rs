use super::*;

#[test]
fn distinct_small_abscissae_interpolate_and_bound_auc_exactly() {
    let (_, y, _, _) = interpolation::interpolate_from_xy(&[0.0, 1.0e-16], &[0.0, 1.0], 5.0e-17);
    assert!((y - 0.5).abs() < 1.0e-12);
    let area = auc_interval(&[0.0, 1.0e-16], &[0.0, 1.0], &[5.0e-17, 1.0e-16]);
    assert!((area - 3.75e-17).abs() < 1.0e-28);
}

#[test]
fn optimal_roc_slope_uses_effective_empirical_class_mass() {
    let options = Options {
        xcrit: Criterion::Fpr,
        ycrit: Criterion::Tpr,
        neg_class: None,
        weights: None,
        prior: PriorOption::Empirical,
        cost: [[0.0, 1.0], [1.0, 0.0]],
        tvals: None,
        xvals: None,
        use_nearest: true,
        process_nan: ProcessNaN::Ignore,
    };
    let mut curve = Curve {
        x: vec![0.0],
        y: vec![0.0],
        thresholds: vec![1.0],
        counts: vec![Counts {
            tp: 0.0,
            fp: 0.0,
            tn: 3.0,
            fn_: 1.0,
        }],
        auc: 0.0,
    };
    assert_eq!(interpolation::roc_slope(&curve, &options), 3.0);
    curve.counts[0] = Counts {
        tp: 0.0,
        fp: 0.0,
        tn: 0.5,
        fn_: 0.5,
    };
    assert_eq!(interpolation::roc_slope(&curve, &options), 1.0);
}
