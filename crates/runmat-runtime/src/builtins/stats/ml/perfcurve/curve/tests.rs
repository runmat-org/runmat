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

#[test]
fn expected_cost_keeps_finite_extreme_weighted_means() {
    let balanced = Counts {
        tp: 1.0,
        fp: 1.0,
        tn: 1.0,
        fn_: 1.0,
    };
    let large = [[1.0e308; 2]; 2];
    let result = core::criterion_value(balanced, Criterion::ExpectedCost, &large);
    assert!((result / 1.0e308 - 1.0).abs() < 1.0e-15);

    let unequal = Counts {
        tp: 0.0,
        fp: 1.0,
        tn: 3.0,
        fn_: 0.0,
    };
    let near_maximum = [[f64::MAX, f64::MAX / 2.0], [0.0, 0.0]];
    let expected = f64::MAX * (7.0 / 8.0);
    let result = core::criterion_value(unequal, Criterion::ExpectedCost, &near_maximum);
    assert!((result / expected - 1.0).abs() < 1.0e-15);

    // A tiny count paired with the largest cost must not erase a smaller,
    // dominant cost in another occupied cell.
    let mixed = Counts {
        tp: 0.0,
        fp: 1.0,
        tn: f64::from_bits(1),
        fn_: 0.0,
    };
    let mixed_cost = [[1.0e308, 1.0e-308], [0.0, 0.0]];
    let expected = 1.0e308 * f64::from_bits(1) + 1.0e-308;
    let result = core::criterion_value(mixed, Criterion::ExpectedCost, &mixed_cost);
    assert!((result / expected - 1.0).abs() < 1.0e-14);

    // These four positive masses have a finite public total, but adding the
    // largest three first overflows binary64. Their exact proportions are
    // approximately 3/4 TN, 1/8 FP, 1/8 TP and a negligible FN. Independent
    // cost arithmetic gives 2*(3/4) + 8*(1/8) = 2.5.
    let reordered_overflow = Counts {
        tp: 2.2471164185778941e307,
        fp: 2.2471164185778941e307,
        tn: 1.3482698511467367e308,
        fn_: 9.9792015476736e291,
    };
    assert!(reordered_overflow.total().is_finite());
    let costs = [[2.0, 8.0], [0.0, 0.0]];
    let result = core::criterion_value(reordered_overflow, Criterion::ExpectedCost, &costs);
    assert!((result - 2.5).abs() < 1.0e-14, "{result}");
}

#[test]
fn extreme_roc_cost_ratio_preserves_finite_slope_and_negative_difference_policy() {
    let mut options = Options {
        xcrit: Criterion::Fpr,
        ycrit: Criterion::Tpr,
        neg_class: None,
        weights: None,
        prior: PriorOption::Empirical,
        cost: [[0.0, 1.0e308], [5.0e307, 0.0]],
        tvals: None,
        xvals: None,
        use_nearest: true,
        process_nan: ProcessNaN::Ignore,
    };
    let mut curve = Curve {
        x: vec![0.0, 0.5],
        y: vec![0.0, 0.5],
        thresholds: vec![1.0, 0.5],
        counts: vec![
            Counts {
                tp: 1.0,
                fp: 1.0,
                tn: 1.0,
                fn_: 1.0,
            },
            Counts {
                tp: 0.0,
                fp: 0.0,
                tn: 2.0,
                fn_: 2.0,
            },
        ],
        auc: 0.0,
    };
    assert!((interpolation::roc_slope(&curve, &options) - 2.0).abs() < 1.0e-15);
    assert_eq!(optimal_point(&curve, &options), (0.0, 0.0));

    options.cost = [[0.0, 1.0e-308], [1.0e-308, 0.0]];
    curve.counts[0] = Counts {
        tp: 0.0,
        fp: 0.0,
        tn: 1.0e-308,
        fn_: 1.0e-308,
    };
    assert!((interpolation::roc_slope(&curve, &options) - 1.0).abs() < 1.0e-15);
    options.cost = [[2.0, 1.0], [1.0, 2.0]];
    assert!(interpolation::roc_slope(&curve, &options).is_nan());
}
