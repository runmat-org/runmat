use super::*;
use runmat_value::IntegerStorage;

#[test]
fn production_evaluator_retains_wide_integer_center_geometry() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
        vec![1, 2],
    )
    .unwrap();
    let args = [runmat_value::Value::Tensor(tensor.clone())];
    let evaluated = futures::executor::block_on(super::super::evaluate_async(
        runmat_value::Value::Tensor(tensor),
        crate::call::FunctionArgumentView::from_slice(&args),
    ))
    .unwrap();
    assert_eq!(evaluated.counts.materialize_f64(), vec![1., 1.]);
    assert_eq!(evaluated.counts.shape(), [1, 2]);
    assert_eq!(
        evaluated.centers.materialize_f64(),
        vec![(u64::MAX - 1) as f64, u64::MAX as f64]
    );
}

#[test]
fn every_integer_class_keeps_adjacent_centers_distinct_at_its_extremes() {
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, i8::MIN + 1]),
        IntegerStorage::I16(vec![i16::MIN, i16::MIN + 1]),
        IntegerStorage::I32(vec![i32::MIN, i32::MIN + 1]),
        IntegerStorage::I64(vec![i64::MIN, i64::MIN + 1]),
        IntegerStorage::U8(vec![u8::MAX - 1, u8::MAX]),
        IntegerStorage::U16(vec![u16::MAX - 1, u16::MAX]),
        IntegerStorage::U32(vec![u32::MAX - 1, u32::MAX]),
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
    ] {
        let bins = IntegerCenterBins::new(&storage).unwrap();
        let tensor = Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap();
        let result = bins
            .evaluate(&tensor, HistNormalization::Count, None, 2.)
            .unwrap();
        assert_eq!(result.counts, vec![1., 1.], "{}", storage.class_name());
        assert_eq!(result.output_shape, vec![1, 2]);
        assert_eq!(bins.realize().widths, vec![1., 1.]);
        assert_eq!(result.centers, storage.to_f64_vec());
    }
}

#[test]
fn exact_midpoint_ties_go_right_and_outer_bins_still_absorb_extremes() {
    let origin = i64::MAX - 5;
    let centers = IntegerStorage::I64(vec![origin, origin + 2, origin + 4]);
    let bins = IntegerCenterBins::new(&centers).unwrap();
    for (value, expected) in [
        (i64::MIN as i128, 0),
        (origin as i128, 0),
        (origin as i128 + 1, 1),
        (origin as i128 + 3, 2),
        (u64::MAX as i128, 2),
    ] {
        assert_eq!(bins.integer_bin(value), expected);
    }
    assert_eq!(bins.realize().widths, vec![2.; 3]);
    assert_eq!(half_label(-1), "-0.5");
    assert_eq!(half_label(-3), "-1.5");
    assert_eq!(half_label(3), "1.5");
}

#[test]
fn floating_samples_use_the_same_exact_midpoint_without_rounding_its_center() {
    let bins = IntegerCenterBins::new(&IntegerStorage::U64(vec![
        9_007_199_254_740_993,
        9_007_199_254_740_994,
    ]))
    .unwrap();
    assert_eq!(bins.floating_bin(9_007_199_254_740_992.), 0);
    assert_eq!(bins.floating_bin(9_007_199_254_740_994.), 1);
    let bins = IntegerCenterBins::new(&IntegerStorage::I8(vec![0, 2])).unwrap();
    for (value, expected) in [
        (f64::NEG_INFINITY, 0),
        (0., 0),
        (0.5, 0),
        (1., 1),
        (f64::INFINITY, 1),
        (f64::NAN, 1),
    ] {
        assert_eq!(bins.floating_bin(value), expected);
    }
}

#[test]
fn exact_geometry_preserves_weighted_matrix_columns_and_pdf_widths() {
    let storage = IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]);
    let bins = IntegerCenterBins::new(&storage).unwrap();
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]),
        vec![2, 2],
    )
    .unwrap();
    let result = bins
        .evaluate(
            &tensor,
            HistNormalization::Count,
            Some(&[1., 2., 3., 4.]),
            10.,
        )
        .unwrap();
    assert_eq!(result.output_shape, vec![2, 2]);
    assert_eq!(result.counts, vec![1., 2., 0., 7.]);
    assert_eq!(result.charts.len(), 2);
    let result = bins
        .evaluate(
            &tensor,
            HistNormalization::Pdf,
            Some(&[1., 2., 3., 4.]),
            10.,
        )
        .unwrap();
    assert_eq!(result.counts, vec![1. / 3., 2. / 3., 0., 1.]);
}

#[test]
fn integer_center_validation_is_exact_and_floating_midpoints_do_not_overflow() {
    for values in [vec![u64::MAX, u64::MAX], vec![u64::MAX, u64::MAX - 1]] {
        assert!(IntegerCenterBins::new(&IntegerStorage::U64(values)).is_err());
    }
    assert!(IntegerCenterBins::new(&IntegerStorage::I64(vec![0, 2, 5])).is_err());
    assert_eq!(super::super::stable_midpoint(1e308, 1.5e308), 1.25e308);
    assert_eq!(super::super::stable_midpoint(-1e308, 1e308), 0.);
    assert_eq!(
        super::super::stable_midpoint(f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        f64::MIN_POSITIVE
    );
    // An arithmetic half-difference or separate half-sums round too early here.
    let tiny = f64::from_bits(1);
    assert_eq!(super::super::stable_midpoint(tiny, 2. * tiny), 2. * tiny);
    assert_eq!(super::super::stable_midpoint(-tiny, 2. * tiny), 0.);
}
