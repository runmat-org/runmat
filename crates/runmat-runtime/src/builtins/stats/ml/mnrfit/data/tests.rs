use super::*;

#[test]
fn adjacent_finite_numeric_labels_remain_distinct_classes() {
    let next = f64::from_bits(1.0f64.to_bits() + 1);
    let response = numeric_outcomes(vec![1.0, next, 1.0, next], vec![4, 1], 4)
        .expect("adjacent representable class values are distinct");
    let ClassNames::Numeric(classes) = response.classes else {
        panic!("numeric Y must retain numeric class labels");
    };
    assert_eq!(classes, vec![1.0, next]);
    assert_eq!(response.outcomes, vec![vec![1.0, 0.0, 1.0, 0.0]]);

    let signed_zero = numeric_outcomes(vec![-0.0, 0.0, 1.0], vec![3, 1], 3)
        .expect("signed zeros denote the same numeric class");
    assert_eq!(signed_zero.classes.len(), 2);
    assert_eq!(signed_zero.outcomes, vec![vec![1.0, 1.0, 0.0]]);
}

#[test]
fn grouped_finite_counts_reject_unrepresentable_row_and_total() {
    let overflowing_row = Tensor::new(vec![1.0e308, 1.0e308], vec![1, 2]).unwrap();
    let err = count_outcomes(&overflowing_row).expect_err("row total must stay finite");
    assert!(err.message().contains("grouped count total"));

    let overflowing_observations =
        Tensor::new(vec![1.0e308, 1.0e308, 0.0, 0.0], vec![2, 2]).unwrap();
    let err = count_outcomes(&overflowing_observations)
        .expect_err("aggregate observation count must stay finite");
    assert!(err.message().contains("grouped count total"));

    let omitted_missing_row = Tensor::new(vec![1.0e308, 1.0, f64::NAN, 1.0], vec![2, 2]).unwrap();
    let response = count_outcomes(&omitted_missing_row)
        .expect("a missing row does not contribute an overflowing total");
    assert_eq!(response.weights, vec![0.0, 2.0]);
}

#[test]
fn finite_weight_products_and_totals_must_remain_representable() {
    let x = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0, 0.0]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![2.0, 2.0],
    };
    let options = Options {
        weights: Some(vec![f64::MAX, 1.0]),
        ..Options::default()
    };
    let err = prepare_data(&x, response, &options)
        .expect_err("finite operand multiplication may not silently drop a row");
    assert!(err.message().contains("weighted observation count"));

    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0, 0.0]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![1.0, 1.0],
    };
    let options = Options {
        weights: Some(vec![-1.0, 1.0]),
        ..Options::default()
    };
    let err = prepare_data(&x, response, &options)
        .expect_err("a negative option weight must not be silently omitted");
    assert!(err.message().contains("nonnegative"));

    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0, 0.0]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![1.0e308, 1.0e308],
    };
    let err = prepare_data(&x, response, &Options::default())
        .expect_err("finite weights with an unrepresentable total must reject");
    assert!(err.message().contains("weights total"));
}
