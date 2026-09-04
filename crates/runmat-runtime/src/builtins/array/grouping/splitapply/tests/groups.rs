use super::*;

#[test]
fn sums_column_oriented_groups_in_numeric_order() {
    let output = call(
        "sum",
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]),
        vec![tensor(vec![2.0, 1.0, 2.0, 1.0], vec![4, 1])],
    )
    .unwrap();
    assert_eq!(numeric(output), vec![6.0, 4.0]);
}

#[test]
fn omits_nan_group_observations_and_rejects_gaps() {
    let output = call(
        "sum",
        tensor(vec![10.0, 20.0, 30.0], vec![3, 1]),
        vec![tensor(vec![1.0, f64::NAN, 2.0], vec![3, 1])],
    )
    .unwrap();
    assert_eq!(numeric(output), vec![10.0, 30.0]);

    let error = call(
        "sum",
        tensor(vec![10.0, 20.0], vec![2, 1]),
        vec![tensor(vec![1.0, 3.0], vec![2, 1])],
    )
    .expect_err("group-number gaps must reject");
    assert!(error.message.contains("cannot have gaps"));
}

#[test]
fn invalid_group_values_and_observation_counts_reject() {
    for value in [0.0, -1.0, 1.5, f64::INFINITY] {
        let error = call("sum", Value::Num(7.0), vec![Value::Num(value)]).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:splitapply:InvalidInput"));
    }
    let error = call(
        "sum",
        tensor(vec![1.0, 2.0], vec![2, 1]),
        vec![tensor(vec![1.0, 1.0, 1.0], vec![3, 1])],
    )
    .unwrap_err();
    assert!(error.message.contains("data has 2 observations"));
}

#[test]
fn row_group_vector_splits_matrix_columns() {
    let output = call(
        "sum",
        tensor(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], vec![2, 4]),
        vec![tensor(vec![1.0, 2.0, 1.0, 2.0], vec![1, 4])],
    )
    .unwrap();
    assert_eq!(numeric(output), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn row_group_vector_preserves_trailing_dimensions() {
    let output = call(
        "sum",
        tensor(
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            vec![2, 2, 2],
        ),
        vec![tensor(vec![1.0, 2.0], vec![1, 2])],
    )
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    assert_eq!(output.shape, vec![2, 1, 2]);
    assert_eq!(output.materialize_f64(), vec![11.0, 22.0, 33.0, 44.0]);
}
