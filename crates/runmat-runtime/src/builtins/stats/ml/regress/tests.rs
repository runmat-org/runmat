use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

mod provider_preflight;

#[test]
fn rank_one_response_is_admitted_as_a_vector() {
    let y = Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0], vec![3]).unwrap());
    let x = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let output = block_on(regress_builtin(
        y,
        x,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::Tensor(beta) = output else {
        panic!("expected numeric coefficient vector")
    };
    assert_eq!(beta.shape(), [2, 1]);
    assert_close(beta.materialize_f64()[0], 1.0);
    assert_close(beta.materialize_f64()[1], 2.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rank_zero_design_preserves_cached_matrix_geometry_at_entry() {
    let y = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
    let design = Tensor::new(vec![2.0], Vec::new()).unwrap();
    assert_eq!(regress_matrix_dimensions(design.shape()), (0, 0));
    let x = Value::Tensor(design);
    let error = block_on(regress_builtin(
        y,
        x,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("rank-zero design has no observation row");
    assert!(error
        .message()
        .contains("length(y) must match the number of rows in X"));
}

#[test]
fn empty_observations_reject_before_a_fit() {
    let y = tensor(Vec::new(), 0, 1);
    let x = tensor(Vec::new(), 0, 2);
    let error = block_on(regress_builtin(
        y,
        x,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(error
        .message()
        .contains("at least one complete observation"));
}

#[test]
fn integer_alpha_compatibility_and_exactness_precede_alpha_range() {
    let y = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let x = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let one = Value::Tensor(Tensor::new_integer(IntegerStorage::I64(vec![1]), vec![1, 1]).unwrap());
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(regress_builtin(
            y.clone(),
            x.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![one]),
        ))
        .expect_err("typed alpha is a RunMat-only extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:RegressIntegerAlphaExtension")
        );
    }
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 54) + 1]), vec![1, 1]).unwrap(),
    );
    let error = block_on(regress_builtin(
        y,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![wide]),
    ))
    .expect_err("wide integer alpha must cross an exact boundary first");
    assert!(error.message().contains("exactly representable"));
}

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize, _poison: f64) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn outputs(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

fn tensor_ref(value: &Value) -> &Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn assert_close(left: f64, right: f64) {
    assert_close_tol(left, right, 1.0e-10);
}

fn assert_close_tol(left: f64, right: f64, tol: f64) {
    assert!(
        (left - right).abs() < tol,
        "{left:?} not close to {right:?}"
    );
}

#[test]
fn regress_fits_coefficients_and_requested_outputs() {
    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let x = tensor(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0], 4, 2);
    let out = outputs(
        block_on(regress_builtin(
            y,
            x,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    let b = tensor_ref(&out[0]);
    assert_eq!(b.shape(), [2, 1]);
    assert_close(b.materialize_f64()[0], 1.0);
    assert_close(b.materialize_f64()[1], 2.0);
    assert_eq!(tensor_ref(&out[1]).shape(), [2, 2]);
    assert_eq!(tensor_ref(&out[2]).shape(), [4, 1]);
    assert_eq!(tensor_ref(&out[3]).shape(), [4, 2]);
    let stats = tensor_ref(&out[4]);
    assert_eq!(stats.shape(), [1, 4]);
    assert_close(stats.materialize_f64()[0], 1.0);
    assert_close(stats.materialize_f64()[3], 0.0);
}

#[test]
fn regress_accepts_typed_integer_design_and_response() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), 4, 1, f64::NAN);
    let x = poisoned_int_tensor(
        IntegerStorage::I16(vec![1, 1, 1, 1, 0, 1, 2, 3]),
        4,
        2,
        f64::NAN,
    );
    let out = outputs(
        block_on(regress_builtin(
            y,
            x,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    let b = tensor_ref(&out[0]);
    assert_eq!(b.shape(), [2, 1]);
    assert_close(b.materialize_f64()[0], 1.0);
    assert_close(b.materialize_f64()[1], 2.0);
}

#[test]
fn regress_rejects_typed_integer_alpha_boundaries() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), 4, 1, 0.0);
    let x = poisoned_int_tensor(IntegerStorage::I16(vec![1, 1, 1, 1, 0, 1, 2, 3]), 4, 2, 0.0);
    let alpha = poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1, 0.5);
    let err = block_on(regress_builtin(
        y,
        x,
        crate::builtins::common::test_support::builtin_arguments(vec![alpha]),
    ))
    .unwrap_err();
    assert!(err.message.contains("alpha must be between 0 and 1"));
}

#[test]
fn regress_omits_nan_rows_and_uses_alpha() {
    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let y = tensor(vec![1.0, f64::NAN, 5.0, 7.0], 4, 1);
    let x = tensor(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0], 4, 2);
    let out = outputs(
        block_on(regress_builtin(
            y,
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.1)]),
        ))
        .unwrap(),
    );
    let b = tensor_ref(&out[0]);
    assert_close(b.materialize_f64()[0], 1.0);
    assert_close(b.materialize_f64()[1], 2.0);
    assert_eq!(tensor_ref(&out[2]).shape(), [3, 1]);
    assert_eq!(tensor_ref(&out[3]).shape(), [3, 2]);
}

#[test]
fn regress_supports_partial_outputs() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let x = tensor(vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0], 4, 2);
    let out = outputs(
        block_on(regress_builtin(
            y,
            x,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(out.len(), 3);
    assert_eq!(tensor_ref(&out[0]).shape(), [2, 1]);
    assert_eq!(tensor_ref(&out[1]).shape(), [2, 2]);
    assert_eq!(tensor_ref(&out[2]).shape(), [4, 1]);
}

#[test]
fn regress_reports_nonzero_intervals_and_stats() {
    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let y = tensor(vec![1.0, 2.1, 2.9, 4.2, 4.8, 6.1], 6, 1);
    let x = tensor(
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // intercept
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, // predictor
        ],
        6,
        2,
    );
    let out = outputs(
        block_on(regress_builtin(
            y,
            x,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );

    let b = tensor_ref(&out[0]);
    assert_close_tol(b.materialize_f64()[0], 1.023809523809523, 1.0e-12);
    assert_close_tol(b.materialize_f64()[1], 0.9971428571428576, 1.0e-12);

    let bint = tensor_ref(&out[1]);
    let expected_bint = [
        0.6933332335194298,
        0.8879901308900255,
        1.3542858140996163,
        1.1062955833956896,
    ];
    for (actual, expected) in bint.materialize_f64().iter().zip(expected_bint) {
        assert_close_tol(*actual, expected, 1.0e-9);
    }

    let rint = tensor_ref(&out[3]);
    let expected_rint = [
        -0.43855599446509025,
        -0.40707908392230313,
        -0.6201647380283666,
        -0.24408652319230167,
        -0.5366307280868483,
        -0.2920062540275898,
        0.39093694684604413,
        0.565174322017542,
        0.38397426183788963,
        0.6136103327161107,
        0.1118688233249418,
        0.4729586349799683,
    ];
    for (actual, expected) in rint.materialize_f64().iter().zip(expected_rint) {
        assert_close_tol(*actual, expected, 1.0e-9);
    }

    let stats = tensor_ref(&out[4]);
    let expected_stats = [
        0.9938206296321479,
        643.3151408450694,
        1.4348829360466553e-05,
        0.027047619047619084,
    ];
    for (actual, expected) in stats.materialize_f64().iter().zip(expected_stats) {
        assert_close_tol(*actual, expected, 1.0e-9);
    }
}

#[test]
fn regress_zeroes_dependent_columns_and_intervals() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let x = tensor(
        vec![
            1.0, 1.0, 1.0, 1.0, // intercept
            0.0, 1.0, 2.0, 3.0, // independent slope
            0.0, 2.0, 4.0, 6.0, // dependent on slope
        ],
        4,
        3,
    );
    let out = outputs(
        block_on(regress_builtin(
            y,
            x,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    let b = tensor_ref(&out[0]);
    assert_close(b.materialize_f64()[0], 1.0);
    assert_close(b.materialize_f64()[1], 0.0);
    assert_close(b.materialize_f64()[2], 1.0);
    let bint = tensor_ref(&out[1]);
    assert_close(bint.materialize_f64()[1], 0.0);
    assert_close(bint.materialize_f64()[4], 0.0);
}

#[test]
fn regress_budgets_only_requested_outputs() {
    ensure_output_budget(MAX_REGRESS_CELLS, 1, 2).unwrap();
    assert!(ensure_output_budget(MAX_REGRESS_CELLS, 1, 3)
        .unwrap_err()
        .message
        .contains("output is too large"));
}

#[test]
fn regress_stats_preserve_negative_model_sum() {
    let stats = stats_values(10.0, 5.0, 2.0, 6, 2);
    assert_close(stats[0], -1.0);
    assert_close(stats[1], -2.5);
    assert!(stats[2].is_nan());
    assert_close(stats[3], 2.0);
}

#[test]
fn regress_resource_guards_cover_input_and_diagnostics() {
    assert!(ensure_input_budget(MAX_REGRESS_CELLS + 1, 1)
        .unwrap_err()
        .message
        .contains("input is too large"));
    assert!(ensure_diagnostic_budget(1_000_000, 2)
        .unwrap_err()
        .message
        .contains("diagnostic work array"));
}

#[test]
fn regress_rejects_bad_dimensions_and_alpha() {
    let err = block_on(regress_builtin(
        tensor(vec![1.0, 2.0], 2, 1),
        tensor(vec![1.0, 1.0, 1.0], 3, 1),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("length(y)"));

    let err = block_on(regress_builtin(
        tensor(vec![1.0, 2.0], 2, 1),
        tensor(vec![1.0, 1.0], 2, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .unwrap_err();
    assert!(err.message.contains("alpha"));
}

#[test]
fn discarded_calls_admit_without_fitting_but_still_validate_alpha() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    for (response, design) in [
        (tensor(Vec::new(), 0, 1), tensor(Vec::new(), 0, 2)),
        (
            tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
            tensor(vec![1.0, 1.0, 1.0], 3, 1),
        ),
    ] {
        let result = block_on(regress_builtin(
            response.clone(),
            design.clone(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("discarded call does not fit");
        assert!(matches!(result, Value::OutputList(values) if values.is_empty()));
        let error = block_on(regress_builtin(
            response,
            design,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect_err("discarded call still admits alpha");
        assert!(error.message().contains("alpha"));
    }
}
