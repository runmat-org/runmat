use super::*;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, StringArray};

#[path = "tests/provider_preflight.rs"]
mod provider_preflight;

#[test]
fn perfcurve_catalog_binding_is_the_only_public_metadata_authority() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("perfcurve")
        .expect("canonical perfcurve catalog entry");
    assert!(runmat_builtins::builtin_function_by_name("perfcurve").is_none());
    assert_eq!(
        entry.descriptor.signatures.len(),
        runmat_builtins::PERFCURVE_DESCRIPTOR.signatures.len()
    );
    assert_eq!(entry.extensions, &runmat_builtins::PERFCURVE_EXTENSIONS);
    assert_eq!(entry.integer_capabilities.len(), 4);
    assert_eq!(entry.bindings.len(), 1);
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("perfcurve");
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
}

#[test]
fn perfcurve_rejects_bad_option_header_before_score_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let score = Tensor::new(vec![0.9, 0.8, 0.4, 0.1], vec![4, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &score).unwrap();
        let err = block_on(perfcurve_builtin(
            tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
            Value::GpuTensor(handle),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("UnknownOption"),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("invalid option must be rejected before score download");
        assert!(err.message().contains("unsupported option 'unknownoption'"));
    });
}

#[test]
fn perfcurve_typed_option_identity_keeps_extension_position() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(perfcurve_builtin(
        tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
        tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("UnknownOption"),
            Value::Num(1.0),
            Value::from("Weights"),
            int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1]), &[4, 1]),
        ]),
    ))
    .expect_err("known later option must retain integer extension gate");
    assert_eq!(
        err.identifier(),
        PERFCURVE_INTEGER_NUMERIC_OPTIONS_EXTENSION.error_identifier
    );
}

#[test]
fn perfcurve_scalar_text_preserves_exact_uint64() {
    assert_eq!(
        scalar_text(&Value::Int(runmat_value::IntValue::U64(u64::MAX)), "option")
            .expect("scalar text"),
        "18446744073709551615"
    );
}

#[test]
fn perfcurve_scalar_bool_rejects_wide_uint64_option() {
    assert!(scalar_bool(&Value::Int(runmat_value::IntValue::U64(u64::MAX)), "option").is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn perfcurve_rank_zero_score_reaches_observation_count_admission() {
    let error = block_on(perfcurve_builtin(
        tensor(&[1.0, 0.0], &[2]),
        tensor(&[0.5], &[]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("one rank-zero score cannot match two labels");
    assert!(error
        .message()
        .contains("labels and scores must contain the same number of observations"));
}

#[test]
fn wide_integer_class_identity_and_exact_score_boundary() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let positive = u64::MAX;
    let negative = u64::MAX - 1;
    let labels = || {
        int_tensor(
            IntegerStorage::U64(vec![positive, negative, positive, negative]),
            &[4, 1],
        )
    };
    let aligned = 1_u64 << 54;
    let result = block_on(perfcurve_builtin(
        labels(),
        int_tensor(
            IntegerStorage::U64(vec![aligned + 8, aligned + 4, aligned, aligned - 4]),
            &[4, 1],
        ),
        Value::Int(runmat_value::IntValue::U64(positive)),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("exactly representable wide scores and adjacent exact class labels");
    match result {
        Value::Tensor(curve) => assert_eq!(curve.shape()[1], 1),
        other => panic!("expected ROC curve, got {other:?}"),
    }
    let error = block_on(perfcurve_builtin(
        labels(),
        int_tensor(
            IntegerStorage::U64(vec![aligned + 1, aligned + 4, aligned, aligned - 4]),
            &[4, 1],
        ),
        Value::Int(runmat_value::IntValue::U64(positive)),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("nonrepresentable wide score must reject");
    assert_eq!(error.identifier(), None);
    assert_eq!(
        error.message(),
        "perfcurve: integer score values must be exactly representable as double"
    );
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn finite_large_priors_normalize_without_overflow() {
    let _outputs = output_count::push_output_count(Some(4)).expect("output-count scope");
    let result = block_on(perfcurve_builtin(
        tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
        tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Prior"),
            tensor(&[1.0e308, 1.0e308], &[1, 2]),
        ]),
    ))
    .expect("finite large prior pair");
    assert!((output_num(&result, 3) - 0.75).abs() < 1.0e-12);
}

#[test]
fn finite_large_observation_weights_have_explicit_empirical_boundary_and_stable_prior() {
    let input = || {
        (
            tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
            tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
            Value::Num(1.0),
        )
    };
    let weights = || vec![Value::from("Weights"), tensor(&[1.0e308; 4], &[4, 1])];
    let (labels, scores, positive) = input();
    let error = block_on(perfcurve_builtin(
        labels,
        scores,
        positive,
        crate::builtins::common::test_support::builtin_arguments(weights()),
    ))
    .expect_err("empirical count total cannot fit in double");
    assert!(error
        .message()
        .contains("weight exceeds finite double range"));

    let _outputs = output_count::push_output_count(Some(4)).expect("output-count scope");
    let (labels, scores, positive) = input();
    let mut controls = weights();
    controls.extend([Value::from("Prior"), Value::from("uniform")]);
    let result = block_on(perfcurve_builtin(
        labels,
        scores,
        positive,
        crate::builtins::common::test_support::builtin_arguments(controls),
    ))
    .expect("normalized prior keeps finite class masses");
    assert!((output_num(&result, 3) - 0.75).abs() < 1.0e-12);
}

#[test]
fn finite_extreme_costs_keep_expected_cost_curve_finite() {
    let _outputs = output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = block_on(perfcurve_builtin(
        tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
        tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("XCrit"),
            Value::from("ecost"),
            Value::from("Cost"),
            tensor(&[1.0e308; 4], &[2, 2]),
        ]),
    ))
    .expect("finite nonnegative cost matrix is admitted");
    for expected_cost in output_tensor(&result, 0).materialize_f64() {
        assert!((expected_cost / 1.0e308 - 1.0).abs() < 1.0e-15);
    }
}

#[test]
fn empirical_and_uniform_priors_select_different_roc_operating_points() {
    let _outputs = output_count::push_output_count(Some(5)).expect("output-count scope");
    let run = |controls| {
        block_on(perfcurve_builtin(
            tensor(&[1.0, 0.0, 0.0, 0.0], &[4, 1]),
            tensor(&[0.8, 0.9, 0.7, 0.1], &[4, 1]),
            Value::Num(1.0),
            controls,
        ))
        .expect("imbalanced ROC curve")
    };
    let empirical = run(crate::builtins::common::test_support::builtin_arguments(
        vec![],
    ));
    let uniform = run(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::from("Prior"), Value::from("uniform")],
    ));
    assert_eq!(
        output_tensor(&empirical, 4).materialize_f64(),
        vec![0.0, 0.0]
    );
    let optimal = output_tensor(&uniform, 4).materialize_f64();
    assert!((optimal[0] - 1.0 / 3.0).abs() < 1.0e-12);
    assert_eq!(optimal[1], 1.0);
}

fn tensor(data: &[f64], shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(data.to_vec(), shape.to_vec()).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: &[usize]) -> Value {
    let tensor = Tensor::new_integer(storage, shape.to_vec()).unwrap();
    Value::Tensor(tensor)
}

fn output_tensor(value: &Value, index: usize) -> &Tensor {
    match value {
        Value::OutputList(values) => match &values[index] {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor at {index}, got {other:?}"),
        },
        other => panic!("expected output list, got {other:?}"),
    }
}

fn output_num(value: &Value, index: usize) -> f64 {
    match value {
        Value::OutputList(values) => match &values[index] {
            Value::Num(value) => *value,
            other => panic!("expected num at {index}, got {other:?}"),
        },
        other => panic!("expected output list, got {other:?}"),
    }
}

#[tokio::test]
async fn default_roc_returns_thresholds_auc_and_optimal_point() {
    let _guard = output_count::push_output_count(Some(5)).expect("output-count scope");
    let result = perfcurve_builtin(
        tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
        tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .await
    .unwrap();
    let x = output_tensor(&result, 0);
    let y = output_tensor(&result, 1);
    let thresholds = output_tensor(&result, 2);
    assert_eq!(x.shape(), &[5, 1]);
    assert_eq!(y.shape(), &[5, 1]);
    assert_eq!(thresholds.shape(), &[5, 1]);
    assert!((output_num(&result, 3) - 0.75).abs() < 1.0e-12);
    assert_eq!(x.materialize_f64()[0], 0.0);
    assert_eq!(y.materialize_f64()[0], 0.0);
    assert_eq!(
        thresholds.materialize_f64()[0],
        thresholds.materialize_f64()[1]
    );
    assert_eq!(x.materialize_f64()[4], 1.0);
    assert_eq!(y.materialize_f64()[4], 1.0);
    let opt = output_tensor(&result, 4);
    assert_eq!(opt.shape(), &[1, 2]);
}

#[tokio::test]
async fn perfcurve_accepts_typed_integer_labels_scores_and_weights() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = output_count::push_output_count(Some(4)).expect("output-count scope");
    let result = perfcurve_builtin(
        int_tensor(IntegerStorage::U8(vec![1, 0, 1, 0]), &[4, 1]),
        int_tensor(IntegerStorage::U16(vec![9, 8, 4, 1]), &[4, 1]),
        Value::Int(runmat_value::IntValue::U8(1)),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Weights".into()),
            int_tensor(IntegerStorage::U8(vec![1, 2, 1, 1]), &[4, 1]),
            Value::String("Prior".into()),
            int_tensor(IntegerStorage::U8(vec![1, 1]), &[1, 2]),
            Value::String("Cost".into()),
            int_tensor(IntegerStorage::U8(vec![0, 1, 1, 0]), &[2, 2]),
        ]),
    )
    .await
    .unwrap();
    assert_eq!(output_tensor(&result, 0).shape(), &[5, 1]);
    assert_eq!(output_tensor(&result, 1).shape(), &[5, 1]);
    assert!(output_num(&result, 3).is_finite());
}

#[tokio::test]
async fn supports_text_labels_negclass_weights_and_precision_recall() {
    let _guard = output_count::push_output_count(Some(7)).expect("output-count scope");
    let labels = Value::StringArray(
        StringArray::new(
            vec![
                "hit".into(),
                "miss".into(),
                "hit".into(),
                "other".into(),
                "miss".into(),
            ],
            vec![5, 1],
        )
        .unwrap(),
    );
    let result = perfcurve_builtin(
        labels,
        tensor(&[0.9, 0.8, 0.4, 0.2, f64::NAN], &[5, 1]),
        Value::String("hit".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("NegClass".into()),
            Value::String("miss".into()),
            Value::String("Weights".into()),
            tensor(&[1.0, 2.0, 1.0, 10.0, 2.0], &[5, 1]),
            Value::String("XCrit".into()),
            Value::String("reca".into()),
            Value::String("YCrit".into()),
            Value::String("prec".into()),
        ]),
    )
    .await
    .unwrap();
    let x = output_tensor(&result, 0);
    let y = output_tensor(&result, 1);
    let suby = output_tensor(&result, 5);
    assert_eq!(x.shape(), &[4, 1]);
    assert_eq!(suby.shape(), &[4, 1]);
    assert!(y
        .materialize_f64()
        .iter()
        .any(|value| (*value - 0.5).abs() < 1.0e-12));
    match &result {
        Value::OutputList(values) => match &values[6] {
            Value::Cell(names) => assert_eq!(
                names.data(),
                vec![Value::CharArray(CharArray::new_row("miss"))]
            ),
            other => panic!("expected cell names, got {other:?}"),
        },
        other => panic!("expected output list, got {other:?}"),
    }
}

#[tokio::test]
async fn xvals_nearest_returns_actual_points_and_interpolation_sorts_targets() {
    let _guard = output_count::push_output_count(Some(4)).expect("output-count scope");
    let result = perfcurve_builtin(
        tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
        tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("XVals".into()),
            tensor(&[0.6, 0.0], &[1, 2]),
        ]),
    )
    .await
    .unwrap();
    let x = output_tensor(&result, 0);
    assert_eq!(x.shape(), &[2, 1]);
    assert_eq!(x.materialize_f64(), vec![0.5, 0.0]);
    assert!(output_num(&result, 3).is_finite());

    let result = perfcurve_builtin(
        tensor(&[1.0, 0.0, 1.0, 0.0], &[4, 1]),
        tensor(&[0.9, 0.8, 0.4, 0.1], &[4, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("XVals".into()),
            tensor(&[0.5, 0.0], &[1, 2]),
            Value::String("UseNearest".into()),
            Value::Bool(false),
        ]),
    )
    .await
    .unwrap();
    assert_eq!(output_tensor(&result, 0).materialize_f64(), vec![0.0, 0.5]);
}

#[tokio::test]
async fn rejects_missing_classes_and_bad_option_shapes() {
    let err = perfcurve_builtin(
        tensor(&[1.0, 1.0], &[2, 1]),
        tensor(&[0.2, 0.3], &[2, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("posclass"));

    let err = perfcurve_builtin(
        tensor(&[1.0, 0.0], &[2, 1]),
        tensor(&[0.2, 0.3], &[2, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Weights".into()),
            tensor(&[1.0], &[1, 1]),
        ]),
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("Weights"));

    let err = perfcurve_builtin(
        tensor(&[1.0, 0.0], &[2, 1]),
        tensor(&[0.2, 0.3], &[2, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("TVals".into()),
            tensor(&[0.2], &[1, 1]),
            Value::String("XVals".into()),
            tensor(&[0.0], &[1, 1]),
        ]),
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("TVals and XVals"));
}

#[tokio::test]
async fn handles_nan_addtofalse_logical_aliases_and_non_roc_optimal_point() {
    let _guard = output_count::push_output_count(Some(5)).expect("output-count scope");
    let result = perfcurve_builtin(
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 1, 0], vec![4, 1]).unwrap()),
        tensor(&[0.9, f64::NAN, f64::NAN, 0.1], &[4, 1]),
        Value::String("true".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("ProcessNaN".into()),
            Value::String("addtofalse".into()),
            Value::String("YCrit".into()),
            Value::String("ecost".into()),
        ]),
    )
    .await
    .unwrap();
    let y = output_tensor(&result, 1);
    assert!(y
        .materialize_f64()
        .iter()
        .any(|value| (*value - 0.5).abs() < 1.0e-12));
    let opt = output_tensor(&result, 4);
    assert!(opt.materialize_f64().iter().all(|value| value.is_nan()));
}
