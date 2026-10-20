use super::*;

#[test]
fn timeit_external_function_handle_surfaces_undefined_function() {
    let err = block_on(timeit_builtin(
        Some(runtime()),
        Value::ExternalFunctionHandle("pkg.missing_callback".into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("unresolved external callback should fail");
    assert_eq!(err.identifier(), Some("RunMat:UndefinedFunction"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn timeit_measures_time() {
    let runtime = runtime();
    let _callbacks = install_test_callbacks(&runtime);
    COUNTER_DEFAULT.store(0, Ordering::SeqCst);
    let result = block_on(timeit_builtin(
        Some(runtime),
        default_handle(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("timeit");
    match result {
        Value::Num(v) => assert!(v >= 0.0),
        other => panic!("expected numeric result, got {other:?}"),
    }
    assert!(
        COUNTER_DEFAULT.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
        "expected at least {} invocations",
        MIN_SAMPLE_COUNT
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn timeit_accepts_num_outputs_argument() {
    let _counter_guard = COUNTER_NUM_OUTPUTS_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = runtime();
    runtime.set_runmat_extensions_enabled(true);
    let _callbacks = install_test_callbacks(&runtime);
    COUNTER_NUM_OUTPUTS.store(0, Ordering::SeqCst);
    let args = vec![Value::Int(IntValue::I32(3))];
    let _ = block_on(timeit_builtin(
        Some(runtime),
        outputs_handle(),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("timeit numOutputs");
    assert!(
        COUNTER_NUM_OUTPUTS.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
        "expected at least {} invocations",
        MIN_SAMPLE_COUNT
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn timeit_supports_zero_outputs() {
    let runtime = runtime();
    runtime.set_runmat_extensions_enabled(true);
    let _callbacks = install_test_callbacks(&runtime);
    COUNTER_ZERO_OUTPUTS.store(0, Ordering::SeqCst);
    let args = vec![Value::Int(IntValue::I32(0))];
    let _ = block_on(timeit_builtin(
        Some(runtime),
        zero_outputs_handle(),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("timeit zero outputs");
    assert!(
        COUNTER_ZERO_OUTPUTS.load(Ordering::SeqCst) >= MIN_SAMPLE_COUNT,
        "expected at least {} invocations",
        MIN_SAMPLE_COUNT
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn timeit_runs_with_wgpu_provider_registered() {
    let runtime = runtime();
    let _callbacks = install_test_callbacks(&runtime);
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let result = block_on(timeit_builtin(
        Some(runtime),
        default_handle(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("timeit with wgpu");
    match result {
        Value::Num(v) => assert!(v >= 0.0),
        other => panic!("expected numeric result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn timeit_rejects_non_function_input() {
    let err = block_on(timeit_builtin(
        Some(runtime()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_timeit_error_contains(&err, "function");
    assert_timeit_error_identifier(&err, TIMEIT_ERROR_FIRST_ARG_KIND.identifier.unwrap());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn timeit_rejects_invalid_num_outputs() {
    let runtime = runtime();
    let _callbacks = install_test_callbacks(&runtime);
    COUNTER_INVALID.store(0, Ordering::SeqCst);
    let err = block_on(timeit_builtin(
        Some(runtime),
        invalid_handle(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-1.0)]),
    ))
    .unwrap_err();
    assert_timeit_error_contains(&err, "nonnegative");
    assert_timeit_error_identifier(&err, TIMEIT_ERROR_NUM_OUTPUTS_NONNEG.identifier.unwrap());
    assert_eq!(COUNTER_INVALID.load(Ordering::SeqCst), 0);
}

#[test]
fn timeit_gates_typed_integer_output_counts_before_invoking_the_callback() {
    let _counter_guard = COUNTER_NUM_OUTPUTS_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = runtime();
    runtime.set_runmat_extensions_enabled(false);
    let _callbacks = install_test_callbacks(&runtime);
    COUNTER_NUM_OUTPUTS.store(0, Ordering::SeqCst);
    let err = block_on(timeit_builtin(
        Some(runtime),
        outputs_handle(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            3,
        ))]),
    ))
    .expect_err("strict compatibility must reject native integer output counts");
    assert_eq!(
        err.identifier(),
        INTEGER_NUM_OUTPUTS_EXTENSION.error_identifier
    );
    assert_eq!(COUNTER_NUM_OUTPUTS.load(Ordering::SeqCst), 0);
}

#[test]
fn timeit_decodes_typed_integer_output_counts_without_an_f64_boundary() {
    let runtime = runtime();
    runtime.set_runmat_extensions_enabled(true);
    assert_eq!(
        block_on(parse_num_outputs(
            &runtime,
            &Value::FunctionHandle("fixture".into()),
            crate::call::FunctionArgumentView::from_slice(&[Value::Int(IntValue::U64(
                usize::MAX as u64
            ))])
        ))
        .expect("platform-sized integer should decode exactly"),
        Some(usize::MAX)
    );
    assert!(block_on(parse_num_outputs(
        &runtime,
        &Value::FunctionHandle("fixture".into()),
        crate::call::FunctionArgumentView::from_slice(&[Value::Int(IntValue::I64(-1))])
    ))
    .is_err());
}

#[test]
fn output_count_extension_policy_is_owned_by_each_invocation() {
    let enabled = runtime();
    enabled.set_runmat_extensions_enabled(true);
    let disabled = runtime();
    disabled.set_runmat_extensions_enabled(false);
    let count = [Value::Int(IntValue::I32(3))];
    assert_eq!(
        block_on(parse_num_outputs(
            &enabled,
            &Value::FunctionHandle("fixture".into()),
            crate::call::FunctionArgumentView::from_slice(&count)
        ))
        .unwrap(),
        Some(3)
    );
    let error = block_on(parse_num_outputs(
        &disabled,
        &Value::FunctionHandle("fixture".into()),
        crate::call::FunctionArgumentView::from_slice(&count),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        INTEGER_NUM_OUTPUTS_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn timeit_rejects_extra_arguments() {
    let err = block_on(timeit_builtin(
        Some(runtime()),
        default_handle(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(1.0),
            Value::from(2.0),
        ]),
    ))
    .unwrap_err();
    assert_timeit_error_contains(&err, "too many");
    assert_timeit_error_identifier(&err, TIMEIT_ERROR_TOO_MANY_INPUTS.identifier.unwrap());
}
