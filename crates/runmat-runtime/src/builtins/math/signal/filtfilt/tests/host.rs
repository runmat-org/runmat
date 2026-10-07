use super::*;

#[test]
fn descriptor_is_registered() {
    assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
    let builtin = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME)
        .expect("canonical filtfilt builtin");
    let descriptor = builtin.descriptor;
    assert_eq!(builtin.bindings, &runmat_builtins::REQUIRED_DEFAULT_BINDING);
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        builtin.binding_identity(&builtin.bindings[0])
    );
    assert_eq!(descriptor.signatures[0].label, "y = filtfilt(b, a, x)");
    assert!(descriptor
        .errors
        .iter()
        .any(|err| err.code == "RM.FILTFILT.INVALID_SIGNAL"));
    assert_eq!(builtin.integer_capabilities.len(), 1);
    assert_eq!(builtin.extensions.len(), 2);
}

#[test]
fn registered_filtfilt_binding_uses_invocation_cancellation() {
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(filtfilt_builtin(
        Some(runtime),
        Value::Num(1.0),
        Value::Num(1.0),
        Value::Num(1.0),
    ))
    .expect_err("cancelled invocation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn operational_source_keeps_filtfilt_as_the_owning_builtin() {
    let mut source = build_runtime_error("readback failed")
        .with_builtin("gather")
        .with_identifier("RunMat:gather:ProviderFailed")
        .build();
    source.gpu_gather_retry = crate::GpuGatherRetry::Never;
    let error = filtfilt_error_with_source(
        &FILTFILT_ERROR_PASS_FAILED,
        "provider result gather failed",
        source,
        true,
    );
    assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let nested = std::error::Error::source(&error).expect("original gather cause");
    assert!(nested.to_string().contains("readback failed"));
}

#[test]
fn compatibility_mode_rejects_integer_and_logical_extensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = call(
        Value::Int(runmat_value::IntValue::I16(1)),
        Value::Num(1.0),
        Value::Num(1.0),
    )
    .expect_err("integer extension must reject");
    assert_eq!(
        integer.identifier(),
        FILTFILT_INTEGER_INPUT_EXTENSION.error_identifier
    );
    let logical = call(Value::Bool(true), Value::Num(1.0), Value::Num(1.0))
        .expect_err("logical extension must reject");
    assert_eq!(
        logical.identifier(),
        FILTFILT_LOGICAL_INPUT_EXTENSION.error_identifier
    );
}

#[test]
fn nonzero_subnormal_denominator_is_not_classified_as_zero() {
    let signal = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let output = call(
        Value::Num(1.0e-300),
        Value::Num(1.0e-300),
        Value::Tensor(signal),
    )
    .expect("finite nonzero denominator");
    let (values, shape) = tensor_data(output);
    assert_eq!(shape, vec![1, 2]);
    assert_eq!(values, vec![1.0, 2.0]);
}

#[test]
fn tiny_nonzero_imaginary_coefficient_remains_complex() {
    let signal = Tensor::new((1..=8).map(f64::from).collect(), vec![1, 8]).unwrap();
    let numerator = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 1.0e-300)], vec![1, 2]).unwrap();
    let output = call(
        Value::ComplexTensor(numerator),
        Value::Num(1.0),
        Value::Tensor(signal),
    )
    .expect("small imaginary coefficient");
    let Value::ComplexTensor(output) = output else {
        panic!("complex coefficient must produce complex output");
    };
    assert_eq!(output.shape(), vec![1, 8]);
    assert!(output.materialize_f64().iter().any(|&(_, im)| im != 0.0));
}

#[test]
fn finite_large_constant_reflection_does_not_overflow_before_identity_filter() {
    let signal = Tensor::new(vec![1.0e308; 8], vec![1, 8]).unwrap();
    let identity_with_state = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let output = call(
        Value::Tensor(identity_with_state),
        Value::Num(1.0),
        Value::Tensor(signal),
    )
    .expect("representable large constant reflection");
    let (values, shape) = tensor_data(output);
    assert_eq!(shape, vec![1, 8]);
    assert_eq!(values, vec![1.0e308; 8]);

    let signal = Tensor::new(
        vec![f64::MAX, -f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1, 8],
    )
    .unwrap();
    let b = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let error = call(Value::Tensor(b), Value::Num(1.0), Value::Tensor(signal))
        .expect_err("unrepresentable endpoint reflection");
    assert_eq!(error.identifier(), FILTFILT_ERROR_INVALID_SIGNAL.identifier);
}

#[test]
fn resident_exact_wide_integer_uses_authoritative_gather_fallback() {
    test_support::with_test_provider(|provider| {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let signal = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_994; 4]),
            vec![1, 4],
        )
        .expect("wide signal");
        let handle = gpu_helpers::upload_tensor(provider, &signal).expect("wide upload");
        let result = block_on(evaluate(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::GpuTensor(handle),
        ))
        .expect("exact resident wide integer");
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
        assert_eq!(gathered.shape(), vec![1, 4]);
    });
}

#[test]
fn single_output_is_native_and_nonfinite_signal_rejects() {
    let single = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let value = call(Value::Num(1.0), Value::Num(1.0), Value::Tensor(single)).unwrap();
    let Value::Tensor(value) = value else {
        panic!("expected tensor")
    };
    assert_eq!(value.numeric_dtype(), NumericDType::F32);

    let nonfinite = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap();
    call(Value::Num(1.0), Value::Num(1.0), Value::Tensor(nonfinite))
        .expect_err("nonfinite signal must reject");
}

#[test]
fn fir_moving_average_preserves_row_shape() {
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
    let (data, shape) =
        tensor_data(call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap());
    assert_eq!(shape, vec![1, 20]);
    approx(
        &data[3..17],
        &(4..=17).map(|v| v as f64).collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn iir_coefficients_are_supported() {
    let b = Tensor::new(vec![0.2], vec![1, 1]).unwrap();
    let a = Tensor::new(vec![1.0, -0.8], vec![1, 2]).unwrap();
    let x = Tensor::new(vec![1.0; 16], vec![16, 1]).unwrap();
    let (data, shape) =
        tensor_data(call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap());
    assert_eq!(shape, vec![16, 1]);
    approx(&data, &[1.0; 16], 1e-8);
}

#[test]
fn rejects_too_short_signal() {
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    let err = call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap_err();
    assert!(err.message().contains("greater than"));
}

#[test]
fn scalar_gain_filters_forward_and_backward() {
    let y = call(Value::Num(2.0), Value::Num(1.0), Value::Num(3.0)).unwrap();
    let (data, shape) = tensor_data(y);
    assert_eq!(shape, vec![1, 1]);
    approx(&data, &[12.0], 1e-12);
}

#[test]
fn complex_host_path() {
    let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
    let value = call(Value::Tensor(b), Value::Tensor(a), Value::ComplexTensor(x)).unwrap();
    let Value::ComplexTensor(out) = value else {
        panic!("expected complex output");
    };
    assert_eq!(out.shape(), vec![1, 2]);
    assert_eq!(out.materialize_f64(), vec![(1.0, 1.0), (2.0, -1.0)]);
}
