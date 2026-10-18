use super::*;

#[test]
fn malformed_logical_signal_is_rejected_with_or_without_an_output() {
    for count in [0, 1] {
        let _outputs =
            crate::output_count::push_output_count(Some(count)).expect("output-count scope");
        let logical = runmat_value::LogicalArray {
            data: runmat_value::HostLogicalBuffer::new(vec![0, 1, 0]),
            shape: vec![1, 4].into(),
        };
        let error = block_on(resample_builtin(
            Value::LogicalArray(logical),
            Value::Num(2.0),
            Value::Num(1.0),
            vec![],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_INPUT.identifier);
    }
}

#[test]
fn discarded_logical_filter_uses_shape_admission_and_prepared_coefficients_stay_numeric() {
    let logical = runmat_value::LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
    assert!(block_on(parse_filter_vector(
        Value::LogicalArray(logical.clone()),
        FilterPreparation::ValidateOnly,
    ))
    .unwrap()
    .is_none());
    let prepared = block_on(parse_filter_vector(
        Value::LogicalArray(logical.clone()),
        FilterPreparation::Prepare,
    ))
    .unwrap()
    .expect("prepared logical filter");
    assert_eq!(
        block_on(prepared.host_values()).unwrap(),
        vec![0.0, 1.0, 0.0]
    );

    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let signal = || tensor(vec![1.0, 2.0], vec![1, 2]);
    // The public one-optional-argument form classifies a nonscalar LogicalArray
    // as an invalid scalar N before the filter-vector branch. Preserve that
    // prepared and discarded call behavior; the direct parser check above
    // verifies the validation-only branch itself.
    let error = block_on(resample_builtin(
        signal(),
        Value::Num(2.0),
        Value::Num(1.0),
        vec![Value::LogicalArray(logical)],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_OPTION.identifier);

    for logical in [
        runmat_value::LogicalArray::new(vec![], vec![0, 1]).unwrap(),
        runmat_value::LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap(),
        runmat_value::LogicalArray {
            data: runmat_value::HostLogicalBuffer::new(vec![0, 1, 0]),
            shape: vec![1, 4].into(),
        },
    ] {
        let error = block_on(parse_filter_vector(
            Value::LogicalArray(logical),
            FilterPreparation::ValidateOnly,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_OPTION.identifier);
    }
}

#[test]
fn discarded_resample_outputs_still_admit_inputs_rates_and_options() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let signal = || tensor(vec![1.0, 2.0], vec![1, 2]);
    let call = |x, p, q, rest| block_on(resample_builtin(x, p, q, rest));

    let invalid_input = call(
        Value::Int(runmat_value::IntValue::I32(1)),
        Value::Num(2.0),
        Value::Num(1.0),
        vec![],
    )
    .unwrap_err();
    assert_eq!(
        invalid_input.identifier(),
        SAMPLE_ERROR_INVALID_INPUT.identifier
    );
    let invalid_rate = call(signal(), Value::Num(0.0), Value::Num(1.0), vec![]).unwrap_err();
    assert_eq!(
        invalid_rate.identifier(),
        SAMPLE_ERROR_INVALID_FACTOR.identifier
    );
    let invalid_q = call(signal(), Value::Num(2.0), Value::Num(0.0), vec![]).unwrap_err();
    assert_eq!(
        invalid_q.identifier(),
        SAMPLE_ERROR_INVALID_FACTOR.identifier
    );
    let invalid_filter = call(
        signal(),
        Value::Num(2.0),
        Value::Num(1.0),
        vec![Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap())],
    )
    .unwrap_err();
    assert_eq!(
        invalid_filter.identifier(),
        SAMPLE_ERROR_INVALID_OPTION.identifier
    );
    let invalid_dimension = call(
        signal(),
        Value::Num(2.0),
        Value::Num(1.0),
        vec![Value::String("Dimension".into()), Value::Bool(true)],
    )
    .unwrap_err();
    assert_eq!(
        invalid_dimension.identifier(),
        SAMPLE_ERROR_INVALID_OPTION.identifier
    );
    let invalid_beta = call(
        signal(),
        Value::Num(2.0),
        Value::Num(1.0),
        vec![Value::Num(2.0), Value::Num(-1.0)],
    )
    .unwrap_err();
    assert_eq!(
        invalid_beta.identifier(),
        SAMPLE_ERROR_INVALID_OPTION.identifier
    );

    let valid = call(signal(), Value::Num(2.0), Value::Num(1.0), vec![]).unwrap();
    assert_eq!(valid, Value::output_list(Vec::new()));
}

#[test]
fn discarded_resample_retains_count_and_extension_precedence() {
    let signal = || tensor(vec![1.0, 2.0], vec![1, 2]);
    {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = block_on(resample_builtin(
            signal(),
            Value::Num(0.0),
            Value::Num(1.0),
            vec![],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), RESAMPLE_ERROR_OUTPUT_COUNT.identifier);
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(resample_builtin(
            signal(),
            Value::Int(runmat_value::IntValue::I32(2)),
            Value::Num(1.0),
            vec![],
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            RESAMPLE_INTEGER_OPTION_EXTENSION.error_identifier
        );
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let factor = 1usize << (usize::BITS - 3);
        let error = block_on(resample_builtin(
            signal(),
            Value::Num(factor as f64),
            Value::Num(1.0),
            vec![],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), SAMPLE_ERROR_SIZE_OVERFLOW.identifier);
    }
}

#[test]
fn discarded_resample_does_not_allocate_provider_output_or_touch_borrowed_source() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let source = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let result = block_on(resample_builtin(
        Value::GpuTensor(source.clone()),
        Value::Num(2.0),
        Value::Num(1.0),
        vec![tensor(vec![0.0, 1.0, 0.0], vec![1, 3])],
    ))
    .unwrap();
    assert_eq!(result, Value::output_list(Vec::new()));
    assert_eq!(provider.probes.lock().unwrap().len(), 1);
    assert_eq!(provider.gathers.load(Ordering::Relaxed), 0);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert_eq!(provider.numeric_uploads.load(Ordering::Relaxed), 0);
    assert!(source.allocation_lease().validate_live().is_ok());
    source.release().unwrap();
}
