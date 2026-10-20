use super::*;

#[test]
fn resident_typed_integer_controls_decode_exactly_without_binary64_roundtrip() {
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    test_support::with_test_provider(|provider| {
        let wide = (1_u64 << 53) + 1;
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(std::slice::from_ref(&wide)),
                shape: &[1, 1],
            })
            .expect("resident exact U64 control");
        let control = Value::GpuTensor(handle.clone());
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(downsample_builtin(
                tensor(vec![1.0, 2.0], vec![1, 2]),
                control.clone(),
                Vec::new(),
            ))
            .expect_err("resident typed factor gate precedes provider access");
            assert_eq!(
                error.identifier(),
                DOWNSAMPLE_INTEGER_FACTOR_EXTENSION.error_identifier
            );
        }
        let factor = block_on(parse_factor(&control, UPSAMPLE_NAME));
        let order = block_on(scalar_integer_option(&control));
        if usize::BITS == 64 {
            assert_eq!(factor.expect("wide factor"), wide as usize);
            assert_eq!(order.expect("wide N option"), Some(wide as usize));
            assert_eq!(
                block_on(parse_factor(&control, RESAMPLE_NAME)).expect("wide P/Q rate"),
                wide as usize
            );
            assert_eq!(
                block_on(parse_phase(&control, wide as usize + 1, DOWNSAMPLE_NAME))
                    .expect("wide phase"),
                wide as usize
            );
        } else {
            assert_eq!(
                factor.expect_err("platform overflow").identifier(),
                SAMPLE_ERROR_INVALID_FACTOR.identifier
            );
            assert_eq!(
                order.expect_err("platform overflow").identifier(),
                SAMPLE_ERROR_INVALID_OPTION.identifier
            );
        }
        assert_eq!(
            block_on(scalar_f64_option(&control))
                .expect_err("inexact integer BETA must reject")
                .identifier(),
            SAMPLE_ERROR_INVALID_OPTION.identifier
        );
        assert!(handle.allocation_lease().validate_live().is_ok());
        handle.release().expect("caller owns scalar");

        let exact = 1_u64 << 53;
        let exact_handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(std::slice::from_ref(&exact)),
                shape: &[1, 1],
            })
            .expect("exact binary64 integer BETA");
        assert_eq!(
            block_on(scalar_f64_option(&Value::GpuTensor(exact_handle.clone())))
                .expect("exact BETA"),
            exact as f64
        );
        exact_handle.release().expect("caller owns BETA");
    });
}

#[test]
fn dimension_runtime_rejects_known_invalid_scalar_kinds() {
    for value in [
        Value::String("2".into()),
        Value::Bool(true),
        Value::Complex(2.0, 0.0),
    ] {
        let error = block_on(parse_resample_options(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Dimension".into()),
                value,
            ]),
            2,
            1,
        ))
        .expect_err("Dimension requires a host real numeric scalar");
        assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_OPTION.identifier);
    }
}

#[test]
fn downsample_typed_controls_are_separately_extension_gated() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let factor_error = block_on(downsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Int(runmat_value::IntValue::U8(2)),
        vec![],
    ))
    .expect_err("integer factor extension");
    assert_eq!(
        factor_error.identifier(),
        DOWNSAMPLE_INTEGER_FACTOR_EXTENSION.error_identifier
    );
    let phase_error = block_on(downsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        vec![Value::Int(runmat_value::IntValue::U8(1))],
    ))
    .expect_err("integer phase extension");
    assert_eq!(
        phase_error.identifier(),
        DOWNSAMPLE_INTEGER_PHASE_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn upsample_typed_controls_are_separately_extension_gated() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let factor_error = block_on(upsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Int(runmat_value::IntValue::U8(2)),
        vec![],
    ))
    .expect_err("integer factor extension");
    assert_eq!(
        factor_error.identifier(),
        UPSAMPLE_INTEGER_FACTOR_EXTENSION.error_identifier
    );
    let phase_error = block_on(upsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        vec![Value::Int(runmat_value::IntValue::U8(1))],
    ))
    .expect_err("integer phase extension");
    assert_eq!(
        phase_error.identifier(),
        UPSAMPLE_INTEGER_PHASE_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn downsample_nd_input_is_extension_gated() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(downsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 1, 2]),
        Value::Num(2.0),
        Vec::new(),
    ))
    .expect_err("N-D input extension gate");
    assert_eq!(
        error.identifier(),
        DOWNSAMPLE_ND_INPUT_EXTENSION.error_identifier
    );
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = block_on(downsample_builtin(
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
        Value::Num(2.0),
        Vec::new(),
    ))
    .expect("N-D input in RunMat mode");
    let Value::Tensor(output) = output else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape(), [1, 1, 2]);
    assert_eq!(output.materialize_f64(), vec![1.0, 3.0]);
    drop(extensions);
}

#[test]
fn sample_rate_rejects_invalid_factor_and_phase() {
    let err = block_on(upsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(0.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

    let err = block_on(upsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0000004),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

    let err = block_on(downsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        vec![Value::Num(2.0)],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_PHASE.identifier);

    let err = block_on(downsample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        vec![Value::Num(1.0000004)],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_PHASE.identifier);
}

#[test]
fn resample_filter_vector_reads_typed_integer_storage_exactly() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let filter = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 0]), vec![1, 3]).unwrap();

    let parsed = block_on(parse_filter_vector(
        Value::Tensor(filter),
        FilterPreparation::Prepare,
    ))
    .unwrap()
    .expect("prepared vector");
    let values = block_on(parsed.host_values()).unwrap();
    assert_eq!(values, vec![0.0, 1.0, 0.0]);

    let out = call_resample(vec![
        tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 0]), vec![1, 3]).unwrap()),
    ]);
    let Value::Tensor(upsampled) = out else {
        panic!("expected tensor");
    };
    assert_eq!(upsampled.shape(), [1, 6]);
    assert_eq!(
        upsampled.materialize_f64(),
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
    );
}

#[test]
fn resample_scalar_integer_option_decodes_exact_native_value() {
    let scalar = Tensor::new_integer(IntegerStorage::U16(vec![12]), vec![1, 1]).expect("scalar");
    let vector =
        Tensor::new_integer(IntegerStorage::U16(vec![12, 14]), vec![1, 2]).expect("vector");

    assert_eq!(
        block_on(scalar_integer_option(&Value::Tensor(scalar))).expect("scalar"),
        Some(12)
    );
    assert_eq!(
        block_on(scalar_integer_option(&Value::Tensor(vector))).expect("vector"),
        None
    );
}

#[test]
fn sample_rate_integer_parsers_read_typed_integer_storage_exactly() {
    let wide = exact_platform_wide_integer();
    let factor = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("factor");
    let phase =
        Tensor::new_integer(IntegerStorage::U64(vec![wide - 1]), vec![1, 1]).expect("phase");

    assert_eq!(
        block_on(parse_factor(&Value::Tensor(factor), UPSAMPLE_NAME)).expect("factor"),
        wide as usize
    );
    assert_eq!(
        block_on(parse_phase(
            &Value::Tensor(phase),
            wide as usize,
            UPSAMPLE_NAME
        ))
        .expect("phase"),
        (wide - 1) as usize
    );
}

#[test]
fn sample_rate_integer_parsers_reject_unrepresentable_double_boundary() {
    let err = parse_nonnegative_integer(
        first_unrepresentable_usize_double(),
        UPSAMPLE_NAME,
        &SAMPLE_ERROR_INVALID_FACTOR,
    )
    .expect_err("unrepresentable integer should fail");

    assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);
}

#[test]
fn resample_dimension_name_value_operates_on_requested_axis() {
    let out = call_resample(vec![
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        Value::Num(2.0),
        Value::Num(1.0),
        tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        Value::CharArray(runmat_value::CharArray::new_row("Dimension")),
        Value::Num(2.0),
    ]);
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [2, 4]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
    );
}

#[test]
fn resample_two_outputs_return_signal_and_filter() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = block_on(resample_builtin(
        tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
        Value::Num(3.0),
        Value::Num(2.0),
        vec![Value::Num(1.0), Value::Num(0.0)],
    ))
    .unwrap();
    let Value::OutputList(values) = result else {
        panic!("expected output list");
    };
    assert_eq!(values.len(), 2);
    let Value::Tensor(signal) = &values[0] else {
        panic!("expected signal tensor");
    };
    let Value::Tensor(filter) = &values[1] else {
        panic!("expected filter tensor");
    };
    assert_eq!(signal.shape(), [5, 1]);
    assert_eq!(filter.shape(), [1, 7]);
    let sum = filter.materialize_f64().iter().sum::<f64>();
    assert!((sum - 3.0).abs() < 1e-10);
}

#[test]
fn resample_default_filter_has_matlab_compatible_length() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = block_on(resample_builtin(
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]),
        Value::Num(4.0),
        Value::Num(2.0),
        Vec::new(),
    ))
    .unwrap();
    let Value::OutputList(values) = result else {
        panic!("expected output list");
    };
    let Value::Tensor(signal) = &values[0] else {
        panic!("expected signal tensor");
    };
    let Value::Tensor(filter) = &values[1] else {
        panic!("expected filter tensor");
    };
    assert_eq!(signal.shape(), [1, 8]);
    assert_eq!(filter.shape(), [1, 41]);
    let sum = filter.materialize_f64().iter().sum::<f64>();
    assert!((sum - 2.0).abs() < 1e-10);
}

#[test]
fn resample_filter_design_rejects_huge_allocation() {
    let err = design_resample_filter(1usize << (usize::BITS - 3), 1, 1, 5.0).unwrap_err();
    assert_eq!(err.identifier(), SAMPLE_ERROR_SIZE_OVERFLOW.identifier);
}

#[test]
fn resample_rejects_invalid_options_and_output_count() {
    let err = block_on(resample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        Value::Num(0.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

    let err = block_on(resample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        Value::Num(1.0),
        vec![Value::CharArray(runmat_value::CharArray::new_row(
            "Dimension",
        ))],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_OPTION.identifier);

    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let err = block_on(resample_builtin(
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        Value::Num(1.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), RESAMPLE_ERROR_OUTPUT_COUNT.identifier);
}
