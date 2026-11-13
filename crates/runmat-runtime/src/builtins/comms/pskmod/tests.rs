use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;

#[test]
fn provider_execution_error_keeps_source_and_borrowed_input_live() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[99.0],
                shape: &[1, 1],
            })
            .unwrap();
        let error = block_on(super::pskmod_builtin(
            Value::GpuTensor(input.clone()),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert!(std::error::Error::source(&error).is_some());
        assert!(error.message().contains("symbols must be in range"));
        assert_eq!(
            block_on(provider.download(&input)).unwrap().data,
            vec![99.0]
        );
    });
}

#[test]
fn malformed_alias_rejection_preserves_original_allocation() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[0.0, 1.0],
                shape: &[1, 2],
            })
            .unwrap();
        assert!(!valid_gpu_output(
            provider,
            &input,
            &input,
            OutputDType::Double,
            &[1, 2]
        ));
        let error = gpu_helpers::reject_gpu_output(input.clone(), pskmod_error("malformed alias"));
        assert!(error.message().contains("malformed alias"));
        assert_eq!(
            block_on(provider.download(&input)).unwrap().data,
            vec![0.0, 1.0]
        );
    });
}

fn pskmod(x: Value, order: usize, rest: Vec<Value>) -> ComplexTensor {
    match block_on(super::pskmod_builtin(
        x,
        Value::Num(order as f64),
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .expect("pskmod")
    {
        Value::ComplexTensor(tensor) => tensor,
        other => panic!("expected ComplexTensor, got {other:?}"),
    }
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).expect("tensor"))
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
}

fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, ((ar, ai), (er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (ar - er).abs() < 1e-12 && (ai - ei).abs() < 1e-12,
            "at {idx}: expected ({er}, {ei}), got ({ar}, {ai})"
        );
    }
}

#[test]
fn pskmod_bpsk_gray_matches_binary() {
    let out = pskmod(tensor(vec![0.0, 1.0], vec![1, 2]), 2, vec![]);
    assert_eq!(out.shape(), [1, 2]);
    assert_complex_close(&out.materialize_f64(), &[(1.0, 0.0), (-1.0, 0.0)]);
}

#[test]
fn pskmod_qpsk_gray_mapping() {
    let out = pskmod(tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]), 4, vec![]);
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)],
    );
}

#[test]
fn pskmod_reads_typed_integer_symbol_storage_exactly() {
    let out = pskmod(
        integer_tensor(IntegerStorage::U32(vec![0, 1, 2, 3]), vec![1, 4]),
        4,
        vec![],
    );
    assert_eq!(out.shape(), [1, 4]);
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)],
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pskmod_preserves_integer_rank_and_bit_grouping_geometry() {
    let rank_zero = pskmod(tensor(vec![0.0], Vec::new()), 4, Vec::new());
    assert!(rank_zero.shape().is_empty());

    let rank_one = pskmod(tensor(vec![0.0, 1.0], vec![2]), 4, Vec::new());
    assert_eq!(rank_one.shape(), [2]);

    let rank_zero_bit = pskmod(
        tensor(vec![0.0], Vec::new()),
        2,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(rank_zero_bit.shape(), [1, 1]);

    let rank_one_bits = pskmod(
        tensor(vec![0.0, 1.0], vec![2]),
        4,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(rank_one_bits.shape(), [1]);
}

#[test]
fn pskmod_accepts_only_documented_integer_symbol_classes() {
    for storage in [
        IntegerStorage::I8(vec![0, 1, 2, 3]),
        IntegerStorage::I16(vec![0, 1, 2, 3]),
        IntegerStorage::I32(vec![0, 1, 2, 3]),
        IntegerStorage::U8(vec![0, 1, 2, 3]),
        IntegerStorage::U16(vec![0, 1, 2, 3]),
        IntegerStorage::U32(vec![0, 1, 2, 3]),
    ] {
        pskmod(integer_tensor(storage, vec![1, 4]), 4, vec![]);
    }
    for storage in [
        IntegerStorage::I64(vec![0, 1, 2, 3]),
        IntegerStorage::U64(vec![0, 1, 2, 3]),
    ] {
        let err = block_on(super::pskmod_builtin(
            integer_tensor(storage, vec![1, 4]),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("unsupported integer X class");
        assert!(err
            .message()
            .contains("int8, int16, int32, uint8, uint16, or uint32"));
    }
}

#[test]
fn pskmod_8psk_gray_mapping() {
    let out = pskmod(
        tensor((0..8).map(|v| v as f64).collect(), vec![1, 8]),
        8,
        vec![],
    );
    let expected_indices = [0usize, 1, 3, 2, 7, 6, 4, 5];
    let expected = expected_indices
        .iter()
        .map(|idx| {
            let theta = TAU * *idx as f64 / 8.0;
            (theta.cos(), theta.sin())
        })
        .collect::<Vec<_>>();
    assert_complex_close(&out.materialize_f64(), &expected);
}

#[test]
fn pskmod_binary_order_is_counterclockwise() {
    let out = pskmod(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
        4,
        vec![Value::Num(0.0), Value::from("bin")],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
    );
}

#[test]
fn pskmod_phase_offset_rotates_constellation() {
    let out = pskmod(
        tensor(vec![0.0, 1.0], vec![1, 2]),
        2,
        vec![Value::Num(std::f64::consts::FRAC_PI_2), Value::from("bin")],
    );
    assert_complex_close(&out.materialize_f64(), &[(0.0, 1.0), (0.0, -1.0)]);
}

#[test]
fn pskmod_custom_mapping_uses_constellation_labels() {
    let mapping = tensor(vec![0.0, 3.0, 1.0, 2.0], vec![1, 4]);
    let out = pskmod(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
        4,
        vec![Value::Num(0.0), mapping],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 1.0)],
    );
}

#[test]
fn pskmod_reads_typed_integer_custom_mapping_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let mapping = integer_tensor(IntegerStorage::U16(vec![0, 3, 1, 2]), vec![1, 4]);
    let out = pskmod(
        integer_tensor(IntegerStorage::U8(vec![0, 1, 2, 3]), vec![1, 4]),
        4,
        vec![Value::Num(0.0), mapping],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 1.0)],
    );
}

#[test]
fn pskmod_bit_input_groups_rows_by_channel() {
    let bits = tensor(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let out = pskmod(bits, 4, vec![Value::from("InputType"), Value::from("bit")]);
    assert_eq!(out.shape(), [2, 2]);
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)],
    );
}

#[test]
fn pskmod_reads_typed_integer_bit_storage_exactly() {
    let bits = integer_tensor(IntegerStorage::U8(vec![0, 0, 0, 1, 1, 0, 1, 1]), vec![4, 2]);
    let out = pskmod(bits, 4, vec![Value::from("InputType"), Value::from("bit")]);
    assert_eq!(out.shape(), [2, 2]);
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)],
    );
}

#[test]
fn pskmod_output_datatype_single_rounds_samples() {
    let out = pskmod(
        tensor(vec![1.0], vec![1, 1]),
        8,
        vec![Value::from("OutputDataType"), Value::from("single")],
    );
    let theta = TAU / 8.0;
    assert_eq!(out.materialize_f64()[0].0, (theta.cos() as f32) as f64);
    assert_eq!(out.materialize_f64()[0].1, (theta.sin() as f32) as f64);
}

#[test]
fn pskmod_single_input_still_defaults_to_double_output() {
    let out = pskmod(
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
        8,
        vec![],
    );
    let theta = TAU / 8.0;
    assert_eq!(out.materialize_f64()[0].0, theta.cos());
    assert_eq!(out.materialize_f64()[0].1, theta.sin());
    assert_ne!(out.materialize_f64()[0].0, (theta.cos() as f32) as f64);
}

#[test]
fn pskmod_rejects_out_of_range_symbols() {
    let err = block_on(super::pskmod_builtin(
        tensor(vec![0.0, 8.0], vec![1, 2]),
        Value::Num(8.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("expected out-of-range error");
    assert!(err.to_string().contains("range [0, 7]"));
}

#[test]
fn pskmod_rejects_default_logical_input() {
    let err = block_on(super::pskmod_builtin(
        Value::LogicalArray(LogicalArray {
            data: vec![0, 1].into(),
            shape: vec![1, 2].into(),
        }),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("expected logical input error");
    assert!(err.to_string().contains("InputType='bit'"));
}

#[test]
fn pskmod_accepts_logical_bit_input() {
    let out = pskmod(
        Value::LogicalArray(LogicalArray {
            data: vec![0, 0, 1, 1].into(),
            shape: vec![2, 2].into(),
        }),
        4,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(out.shape(), [1, 2]);
    assert_complex_close(&out.materialize_f64(), &[(1.0, 0.0), (-1.0, 0.0)]);
}

#[test]
fn pskmod_inprocess_gpu_bit_input_returns_complex_resident_output() {
    use runmat_accelerate_api::{GpuTensorStorage, HostTensorView};

    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let expected = pskmod(
            Value::Tensor(input.clone()),
            4,
            vec![Value::from("InputType"), Value::from("bit")],
        );
        let input_handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload");
        let result = block_on(super::pskmod_builtin(
            Value::GpuTensor(input_handle.clone()),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("InputType"),
                Value::from("bit"),
            ]),
        ))
        .expect("gpu pskmod bit");
        let Value::GpuTensor(output_handle) = result else {
            panic!("expected resident gpu output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(output_handle.clone()),
        ))
        .expect("gather output");
        let Value::ComplexTensor(actual) = gathered else {
            panic!("expected gathered complex tensor");
        };
        assert_eq!(actual.shape(), expected.shape());
        assert_complex_close(&actual.materialize_f64(), &expected.materialize_f64());
        drop(input_handle);
        drop(output_handle);
    });
}

#[test]
fn pskmod_gpu_enforces_integer_symbol_classes_and_output_precision() {
    test_support::with_test_provider(|provider| {
        let accepted =
            Tensor::new_integer(IntegerStorage::U32(vec![0, 1, 2, 3]), vec![1, 4]).unwrap();
        let accepted_handle =
            gpu_helpers::upload_tensor(provider, &accepted).expect("accepted integer upload");
        let result = block_on(super::pskmod_builtin(
            Value::GpuTensor(accepted_handle),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("supported integer gpuArray");
        let Value::GpuTensor(result_handle) = result else {
            panic!("expected resident modulation output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&result_handle),
            Some(ProviderPrecision::F64)
        );

        let rejected =
            Tensor::new_integer(IntegerStorage::U64(vec![0, 1, 2, 3]), vec![1, 4]).unwrap();
        let rejected_handle =
            gpu_helpers::upload_tensor(provider, &rejected).expect("rejected integer upload");
        let err = block_on(super::pskmod_builtin(
            Value::GpuTensor(rejected_handle),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("unsupported integer gpuArray class");
        assert!(err
            .message()
            .contains("int8, int16, int32, uint8, uint16, or uint32"));

        let single = Tensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap();
        let single_handle = gpu_helpers::upload_tensor(provider, &single).expect("single upload");
        let result = block_on(super::pskmod_builtin(
            Value::GpuTensor(single_handle),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("OutputDataType"),
                Value::from("single"),
            ]),
        ))
        .expect("explicit single gpuArray output");
        let Value::GpuTensor(result_handle) = result else {
            panic!("expected resident single modulation output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&result_handle),
            Some(ProviderPrecision::F32)
        );
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn pskmod_wgpu_input_returns_complex_resident_output() {
    use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostTensorView};

    match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) {
        Ok(provider) => {
            let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap();
            let expected = pskmod(Value::Tensor(input.clone()), 4, vec![]);
            let view = HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            };
            let input_handle = provider.upload(&view).expect("upload");
            let result = block_on(super::pskmod_builtin(
                Value::GpuTensor(input_handle.clone()),
                Value::Num(4.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("gpu pskmod");
            let Value::GpuTensor(output_handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(output_handle.clone()),
            ))
            .expect("gather output");
            let Value::ComplexTensor(actual) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(actual.shape(), expected.shape());
            assert_complex_close(&actual.materialize_f64(), &expected.materialize_f64());
            drop(input_handle);
            drop(output_handle);
        }
        Err(err) => {
            tracing::warn!("Skipping pskmod_wgpu_input_returns_complex_resident_output: {err}");
        }
    }
    runmat_accelerate::simple_provider::register_inprocess_provider();
}

#[test]
#[cfg(feature = "wgpu")]
fn pskmod_wgpu_bit_input_returns_complex_resident_output() {
    use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostTensorView};

    match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) {
        Ok(provider) => {
            let input = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
            let expected = pskmod(
                Value::Tensor(input.clone()),
                4,
                vec![Value::from("InputType"), Value::from("bit")],
            );
            let view = HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            };
            let input_handle = provider.upload(&view).expect("upload");
            let result = block_on(super::pskmod_builtin(
                Value::GpuTensor(input_handle.clone()),
                Value::Num(4.0),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("InputType"),
                    Value::from("bit"),
                ]),
            ))
            .expect("gpu pskmod bit");
            let Value::GpuTensor(output_handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(output_handle.clone()),
            ))
            .expect("gather output");
            let Value::ComplexTensor(actual) = gathered else {
                panic!("expected gathered complex tensor");
            };
            assert_eq!(actual.shape(), expected.shape());
            assert_complex_close(&actual.materialize_f64(), &expected.materialize_f64());
            drop(input_handle);
            drop(output_handle);
        }
        Err(err) => {
            tracing::warn!("Skipping pskmod_wgpu_bit_input_returns_complex_resident_output: {err}");
        }
    }
    runmat_accelerate::simple_provider::register_inprocess_provider();
}

#[test]
fn pskmod_rejects_plot_constellation_true() {
    let err = block_on(super::pskmod_builtin(
        Value::Num(0.0),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("PlotConstellation"),
            Value::Bool(true),
        ]),
    ))
    .expect_err("expected plot error");
    assert!(err.to_string().contains("PlotConstellation"));
}

#[test]
fn pskmod_integer_scalars_preserve_exact_symbol_bounds() {
    assert_eq!(
        parse_modulation_order(&Value::Int(runmat_value::IntValue::U16(4))).unwrap(),
        4
    );
    assert_eq!(
        SymbolInput::from_value(
            Value::Int(runmat_value::IntValue::U16(3)),
            4,
            InputType::Integer,
        )
        .unwrap()
        .data,
        vec![3]
    );
    for value in [
        Value::Int(runmat_value::IntValue::I8(-1)),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(parse_modulation_order(&value).is_err());
    }
    assert!(number_to_symbol_with_name(usize::MAX as f64, "M").is_err());
    assert!(number_to_symbol_with_name(1.0e300, "M").is_err());
}

#[test]
fn pskmod_typed_integer_tensors_preserve_exact_symbol_values() {
    let order = Tensor::new_integer(IntegerStorage::U64(vec![4]), vec![1, 1]).unwrap();
    assert_eq!(parse_modulation_order(&Value::Tensor(order)).unwrap(), 4);

    let phase = Tensor::new_integer(IntegerStorage::I16(vec![3]), vec![1, 1]).unwrap();
    assert_eq!(scalar_number(&Value::Tensor(phase), "phase").unwrap(), 3.0);

    let out = pskmod(
        integer_tensor(IntegerStorage::U32(vec![0, 1, 2, 3]), vec![1, 4]),
        4,
        vec![],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)],
    );

    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let custom = integer_tensor(IntegerStorage::U16(vec![0, 3, 1, 2]), vec![1, 4]);
        let out = pskmod(
            integer_tensor(IntegerStorage::U8(vec![0, 1, 2, 3]), vec![1, 4]),
            4,
            vec![Value::Num(0.0), custom],
        );
        assert_complex_close(
            &out.materialize_f64(),
            &[(1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 1.0)],
        );
    }

    for storage in [
        IntegerStorage::I8(vec![4]),
        IntegerStorage::I16(vec![4]),
        IntegerStorage::I32(vec![4]),
        IntegerStorage::I64(vec![4]),
        IntegerStorage::U8(vec![4]),
        IntegerStorage::U16(vec![4]),
        IntegerStorage::U32(vec![4]),
        IntegerStorage::U64(vec![4]),
    ] {
        let order = Tensor::new_integer(storage, vec![1, 1]).expect("typed order");
        assert_eq!(parse_modulation_order(&Value::Tensor(order)).unwrap(), 4);
    }
}

#[test]
fn pskmod_rejects_near_integer_controls_and_matrix_symbol_order() {
    assert!(number_to_symbol_with_name(1.0 + 5e-10, "X").is_err());
    assert!(number_to_symbol_with_name(4.0 + 5e-10, "M").is_err());
    assert!(value_as_bool(&Value::Num(1.0 - 5e-10), "InputType").is_err());
    let matrix = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    assert!(parse_symbol_mapping(&Value::Tensor(matrix), 4).is_err());
}

#[test]
fn pskmod_typed_integer_controls_follow_compatibility_mode() {
    let custom = || {
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![0, 3, 1, 2]), vec![1, 4])
                .expect("typed custom order"),
        )
    };
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::pskmod_builtin(
            Value::Num(0.0),
            Value::Int(IntValue::U16(4)),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("MATLAB mode rejects typed-integer M");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PskmodIntegerModulationOrderExtension")
        );

        let error = block_on(super::pskmod_builtin(
            Value::Num(0.0),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I16(0)),
                Value::from("bin"),
            ]),
        ))
        .expect_err("MATLAB mode rejects typed-integer phase");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PskmodIntegerPhaseOffsetExtension")
        );

        let error = block_on(super::pskmod_builtin(
            Value::Num(0.0),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                custom(),
            ]),
        ))
        .expect_err("MATLAB mode rejects typed-integer custom order");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:PskmodIntegerCustomOrderExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(block_on(super::pskmod_builtin(
            Value::Num(0.0),
            Value::Int(IntValue::U16(4)),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::I16(0)),
                custom()
            ]),
        ))
        .is_ok());
    }
}
