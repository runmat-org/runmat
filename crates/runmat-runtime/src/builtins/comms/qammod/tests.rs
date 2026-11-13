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
        let error = block_on(super::qammod_builtin(
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
        let error = gpu_helpers::reject_gpu_output(input.clone(), qammod_error("malformed alias"));
        assert!(error.message().contains("malformed alias"));
        assert_eq!(
            block_on(provider.download(&input)).unwrap().data,
            vec![0.0, 1.0]
        );
    });
}

fn qammod(x: Value, order: usize, rest: Vec<Value>) -> ComplexTensor {
    match block_on(super::qammod_builtin(
        x,
        Value::Num(order as f64),
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .expect("qammod")
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
fn qammod_4_gray_mapping() {
    let out = qammod(tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]), 4, vec![]);
    assert_eq!(out.shape(), [1, 4]);
    assert_complex_close(
        &out.materialize_f64(),
        &[(-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)],
    );
}

#[test]
fn qammod_reads_typed_integer_symbol_storage_exactly() {
    let out = qammod(
        integer_tensor(IntegerStorage::U16(vec![0, 1, 2, 3]), vec![1, 4]),
        4,
        vec![],
    );
    assert_eq!(out.shape(), [1, 4]);
    assert_complex_close(
        &out.materialize_f64(),
        &[(-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)],
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn qammod_preserves_integer_rank_and_bit_grouping_geometry() {
    let rank_zero = qammod(tensor(vec![0.0], Vec::new()), 4, Vec::new());
    assert!(rank_zero.shape().is_empty());

    let rank_one = qammod(tensor(vec![0.0, 1.0], vec![2]), 4, Vec::new());
    assert_eq!(rank_one.shape(), [2]);

    let rank_zero_bit = qammod(
        tensor(vec![0.0], Vec::new()),
        2,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(rank_zero_bit.shape(), [1, 1]);

    let rank_one_bits = qammod(
        tensor(vec![0.0, 1.0], vec![2]),
        4,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(rank_one_bits.shape(), [1]);
}

#[test]
fn qammod_accepts_only_documented_integer_symbol_classes() {
    for storage in [
        IntegerStorage::I8(vec![0, 1, 2, 3]),
        IntegerStorage::I16(vec![0, 1, 2, 3]),
        IntegerStorage::U8(vec![0, 1, 2, 3]),
        IntegerStorage::U16(vec![0, 1, 2, 3]),
    ] {
        qammod(integer_tensor(storage, vec![1, 4]), 4, vec![]);
    }
    for storage in [
        IntegerStorage::I32(vec![0, 1, 2, 3]),
        IntegerStorage::I64(vec![0, 1, 2, 3]),
        IntegerStorage::U32(vec![0, 1, 2, 3]),
        IntegerStorage::U64(vec![0, 1, 2, 3]),
    ] {
        let err = block_on(super::qammod_builtin(
            integer_tensor(storage, vec![1, 4]),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("unsupported integer X class");
        assert!(err.message().contains("int8, int16, uint8, or uint16"));
    }
}

#[test]
fn qammod_16_gray_mapping() {
    let input: Vec<f64> = (0..16).map(|v| v as f64).collect();
    let out = qammod(tensor(input, vec![1, 16]), 16, vec![]);
    assert_complex_close(
        &out.materialize_f64(),
        &[
            (-3.0, 3.0),
            (-3.0, 1.0),
            (-3.0, -3.0),
            (-3.0, -1.0),
            (-1.0, 3.0),
            (-1.0, 1.0),
            (-1.0, -3.0),
            (-1.0, -1.0),
            (3.0, 3.0),
            (3.0, 1.0),
            (3.0, -3.0),
            (3.0, -1.0),
            (1.0, 3.0),
            (1.0, 1.0),
            (1.0, -3.0),
            (1.0, -1.0),
        ],
    );
}

#[test]
fn qammod_64_gray_has_expected_edges() {
    let out = qammod(
        tensor(vec![0.0, 1.0, 2.0, 3.0, 63.0], vec![1, 5]),
        64,
        vec![],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[
            (-7.0, 7.0),
            (-7.0, 5.0),
            (-7.0, 1.0),
            (-7.0, 3.0),
            (1.0, -1.0),
        ],
    );
}

#[test]
fn qammod_binary_mapping_is_sequential_columnwise() {
    let out = qammod(
        tensor(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![1, 5]),
        16,
        vec![Value::from("bin")],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[
            (-3.0, 3.0),
            (-3.0, 1.0),
            (-3.0, -1.0),
            (-3.0, -3.0),
            (-1.0, 3.0),
        ],
    );
}

#[test]
fn qammod_bit_input_groups_rows_by_channel() {
    let out = qammod(
        tensor(
            vec![
                0.0, 0.0, 0.0, 0.0, // channel 1 -> symbol 0
                1.0, 1.0, 1.0, 1.0, // channel 2 -> symbol 15
            ],
            vec![4, 2],
        ),
        16,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(out.shape(), [1, 2]);
    assert_complex_close(&out.materialize_f64(), &[(-3.0, 3.0), (1.0, -1.0)]);
}

#[test]
fn qammod_reads_typed_integer_bit_storage_exactly() {
    let out = qammod(
        integer_tensor(
            IntegerStorage::U8(vec![
                0, 0, 0, 0, // channel 1 -> symbol 0
                1, 1, 1, 1, // channel 2 -> symbol 15
            ]),
            vec![4, 2],
        ),
        16,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(out.shape(), [1, 2]);
    assert_complex_close(&out.materialize_f64(), &[(-3.0, 3.0), (1.0, -1.0)]);
}

#[test]
fn qammod_bit_input_honors_binary_mapping() {
    let out = qammod(
        tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
        16,
        vec![
            Value::from("bin"),
            Value::from("InputType"),
            Value::from("bit"),
        ],
    );
    assert_eq!(out.shape(), [1, 1]);
    assert_complex_close(&out.materialize_f64(), &[(-3.0, -3.0)]);
}

#[test]
fn qammod_logical_bit_input_preserves_channels() {
    let logical = LogicalArray::new(vec![0, 0, 1, 1], vec![2, 2]).expect("logical");
    let out = qammod(
        Value::LogicalArray(logical),
        4,
        vec![Value::from("InputType"), Value::from("bit")],
    );
    assert_eq!(out.shape(), [1, 2]);
    assert_complex_close(&out.materialize_f64(), &[(-1.0, 1.0), (1.0, -1.0)]);
}

#[test]
fn qammod_inprocess_gpu_bit_input_returns_complex_resident_output() {
    use runmat_accelerate_api::{GpuTensorStorage, HostTensorView};

    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]).unwrap();
        let expected = qammod(
            Value::Tensor(input.clone()),
            16,
            vec![Value::from("InputType"), Value::from("bit")],
        );
        let input_handle = provider
            .upload(&HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            })
            .expect("upload");
        let result = block_on(super::qammod_builtin(
            Value::GpuTensor(input_handle.clone()),
            Value::Num(16.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("InputType"),
                Value::from("bit"),
            ]),
        ))
        .expect("gpu qammod bit");
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
fn qammod_gpu_enforces_integer_symbol_classes_and_output_precision() {
    test_support::with_test_provider(|provider| {
        let accepted =
            Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 2, 3]), vec![1, 4]).unwrap();
        let accepted_handle =
            gpu_helpers::upload_tensor(provider, &accepted).expect("accepted integer upload");
        let result = block_on(super::qammod_builtin(
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
            Tensor::new_integer(IntegerStorage::U32(vec![0, 1, 2, 3]), vec![1, 4]).unwrap();
        let rejected_handle =
            gpu_helpers::upload_tensor(provider, &rejected).expect("rejected integer upload");
        let err = block_on(super::qammod_builtin(
            Value::GpuTensor(rejected_handle),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("unsupported integer gpuArray class");
        assert!(err.message().contains("int8, int16, uint8, or uint16"));

        let single = Tensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap();
        let single_handle = gpu_helpers::upload_tensor(provider, &single).expect("single upload");
        let result = block_on(super::qammod_builtin(
            Value::GpuTensor(single_handle),
            Value::Num(4.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("single gpuArray");
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
fn qammod_bit_input_rejects_partial_symbols() {
    let err = block_on(super::qammod_builtin(
        tensor(vec![0.0, 1.0, 0.0], vec![3, 1]),
        Value::Num(16.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("InputType"),
            Value::from("bit"),
        ]),
    ))
    .expect_err("expected bit grouping error");
    assert!(err.to_string().contains("multiple of log2(M)"));
}

#[test]
fn qammod_bit_input_rejects_non_bits() {
    let err = block_on(super::qammod_builtin(
        tensor(vec![0.0, 2.0, 0.0, 1.0], vec![4, 1]),
        Value::Num(16.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("InputType"),
            Value::from("bit"),
        ]),
    ))
    .expect_err("expected bit value error");
    assert!(err.to_string().contains("bit values must be 0 or 1"));
}

#[test]
fn qammod_custom_mapping_uses_columnwise_symbol_order() {
    let mapping = tensor(vec![0.0, 3.0, 1.0, 2.0], vec![1, 4]);
    let out = qammod(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
        4,
        vec![mapping],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)],
    );
}

#[test]
fn qammod_reads_typed_integer_custom_mapping_exactly() {
    let mapping = integer_tensor(IntegerStorage::U16(vec![0, 3, 1, 2]), vec![1, 4]);
    let out = qammod(
        integer_tensor(IntegerStorage::U8(vec![0, 1, 2, 3]), vec![1, 4]),
        4,
        vec![mapping],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)],
    );
}

#[test]
fn qammod_unit_average_power_normalizes_constellation() {
    let input: Vec<f64> = (0..16).map(|v| v as f64).collect();
    let out = qammod(
        tensor(input, vec![1, 16]),
        16,
        vec![Value::from("UnitAveragePower"), Value::Bool(true)],
    );
    let avg_power = out
        .materialize_f64()
        .iter()
        .map(|(re, im)| re * re + im * im)
        .sum::<f64>()
        / out.materialize_f64().len() as f64;
    assert!((avg_power - 1.0).abs() < 1e-12, "{avg_power}");
}

#[test]
fn qammod_output_datatype_single_rounds_samples() {
    let out = qammod(
        tensor(vec![0.0], vec![1, 1]),
        16,
        vec![
            Value::from("UnitAveragePower"),
            Value::Bool(true),
            Value::from("OutputDataType"),
            Value::from("single"),
        ],
    );
    let expected = (-3.0 / 10.0_f64.sqrt()) as f32 as f64;
    assert_eq!(out.materialize_f64()[0].0, expected);
    assert_eq!(out.materialize_f64()[0].1, -expected);
}

#[test]
fn qammod_single_input_defaults_to_single_computation() {
    let input = Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
    let out = qammod(
        input,
        16,
        vec![Value::from("UnitAveragePower"), Value::Bool(true)],
    );
    let expected = (-3.0 / 10.0_f64.sqrt()) as f32 as f64;
    assert_eq!(out.materialize_f64()[0], (expected, -expected));
}

#[test]
fn qammod_rejects_out_of_range_symbols() {
    let err = block_on(super::qammod_builtin(
        tensor(vec![0.0, 16.0], vec![1, 2]),
        Value::Num(16.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("expected out-of-range error");
    assert!(err.to_string().contains("range [0, 15]"));
}

#[test]
#[cfg(feature = "wgpu")]
fn qammod_wgpu_input_returns_complex_resident_output() {
    use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostTensorView};

    match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) {
        Ok(provider) => {
            let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).unwrap();
            let expected = qammod(Value::Tensor(input.clone()), 4, vec![]);
            let view = HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            };
            let input_handle = provider.upload(&view).expect("upload");
            let result = block_on(super::qammod_builtin(
                Value::GpuTensor(input_handle.clone()),
                Value::Num(4.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("gpu qammod");
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
            tracing::warn!("Skipping qammod_wgpu_input_returns_complex_resident_output: {err}");
        }
    }
    runmat_accelerate::simple_provider::register_inprocess_provider();
}

#[test]
#[cfg(feature = "wgpu")]
fn qammod_wgpu_bit_input_returns_complex_resident_output() {
    use runmat_accelerate_api::{AccelProvider, GpuTensorStorage, HostTensorView};

    match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) {
        Ok(provider) => {
            let input = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]).unwrap();
            let expected = qammod(
                Value::Tensor(input.clone()),
                16,
                vec![Value::from("InputType"), Value::from("bit")],
            );
            let view = HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            };
            let input_handle = provider.upload(&view).expect("upload");
            let result = block_on(super::qammod_builtin(
                Value::GpuTensor(input_handle.clone()),
                Value::Num(16.0),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("InputType"),
                    Value::from("bit"),
                ]),
            ))
            .expect("gpu qammod bit");
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
            tracing::warn!("Skipping qammod_wgpu_bit_input_returns_complex_resident_output: {err}");
        }
    }
    runmat_accelerate::simple_provider::register_inprocess_provider();
}

#[test]
fn qammod_integer_scalars_preserve_exact_symbol_bounds() {
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
fn qammod_typed_integer_tensors_preserve_exact_symbol_values() {
    let order = Tensor::new_integer(IntegerStorage::U64(vec![4]), vec![1, 1]).unwrap();
    assert_eq!(parse_modulation_order(&Value::Tensor(order)).unwrap(), 4);

    let offset = Tensor::new_integer(IntegerStorage::I16(vec![2]), vec![1, 1]).unwrap();
    assert_eq!(
        scalar_number(&Value::Tensor(offset), "offset").unwrap(),
        2.0
    );

    let out = qammod(
        integer_tensor(IntegerStorage::U16(vec![0, 1, 2, 3]), vec![1, 4]),
        4,
        vec![],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)],
    );

    let custom = integer_tensor(IntegerStorage::U16(vec![0, 3, 1, 2]), vec![1, 4]);
    let out = qammod(
        integer_tensor(IntegerStorage::U8(vec![0, 1, 2, 3]), vec![1, 4]),
        4,
        vec![custom],
    );
    assert_complex_close(
        &out.materialize_f64(),
        &[(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)],
    );

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
fn qammod_rejects_near_integer_controls_and_matrix_symbol_order() {
    assert!(number_to_symbol_with_name(1.0 + 5e-10, "X").is_err());
    assert!(number_to_symbol_with_name(4.0 + 5e-10, "M").is_err());
    assert!(value_as_bool(&Value::Num(1.0 - 5e-10), "InputType").is_err());
    let matrix = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    assert!(parse_symbol_mapping(&Value::Tensor(matrix), 4).is_err());
}
