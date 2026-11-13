use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, SymbolicExpr, Tensor};

fn diff_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    block_on(super::diff_builtin(None, value, rest))
}

#[test]
fn diff_typed_order_parser_preserves_platform_uint64_range() {
    assert_eq!(
        parse_order(&Value::Int(IntValue::U64(3))).expect("order"),
        Some(3)
    );
    match usize::try_from(u64::MAX) {
        Ok(expected) => assert_eq!(
            parse_order(&Value::Int(IntValue::U64(u64::MAX))).expect("uint64 order"),
            Some(expected)
        ),
        Err(_) => assert!(parse_order(&Value::Int(IntValue::U64(u64::MAX))).is_err()),
    }
    assert!(parse_order(&Value::Int(IntValue::I64(-1))).is_err());
}

#[test]
fn diff_typed_integer_tensor_order_and_dimension_parse_exactly() {
    let large = 9_007_199_254_740_993_u64;
    let order =
        Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).expect("typed order");
    assert_eq!(
        parse_order(&Value::Tensor(order)).expect("typed order"),
        Some(large as usize)
    );

    let dim = Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1]).expect("typed dim");
    assert_eq!(
        parse_dimension_arg(&Value::Tensor(dim)).expect("typed dim"),
        Some(3)
    );
}

#[test]
fn diff_typed_integer_tensor_order_and_dimension_reject_negative_values() {
    let order =
        Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative order");
    assert!(parse_order(&Value::Tensor(order)).is_err());

    let dim = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative dim");
    assert!(parse_dimension_arg(&Value::Tensor(dim)).is_err());
}

#[test]
fn diff_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = DIFF_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"B = diff(X)"));
    assert!(labels.contains(&"B = diff(X, n)"));
    assert!(labels.contains(&"B = diff(X, n, dim)"));
}

#[test]
fn diff_descriptor_errors_have_stable_codes() {
    assert!(DIFF_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == DIFF_ERROR_INVALID_ARGUMENT.code));
    assert!(DIFF_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == DIFF_ERROR_INVALID_INPUT.code));
    assert!(DIFF_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == DIFF_ERROR_INTERNAL.code));
    assert!(DIFF_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == DIFF_ERROR_TOO_MANY_OUTPUTS.code));
}

#[test]
fn diff_symbolic_function_with_explicit_variable() {
    let y = SymbolicExpr::function_reference("Y", vec!["X".to_string()]);

    let result = diff_builtin(
        Value::Symbolic(y),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Symbolic(
            SymbolicExpr::variable("X"),
        )]),
    )
    .expect("diff");

    assert_eq!(result.to_string(), "diff(Y(X), X)");
}

#[test]
fn diff_symbolic_function_accepts_order_before_variable() {
    let y = SymbolicExpr::function_reference("Y", vec!["X".to_string()]);

    let result = diff_builtin(
        Value::Symbolic(y),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::I32(2)),
            Value::Symbolic(SymbolicExpr::variable("X")),
        ]),
    )
    .expect("diff");

    assert_eq!(result.to_string(), "diff(Y(X), X, 2)");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_row_vector_default_dimension() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let result = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("diff");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 2]);
            assert_eq!(out.materialize_f64(), vec![3.0, 5.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn nonleading_axis_differences_emit_column_major_lanes_in_native_classes() {
    let values = vec![1., 10., 4., 30., 9., 60.];
    let double =
        diff_tensor_host(Tensor::new(values.clone(), vec![2, 3]).unwrap(), 1, Some(2)).unwrap();
    assert_eq!(double.materialize_f64(), vec![3., 20., 5., 30.]);
    let single = diff_tensor_host(
        Tensor::from_f32(
            values.iter().map(|value| *value as f32).collect(),
            vec![2, 3],
        )
        .unwrap(),
        2,
        Some(2),
    )
    .unwrap();
    assert_eq!(single.as_f32_slice().unwrap(), &[2., 10.]);
    let integer = diff_tensor_host(
        Tensor::new_integer(IntegerStorage::I16(vec![1, 10, 4, 30, 9, 60]), vec![2, 3]).unwrap(),
        1,
        Some(2),
    )
    .unwrap();
    assert_eq!(
        integer.integer_storage(),
        Some(&IntegerStorage::I16(vec![3, 20, 5, 30]))
    );
    let complex = ComplexTensor::from_f32(
        vec![
            (1., 2.),
            (10., 20.),
            (4., 8.),
            (30., 60.),
            (9., 18.),
            (60., 120.),
        ],
        vec![2, 3],
    )
    .unwrap();
    let complex = diff_complex_tensor(complex, 1, Some(2)).unwrap();
    let actual = complex
        .as_f32_slice()
        .unwrap()
        .iter()
        .map(|value| (value.0, value.1))
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![(3., 6.), (20., 40.), (5., 10.), (30., 60.)]);
}
#[test]
fn empty_payload_still_consumes_the_entire_selected_difference_order() {
    let input = Tensor::from_f32(vec![], vec![0, 5, 2]).unwrap();
    let result = diff_tensor_host(input, 2, Some(2)).unwrap();
    assert_eq!(result.shape(), &[0, 3, 2]);
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
    let input = ComplexTensor::from_f32(vec![], vec![0, 5, 2]).unwrap();
    let result = diff_complex_tensor(input, 3, Some(2)).unwrap();
    assert_eq!(result.shape(), &[0, 2, 2]);
    assert!(result.as_f32_slice().is_some());
    assert!(parse_numeric_order(usize::MAX as f64).is_err() || usize::BITS < 64);
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_column_vector_second_order() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
    let args = vec![Value::Int(IntValue::I32(2))];
    let result = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .expect("diff");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [2, 1]);
            assert_eq!(out.materialize_f64(), vec![2.0, 2.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_matrix_along_columns() {
    let tensor = Tensor::new(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap();
    let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))];
    let result = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .expect("diff");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [3, 1]);
            assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 1.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_handles_empty_when_order_exceeds_dimension() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let args = vec![Value::Int(IntValue::I32(5))];
    let result = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .expect("diff");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape()[0], 0);
            assert!(out.materialize_f64().is_empty());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn diff_high_order_does_not_cascade_to_later_dimensions() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("matrix");
    let output = diff_tensor_host(tensor, 3, None).expect("diff");
    assert_eq!(output.shape(), [0, 3]);
    assert!(output.materialize_f64().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_char_array_promotes_to_double() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let chars = CharArray::from_code_units(
        [0xD800, 0xD801, 0xDC00, 0xDC02]
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .to_vec(),
        1,
        4,
    )
    .unwrap();
    let result = diff_builtin(
        Value::CharArray(chars),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("diff");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(out.as_f64_slice(), Some(&[1.0, 1023.0, 2.0][..]));
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_complex_tensor_preserves_type() {
    let tensor = ComplexTensor::new(vec![(1.0, 1.0), (3.0, 2.0), (6.0, 5.0)], vec![1, 3]).unwrap();
    let result = diff_builtin(
        Value::ComplexTensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("diff");
    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape(), [1, 2]);
            assert_eq!(out.materialize_f64(), vec![(2.0, 1.0), (3.0, 3.0)]);
        }
        other => panic!("expected complex tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_zero_order_returns_input() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let args = vec![Value::Int(IntValue::I32(0))];
    let result = diff_builtin(
        Value::Tensor(tensor.clone()),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .expect("diff");
    assert_eq!(result, Value::Tensor(tensor));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_preserves_native_single_storage() {
    let tensor = Tensor::from_f32(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let result = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("diff");
    match result {
        Value::Tensor(output) => assert_eq!(
            output.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![3.0, 5.0])
        ),
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_preserves_all_exact_integer_classes_and_saturates() {
    let cases = [
        (
            IntegerStorage::I8(vec![i8::MIN, i8::MAX, i8::MAX]),
            IntegerStorage::I8(vec![i8::MAX, 0]),
        ),
        (
            IntegerStorage::I16(vec![i16::MAX, i16::MIN, i16::MIN]),
            IntegerStorage::I16(vec![i16::MIN, 0]),
        ),
        (
            IntegerStorage::I32(vec![i32::MIN, i32::MAX, i32::MAX]),
            IntegerStorage::I32(vec![i32::MAX, 0]),
        ),
        (
            IntegerStorage::I64(vec![i64::MAX, i64::MIN, i64::MIN]),
            IntegerStorage::I64(vec![i64::MIN, 0]),
        ),
        (
            IntegerStorage::U8(vec![5, 0, 0]),
            IntegerStorage::U8(vec![0, 0]),
        ),
        (
            IntegerStorage::U16(vec![0, u16::MAX, u16::MAX]),
            IntegerStorage::U16(vec![u16::MAX, 0]),
        ),
        (
            IntegerStorage::U32(vec![4, 0, 0]),
            IntegerStorage::U32(vec![0, 0]),
        ),
        (
            IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, u64::MAX - 2]),
            IntegerStorage::U64(vec![1, 0]),
        ),
    ];

    for (storage, expected) in cases {
        let input = Tensor::new_integer(storage, vec![1, expected.len() + 1]).unwrap();
        let result = diff_builtin(
            Value::Tensor(input),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("diff");
        match result {
            Value::Tensor(output) => assert_eq!(output.integer_storage(), Some(&expected)),
            other => panic!("expected exact integer tensor, got {other:?}"),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_exact_integer_order_dimension_and_empty_output_preserve_storage() {
    let input = Tensor::new_integer(IntegerStorage::I64(vec![1, 4, 10]), vec![3, 1]).unwrap();
    let result = diff_builtin(
        Value::Tensor(input),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            2,
        ))]),
    )
    .expect("second order diff");
    assert_eq!(result, Value::Int(IntValue::I64(3)));

    let input = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX, u64::MAX - 1]),
        vec![2, 1],
    )
    .unwrap();
    let result = diff_builtin(
        Value::Tensor(input),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::I32(1)),
            Value::Int(IntValue::I32(5)),
        ]),
    )
    .expect("trailing dimension diff");
    match result {
        Value::Tensor(output) => {
            assert_eq!(output.shape(), [2, 1, 1, 1, 0]);
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U64(Vec::new()))
            );
        }
        other => panic!("expected typed empty tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_accepts_empty_order_argument() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
    let baseline = diff_builtin(
        Value::Tensor(tensor.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("diff");
    let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
    let result = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(empty)]),
    )
    .expect("diff");
    assert_eq!(result, baseline);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_accepts_empty_dimension_argument() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![1, 4]).unwrap();
    let baseline = diff_builtin(
        Value::Tensor(tensor.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            1,
        ))]),
    )
    .expect("diff");
    let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
    let result = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::I32(1)),
            Value::Tensor(empty),
        ]),
    )
    .expect("diff");
    assert_eq!(result, baseline);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_rejects_negative_order() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let args = vec![Value::Int(IntValue::I32(-1))];
    let err = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), DIFF_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("non-negative"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_rejects_non_integer_order() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let args = vec![Value::Num(1.5)];
    let err = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), DIFF_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("non-negative integer"));
}

#[test]
fn diff_rejects_logical_order_but_accepts_logical_data() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
    let error = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
    )
    .expect_err("logical order is unsupported");
    assert_eq!(error.identifier(), DIFF_ERROR_INVALID_ARGUMENT.identifier);

    match diff_builtin(
        Value::Bool(true),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("logical data")
    {
        Value::Tensor(output) => assert!(output.materialize_f64().is_empty()),
        other => panic!("expected empty tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_rejects_invalid_dimension() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(0))];
    let err = diff_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), DIFF_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("dimension must be >= 1"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn diff_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = diff_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("diff");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), [2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![3.0, 5.0]);
    });
}

#[test]
fn diff_resident_complex_uses_owner_resolved_exact_fallback() {
    test_support::with_test_provider(|provider| {
        let tensor = ComplexTensor::new(vec![(1.0, 1.0), (3.0, 2.0), (6.0, 5.0)], vec![1, 3])
            .expect("complex double");
        let input = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
        let output = diff_builtin(
            Value::GpuTensor(input.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("diff");
        let Value::GpuTensor(handle) = &output else {
            panic!("expected resident result")
        };
        assert_eq!(handle.shape(), vec![1, 2]);
        assert_eq!(handle.device_id(), input.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_storage(handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(handle),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        match block_on(gpu_helpers::gather_value_async(&output)).expect("gather") {
            Value::ComplexTensor(result) => {
                assert_eq!(result.materialize_f64(), vec![(2.0, 1.0), (3.0, 3.0)])
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn diff_wgpu_matches_cpu() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
    let args = vec![Value::Int(IntValue::I32(2))];

    let cpu_result = diff_builtin(
        Value::Tensor(tensor.clone()),
        crate::builtins::common::test_support::builtin_arguments(args.clone()),
    )
    .expect("diff");
    let expected = match cpu_result {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };

    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let gpu_value = diff_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .expect("diff");
    let gathered = test_support::gather(gpu_value).expect("gather");

    assert_eq!(gathered.shape(), expected.shape());
    let tol = if matches!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F32
    ) {
        1e-5
    } else {
        1e-12
    };
    for (a, b) in gathered
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
    }
}
