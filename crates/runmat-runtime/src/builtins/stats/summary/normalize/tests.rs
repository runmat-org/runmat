use super::*;
use crate::builtins::common::gpu_helpers;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage};

fn call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(normalize_builtin(
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn complex_int_tensor(real: IntegerStorage, imag: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor =
        ComplexTensor::new_integer(IntegerComplexStorage::new(real, imag).unwrap(), shape).unwrap();
    Value::ComplexTensor(tensor)
}

fn expect_tensor(value: Value) -> Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn default_zscore_normalizes_first_non_singleton_dimension() {
    let out = expect_tensor(call(tensor(vec![1., 2., 3.], vec![3, 1]), vec![]).unwrap());
    assert_close(out.materialize_f64()[0], -1.0);
    assert_close(out.materialize_f64()[1], 0.0);
    assert_close(out.materialize_f64()[2], 1.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn normalize_preserves_rank_zero_and_rank_one_input_boundaries() {
    let NumericInput::Real { shape, data } =
        block_on(value_to_input(tensor(vec![7.0], Vec::new()))).unwrap()
    else {
        panic!("expected real scalar input");
    };
    assert_eq!(shape, vec![1, 1]);
    assert_eq!(data, vec![7.0]);
    assert!(matches!(
        call(tensor(vec![7.0], Vec::new()), Vec::new()).unwrap(),
        Value::Num(value) if value.is_nan()
    ));

    let NumericInput::Real { shape, data } =
        block_on(value_to_input(tensor(vec![1.0, 2.0, 3.0], vec![3]))).unwrap()
    else {
        panic!("expected real rank-one input");
    };
    assert_eq!(shape, vec![3]);
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
    let out = expect_tensor(call(tensor(vec![1.0, 2.0, 3.0], vec![3]), Vec::new()).unwrap());
    assert_eq!(out.shape(), [3, 1]);
    assert_eq!(out.materialize_f64(), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn normalization_keyword_roles_preserve_public_diagnostics() {
    let input = || tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let accepted = call(
        input(),
        vec![
            Value::from("CENTER"),
            Value::from("MEAN"),
            Value::from("SCALE"),
            Value::from("STD"),
        ],
    )
    .expect("case-insensitive center and scale tokens");
    assert_eq!(expect_tensor(accepted).shape(), [3, 1]);

    for (arguments, expected) in [
        (vec![Value::from("unknown")], "unsupported method 'unknown'"),
        (
            vec![Value::from("ZSCORE"), Value::from("mystery")],
            "unsupported zscore method type 'mystery'",
        ),
        (
            vec![Value::from("CENTER"), Value::from("mystery")],
            "unsupported center type 'mystery'",
        ),
        (
            vec![Value::from("SCALE"), Value::from("mystery")],
            "unsupported scale type 'mystery'",
        ),
        (
            vec![Value::from("DataVariables")],
            "table name-value options are not supported",
        ),
    ] {
        let error = call(input(), arguments).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:normalize:InvalidArgument"));
        assert!(error.message().contains(expected), "{}", error.message());
    }
}

#[test]
fn normalize_reads_typed_integer_input_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = expect_tensor(
        call(
            int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
            vec![],
        )
        .unwrap(),
    );
    assert_close(out.materialize_f64()[0], -1.0);
    assert_close(out.materialize_f64()[1], 0.0);
    assert_close(out.materialize_f64()[2], 1.0);
}

#[test]
fn normalize_integer_data_gates_and_wide_values_reject() {
    let data = int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]);
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(data, Vec::new()).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NormalizeIntegerDataExtension")
        );
    }
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
    let error = call(wide, Vec::new()).unwrap_err();
    assert!(error.message.contains("exactly representable"));
}

#[test]
fn exact_wide_integer_boundaries_apply_to_scalar_dense_and_complex_storage() {
    let exact = 1_u64 << 54;
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let NumericInput::Real { data, .. } =
        block_on(value_to_input(Value::Int(IntValue::U64(exact)))).unwrap()
    else {
        panic!("expected real scalar input");
    };
    assert_eq!(data, vec![exact as f64]);

    let NumericInput::Real { data, .. } = block_on(value_to_input(mirrorless_int_tensor(
        IntegerStorage::I64(vec![i64::MIN, exact as i64]),
        vec![2, 1],
    )))
    .unwrap() else {
        panic!("expected real dense input");
    };
    assert_eq!(data, vec![i64::MIN as f64, exact as f64]);

    let NumericInput::Complex { data, .. } = block_on(value_to_input(complex_int_tensor(
        IntegerStorage::I64(vec![i64::MIN]),
        IntegerStorage::I64(vec![exact as i64]),
        vec![1, 1],
    )))
    .unwrap() else {
        panic!("expected complex input");
    };
    assert_eq!(data, vec![(i64::MIN as f64, exact as f64)]);

    for value in [
        Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        mirrorless_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
        complex_int_tensor(
            IntegerStorage::I64(vec![i64::MIN]),
            IntegerStorage::I64(vec![((1_u64 << 53) + 1) as i64]),
            vec![1, 1],
        ),
    ] {
        assert!(block_on(value_to_input(value))
            .err()
            .expect("inexact integer boundary must reject")
            .message()
            .contains("exactly representable"));
    }
}

#[test]
fn structural_dimension_is_not_forced_through_the_floating_boundary() {
    let input = tensor(vec![1.0, 2.0], vec![2, 1]);
    let error = call(input, vec![Value::Int(IntValue::U64(1_u64 << 54))]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:normalize:InvalidArgument"));
    assert!(error.message().contains("dimension must be <="));
    assert!(!error.message().contains("exactly representable"));

    let output = call(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        vec![Value::from("range"), Value::Int(IntValue::U64(1_u64 << 54))],
    )
    .expect("exact wide range boundary");
    assert_eq!(
        expect_tensor(output).materialize_f64(),
        vec![0.0, (1_u64 << 54) as f64]
    );
}

#[test]
fn normalize_single_input_preserves_single_outputs() {
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let Value::OutputList(values) = call(input, Vec::new()).unwrap() else {
        panic!("expected three outputs");
    };
    for value in values.try_into_owned().expect("unique outputs") {
        let Value::Tensor(tensor) = value else {
            panic!("single normalize outputs must be tensors");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    }
}

#[test]
fn normalize_restores_only_requested_resident_output_forms() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        for requested in 0..=3 {
            let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            let source = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let _outputs = crate::output_count::push_output_count(Some(requested))
                .expect("output-count scope");
            let Value::OutputList(outputs) =
                call(Value::GpuTensor(source.clone()), Vec::new()).unwrap()
            else {
                panic!("normalize must return the selected output list");
            };
            assert_eq!(outputs.len(), requested);
            for output in outputs.try_into_owned().expect("unique outputs") {
                let Value::GpuTensor(result) = output else {
                    panic!("selected normalize output must retain explicit residency");
                };
                assert!(runmat_accelerate_api::handle_is_explicit(&result));
                assert_eq!(result.device_id(), source.device_id());
                assert!(!result.shares_allocation_with(&source));
            }
            source.allocation_lease().validate_live().unwrap();
        }
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn normalize_wgpu_integer_fallback_preserves_class_and_explicit_intent() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("wgpu provider");
    let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
    let out = call(Value::GpuTensor(handle.clone()), Vec::new()).expect("normalize");
    let Value::GpuTensor(out) = out else {
        panic!("expected automatic resident double output");
    };
    let host = crate::builtins::common::test_support::gather(Value::GpuTensor(out))
        .expect("gather automatic output");
    assert_eq!(host.numeric_dtype(), NumericDType::F64);
    assert_close(host.materialize_f64()[0], -1.0);
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let out = call(Value::GpuTensor(handle), Vec::new()).expect("explicit resident double output");
    let Value::GpuTensor(out) = out else {
        panic!("expected explicit resident output");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&out));
    assert_eq!(
        runmat_accelerate_api::handle_precision(&out),
        Some(runmat_accelerate_api::ProviderPrecision::F64)
    );
}

#[test]
fn normalize_real_input_shape_uses_typed_integer_storage_not_mirror() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = expect_tensor(
        call(
            mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]),
            vec![],
        )
        .unwrap(),
    );
    assert_eq!(out.shape(), [3, 1]);
    assert_close(out.materialize_f64()[0], -1.0);
    assert_close(out.materialize_f64()[1], 0.0);
    assert_close(out.materialize_f64()[2], 1.0);
}

#[test]
fn dim_argument_normalizes_rows_independently() {
    let out = expect_tensor(
        call(
            tensor(vec![1., 2., 3., 5., 5., 9.], vec![3, 2]),
            vec![Value::Num(2.0)],
        )
        .unwrap(),
    );
    assert_eq!(out.shape(), [3, 2]);
    assert_close(out.materialize_f64()[0], -std::f64::consts::FRAC_1_SQRT_2);
    assert_close(out.materialize_f64()[3], std::f64::consts::FRAC_1_SQRT_2);
}

#[test]
fn range_method_maps_to_requested_bounds() {
    let out = expect_tensor(
        call(
            tensor(vec![2., 4., 6.], vec![3, 1]),
            vec![
                Value::from("range"),
                Value::Tensor(Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap()),
            ],
        )
        .unwrap(),
    );
    assert_eq!(out.materialize_f64(), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn normalize_reads_typed_integer_range_bounds_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = expect_tensor(
        call(
            tensor(vec![2., 4., 6.], vec![3, 1]),
            vec![
                Value::from("range"),
                int_tensor(IntegerStorage::I16(vec![-1, 1]), vec![1, 2]),
            ],
        )
        .unwrap(),
    );
    assert_eq!(out.materialize_f64(), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn norm_method_accepts_p_norm() {
    let out = expect_tensor(
        call(
            tensor(vec![3., 4.], vec![2, 1]),
            vec![Value::from("norm"), Value::Num(2.0)],
        )
        .unwrap(),
    );
    assert_close(out.materialize_f64()[0], 0.6);
    assert_close(out.materialize_f64()[1], 0.8);
}

#[test]
fn center_scale_outputs_can_be_reused() {
    let values = {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = call(
            tensor(vec![1., 2., 3., 5.], vec![2, 2]),
            vec![
                Value::Num(1.0),
                Value::from("center"),
                Value::from("mean"),
                Value::from("scale"),
                Value::from("std"),
            ],
        )
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected outputs");
        };
        values
    };
    assert_eq!(values.len(), 3);
    let n = expect_tensor(values[0].clone());
    let c = expect_tensor(values[1].clone());
    let s = expect_tensor(values[2].clone());
    assert_eq!(n.shape(), [2, 2]);
    assert_eq!(c.shape(), [1, 2]);
    assert_eq!(s.shape(), [1, 2]);
    let reused = expect_tensor(
        call(
            tensor(vec![1., 2., 3., 5.], vec![2, 2]),
            vec![
                Value::from("center"),
                values[1].clone(),
                Value::from("scale"),
                values[2].clone(),
            ],
        )
        .unwrap(),
    );
    assert_eq!(n.materialize_f64(), reused.materialize_f64());
}

#[test]
fn normalize_reads_typed_integer_explicit_center_and_scale_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = expect_tensor(
        call(
            int_tensor(IntegerStorage::I16(vec![2, 4, 6]), vec![3, 1]),
            vec![
                Value::from("center"),
                int_tensor(IntegerStorage::I16(vec![2]), vec![1, 1]),
                Value::from("scale"),
                int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
            ],
        )
        .unwrap(),
    );
    assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 2.0]);
}

#[test]
fn normalize_explicit_param_shape_uses_typed_integer_storage_not_mirror() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = expect_tensor(
        call(
            mirrorless_int_tensor(IntegerStorage::I16(vec![2, 4, 6]), vec![3, 1]),
            vec![
                Value::from("center"),
                mirrorless_int_tensor(IntegerStorage::I16(vec![2, 2, 2]), vec![3, 1]),
                Value::from("scale"),
                mirrorless_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
            ],
        )
        .unwrap(),
    );
    assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 2.0]);
}

#[test]
fn zero_scale_output_can_be_reused() {
    let values = {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = call(tensor(vec![4., 4.], vec![2, 1]), vec![]).unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected outputs");
        };
        values
    };
    let s = expect_tensor(values[2].clone());
    assert_eq!(s.materialize_f64(), vec![0.0]);

    let reused = expect_tensor(
        call(
            tensor(vec![4., 4.], vec![2, 1]),
            vec![
                Value::from("center"),
                values[1].clone(),
                Value::from("scale"),
                values[2].clone(),
            ],
        )
        .unwrap(),
    );
    assert!(reused.materialize_f64().iter().all(|value| value.is_nan()));
}

#[test]
fn explicit_nan_scale_is_accepted_and_propagates() {
    let out = expect_tensor(
        call(
            tensor(vec![1., 2.], vec![2, 1]),
            vec![Value::from("scale"), Value::Num(f64::NAN)],
        )
        .unwrap(),
    );
    assert!(out.materialize_f64().iter().all(|value| value.is_nan()));
}

#[test]
fn empty_inputs_preserve_empty_shapes() {
    let values = {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = call(tensor(vec![], vec![0, 3]), vec![]).unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected outputs");
        };
        values
    };
    let n = expect_tensor(values[0].clone());
    let c = expect_tensor(values[1].clone());
    let s = expect_tensor(values[2].clone());
    assert_eq!(n.shape(), [0, 3]);
    assert_eq!(c.shape(), [0, 1]);
    assert_eq!(s.shape(), [0, 1]);
    assert!(n.materialize_f64().is_empty());
    assert!(c.materialize_f64().is_empty());
    assert!(s.materialize_f64().is_empty());
}

#[test]
fn nan_values_are_omitted_from_parameters_and_preserved_in_output() {
    let out = expect_tensor(
        call(
            tensor(vec![1., f64::NAN, 3.], vec![3, 1]),
            vec![Value::from("center"), Value::from("mean")],
        )
        .unwrap(),
    );
    assert_eq!(out.materialize_f64()[0], -1.0);
    assert!(out.materialize_f64()[1].is_nan());
    assert_eq!(out.materialize_f64()[2], 1.0);
}

#[test]
fn complex_norm_uses_magnitudes_but_preserves_phase() {
    let input = Value::ComplexTensor(
        ComplexTensor::new(vec![(3.0, 4.0), (0.0, 12.0)], vec![2, 1]).unwrap(),
    );
    let Value::ComplexTensor(out) =
        call(input, vec![Value::from("norm"), Value::Num(2.0)]).unwrap()
    else {
        panic!("expected complex output");
    };
    assert_close(out.materialize_f64()[0].0, 3.0 / 13.0);
    assert_close(out.materialize_f64()[0].1, 4.0 / 13.0);
    assert_close(out.materialize_f64()[1].1, 12.0 / 13.0);
}

#[test]
fn normalize_reads_typed_complex_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = complex_int_tensor(
        IntegerStorage::I16(vec![3, 0]),
        IntegerStorage::I16(vec![4, 12]),
        vec![2, 1],
    );
    let Value::ComplexTensor(out) =
        call(input, vec![Value::from("norm"), Value::Num(2.0)]).unwrap()
    else {
        panic!("expected complex output");
    };
    assert_close(out.materialize_f64()[0].0, 3.0 / 13.0);
    assert_close(out.materialize_f64()[0].1, 4.0 / 13.0);
    assert_close(out.materialize_f64()[1].0, 0.0);
    assert_close(out.materialize_f64()[1].1, 12.0 / 13.0);
}

#[test]
fn normalize_reuses_typed_complex_integer_center_from_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input =
        Value::ComplexTensor(ComplexTensor::new(vec![(3.0, 4.0), (5.0, 8.0)], vec![2, 1]).unwrap());
    let center = complex_int_tensor(
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I16(vec![2]),
        vec![1, 1],
    );
    let Value::ComplexTensor(out) = call(
        input,
        vec![
            Value::from("center"),
            center,
            Value::from("scale"),
            Value::Num(2.0),
        ],
    )
    .unwrap() else {
        panic!("expected complex output");
    };
    assert_eq!(out.shape(), [2, 1]);
    assert_eq!(out.materialize_f64(), vec![(1.0, 1.0), (2.0, 3.0)]);
}

#[test]
fn rejects_multiple_top_level_methods() {
    let err = call(
        tensor(vec![1., 2., 3.], vec![3, 1]),
        vec![
            Value::from("center"),
            Value::from("mean"),
            Value::from("range"),
        ],
    )
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
}

#[test]
fn rejects_unbounded_dimension() {
    let err = call(
        tensor(vec![1., 2.], vec![2, 1]),
        vec![Value::Num((MAX_NORMALIZE_DIM + 1) as f64)],
    )
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
}

#[test]
fn rejects_complex_explicit_center_for_real_input() {
    let err = call(
        tensor(vec![1., 2.], vec![2, 1]),
        vec![
            Value::from("center"),
            Value::Complex(1.0, 1.0),
            Value::from("scale"),
            Value::Num(1.0),
        ],
    )
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
}

#[test]
fn rejects_table_only_name_value_options_for_arrays() {
    let err = call(
        tensor(vec![1., 2.], vec![2, 1]),
        vec![Value::from("DataVariables"), Value::from("x")],
    )
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:normalize:InvalidArgument"));
}
