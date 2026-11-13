use super::*;

#[test]
fn logical_sparse_complex_rejects_even_zero_imaginary_storage() {
    let sparse = SparseTensor::new_complex(1, 1, vec![0, 1], vec![0], vec![(1.0, 0.0)]).unwrap();
    let error = logical_builtin(Value::SparseTensor(sparse), vec![]).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:logical:ConversionNotPossible")
    );
}

#[test]
fn logical_empty_character_pages_preserve_geometry() {
    let shape = vec![0, 3, 2];
    let chars = CharArray::from_unicode_scalars_with_shape(vec![], shape.clone()).unwrap();
    let Value::LogicalArray(result) = logical_builtin(Value::CharArray(chars), vec![]).unwrap()
    else {
        panic!("logical array")
    };
    assert_eq!(result.shape, shape);
    assert!(result.data.is_empty());
}

#[test]
fn logical_provider_output_validation_rejects_alias_shape_and_storage() {
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{GpuTensorStorage, ThreadProviderGuard};
    let _state = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(InProcessProvider::new()));
    let _provider = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[0.0, 1.0],
            shape: &[1, 2],
        })
        .unwrap();
    let output = provider
        .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    assert!(valid_logical_gpu_output(&output, &input, provider));
    assert!(!valid_logical_gpu_output(&input, &input, provider));
    assert!(!valid_logical_gpu_output(&output, &output, provider));
    let numeric_mask = provider
        .upload(&HostTensorView {
            data: &[0.0, 1.0],
            shape: &[1, 2],
        })
        .unwrap();
    assert!(
        !valid_logical_gpu_output(&numeric_mask, &input, provider),
        "boolean-looking numeric storage is not producer-admitted logical"
    );
    let wrong_shape = output
        .with_view(output.view().clone().with_shape(vec![2, 1]))
        .unwrap();
    assert!(!valid_logical_gpu_output(&wrong_shape, &input, provider));
    let complex = provider
        .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F64(&[0.0, 0.0, 1.0, 0.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    assert!(!valid_logical_gpu_output(&complex, &input, provider));
}

#[test]
fn logical_symbolic_extension_gate_precedes_conversion() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = logical_builtin(Value::Symbolic(SymbolicExpr::constant(2.0)), vec![]).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:LogicalSymbolicConstantInputExtension")
    );
}

#[test]
fn logical_character_pages_preserve_shape_and_column_major_order() {
    let shape = vec![2, 2, 2];
    let chars = CharArray::from_unicode_scalars_with_shape(
        vec!['A', '\0', 'B', 'C', '\0', 'D', '\0', 'E'],
        shape.clone(),
    )
    .unwrap();
    let Value::LogicalArray(result) = logical_builtin(Value::CharArray(chars), vec![]).unwrap()
    else {
        panic!("logical array")
    };
    assert_eq!(result.shape, shape);
    assert_eq!(result.data, vec![1, 1, 0, 1, 0, 0, 1, 1]);
}

#[test]
fn logical_output_arity_precedes_input_conversion_and_discard_still_validates() {
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = logical_builtin(Value::Num(f64::NAN), vec![]).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:logical:TooManyOutputs"));
    }
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = logical_builtin(Value::Num(f64::NAN), vec![]).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:logical:ConversionNotPossible")
    );
}
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{
    CellArray, IntValue, IntegerComplexStorage, IntegerStorage, MException, ObjectInstance,
    SparseTensor, SymbolicExpr,
};

fn logical_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    // This test-only terminal projection is confined to logical's scalar,
    // array, sparse-numeric, and external GPU outputs; it cannot export a
    // managed GC child after the canonical owner retires.
    block_on(super::logical_builtin(
        crate::context::legacy::active(),
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .map(|output| output.with_owned(|value| value))
}

fn assert_error_message(err: &crate::RuntimeError, expected: &str) {
    assert_eq!(err.message(), expected);
}

fn assert_error_contains(err: &crate::RuntimeError, expected: &str) {
    assert!(
        err.message().contains(expected),
        "unexpected error: {}",
        err.message()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_scalar_num() {
    let result = logical_builtin(Value::Num(5.0), Vec::new()).expect("logical");
    assert_eq!(result, Value::Bool(true));

    let zero_result = logical_builtin(Value::Num(0.0), Vec::new()).expect("logical");
    assert_eq!(zero_result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_converts_symbolic_constants() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let nonzero =
        logical_builtin(Value::Symbolic(SymbolicExpr::constant(2.0)), Vec::new()).expect("logical");
    assert_eq!(nonzero, Value::Bool(true));

    let zero =
        logical_builtin(Value::Symbolic(SymbolicExpr::constant(0.0)), Vec::new()).expect("logical");
    assert_eq!(zero, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_rejects_symbolic_variables() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let err = logical_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
        .expect_err("symbolic variable should not convert");

    assert_eq!(
        err.identifier(),
        LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
    );
    assert!(err.message().contains("logical from sym"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_rejects_nan() {
    let tensor = Tensor::new(vec![0.0, f64::NAN, -0.0], vec![1, 3]).unwrap();
    let error =
        logical_builtin(Value::Tensor(tensor), Vec::new()).expect_err("NaN conversion must fail");
    assert!(error.message().contains("NaN"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_tensor_matrix() {
    let tensor = Tensor::new(vec![0.0, 2.0, -3.0, 0.0], vec![2, 2]).unwrap();
    let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 2]);
            assert_eq!(array.data, vec![0, 1, 1, 0]);
        }
        other => panic!("expected logical array, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_sparse_tensor_preserves_sparse_storage() {
    let sparse = SparseTensor::new(3, 2, vec![0, 1, 2], vec![1, 2], vec![4.0, -1.0]).unwrap();
    let result = logical_builtin(Value::SparseTensor(sparse), Vec::new()).expect("logical");
    match result {
        Value::SparseTensor(sparse) => {
            assert!(sparse.is_logical());
            assert_eq!(sparse.shape(), vec![3, 2]);
            assert_eq!(sparse.col_ptrs, vec![0, 1, 2]);
            assert_eq!(sparse.row_indices, vec![1, 2]);
            assert_eq!(
                sparse.to_dense_logical().expect("dense logical").data,
                vec![0, 1, 0, 0, 0, 1]
            );
        }
        other => panic!("expected logical sparse tensor, got {other:?}"),
    }
}

#[test]
fn logical_sparse_nan_rejects_like_dense_nan() {
    let sparse = SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap();
    let error = logical_builtin(Value::SparseTensor(sparse), Vec::new())
        .expect_err("sparse NaN must reject");
    assert_eq!(
        error.identifier(),
        Some("RunMat:logical:ConversionNotPossible")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_rejects_complex_conversion() {
    let complex = ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)], vec![3, 1]).unwrap();
    let error = logical_builtin(Value::ComplexTensor(complex), Vec::new())
        .expect_err("complex conversion must fail");
    assert!(error.message().contains("complex"));
}

#[test]
fn logical_rejects_typed_complex_integer_components() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![0, u64::MAX, 0]),
        IntegerStorage::U64(vec![0, 0, 1_u64 << 63]),
    )
    .expect("matching components");
    let tensor = ComplexTensor::new_integer(storage, vec![3, 1]).expect("typed complex");

    let error = logical_builtin(Value::ComplexTensor(tensor), Vec::new())
        .expect_err("complex conversion must fail");
    assert!(error.message().contains("complex"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_char_array_conversion() {
    let chars = CharArray::from_unicode_scalars(vec!['A', '\0', 'C'], 1, 3).unwrap();
    let result = logical_builtin(Value::CharArray(chars), Vec::new()).expect("logical");
    match result {
        Value::LogicalArray(array) => assert_eq!(array.data, vec![1, 0, 1]),
        other => panic!("expected logical array, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_string_error() {
    let err = logical_builtin(Value::String("runmat".to_string().into()), Vec::new()).unwrap_err();
    assert_error_message(
        &err,
        "logical: conversion to logical from string is not possible",
    );
    assert_eq!(
        err.identifier(),
        LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_struct_error() {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert("field", Value::Num(1.0));
    let err = logical_builtin(Value::Struct(st.finish()), Vec::new()).unwrap_err();
    assert_error_contains(&err, "struct");
    assert_eq!(
        err.identifier(),
        LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_cell_error() {
    let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell creation");
    let err = logical_builtin(Value::Cell(cell), Vec::new()).unwrap_err();
    assert_error_message(
        &err,
        "logical: conversion to logical from cell is not possible",
    );
    assert_eq!(
        err.identifier(),
        LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_function_handle_error() {
    let err = logical_builtin(Value::FunctionHandle("foo".into()), Vec::new()).unwrap_err();
    assert_error_message(
        &err,
        "logical: conversion to logical from function_handle is not possible",
    );
    assert_eq!(
        err.identifier(),
        LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_object_error() {
    let obj = ObjectInstance::new("DemoClass".to_string());
    let err = logical_builtin(Value::Object(obj), Vec::new()).unwrap_err();
    assert_error_contains(&err, "DemoClass");
    assert_eq!(
        err.identifier(),
        LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_mexception_error() {
    let mex = MException::new("id:logical".into(), "message".into());
    let err = logical_builtin(Value::MException(mex), Vec::new()).unwrap_err();
    assert_error_message(
        &err,
        "logical: conversion to logical from MException is not possible",
    );
    assert_eq!(
        err.identifier(),
        LOGICAL_ERROR_CONVERSION_NOT_POSSIBLE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_too_many_inputs_error() {
    let err = logical_builtin(Value::Bool(true), vec![Value::Bool(false)]).unwrap_err();
    assert_error_message(&err, LOGICAL_ERROR_TOO_MANY_INPUTS.message);
    assert_eq!(err.identifier(), LOGICAL_ERROR_TOO_MANY_INPUTS.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            logical_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("logical");
        let gathered = test_support::gather(result.clone()).expect("gather");
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 1.0]);
        if let Value::GpuTensor(out) = result {
            assert!(runmat_accelerate_api::handle_is_logical(&out));
        } else {
            panic!("expected gpu tensor output");
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_gpu_passthrough_for_logical_handle() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1],
                shape: &[2, 1],
            })
            .expect("logical producer upload");
        let result =
            logical_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("logical");
        match result {
            Value::GpuTensor(out) => assert_eq!(out, handle),
            other => panic!("expected gpu tensor, got {:?}", other),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_bool_and_logical_inputs_passthrough() {
    let res_bool = logical_builtin(Value::Bool(true), Vec::new()).expect("logical");
    assert_eq!(res_bool, Value::Bool(true));

    let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
    let res_array =
        logical_builtin(Value::LogicalArray(logical.clone()), Vec::new()).expect("logical");
    assert_eq!(res_array, Value::LogicalArray(logical));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_empty_tensor_preserves_shape() {
    let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
    match result {
        Value::LogicalArray(array) => {
            assert!(array.data.is_empty());
            assert_eq!(array.shape, vec![0, 3]);
        }
        other => panic!("expected logical array, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_integer_scalar() {
    let res = logical_builtin(Value::Int(IntValue::I32(0)), Vec::new()).expect("logical");
    assert_eq!(res, Value::Bool(false));

    let res_nonzero = logical_builtin(Value::Int(IntValue::I32(-5)), Vec::new()).expect("logical");
    assert_eq!(res_nonzero, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn logical_wgpu_matches_cpu_conversion() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };

    let tensor = Tensor::new(vec![0.0, 2.0, -3.0, 1.0], vec![2, 2]).unwrap();
    let cpu = logical_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");

    let gpu_value = logical_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
    let out_handle = match gpu_value {
        Value::GpuTensor(ref h) => {
            assert!(runmat_accelerate_api::handle_is_logical(h));
            h.clone()
        }
        other => panic!("expected gpu tensor, got {other:?}"),
    };

    let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather");

    let (expected, expected_shape): (Vec<f64>, Vec<usize>) = match cpu {
        Value::LogicalArray(arr) => (
            arr.data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect(),
            arr.shape.to_vec(),
        ),
        Value::Bool(flag) => (vec![if flag { 1.0 } else { 0.0 }], vec![1, 1]),
        other => panic!("unexpected cpu result {other:?}"),
    };

    assert_eq!(gathered.shape(), expected_shape);
    assert_eq!(gathered.materialize_f64(), expected);
}
