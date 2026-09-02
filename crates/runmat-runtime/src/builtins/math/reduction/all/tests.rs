use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{CharArray, ComplexTensor, IntValue, IntegerStorage, Tensor};

fn all_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::all_builtin(value, rest))
}

fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
    error.identifier()
}

#[test]
fn all_descriptor_signatures_and_errors() {
    let labels: Vec<&str> = ALL_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"B = all(A)"));
    assert!(labels.contains(&"B = all(A, dim)"));
    assert!(labels.contains(&"B = all(A, \"all\")"));
    assert!(labels.contains(&"B = all(A, nanflag)"));
    assert!(labels.contains(&"B = all(A, dim, nanflag)"));
    assert!(labels.contains(&"B = all(A, nanflag, dim)"));
    assert!(ALL_DESCRIPTOR
        .errors
        .iter()
        .any(|err| err.code == ALL_ERROR_INVALID_ARGUMENT.code));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_matrix_default_dimension() {
    let tensor = Tensor::new(vec![1.0, 1.0, 4.0, 5.0, 0.0, 6.0], vec![2, 3]).unwrap();
    let result = all_builtin(Value::Tensor(tensor), Vec::new()).expect("all");
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![1, 3]);
            assert_eq!(out.data, vec![1, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[test]
fn all_reads_all_integer_storage_classes_without_floating_materialization() {
    let storages = [
        IntegerStorage::I8(vec![1, 1, 4, 5, 0, 6]),
        IntegerStorage::I16(vec![1, 1, 4, 5, 0, 6]),
        IntegerStorage::I32(vec![1, 1, 4, 5, 0, 6]),
        IntegerStorage::I64(vec![1, 1, 4, 5, 0, i64::MAX]),
        IntegerStorage::U8(vec![1, 1, 4, 5, 0, 6]),
        IntegerStorage::U16(vec![1, 1, 4, 5, 0, 6]),
        IntegerStorage::U32(vec![1, 1, 4, 5, 0, 6]),
        IntegerStorage::U64(vec![1, 1, 4, 5, 0, u64::MAX]),
    ];
    for storage in storages {
        let tensor = Tensor::new_integer(storage, vec![2, 3]).expect("typed integer input");
        let result = all_builtin(Value::Tensor(tensor), Vec::new()).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }
}

#[test]
fn all_nanflag_is_a_declared_runmat_only_extension() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = all_builtin(Value::Num(1.0), vec![Value::from("omitnan")])
            .expect_err("MATLAB mode rejects nanflag");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:AllNanflagExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        all_builtin(Value::Num(1.0), vec![Value::from("omitnan")])
            .expect("RunMat mode accepts nanflag");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_zero_column_matrix_returns_empty_row() {
    let tensor = Tensor::new(Vec::<f64>::new(), vec![2, 0]).unwrap();
    let result = all_builtin(Value::Tensor(tensor), Vec::new()).expect("all");
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![1, 0]);
            assert!(out.data.is_empty());
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_row_dimension() {
    let tensor = Tensor::new(vec![1.0, 1.0, 4.0, 5.0, 0.0, 6.0], vec![2, 3]).unwrap();
    let args = vec![Value::Int(IntValue::I32(2))];
    let result = all_builtin(Value::Tensor(tensor), args).expect("all");
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![2, 1]);
            assert_eq!(out.data, vec![0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_zero_row_matrix_dim_two() {
    let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
    let args = vec![Value::Int(IntValue::I32(2))];
    let result = all_builtin(Value::Tensor(tensor), args).expect("all");
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![0, 1]);
            assert!(out.data.is_empty());
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_vecdim_multiple_axes() {
    let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
    let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let result = all_builtin(Value::Tensor(tensor), vec![Value::Tensor(vecdim)]).expect("all");
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![1, 1, 2]);
            assert_eq!(out.data, vec![1, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_all_option_returns_scalar() {
    let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
    let result = all_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("all");
    match result {
        Value::Bool(flag) => assert!(!flag),
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_all_on_empty_returns_true() {
    let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
    let result = all_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("all");
    match result {
        Value::Bool(flag) => assert!(flag),
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_handles_nan_modes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![f64::NAN, f64::NAN, 1.0, 0.0], vec![2, 2]).unwrap();
    let includenan = all_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("all");
    match includenan {
        Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0]),
        other => panic!("expected logical array, got {other:?}"),
    }

    let omit = all_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("all omit");
    match omit {
        Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0]),
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_char_array_support() {
    let chars = CharArray::new("a\0c".chars().collect(), 1, 3).unwrap();
    let result =
        all_builtin(Value::CharArray(chars), vec![Value::Int(IntValue::I32(1))]).expect("all");
    match result {
        Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0, 1]),
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_includenan_keyword_allowed() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new(vec![f64::NAN, 1.0], vec![2, 1]).unwrap();
    let result = all_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("all");
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![1, 1]);
            assert_eq!(out.data, vec![1]);
        }
        Value::Bool(flag) => assert!(flag),
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_complex_tensor_with_omitnan() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let complex = ComplexTensor::new(vec![(f64::NAN, 0.0), (1.0, 0.0)], vec![2, 1]).unwrap();
    let tensor_value = Value::ComplexTensor(complex);
    let omit = all_builtin(tensor_value.clone(), vec![Value::from("omitnan")]).expect("all");
    match omit {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![1, 1]);
            assert_eq!(out.data, vec![1]);
        }
        Value::Bool(flag) => assert!(flag),
        other => panic!("expected logical array, got {other:?}"),
    }
    let include = all_builtin(tensor_value, Vec::new()).expect("all include");
    match include {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![1, 1]);
            assert_eq!(out.data, vec![1]);
        }
        Value::Bool(flag) => assert!(flag),
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_vecdim_with_omitnan() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut data = vec![0.0; 8];
    data[7] = f64::NAN;
    let tensor = Tensor::new(data, vec![2, 2, 2]).unwrap();
    let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let args = vec![Value::Tensor(vecdim), Value::from("omitnan")];
    let result = all_builtin(Value::Tensor(tensor), args).expect("all vecdim omitnan");
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![1, 1, 2]);
            assert_eq!(out.data, vec![0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_all_with_dim_errors() {
    let tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
    let args = vec![Value::from("all"), Value::Int(IntValue::I32(1))];
    let err = all_builtin(Value::Tensor(tensor), args).unwrap_err();
    assert_eq!(
        error_identifier(&err),
        ALL_ERROR_INVALID_ARGUMENT.identifier
    );
    assert!(
        err.message().contains(ALL_ERROR_INVALID_ARGUMENT.message),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = all_builtin(Value::GpuTensor(handle), Vec::new()).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_gpu_provider_omitnan_roundtrip() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN, 1.0, 0.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            all_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    });
}

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_wgpu_default_matches_cpu() {
    let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
    }));
    let Ok(reg_result) = init else {
        tracing::warn!("skipping all_wgpu_default_matches_cpu: wgpu provider panicked during init");
        return;
    };
    if reg_result.is_err() {
        tracing::warn!("skipping all_wgpu_default_matches_cpu: wgpu provider unavailable");
        return;
    }
    let tensor = Tensor::new(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 3]).unwrap();
    let cpu = block_on(all_host(
        Value::Tensor(tensor.clone()),
        ReductionSpec::Default,
        ReductionNaN::Include,
    ))
    .unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => {
            tracing::warn!("skipping all_wgpu_default_matches_cpu: provider not registered");
            return;
        }
    };
    let handle = match provider.upload(&view) {
        Ok(h) => h,
        Err(err) => {
            tracing::warn!("skipping all_wgpu_default_matches_cpu: upload failed: {err}");
            return;
        }
    };
    let gpu = all_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
    match (cpu, gpu) {
        (Value::LogicalArray(expected), Value::LogicalArray(actual)) => {
            assert_eq!(expected.shape, actual.shape);
            assert_eq!(expected.data, actual.data);
        }
        _ => panic!("unexpected shapes"),
    }
}

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn all_wgpu_omitnan_matches_cpu() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
    }));
    let Ok(reg_result) = init else {
        tracing::warn!("skipping all_wgpu_omitnan_matches_cpu: wgpu provider panicked during init");
        return;
    };
    if reg_result.is_err() {
        tracing::warn!("skipping all_wgpu_omitnan_matches_cpu: wgpu provider unavailable");
        return;
    }
    let tensor = Tensor::new(vec![f64::NAN, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
    let cpu = block_on(all_host(
        Value::Tensor(tensor.clone()),
        ReductionSpec::Default,
        ReductionNaN::Omit,
    ))
    .unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => {
            tracing::warn!("skipping all_wgpu_omitnan_matches_cpu: provider not registered");
            return;
        }
    };
    let handle = match provider.upload(&view) {
        Ok(h) => h,
        Err(err) => {
            tracing::warn!("skipping all_wgpu_omitnan_matches_cpu: upload failed: {err}");
            return;
        }
    };
    let gpu = all_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).unwrap();
    match (cpu, gpu) {
        (Value::LogicalArray(expected), Value::LogicalArray(actual)) => {
            assert_eq!(expected.shape, actual.shape);
            assert_eq!(expected.data, actual.data);
        }
        _ => panic!("unexpected shapes"),
    }
}
