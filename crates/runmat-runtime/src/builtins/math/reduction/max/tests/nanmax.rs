use super::*;
#[cfg(feature = "wgpu")]
use crate::builtins::common::test_support;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, HostTensorView};
use runmat_value::IntValue;
#[cfg(feature = "wgpu")]
use runmat_value::IntegerStorage;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nanmax_reduction_omits_nan_by_default() {
    let tensor = Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN], vec![2, 2]).unwrap();
    let result = nanmax_builtin(Value::Tensor(tensor), Vec::new()).expect("nanmax");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [1, 2]);
            assert_eq!(t.materialize_f64()[0], 4.0);
            assert_eq!(t.materialize_f64()[1], 2.0);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn nanmax_typed_integer_data_follows_compatibility_mode() {
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = nanmax_builtin(Value::Int(IntValue::U64(u64::MAX)), Vec::new())
            .expect_err("strict nanmax typed-integer input");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanmaxTypedIntegerInputExtension")
        );
        let error = nanmax_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("floating input")),
            vec![placeholder(), Value::Int(IntValue::U8(2))],
        )
        .expect_err("strict nanmax typed-integer dimension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:NanmaxTypedIntegerInputExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = nanmax_builtin(Value::Int(IntValue::U64(u64::MAX)), Vec::new())
            .expect("RunMat nanmax typed-integer input");
        assert_eq!(result, Value::Int(IntValue::U64(u64::MAX)));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nanmax_reduction_accepts_empty_placeholder_and_dim() {
    let tensor = Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN], vec![2, 2]).unwrap();
    let result = nanmax_builtin(Value::Tensor(tensor), vec![placeholder(), Value::Num(2.0)])
        .expect("nanmax dim");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [2, 1]);
            assert_eq!(t.materialize_f64()[0], 2.0);
            assert_eq!(t.materialize_f64()[1], 4.0);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nanmax_output_count_returns_indices_and_zero_output_list() {
    let tensor = Tensor::new(vec![f64::NAN, 4.0, 2.0], vec![3, 1]).unwrap();
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = nanmax_builtin(Value::Tensor(tensor), Vec::new()).expect("nanmax outputs");
    match result {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 2);
            assert_eq!(values[0], Value::Num(4.0));
            assert_eq!(values[1], Value::Num(2.0));
        }
        other => panic!("expected output list, got {other:?}"),
    }
    drop(_guard);

    let _guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let result = nanmax_builtin(Value::Num(1.0), Vec::new()).expect("nanmax zero outputs");
    assert_eq!(result, Value::output_list(Vec::new()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nanmax_elementwise_omits_nan_by_default() {
    let lhs = Tensor::new(vec![f64::NAN, 2.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![3.0, f64::NAN], vec![2, 1]).unwrap();
    let result = nanmax_builtin(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).expect("nanmax");
    match result {
        Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![3.0, 2.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn nanmax_gpu_omitnan_gathers_to_host_result() {
    let tensor = Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN], vec![2, 2]).unwrap();
    test_support::with_test_provider(|provider| {
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = nanmax_builtin(Value::GpuTensor(handle), Vec::new()).expect("nanmax gpu");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), [1, 2]);
                assert_eq!(t.materialize_f64(), vec![4.0, 2.0]);
            }
            other => panic!("expected host tensor fallback, got {other:?}"),
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn nanmax_integer_gpu_extension_gathers_exact_host_result() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let data = [u64::MAX - 1, u64::MAX, 7, 9];
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&data),
                shape: &[2, 2],
            })
            .expect("integer upload");
        let result = nanmax_builtin(Value::GpuTensor(handle), Vec::new()).expect("integer nanmax");
        let Value::Tensor(tensor) = result else {
            panic!("expected gathered integer tensor");
        };
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 9]))
        );
    });
}
