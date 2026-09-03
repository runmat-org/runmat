//! Factorial runtime tests.

use super::*;

mod cases {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::FACTORIAL_LIKE_EXTENSION;
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, NumericStorage, Tensor};

    fn factorial_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::factorial_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_scalar_positive() {
        let result = factorial_builtin(Value::Num(5.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(120.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_zero_is_one() {
        let result = factorial_builtin(Value::Num(0.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_vector_inputs() {
        let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.materialize_f64(), vec![1.0, 1.0, 6.0, 120.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_preserves_uint64_storage_and_saturates() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![3, 5, 171]), vec![3, 1])
            .expect("integer tensor");

        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(
                    out.into_numeric_storage().unwrap(),
                    NumericStorage::U64(vec![6, 120, u64::MAX])
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn factorial_preserves_all_native_classes_and_documented_saturation_thresholds() {
        let cases = [
            (
                IntegerStorage::I8(vec![5, 6]),
                IntegerStorage::I8(vec![120, i8::MAX]),
            ),
            (
                IntegerStorage::I16(vec![7, 8]),
                IntegerStorage::I16(vec![5_040, i16::MAX]),
            ),
            (
                IntegerStorage::I32(vec![12, 13]),
                IntegerStorage::I32(vec![479_001_600, i32::MAX]),
            ),
            (
                IntegerStorage::I64(vec![20, 21]),
                IntegerStorage::I64(vec![2_432_902_008_176_640_000, i64::MAX]),
            ),
            (
                IntegerStorage::U8(vec![5, 6]),
                IntegerStorage::U8(vec![120, u8::MAX]),
            ),
            (
                IntegerStorage::U16(vec![8, 9]),
                IntegerStorage::U16(vec![40_320, u16::MAX]),
            ),
            (
                IntegerStorage::U32(vec![12, 13]),
                IntegerStorage::U32(vec![479_001_600, u32::MAX]),
            ),
            (
                IntegerStorage::U64(vec![20, 21]),
                IntegerStorage::U64(vec![2_432_902_008_176_640_000, u64::MAX]),
            ),
        ];
        for (input, expected) in cases {
            let tensor = Tensor::new_integer(input, vec![1, 2]).unwrap();
            let Value::Tensor(output) =
                factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial")
            else {
                panic!("expected typed tensor");
            };
            assert_eq!(output.integer_storage(), Some(&expected));
        }

        let single = Tensor::from_f32(vec![5.0, 34.0, 35.0], vec![1, 3]).unwrap();
        let Value::Tensor(output) =
            factorial_builtin(Value::Tensor(single), Vec::new()).expect("single factorial")
        else {
            panic!("expected single tensor");
        };
        let NumericStorage::F32(values) = output.into_numeric_storage().unwrap() else {
            panic!("expected native single storage");
        };
        assert_eq!(values[0], 120.0);
        assert!(values[1].is_finite());
        assert!(values[2].is_infinite());

        let empty = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::Tensor(output) =
            factorial_builtin(Value::Tensor(empty), Vec::new()).expect("empty factorial")
        else {
            panic!("expected empty single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(Vec::new())
        );
    }

    #[test]
    fn factorial_rejects_negative_signed_integer_and_64_bit_integer_gpu_inputs() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 3]), vec![1, 2]).unwrap();
        let err = factorial_builtin(Value::Tensor(tensor), Vec::new())
            .expect_err("negative integer must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);

        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![5, 20]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let err = factorial_builtin(Value::GpuTensor(handle), Vec::new())
                .expect_err("uint64 GPU factorial must reject");
            assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
            assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
            assert!(err.message().contains("64-bit integer GPU"));
        });
    }

    #[test]
    fn public_dispatch_preserves_64_bit_integer_gpu_factorial_rejection() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![5, 20]), vec![1, 2])
                .expect("integer tensor");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let error = crate::dispatcher::call_builtin("factorial", &[Value::GpuTensor(handle)])
                .expect_err("public dispatch must not gather unsupported uint64 GPU input");
            assert_eq!(error.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
            assert!(error.message().contains("64-bit integer GPU"));
        });
    }

    #[test]
    fn factorial_integer_gpu_preserves_class_and_residency_without_floating_hook() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U32(vec![5, 13]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result =
                factorial_builtin(Value::GpuTensor(handle), Vec::new()).expect("factorial");
            assert!(matches!(result, Value::GpuTensor(_)));
            let output = test_support::gather(result).expect("gather");
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U32(vec![120, u32::MAX]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_non_integer_errors() {
        let err =
            factorial_builtin(Value::Num(2.5), Vec::new()).expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_negative_errors() {
        let tensor = Tensor::new(vec![-1.0, 3.0], vec![2, 1]).unwrap();
        let err = factorial_builtin(Value::Tensor(tensor), Vec::new())
            .expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_small_positive_non_integer_errors() {
        let err =
            factorial_builtin(Value::Num(1e-12), Vec::new()).expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_overflow_returns_inf() {
        let result = factorial_builtin(Value::Num(171.0), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_infinite()),
            other => panic!("expected scalar Inf, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_missing_prototype_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = factorial_builtin(Value::Num(3.0), vec![Value::from("like")])
            .expect_err("expected error");
        assert!(err.message().contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_gpu_prototype_uploads() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = factorial_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("factorial");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.materialize_f64(), vec![6.0, 24.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn factorial_like_gpu_prototype_preserves_integer_scalar_class() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let prototype = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .expect("upload prototype");
            let result = factorial_builtin(
                Value::Int(IntValue::U16(5)),
                vec![Value::from("like"), Value::GpuTensor(prototype)],
            )
            .expect("factorial like");
            let output = test_support::gather(result).expect("gather");
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U16(vec![120]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(Value::GpuTensor(handle), Vec::new()).expect("fact");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0, 6.0, 120.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_host_with_gpu_input_gathers() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("factorial");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.materialize_f64(), vec![6.0, 24.0]);
                }
                other => panic!("expected host tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_logical_input_promotes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = factorial_builtin(Value::LogicalArray(logical), Vec::new()).expect("fact");
        match result {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn matlab_compatibility_rejects_runmat_only_forms() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

        let logical_error = factorial_builtin(Value::Bool(true), Vec::new())
            .expect_err("logical input is a RunMat extension");
        assert_eq!(
            logical_error.identifier(),
            FACTORIAL_LOGICAL_EXTENSION.error_identifier
        );

        let like_error =
            factorial_builtin(Value::Num(3.0), vec![Value::from("like"), Value::Num(0.0)])
                .expect_err("the like form is a RunMat extension");
        assert_eq!(
            like_error.identifier(),
            FACTORIAL_LIKE_EXTENSION.error_identifier
        );
    }

    #[test]
    fn factorial_rejects_more_than_one_output() {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error = factorial_builtin(Value::Num(5.0), Vec::new())
            .expect_err("factorial defines one output");
        assert_eq!(
            error.identifier(),
            FACTORIAL_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_int_input_preserves_class() {
        let value = Value::Int(IntValue::U16(5));
        let result = factorial_builtin(value, Vec::new()).expect("factorial");
        assert_eq!(result, Value::Int(IntValue::U16(120)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_nan_errors() {
        let err =
            factorial_builtin(Value::Num(f64::NAN), Vec::new()).expect_err("factorial must reject");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_complex_input_errors() {
        let err = factorial_builtin(Value::Complex(1.0, 0.5), Vec::new())
            .expect_err("expected complex rejection");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_string_input_errors() {
        let err = factorial_builtin(Value::from("hello"), Vec::new())
            .expect_err("expected string rejection");
        assert_eq!(err.identifier(), FACTORIAL_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_complex_prototype_rejected() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = factorial_builtin(
            Value::Num(3.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected complex prototype rejection");
        assert!(err.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn factorial_wgpu_matches_cpu_after_gather() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0], vec![3, 1]).unwrap();
        let cpu = factorial_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).unwrap();
        let gpu = block_on(factorial_gpu(handle)).expect("gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.materialize_f64(), cpu_tensor.materialize_f64());
    }
}
