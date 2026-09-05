use super::*;

#[test]
fn complex_mixed_gpu_path_preserves_integer_class_rule_without_floating_upload() {
    test_support::with_test_provider(|provider| {
        let real = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("typed integer tensor");
        let imag = Tensor::new(vec![10.0, 20.0], vec![1, 2]).expect("imag tensor");
        let imag_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &imag.materialize_f64(),
                shape: &imag.shape,
            })
            .expect("upload imag");

        let error = complex_call(Value::Tensor(real), vec![Value::GpuTensor(imag_handle)])
            .expect_err("nonscalar floating peer must not erase integer class");
        assert_eq!(error.identifier(), COMPLEX_ERROR_INTEGER_CLASS.identifier);
    });
}

#[test]
fn complex_host_integer_with_resident_scalar_double_restores_exact_owner_output() {
    test_support::with_test_provider(|provider| {
        let scalar = Tensor::new(vec![3.0], vec![1, 1]).expect("scalar");
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &scalar.materialize_f64(),
                shape: &scalar.shape,
            })
            .expect("upload");
        let real =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("integer");
        let Value::GpuTensor(result) =
            complex_call(Value::Tensor(real), vec![Value::GpuTensor(handle)]).expect("complex")
        else {
            panic!("expected resident complex integer");
        };
        let Value::ComplexTensor(result) =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(result))).expect("gather")
        else {
            panic!("expected complex integer tensor");
        };
        assert_eq!(
            result.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![3])
                )
                .expect("storage")
            )
        );
    });
}

#[test]
fn complex_resident_integer_fallback_restores_exact_class_to_owner() {
    test_support::with_test_provider(|provider| {
        let real =
            Tensor::new_integer(IntegerStorage::U32(vec![u32::MAX]), vec![1, 1]).expect("integer");
        let handle = gpu_helpers::upload_tensor(provider, &real).expect("upload");
        let result = complex_call(Value::GpuTensor(handle), Vec::new()).expect("complex");
        let Value::GpuTensor(result) = result else {
            panic!("expected resident complex integer");
        };
        let Value::ComplexTensor(result) =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(result))).expect("gather")
        else {
            panic!("expected complex integer tensor");
        };
        assert_eq!(
            result.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U32(vec![u32::MAX]),
                    IntegerStorage::U32(vec![0])
                )
                .expect("complex storage")
            )
        );
    });
}

#[test]
fn complex_unary_gpu_stays_resident() {
    test_support::with_test_provider(|provider| {
        let real = Tensor::new(vec![1.0, -2.0, 3.5], vec![3, 1]).unwrap();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &real.materialize_f64(),
                shape: &real.shape,
            })
            .expect("upload");
        let result = complex_call(Value::GpuTensor(handle), Vec::new()).expect("complex");
        let Value::GpuTensor(out) = result else {
            panic!("expected resident complex gpuArray");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
        let Value::ComplexTensor(ct) = gathered else {
            panic!("expected gathered complex tensor");
        };
        assert_eq!(ct.shape, vec![3, 1]);
        assert_eq!(
            ct.materialize_f64(),
            vec![(1.0, 0.0), (-2.0, 0.0), (3.5, 0.0)]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_binary_gpu_stays_resident_with_scalar_expansion() {
    test_support::with_test_provider(|provider| {
        let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let real_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &real.materialize_f64(),
                shape: &real.shape,
            })
            .expect("upload real");
        let result =
            complex_call(Value::GpuTensor(real_handle), vec![Value::Num(-4.0)]).expect("complex");
        let Value::GpuTensor(out) = result else {
            panic!("expected resident complex gpuArray");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
        let Value::ComplexTensor(ct) = gathered else {
            panic!("expected gathered complex tensor");
        };
        assert_eq!(ct.shape, vec![1, 3]);
        assert_eq!(
            ct.materialize_f64(),
            vec![(1.0, -4.0), (2.0, -4.0), (3.0, -4.0)]
        );
    });
}

#[test]
fn complex_typed_integer_gpu_composition_stays_exact_and_resident() {
    test_support::with_test_provider(|provider| {
        let real = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("real");
        let imaginary =
            Tensor::new_integer(IntegerStorage::U64(vec![7, 11]), vec![1, 2]).expect("imaginary");
        let real = gpu_helpers::upload_tensor(provider, &real).expect("upload real");
        let imaginary = gpu_helpers::upload_tensor(provider, &imaginary).expect("upload imaginary");
        let Value::GpuTensor(output) =
            complex_call(Value::GpuTensor(real), vec![Value::GpuTensor(imaginary)])
                .expect("complex")
        else {
            panic!("expected resident complex integer");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let Value::ComplexTensor(gathered) =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).expect("gather")
        else {
            panic!("expected complex integer tensor");
        };
        let storage = gathered.integer_storage().expect("integer storage");
        assert_eq!(
            storage.real,
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])
        );
        assert_eq!(storage.imag, IntegerStorage::U64(vec![7, 11]));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_empty_gpu_inputs_stay_resident() {
    test_support::with_test_provider(|provider| {
        let real = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let imag = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let real_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &real.materialize_f64(),
                shape: &real.shape,
            })
            .expect("upload real");
        let imag_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &imag.materialize_f64(),
                shape: &imag.shape,
            })
            .expect("upload imag");
        let result = complex_call(
            Value::GpuTensor(real_handle),
            vec![Value::GpuTensor(imag_handle)],
        )
        .expect("complex");
        let Value::GpuTensor(out) = result else {
            panic!("expected resident complex gpuArray");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
        let Value::ComplexTensor(ct) = gathered else {
            panic!("expected gathered complex tensor");
        };
        assert_eq!(ct.shape, vec![0, 3]);
        assert!(ct.materialize_f64().is_empty());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_gpu_shape_mismatch_fallback_reports_size_error() {
    test_support::with_test_provider(|provider| {
        let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let imag = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let real_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &real.materialize_f64(),
                shape: &real.shape,
            })
            .expect("upload real");
        let imag_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &imag.materialize_f64(),
                shape: &imag.shape,
            })
            .expect("upload imag");
        let err = complex_call(
            Value::GpuTensor(real_handle),
            vec![Value::GpuTensor(imag_handle)],
        )
        .unwrap_err();
        let message = err.message();
        assert!(
            message.contains("same size") || message.contains("scalar"),
            "unexpected error: {message}"
        );
        assert!(
            !message.contains("GpuTensor"),
            "fallback leaked gpuArray host-conversion error: {message}"
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_binary_rejects_complex_gpu_input() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let err = complex_call(Value::GpuTensor(handle), vec![Value::Num(0.0)]).unwrap_err();
        assert!(
            err.message().contains("must be real"),
            "unexpected error: {}",
            err.message()
        );
    });
}
