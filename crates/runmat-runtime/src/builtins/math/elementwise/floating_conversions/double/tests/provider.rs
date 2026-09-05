use super::*;

#[test]
fn double_like_is_a_gated_runmat_extension() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error =
        double_builtin(Value::Num(1.0), vec![Value::from("like"), Value::Num(0.0)]).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DoubleLikePrototypeExtension")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = double_builtin(Value::GpuTensor(handle), Vec::new()).expect("double");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_gpu_integer_handle_converts_to_double_resident_output() {
    test_support::with_test_provider(|provider| {
        let data = [1_u64 << 63, u64::MAX];
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&data),
                shape: &[1, 2],
            })
            .expect("upload integer");

        let result = double_builtin(Value::GpuTensor(handle), Vec::new()).expect("double");
        let Value::GpuTensor(result_handle) = result else {
            panic!("expected f64-capable provider to keep double output resident");
        };
        assert!(runmat_accelerate_api::handle_integer_type(&result_handle).is_none());

        let gathered =
            test_support::gather(Value::GpuTensor(result_handle)).expect("gather double");
        assert_eq!(gathered.shape, vec![1, 2]);
        assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
        assert!(gathered.integer_storage().is_none());
        assert_eq!(
            gathered.materialize_f64(),
            vec![(1_u64 << 63) as f64, u64::MAX as f64]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_like_gpu_prototype_keeps_residency() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let proto = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .expect("upload");
        let result = double_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("like"), Value::GpuTensor(proto.clone())],
        )
        .expect("double");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather");
                assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    });
}

#[test]
fn double_like_rejects_hostile_upload_metadata_and_frees_only_resolved_owner() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .expect("prototype upload");

        let make_output = || {
            provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[2, 1],
                })
                .expect("result upload")
        };

        let wrong_shape = {
            let mut handle = make_output();
            handle.shape = vec![1, 2];
            handle
        };
        assert!(!valid_double_like_output(
            &wrong_shape,
            &prototype,
            provider,
            &[2, 1],
            false,
        ));
        provider
            .free(&wrong_shape)
            .expect("free wrong-shape result");

        let mut wrong_storage = make_output();
        wrong_storage.descriptor.storage =
            Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        assert!(!valid_double_like_output(
            &wrong_storage,
            &prototype,
            provider,
            &[2, 1],
            false,
        ));
        provider
            .free(&wrong_storage)
            .expect("free wrong-storage result");

        let mut wrong_precision = make_output();
        wrong_precision.descriptor.element_type =
            Some(runmat_accelerate_api::NumericElementType::F32);
        assert!(!valid_double_like_output(
            &wrong_precision,
            &prototype,
            provider,
            &[2, 1],
            false,
        ));
        provider
            .free(&wrong_precision)
            .expect("free wrong-precision result");

        let mut integer = make_output();
        integer.descriptor.element_type = Some(runmat_accelerate_api::NumericElementType::U8);
        assert!(!valid_double_like_output(
            &integer,
            &prototype,
            provider,
            &[2, 1],
            false,
        ));
        provider.free(&integer).expect("free integer result");

        let logical = make_output();
        runmat_accelerate_api::set_handle_logical(&logical, true);
        assert!(!valid_double_like_output(
            &logical,
            &prototype,
            provider,
            &[2, 1],
            false,
        ));
        provider.free(&logical).expect("free logical result");

        let mut owned_rejection = make_output();
        owned_rejection.descriptor.storage =
            Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        free_rejected_double_handle(&owned_rejection, &[]);
        assert!(block_on(provider.download(&owned_rejection)).is_err());

        assert!(!valid_double_like_output(
            &prototype,
            &prototype,
            provider,
            &[1, 1],
            false,
        ));
        assert!(!valid_double_gpu_output(
            &prototype, &prototype, provider, false,
        ));
        free_rejected_double_handle(&prototype, &[&prototype]);
        assert!(block_on(provider.download(&prototype)).is_ok());

        let unowned_rejection = runmat_accelerate_api::GpuTensorHandle {
            device_id: prototype.device_id.wrapping_add(10_000),
            buffer_id: prototype.buffer_id,
            shape: vec![2, 1],
            descriptor: Default::default(),
        };
        assert!(!valid_double_like_output(
            &unowned_rejection,
            &prototype,
            provider,
            &[2, 1],
            false,
        ));
        assert!(resolved_actual_double_owner(&unowned_rejection).is_none());
        free_rejected_double_handle(&unowned_rejection, &[]);
        assert!(block_on(provider.download(&prototype)).is_ok());
        provider.free(&prototype).expect("free prototype");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_like_host_gathers_gpu_input() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = double_builtin(
            Value::GpuTensor(handle),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("double");
        match result {
            Value::Num(n) => assert_eq!(n, 3.0),
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.materialize_f64(), vec![3.0]);
            }
            other => panic!("expected scalar host value, got {other:?}"),
        }
    });
}
