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
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = double_builtin(Value::GpuTensor(handle), Vec::new()).expect("double");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), vec![2, 2]);
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
        assert_eq!(gathered.shape(), vec![1, 2]);
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
fn double_like_rejects_genuine_wrong_outputs_without_consuming_prototype() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .unwrap();
        let wrong_shape =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap())
                .unwrap();
        let wrong_storage = gpu_helpers::upload_complex_tensor(
            provider,
            &ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![2, 1]).unwrap(),
        )
        .unwrap();
        let wrong_precision = gpu_helpers::upload_tensor(
            provider,
            &Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )
        .unwrap();
        let integer = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap(),
        )
        .unwrap();
        let logical = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1, 0],
                shape: &[2, 1],
            })
            .unwrap();
        for output in [
            wrong_shape,
            wrong_storage,
            wrong_precision,
            integer,
            logical,
        ] {
            assert!(!valid_double_like_output(
                &output,
                &prototype,
                provider,
                &[2, 1],
                false
            ));
            let physical = output.allocation_lease().downgrade();
            output.release().expect("release owned rejected result");
            assert!(physical.acquire().is_none());
            prototype
                .allocation_lease()
                .validate_live()
                .expect("prototype survives");
        }
        assert!(!valid_double_like_output(
            &prototype,
            &prototype,
            provider,
            &[1, 1],
            false
        ));
        // A plain immutable same-type cast may adopt the input, unlike a
        // newly uploaded result aliasing an unrelated like-prototype.
        assert!(valid_double_gpu_output(
            &prototype, &prototype, provider, false
        ));
        prototype.clone().release().expect("release owned alias");
        assert!(block_on(provider.download(&prototype)).is_ok());

        let foreign_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let foreign = foreign_owner
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(!valid_double_like_output(
            &foreign.handle,
            &prototype,
            provider,
            &[2, 1],
            false
        ));
        assert!(resolved_actual_double_owner(&foreign.handle).is_none());
        drop(foreign.handle);
        assert_eq!(foreign.probe.retirements(), 1);
        assert!(block_on(provider.download(&prototype)).is_ok());
        let physical = prototype.allocation_lease().downgrade();
        prototype.release().expect("release prototype");
        assert!(physical.acquire().is_none());
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
            shape: tensor.shape(),
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
                assert_eq!(t.shape(), &[1, 1]);
                assert_eq!(t.materialize_f64(), vec![3.0]);
            }
            other => panic!("expected scalar host value, got {other:?}"),
        }
    });
}
