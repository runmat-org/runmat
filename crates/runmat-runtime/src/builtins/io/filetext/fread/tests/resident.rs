use super::*;
#[test]
fn fread_like_preserves_every_exact_integer_class() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let prototypes = [
        IntegerStorage::I8(vec![0]),
        IntegerStorage::I16(vec![0]),
        IntegerStorage::I32(vec![0]),
        IntegerStorage::I64(vec![0]),
        IntegerStorage::U8(vec![0]),
        IntegerStorage::U16(vec![0]),
        IntegerStorage::U32(vec![0]),
        IntegerStorage::U64(vec![0]),
    ];

    for storage in prototypes {
        let expected = storage
            .from_same_class_values(
                [1.0, 2.5, -3.0]
                    .into_iter()
                    .map(|value| storage.cast_f64_assignment(value))
                    .collect(),
            )
            .expect("expected storage");
        let data = Value::Tensor(Tensor::new(vec![1.0, 2.5, -3.0], vec![3, 1]).unwrap());
        let prototype_tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        let prototype = Value::Tensor(prototype_tensor);
        let output = adjust_output_for_like(data, &prototype, PrecisionSpec::default())
            .expect("integer like output");
        let Value::Tensor(output) = output else {
            panic!("expected tensor output");
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }
}

#[test]
fn fread_resident_like_preserves_every_exact_integer_class() {
    let _guard = test_support::accel_test_lock();
    let owner: &'static FreadOwnerProvider = Box::leak(Box::new(FreadOwnerProvider::new(
        runmat_accelerate_api::ProviderPrecision::F64,
    )));
    unsafe { runmat_accelerate_api::register_provider(owner) };
    let prototypes = [
        IntegerStorage::I8(vec![0]),
        IntegerStorage::I16(vec![0]),
        IntegerStorage::I32(vec![0]),
        IntegerStorage::I64(vec![0]),
        IntegerStorage::U8(vec![0]),
        IntegerStorage::U16(vec![0]),
        IntegerStorage::U32(vec![0]),
        IntegerStorage::U64(vec![0]),
    ];
    for storage in prototypes {
        let expected_type = match storage.numeric_dtype() {
            NumericDType::I8 => runmat_accelerate_api::IntegerElementType::I8,
            NumericDType::I16 => runmat_accelerate_api::IntegerElementType::I16,
            NumericDType::I32 => runmat_accelerate_api::IntegerElementType::I32,
            NumericDType::I64 => runmat_accelerate_api::IntegerElementType::I64,
            NumericDType::U8 => runmat_accelerate_api::IntegerElementType::U8,
            NumericDType::U16 => runmat_accelerate_api::IntegerElementType::U16,
            NumericDType::U32 => runmat_accelerate_api::IntegerElementType::U32,
            NumericDType::U64 => runmat_accelerate_api::IntegerElementType::U64,
            other => panic!("unexpected dtype {other:?}"),
        };
        let prototype_tensor = Tensor::new_integer(storage, vec![1, 1]).expect("prototype tensor");
        let prototype =
            gpu_helpers::upload_tensor(owner, &prototype_tensor).expect("prototype integer upload");
        let output = tensor_to_gpu_value(
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("read tensor"),
            &prototype,
        )
        .expect("resident integer like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(expected_type)
        );
        assert_eq!(output.device_id(), prototype.device_id());
        assert_ne!(output.buffer_id(), prototype.buffer_id());
    }
}

#[test]
fn fread_resident_like_routes_to_prototype_owner_not_ambient_provider() {
    let _guard = test_support::accel_test_lock();
    let owner: &'static FreadOwnerProvider = Box::leak(Box::new(FreadOwnerProvider::new(
        runmat_accelerate_api::ProviderPrecision::F32,
    )));
    let ambient: &'static FreadOwnerProvider = Box::leak(Box::new(FreadOwnerProvider::new(
        runmat_accelerate_api::ProviderPrecision::F64,
    )));
    unsafe {
        runmat_accelerate_api::register_provider(owner);
        runmat_accelerate_api::register_provider(ambient);
    }
    let _ambient = runmat_accelerate_api::ThreadProviderGuard::set(Some(ambient));
    let prototype_data = [0.0_f32];
    let prototype = owner
        .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F32(&prototype_data),
            shape: &[1, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .expect("prototype upload");
    let output = tensor_to_gpu_value(
        Tensor::new(vec![1.25, 2.5], vec![2, 1]).expect("tensor"),
        &prototype,
    )
    .expect("owner upload");
    let Value::GpuTensor(output) = output else {
        panic!("expected resident output");
    };
    assert_eq!(output.device_id(), owner.device_id());
    assert_ne!(output.device_id(), ambient.device_id());
    assert_ne!(output.buffer_id(), prototype.buffer_id());
    assert_eq!(output.shape(), vec![2, 1]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
}

#[test]
fn fread_resident_like_rejects_and_cleans_malformed_outputs_without_freeing_prototype() {
    let _guard = test_support::accel_test_lock();
    let owner = Box::leak(Box::new(FreadOwnerProvider::new(
        runmat_accelerate_api::ProviderPrecision::F64,
    )));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let prototype = owner
        .upload(&HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        })
        .unwrap();
    let malformed = owner
        .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
            data: runmat_accelerate_api::HostNumericDataView::F64(&[1.0, 2.0]),
            shape: &[1, 1],
            storage: runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let wire = malformed.wire_descriptor();
    let message = validate_resident_like_output(
        &prototype,
        &malformed,
        &[1, 1],
        owner,
        Some(runmat_accelerate_api::ProviderPrecision::F64),
        None,
        false,
    )
    .unwrap_err();
    assert!(message.contains("invalid or aliased"));
    drop(gpu_helpers::reject_gpu_output(
        malformed,
        fread_error_with_detail(&FREAD_ERROR_INVALID_OPTION, &message),
    ));
    assert!(owner.inner.acquire_live(&wire).is_err());
    let message = validate_resident_like_output(
        &prototype,
        &prototype,
        &[1, 1],
        owner,
        Some(runmat_accelerate_api::ProviderPrecision::F64),
        None,
        false,
    )
    .unwrap_err();
    drop(gpu_helpers::reject_gpu_output(
        prototype.clone(),
        fread_error_with_detail(&FREAD_ERROR_INVALID_OPTION, &message),
    ));
    assert!(owner
        .inner
        .acquire_live(&prototype.wire_descriptor())
        .is_ok());
}
