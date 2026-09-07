use super::super::*;
#[test]
fn ldivide_resident_integer_fallback_preserves_source_and_residency() {
    test_support::with_test_provider(|provider| {
        let divisor =
            Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).expect("divisor");
        let source = gpu_helpers::upload_tensor(provider, &divisor).expect("upload");
        let source = source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let result = ldivide_builtin(
            Value::GpuTensor(source.clone()),
            Value::Num(10.0),
            Vec::new(),
        )
        .expect("resident ldivide");
        let Value::GpuTensor(output) = result else {
            panic!("explicit gpuArray result must remain resident");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert!(!gpu_helpers::same_gpu_handle(&source, &output));
        let original = test_support::gather(Value::GpuTensor(source)).expect("source survives");
        assert_eq!(
            original.integer_storage(),
            Some(&IntegerStorage::I32(vec![2, 4]))
        );
    });
}

#[test]
fn ldivide_fallback_prefers_explicit_second_resident_operand() {
    test_support::with_test_provider(|provider| {
        let scalar = Tensor::new(vec![2.0], vec![1, 1]).expect("scalar");
        let automatic = gpu_helpers::upload_tensor(provider, &scalar).expect("automatic");
        let automatic =
            automatic.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let integer =
            Tensor::new_integer(IntegerStorage::I32(vec![6, 10]), vec![2, 1]).expect("integer");
        let explicit = gpu_helpers::upload_tensor(provider, &integer).expect("explicit");
        let explicit =
            explicit.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);

        let result = ldivide_builtin(
            Value::GpuTensor(automatic),
            Value::GpuTensor(explicit.clone()),
            Vec::new(),
        )
        .expect("fallback ldivide");
        let Value::GpuTensor(output) = result else {
            panic!("explicit second operand must preserve resident output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::I32)
        );
        let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::I32(vec![3, 5]))
        );
        let original = test_support::gather(Value::GpuTensor(explicit)).expect("source");
        assert_eq!(
            original.integer_storage(),
            Some(&IntegerStorage::I32(vec![6, 10]))
        );
    });
}
