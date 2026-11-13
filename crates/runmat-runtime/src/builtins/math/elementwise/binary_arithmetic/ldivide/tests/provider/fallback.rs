use super::super::*;
#[test]
fn ldivide_uncertified_integer_scalar_refuses_before_upload() {
    test_support::with_uncertified_integer_provider(|provider, fixture| {
        let divisor =
            Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).expect("divisor");
        let source = gpu_helpers::upload_tensor(provider, &divisor).expect("upload");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit view");
        provider.reset_telemetry();
        let error = ldivide_builtin(
            Value::GpuTensor(source.clone()),
            Value::Num(10.0),
            Vec::new(),
        )
        .expect_err("uncertified integer scalar route must refuse before effects");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ExplicitProviderRequired")
        );
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(
            fixture.upload_evidence_queries() > 0,
            "upload evidence gate queried"
        );
        source
            .allocation_lease()
            .validate_live()
            .expect("source live");
        let original = test_support::gather(Value::GpuTensor(source)).expect("source survives");
        assert_eq!(
            original.integer_storage(),
            Some(&IntegerStorage::I32(vec![2, 4]))
        );
    });
}

#[test]
fn ldivide_uncertified_mixed_integer_fallback_refuses_before_download() {
    test_support::with_uncertified_integer_provider(|provider, fixture| {
        let scalar = Tensor::new(vec![2.0], vec![1, 1]).expect("scalar");
        let automatic = gpu_helpers::upload_tensor(provider, &scalar).expect("automatic");
        let automatic = automatic
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("automatic view");
        let integer =
            Tensor::new_integer(IntegerStorage::I32(vec![6, 10]), vec![2, 1]).expect("integer");
        let explicit = gpu_helpers::upload_tensor(provider, &integer).expect("explicit");
        let explicit = explicit
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit view");

        provider.reset_telemetry();
        let error = ldivide_builtin(
            Value::GpuTensor(automatic),
            Value::GpuTensor(explicit.clone()),
            Vec::new(),
        )
        .expect_err("uncertified explicit integer fallback must refuse");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ExplicitProviderRequired")
        );
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(
            fixture.upload_evidence_queries() > 0,
            "upload evidence gate queried"
        );
        explicit
            .allocation_lease()
            .validate_live()
            .expect("source live");
        let original = test_support::gather(Value::GpuTensor(explicit)).expect("source");
        assert_eq!(
            original.integer_storage(),
            Some(&IntegerStorage::I32(vec![6, 10]))
        );
    });
}
