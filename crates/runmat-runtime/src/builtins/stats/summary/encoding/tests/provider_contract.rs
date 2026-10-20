//! Registered-owner witnesses for encoding admission and gather failures.
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NumericElementType, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct RejectDownloads {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for RejectDownloads {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "encoding registered-owner admission witness".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("encoding admission must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("encoding fixture rejected download").into()) })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(std::io::Error::other("encoding fixture rejected download").into()) })
    }
}

fn encoded() -> Value {
    tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])
}

fn classes() -> Value {
    tensor(vec![1.0, 2.0], vec![1, 2])
}

fn labels() -> Value {
    tensor(vec![1.0, 2.0], vec![2, 1])
}

fn assert_gather_cause(error: &RuntimeError, owner: EncodingBuiltin) {
    assert_eq!(error.identifier(), owner.invalid().identifier);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let gather = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("typed gather cause is retained");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(error.span, gather.span);
    assert_eq!(error.gpu_gather_retry, gather.gpu_gather_retry);
    let mut source: &(dyn std::error::Error + 'static) = error;
    while !source.is::<std::io::Error>() {
        source = source
            .source()
            .expect("original provider failure remains in the cause chain");
    }
    assert_eq!(source.to_string(), "encoding fixture rejected download");
}

fn assert_failed_download(
    result: BuiltinResult<Value>,
    owner: EncodingBuiltin,
    expected_count: usize,
    provider: &RejectDownloads,
    allocation: &runmat_accelerate_api::test_support::TestAllocation,
) {
    let error = result.expect_err("registered fixture rejects the exact requested download");
    assert_gather_cause(&error, owner);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), expected_count);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn known_invalid_encoding_calls_do_not_download_from_registered_owner() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());
    assert!(runmat_accelerate_api::same_provider_instance(
        runmat_accelerate_api::provider_for_handle(&allocation.handle).unwrap(),
        provider
    ));

    {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = block_on(onehotencode_builtin(
            resident(),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            EncodingBuiltin::OnehotEncode.invalid().identifier
        );
        assert!(error.message().contains("featureDim"), "{error}");

        // A deferred positional typename cannot force a transfer before a
        // later known-invalid host option is admitted.
        let error = block_on(onehotencode_builtin(
            classes(),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                resident(),
                Value::from("NotAnEncodingOption"),
                Value::from("double"),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("unsupported option"), "{error}");

        let error = block_on(onehotdecode_builtin(
            resident(),
            classes(),
            Value::Num(0.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            EncodingBuiltin::OnehotDecode.invalid().identifier
        );
        assert!(error.message().contains("featureDim"), "{error}");

        // The first option name is resident; the later host name is already
        // invalid and must be rejected without downloading either value.
        let error = block_on(onehotdecode_builtin(
            encoded(),
            classes(),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                resident(),
                Value::from("double"),
                Value::from("NotAnEncodingOption"),
                Value::from("double"),
            ]),
        ))
        .unwrap_err();
        assert!(error.message().contains("unsupported option"), "{error}");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(dummyvar_builtin(resident())).unwrap_err();
        assert_eq!(
            error.identifier(),
            DUMMYVAR_GPU_GROUP_EXTENSION.error_identifier
        );

        let error = block_on(onehotencode_builtin(
            resident(),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("OutputType"),
                Value::from("double"),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            ONEHOTENCODE_OUTPUT_TYPE_ALIAS_EXTENSION.error_identifier
        );

        let error = block_on(onehotdecode_builtin(
            resident(),
            classes(),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("OutputType"),
                Value::from("double"),
            ]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION.error_identifier
        );

        let explicit = Value::GpuTensor(
            allocation
                .handle
                .clone()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap(),
        );
        let error = block_on(onehotencode_builtin(
            explicit.clone(),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            ONEHOTENCODE_EXPLICIT_GPU_EXTENSION.error_identifier
        );
        let error = block_on(onehotdecode_builtin(
            explicit,
            classes(),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            ONEHOTDECODE_EXPLICIT_GPU_EXTENSION.error_identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }

    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn encoding_download_failures_keep_cause_retry_and_caller_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let resident = || Value::GpuTensor(allocation.handle.clone());

    assert_failed_download(
        block_on(dummyvar_builtin(resident())),
        EncodingBuiltin::Dummyvar,
        1,
        provider,
        &allocation,
    );
    assert_failed_download(
        block_on(onehotencode_builtin(
            resident(),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )),
        EncodingBuiltin::OnehotEncode,
        2,
        provider,
        &allocation,
    );
    assert_failed_download(
        block_on(onehotdecode_builtin(
            resident(),
            classes(),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        )),
        EncodingBuiltin::OnehotDecode,
        3,
        provider,
        &allocation,
    );
    assert_failed_download(
        block_on(onehotencode_builtin(
            labels(),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![resident()]),
        )),
        EncodingBuiltin::OnehotEncode,
        4,
        provider,
        &allocation,
    );
    assert_failed_download(
        block_on(onehotdecode_builtin(
            encoded(),
            classes(),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![resident()]),
        )),
        EncodingBuiltin::OnehotDecode,
        5,
        provider,
        &allocation,
    );
}
