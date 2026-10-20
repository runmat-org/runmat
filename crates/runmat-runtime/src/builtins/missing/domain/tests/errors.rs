use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for FailingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "missing-data download failure fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("failure paths must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(std::io::Error::other("injected missing-data download failure").into())
        })
    }
}

#[test]
fn gather_diagnostics_preserve_all_retry_policies_and_source_identity() {
    for retry in [
        crate::GpuGatherRetry::Legacy,
        crate::GpuGatherRetry::Never,
        crate::GpuGatherRetry::Requested,
    ] {
        let source = build_runtime_error("delegated failure")
            .with_identifier("RunMat:test:DelegatedFailure")
            .with_gpu_gather_retry(retry)
            .with_source(std::io::Error::other("underlying operation"))
            .build();
        let error = gather_error("ismissing", "input", source);
        assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
        assert_eq!(error.context.builtin.as_deref(), Some("ismissing"));
        assert_eq!(error.gpu_gather_retry(), retry);
        assert_eq!(error.message(), "missing-value builtin: invalid argument: ismissing: failed to gather input: delegated failure");
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<RuntimeError>()
            .unwrap();
        assert_eq!(source.identifier(), Some("RunMat:test:DelegatedFailure"));
        assert_eq!(source.source().unwrap().to_string(), "underlying operation");
    }
}

#[derive(Clone, Copy)]
enum Invocation {
    Missing,
    IsMissing,
    AnyMissing,
    StandardizeInput,
    StandardizeIndicators,
    Remove,
    Fill,
    MovingMad,
}

impl Invocation {
    fn builtin(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::IsMissing => "ismissing",
            Self::AnyMissing => "anymissing",
            Self::StandardizeInput | Self::StandardizeIndicators => "standardizeMissing",
            Self::Remove => "rmmissing",
            Self::Fill => "fillmissing",
            Self::MovingMad => "movmad",
        }
    }

    fn invoke(self, resident: Value) -> BuiltinResult<Value> {
        use super::super::entrypoints::*;
        match self {
            Self::Missing => block_on(missing_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![resident]),
            )),
            Self::IsMissing => block_on(ismissing_builtin(resident)),
            Self::AnyMissing => block_on(anymissing_builtin(resident)),
            Self::StandardizeInput => block_on(standardize_missing_builtin(
                None,
                resident,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
            )),
            Self::StandardizeIndicators => block_on(standardize_missing_builtin(
                None,
                Value::Num(1.0),
                crate::builtins::common::test_support::builtin_arguments(vec![resident]),
            )),
            Self::Remove => block_on(rmmissing_builtin(
                None,
                resident,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
            Self::Fill => block_on(fillmissing_builtin(
                None,
                resident,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("constant"),
                    Value::Num(0.0),
                ]),
            )),
            Self::MovingMad => block_on(movmad_builtin(
                resident,
                Value::Num(3.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            )),
        }
    }
}

#[test]
fn every_missing_gather_entrypoint_retains_provider_failure_without_replay() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(FailingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let _provider = ThreadProviderGuard::set(Some(provider));
    for (index, invocation) in [
        Invocation::Missing,
        Invocation::IsMissing,
        Invocation::AnyMissing,
        Invocation::StandardizeInput,
        Invocation::StandardizeIndicators,
        Invocation::Remove,
        Invocation::Fill,
        Invocation::MovingMad,
    ]
    .into_iter()
    .enumerate()
    {
        let error = invocation
            .invoke(Value::GpuTensor(allocation.handle.clone()))
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
        assert_eq!(error.context.builtin.as_deref(), Some(invocation.builtin()));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let source = error.source().expect("delegated gather failure");
        assert!(source
            .to_string()
            .contains("injected missing-data download failure"));
        assert!(
            source.source().is_some(),
            "underlying provider cause must survive"
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), index + 1);
        assert_eq!(allocation.probe.retirements(), 0);
    }
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}
