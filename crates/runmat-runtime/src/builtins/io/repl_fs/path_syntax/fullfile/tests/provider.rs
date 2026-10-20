mod fixture;

use fixture::{ControlledDownloadProvider, DownloadBehavior};
use runmat_accelerate_api::ThreadProviderGuard;
use runmat_value::{CharArray, Tensor, Value};
use std::future::Future;
use std::sync::atomic::Ordering;

#[test]
fn resident_numeric_character_row_gathers_without_kernel_dispatch() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor =
            Tensor::new(vec![100.0, 97.0, 116.0, 97.0], vec![1, 4]).expect("numeric character row");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("resident numeric character row");
        let before = provider.telemetry_snapshot();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);

        let output = super::run(vec![
            Value::GpuTensor(handle.clone()),
            Value::CharArray(CharArray::new_row("sample.dat")),
        ])
        .expect("resident character row gathers");

        let path = String::try_from(&output).expect("character path");
        let expected = std::path::Path::new("data")
            .join("sample.dat")
            .to_string_lossy()
            .into_owned();
        assert_eq!(path, expected);
        let after = provider.telemetry_snapshot();
        assert!(after.download_bytes > before.download_bytes);
        assert_eq!(after.kernel_launches, before.kernel_launches);
        handle.allocation_lease().validate_live().unwrap();
    });
}

fn controlled(
    behavior: DownloadBehavior,
) -> (
    &'static ControlledDownloadProvider,
    ThreadProviderGuard,
    runmat_accelerate_api::test_support::TestAllocation,
) {
    let provider = Box::leak(Box::new(ControlledDownloadProvider::new(behavior)));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider.numeric_character_row();
    (provider, selected, allocation)
}

#[test]
fn matlab_mode_rejects_resident_numeric_extension_before_provider_lookup() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let allocation =
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 2]);
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(false);

    let error = super::run(vec![Value::GpuTensor(allocation.handle.clone())])
        .expect_err("MATLAB mode rejects numeric character codes");

    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FullfileNumericCharacterCodesExtension")
    );
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn resident_provider_failure_preserves_gather_source_and_borrowed_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let (provider, _selected, allocation) = controlled(DownloadBehavior::Fail);
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);

    let error = super::run(vec![Value::GpuTensor(allocation.handle.clone())])
        .expect_err("provider download fails");

    assert_eq!(error.identifier(), Some("RunMat:fullfile:ProviderFailed"));
    let gather = error
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<crate::RuntimeError>())
        .expect("fullfile retains the gather failure");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    let provider_error = gather
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>())
        .expect("gather retains the provider failure");
    let io_error = std::error::Error::source(provider_error)
        .and_then(|source| source.downcast_ref::<std::io::Error>())
        .expect("typed provider source");
    assert_eq!(io_error.kind(), std::io::ErrorKind::Other);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn unpolled_and_abandoned_resident_gather_has_no_replay_and_exact_lifetime() {
    use std::task::{Context, Poll};

    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let (provider, _selected, allocation) = controlled(DownloadBehavior::Pending);
    let retained = allocation.handle.allocation_lease().downgrade();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);

    let unpolled =
        super::super::execute::run(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::GpuTensor(allocation.handle.clone())],
        ));
    drop(unpolled);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert!(retained.acquire().is_some());

    let mut pending = Box::pin(super::super::execute::run(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            allocation.handle.clone(),
        )]),
    ));
    drop(allocation.handle);
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert!(retained.acquire().is_some());

    drop(pending);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert!(retained.acquire().is_none());
    assert_eq!(allocation.probe.retirements(), 1);
}
