//! Known host controls must be rejected before resident materialization or I/O.

use super::publication::{with_lifecycle, LifecycleBehavior, LifecycleEvidence};
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

struct PendingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for PendingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "xlswrite admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { futures::future::pending().await })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { futures::future::pending().await })
    }
}

fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    Future::poll(
        future,
        &mut Context::from_waker(futures::task::noop_waker_ref()),
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_host_controls_precede_resident_download_and_filesystem_effects() {
    let _accel = crate::builtins::common::test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(PendingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident data");
    let resident = Value::GpuTensor(allocation.handle.clone());

    let evidence = Arc::new(LifecycleEvidence::default());
    with_lifecycle(
        LifecycleBehavior::Pending(evidence),
        0,
        |memory, evidence| {
            for (arguments, expected) in [
                (vec![Value::Num(0.0)], XLSWRITE_ERROR_SHEET.identifier),
                (
                    vec![Value::from("Sheet1"), Value::from("A0")],
                    XLSWRITE_ERROR_RANGE.identifier,
                ),
                (
                    vec![
                        Value::from("Sheet1"),
                        Value::from("A1"),
                        Value::from("unexpected"),
                    ],
                    XLSWRITE_ERROR_INVALID_ARGUMENT.identifier,
                ),
                (
                    vec![Value::Num(1.0); 4],
                    XLSWRITE_ERROR_INVALID_ARGUMENT.identifier,
                ),
                (
                    vec![resident.clone(), Value::from("A0")],
                    XLSWRITE_ERROR_RANGE.identifier,
                ),
            ] {
                let mut call = Box::pin(xlswrite_builtin(
                    None,
                    Value::from("out.xlsx"),
                    resident.clone(),
                    crate::builtins::common::test_support::builtin_arguments(arguments),
                ));
                let Poll::Ready(Err(error)) = poll_once(call.as_mut()) else {
                    panic!("known-invalid host control must reject before download")
                };
                assert_eq!(error.identifier(), expected);
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                assert_eq!(evidence.begins.load(Ordering::SeqCst), 0);
                assert_eq!(evidence.resource_probes.load(Ordering::SeqCst), 0);
                assert!(memory
                    .read_project_path(std::path::Path::new("/publication/out.xlsx"))
                    .is_err());
            }
        },
    );
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn valid_resident_data_still_reaches_the_existing_download_route() {
    let _accel = crate::builtins::common::test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(PendingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident data");
    let evidence = Arc::new(LifecycleEvidence::default());
    with_lifecycle(
        LifecycleBehavior::Pending(evidence),
        0,
        |_memory, evidence| {
            let mut call = Box::pin(xlswrite_builtin(
                None,
                Value::from("out.xlsx"),
                Value::GpuTensor(allocation.handle.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("Sheet1"),
                    Value::from("A1"),
                ]),
            ));
            assert!(poll_once(call.as_mut()).is_pending());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            drop(call);
            assert_eq!(evidence.begins.load(Ordering::SeqCst), 0);
            assert!(evidence.resource_probes.load(Ordering::SeqCst) > 0);
        },
    );
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn host_known_invalid_data_precedes_resident_cell_download_and_fs_admission() {
    let _accel = crate::builtins::common::test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(PendingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident cell member");
    let mut invalid = runmat_value::UnboundedStructValueBuilder::new();
    invalid.insert("x", Value::Num(1.0));
    let invalid = invalid.finish();
    let resident_then_invalid = Value::Cell(
        CellArray::new(
            vec![
                Value::GpuTensor(allocation.handle.clone()),
                Value::Struct(invalid.clone()),
            ],
            1,
            2,
        )
        .expect("cell fixture"),
    );
    let evidence = Arc::new(LifecycleEvidence::default());
    with_lifecycle(
        LifecycleBehavior::Pending(evidence),
        0,
        |memory, evidence| {
            for data in [Value::Struct(invalid), resident_then_invalid] {
                let mut call = Box::pin(xlswrite_builtin(
                    None,
                    Value::from("out.xlsx"),
                    data,
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::from("Sheet1"),
                        Value::from("A1"),
                    ]),
                ));
                let Poll::Ready(Err(error)) = poll_once(call.as_mut()) else {
                    panic!("known-invalid data must reject before resident download")
                };
                assert_eq!(error.identifier(), XLSWRITE_ERROR_DATA.identifier);
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                assert_eq!(evidence.resource_probes.load(Ordering::SeqCst), 0);
                assert_eq!(evidence.begins.load(Ordering::SeqCst), 0);
                assert!(memory
                    .read_project_path(std::path::Path::new("/publication/out.xlsx"))
                    .is_err());
            }
        },
    );
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn admitted_resident_cell_member_still_enters_download() {
    let _accel = crate::builtins::common::test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(PendingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident cell member");
    let data = Value::Cell(
        CellArray::new(vec![Value::GpuTensor(allocation.handle.clone())], 1, 1)
            .expect("cell fixture"),
    );
    let evidence = Arc::new(LifecycleEvidence::default());
    with_lifecycle(
        LifecycleBehavior::Pending(evidence),
        0,
        |_memory, evidence| {
            let mut call = Box::pin(xlswrite_builtin(
                None,
                Value::from("out.xlsx"),
                data,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("Sheet1"),
                    Value::from("A1"),
                ]),
            ));
            assert!(poll_once(call.as_mut()).is_pending());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            drop(call);
            assert_eq!(evidence.begins.load(Ordering::SeqCst), 0);
            assert!(evidence.resource_probes.load(Ordering::SeqCst) > 0);
        },
    );
    assert_eq!(allocation.probe.retirements(), 0);
}
