//! Registered-provider admission and borrowed-resource witnesses.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned,
    HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
struct RejectDownloads {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
    pending: AtomicBool,
    succeed: AtomicBool,
}

impl AccelProvider for RejectDownloads {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "scatterhist registered-owner fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("scatterhist host admission must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        Box::pin(async {
            Err(std::io::Error::other("scatterhist fixture rejected download").into())
        })
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.pending.load(Ordering::SeqCst) {
            return Box::pin(futures::future::pending());
        }
        if self.succeed.load(Ordering::SeqCst) {
            return Box::pin(async move {
                Ok(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(vec![1.0; handle.shape().iter().product()]),
                    shape: handle.shape().to_vec(),
                    storage: GpuTensorStorage::Real,
                })
            });
        }
        Box::pin(async {
            Err(std::io::Error::other("scatterhist fixture rejected download").into())
        })
    }
}

#[test]
fn nested_group_preflight_rejects_known_invalid_cells_before_any_download() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let scalar = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let vector = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let complex = provider
        .owner
        .allocate_zeroed(
            vec![1, 1],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap();
    for bad in [
        Value::StringArray(
            runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
        ),
        Value::Cell(CellArray::new(vec![Value::String("nested".into())], 1, 1).unwrap()),
        Value::GpuTensor(vector.handle.clone()),
        Value::GpuTensor(complex.handle.clone()),
    ] {
        let group = Value::Cell(
            CellArray::new(vec![Value::GpuTensor(scalar.handle.clone()), bad], 2, 1).unwrap(),
        );
        let observations = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let err = block_on(scatterhist_builtin(vec![
            observations(),
            observations(),
            Value::String("Group".into()),
            group,
        ]))
        .expect_err("known-invalid nested group must preflight");
        assert_eq!(
            err.identifier(),
            SCATTERHIST_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    for handle in [&scalar.handle, &vector.handle, &complex.handle] {
        handle.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn nested_scalar_resident_group_decodes_through_typed_key() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    provider.succeed.store(true, Ordering::SeqCst);
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let scalar = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let group =
        Value::Cell(CellArray::new(vec![Value::GpuTensor(scalar.handle.clone())], 1, 1).unwrap());
    let options = block_on(
        ScatterhistOptionPlan::admit(
            crate::call::BuiltinArguments::from_borrowed(
                &[Value::String("Group".into()), group],
                None,
            )
            .unwrap(),
            1,
        )
        .unwrap()
        .evaluate(1),
    )
    .unwrap();
    assert_eq!(options.group, Some(vec![Some(KeyAtom::Number(1.0))]));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    scalar.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(scalar.probe.retirements(), 0);
}

fn sample() -> Value {
    Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap())
}

#[test]
fn known_invalid_inputs_and_later_options_preflight_without_download() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let option = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for args in [
        vec![
            Value::GpuTensor(resident.handle.clone()),
            sample(),
            Value::String("NBins".into()),
            Value::Num(1.0),
        ],
        vec![
            sample(),
            Value::GpuTensor(resident.handle.clone()),
            Value::String("DefinitelyUnknown".into()),
            Value::Num(2.0),
        ],
        vec![
            sample(),
            sample(),
            Value::String("NBins".into()),
            Value::GpuTensor(option.handle.clone()),
            Value::String("MarkerSize".into()),
            Value::Num(-1.0),
        ],
        vec![
            Value::String("bad x".into()),
            Value::GpuTensor(resident.handle.clone()),
        ],
    ] {
        let err = block_on(scatterhist_builtin(args)).expect_err("known-invalid call");
        assert_eq!(
            err.identifier(),
            SCATTERHIST_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        resident.handle.allocation_lease().validate_live().unwrap();
        option.handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(resident.probe.retirements(), 0);
    assert_eq!(option.probe.retirements(), 0);
}

#[test]
fn nested_resident_group_is_compatibility_gated_before_download() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let group = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let nested =
        Value::Cell(CellArray::new(vec![Value::GpuTensor(group.handle.clone())], 1, 1).unwrap());
    let err = block_on(scatterhist_builtin(vec![
        sample(),
        sample(),
        Value::String("Group".into()),
        nested,
    ]))
    .expect_err("nested resident group is a RunMat extension");
    assert_eq!(
        err.identifier(),
        SCATTERHIST_GPU_INPUT_EXTENSION.error_identifier
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    group.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn failed_download_preserves_source_and_borrowed_lease() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let err = block_on(scatterhist_builtin(vec![
        Value::GpuTensor(resident.handle.clone()),
        sample(),
    ]))
    .expect_err("provider rejects download");
    assert_eq!(
        err.identifier(),
        SCATTERHIST_ERROR_INVALID_ARGUMENT.identifier
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    let gather = err
        .source
        .as_ref()
        .and_then(|source| source.downcast_ref::<RuntimeError>())
        .expect("nested gather error");
    assert_eq!(gather.identifier(), Some("RunMat:gather:DownloadFailed"));
    assert_eq!(err.span, gather.span);
    assert_eq!(err.gpu_gather_retry, gather.gpu_gather_retry);
    let mut source: &(dyn std::error::Error + 'static) = &err;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("provider cause retained");
    }
    assert_eq!(source.to_string(), "scatterhist fixture rejected download");
    resident.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(resident.probe.retirements(), 0);
}

#[test]
fn unpolled_and_abandoned_pending_gather_preserve_borrowed_input() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectDownloads::default()));
    provider.pending.store(true, Ordering::SeqCst);
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let value = || Value::GpuTensor(resident.handle.clone());
    drop(scatterhist_builtin(vec![value(), sample()]));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let mut pending = Box::pin(scatterhist_builtin(vec![value(), sample()]));
    let waker = futures::task::noop_waker();
    let mut context = std::task::Context::from_waker(&waker);
    assert!(matches!(
        pending.as_mut().poll(&mut context),
        std::task::Poll::Pending
    ));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(pending);
    resident.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(resident.probe.retirements(), 0);
}
