use super::*;
use runmat_accelerate_api::{AccelDownloadFuture, HostTensorView, ThreadProviderGuard};
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll};

struct Owner(Mutex<Vec<u64>>);
static OWNER: Owner = Owner(Mutex::new(Vec::new()));
static SERIAL: Mutex<()> = Mutex::new(());
static UPLOAD_RESULT: AtomicU64 = AtomicU64::new(0);
impl AccelProvider for Owner {
    fn device_id(&self) -> u32 {
        u32::MAX - 283
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "fusion-output-owner".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn upload_numeric_raw(
        &self,
        _: &runmat_accelerate_api::HostNumericTensorView<'_>,
    ) -> anyhow::Result<GpuTensorHandle> {
        let output = handle(UPLOAD_RESULT.load(Ordering::SeqCst));
        runmat_accelerate_api::set_handle_logical(&output, true);
        Ok(output)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(handle.buffer_id);
        Ok(())
    }
}
fn handle(id: u64) -> GpuTensorHandle {
    GpuTensorHandle::new(vec![1, 2], OWNER.device_id(), id)
        .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real)
}

#[test]
fn pending_drop_releases_owned_outputs_once_and_retains_borrowed_aliases() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&OWNER));
    OWNER.0.lock().unwrap().clear();
    let borrowed = handle(1);
    let mut owner = FusionAllocations::new(&[Value::GpuTensor(borrowed.clone())]);
    assert!(owner
        .admit(borrowed, &OWNER, &[1, 2], ProviderPrecision::F64)
        .is_err());
    owner
        .admit(handle(2), &OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    owner
        .admit(handle(2), &OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    let mut future = Box::pin(async move {
        let _owner = owner;
        clear_handle_metadata(&handle(2));
        futures::future::pending::<()>().await;
    });
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    drop(future);
    assert_eq!(*OWNER.0.lock().unwrap(), vec![2]);
}

#[test]
fn malformed_batch_releases_later_outputs_without_double_free() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&OWNER));
    OWNER.0.lock().unwrap().clear();
    let mut owner = FusionAllocations::new(&[]);
    let mut invalid = handle(4);
    invalid.descriptor.storage = None;
    assert!(owner
        .admit_batch(
            vec![handle(3), invalid, handle(5), handle(5)],
            &OWNER,
            4,
            &[1, 2],
            ProviderPrecision::F64
        )
        .is_err());
    drop(owner);
    let mut freed = OWNER.0.lock().unwrap().clone();
    freed.sort_unstable();
    assert_eq!(freed, vec![3, 4, 5]);
}

#[test]
fn publication_keeps_only_outputs_and_releases_intermediates() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&OWNER));
    OWNER.0.lock().unwrap().clear();
    let mut owner = FusionAllocations::new(&[]);
    owner
        .admit(handle(6), &OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    let output = owner
        .admit(handle(7), &OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    owner.publish(&[&output]);
    assert_eq!(*OWNER.0.lock().unwrap(), vec![6]);
}

#[test]
fn upload_rejection_protects_both_originals_and_prior_owned_staging_metadata() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&OWNER));
    OWNER.0.lock().unwrap().clear();
    let borrowed = handle(10);
    let mut owner = FusionAllocations::new(&[Value::GpuTensor(borrowed.clone())]);
    let staged = owner
        .admit(handle(11), &OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    for protected in [&borrowed, &staged] {
        let before = runmat_accelerate_api::snapshot_handle_metadata(protected);
        UPLOAD_RESULT.store(protected.buffer_id, Ordering::SeqCst);
        let error = owner.upload(&OWNER, &[1.0, 2.0], &[1, 2]).unwrap_err();
        assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
            &error
        ));
        assert_eq!(
            runmat_accelerate_api::snapshot_handle_metadata(protected),
            before
        );
        assert!(OWNER.0.lock().unwrap().is_empty());
    }
    drop(owner);
    assert_eq!(*OWNER.0.lock().unwrap(), vec![11]);
}
