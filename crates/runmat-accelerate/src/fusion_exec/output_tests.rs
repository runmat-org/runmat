use super::*;
use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, GpuTensorLayout, GpuTensorView, HostTensorView, ThreadProviderGuard,
    TransposeInfo,
};
use std::future::Future;
use std::sync::{LazyLock, Mutex};
use std::task::{Context, Poll};

struct Owner {
    allocations: TestAllocationOwner,
    upload_result: Mutex<Option<WeakGpuTensorHandle>>,
}
static OWNER: LazyLock<Owner> = LazyLock::new(|| Owner {
    allocations: TestAllocationOwner::new(),
    upload_result: Mutex::new(None),
});
static SERIAL: Mutex<()> = Mutex::new(());
impl AccelProvider for Owner {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "fusion-output-owner".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn upload_numeric_raw(
        &self,
        _: &runmat_accelerate_api::HostNumericTensorView<'_>,
    ) -> anyhow::Result<GpuTensorHandle> {
        let weak = self
            .upload_result
            .lock()
            .unwrap()
            .clone()
            .expect("configured upload");
        Ok(weak.acquire()?)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
}
fn allocation() -> TestAllocation {
    OWNER
        .allocations
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap()
}
fn facts(
    handle: &GpuTensorHandle,
) -> (runmat_accelerate_api::GpuPhysicalDescriptor, GpuTensorView) {
    (*handle.physical(), handle.view().clone())
}

#[test]
fn pending_and_unpolled_drop_release_actual_outputs_once_and_retain_borrowed_aliases() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&*OWNER));
    for poll in [false, true] {
        let TestAllocation {
            handle: borrowed,
            probe: borrowed_probe,
        } = allocation();
        let mut admission = FusionOutputAdmission::new(&[Value::GpuTensor(borrowed.clone())]);
        // A different lawful view is still the same borrowed physical allocation.
        let alias = borrowed
            .with_view(GpuTensorView::contiguous(vec![1, 2]))
            .unwrap();
        assert!(admission
            .admit(alias, &*OWNER, &[1, 2], ProviderPrecision::F64)
            .is_err());
        let TestAllocation {
            handle: output,
            probe: output_probe,
        } = allocation();
        let output = admission
            .admit(output, &*OWNER, &[1, 2], ProviderPrecision::F64)
            .unwrap();
        let duplicate = admission
            .admit(output.clone(), &*OWNER, &[1, 2], ProviderPrecision::F64)
            .unwrap();
        let mut future = Box::pin(async move {
            let _admission = admission;
            let _outputs = (output, duplicate);
            futures::future::pending::<()>().await;
        });
        if poll {
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        }
        assert_eq!(output_probe.retirements(), 0);
        drop(future);
        assert_eq!(output_probe.retirements(), 1);
        assert_eq!(borrowed_probe.retirements(), 0);
        drop(borrowed);
        assert_eq!(borrowed_probe.retirements(), 1);
    }
}

#[test]
fn malformed_batch_releases_later_outputs_without_double_retirement() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&*OWNER));
    // Malformed operation result, not fabricated/mutable physical metadata.
    for wrong_count in [false, true] {
        let mut admission = FusionOutputAdmission::new(&[]);
        let first = allocation();
        let invalid = OWNER
            .allocations
            .allocate_zeroed(
                vec![1, 2],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let last = allocation();
        assert!(admission
            .admit_batch(
                vec![
                    first.handle,
                    invalid.handle,
                    last.handle.clone(),
                    last.handle
                ],
                &*OWNER,
                if wrong_count { 3 } else { 4 },
                &[1, 2],
                ProviderPrecision::F64,
            )
            .is_err());
        assert_eq!(first.probe.retirements(), 1);
        assert_eq!(invalid.probe.retirements(), 1);
        assert_eq!(last.probe.retirements(), 1);
        drop(admission);
        assert_eq!(last.probe.retirements(), 1);
    }
}

#[test]
fn validator_drop_preserves_retained_outputs_and_does_not_own_intermediates() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&*OWNER));
    let mut admission = FusionOutputAdmission::new(&[]);
    let intermediate = allocation();
    let output = allocation();
    let intermediate_handle = admission
        .admit(
            intermediate.handle,
            &*OWNER,
            &[1, 2],
            ProviderPrecision::F64,
        )
        .unwrap();
    let output_handle = admission
        .admit(output.handle, &*OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    drop(intermediate_handle);
    assert_eq!(
        intermediate.probe.retirements(),
        1,
        "validator must not own outputs"
    );
    assert_eq!(output.probe.retirements(), 0);
    drop(admission);
    assert_eq!(
        output.probe.retirements(),
        0,
        "caller still owns its output"
    );
    drop(output_handle);
    assert_eq!(output.probe.retirements(), 1);
}

#[test]
fn upload_rejection_protects_originals_and_live_staging_with_immutable_facts() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&*OWNER));
    let borrowed = allocation();
    let mut admission = FusionOutputAdmission::new(&[Value::GpuTensor(borrowed.handle.clone())]);
    let staged = allocation();
    let staged_handle = admission
        .admit(staged.handle, &*OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    for protected in [&borrowed.handle, &staged_handle] {
        let before = facts(protected);
        *OWNER.upload_result.lock().unwrap() = Some(protected.downgrade());
        let error = admission.upload(&*OWNER, &[1.0, 2.0], &[1, 2]).unwrap_err();
        assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
            &error
        ));
        assert_eq!(facts(protected), before);
        assert_eq!(borrowed.probe.retirements(), 0);
        assert_eq!(staged.probe.retirements(), 0);
    }
    drop(OWNER.upload_result.lock().unwrap().take());
    drop(admission);
    assert_eq!(staged.probe.retirements(), 0);
    drop(staged_handle);
    assert_eq!(staged.probe.retirements(), 1);
    drop(borrowed.handle);
    assert_eq!(borrowed.probe.retirements(), 1);
}

#[test]
fn expired_produced_weak_views_do_not_retain_or_block_later_uploads() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&*OWNER));
    let mut admission = FusionOutputAdmission::new(&[]);
    let previous = allocation();
    let previous_handle = admission
        .admit(previous.handle, &*OWNER, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    drop(previous_handle);
    assert_eq!(previous.probe.retirements(), 1);

    let next = allocation();
    *OWNER.upload_result.lock().unwrap() = Some(next.handle.downgrade());
    let uploaded = admission.upload(&*OWNER, &[0.0, 0.0], &[1, 2]).unwrap();
    assert!(uploaded.shares_allocation_with(&next.handle));
    drop(next.handle);
    assert_eq!(next.probe.retirements(), 0);
    drop(uploaded);
    assert_eq!(next.probe.retirements(), 1);
    drop(admission);
    assert_eq!(previous.probe.retirements(), 1);
    assert_eq!(next.probe.retirements(), 1);
    drop(OWNER.upload_result.lock().unwrap().take());
}

#[test]
fn admission_rejects_actual_wrong_owner_class_shape_and_layout() {
    let _serial = SERIAL.lock().unwrap();
    let _selected = ThreadProviderGuard::set(Some(&*OWNER));
    let other_owner = TestAllocationOwner::new();
    let cases = [
        other_owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap(),
        OWNER
            .allocations
            .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap(),
        OWNER
            .allocations
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap(),
        OWNER
            .allocations
            .allocate_logical_zeroed(vec![1, 2], ProviderPrecision::F64)
            .unwrap(),
    ];
    let mut admission = FusionOutputAdmission::new(&[]);
    for allocation in cases {
        assert!(admission
            .admit(allocation.handle, &*OWNER, &[1, 2], ProviderPrecision::F64)
            .is_err());
        assert_eq!(allocation.probe.retirements(), 1);
    }
    let allocation = allocation();
    let transposed =
        allocation
            .handle
            .with_view(GpuTensorView::contiguous(vec![1, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 1,
                }),
            ))
            .unwrap();
    drop(allocation.handle);
    assert!(admission
        .admit(transposed, &*OWNER, &[1, 2], ProviderPrecision::F64)
        .is_err());
    assert_eq!(allocation.probe.retirements(), 1);
}
