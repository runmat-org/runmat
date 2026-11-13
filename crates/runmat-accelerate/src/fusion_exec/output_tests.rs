use super::*;
use runmat_accelerate_api::{
    test_support::{TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, GpuTensorLayout, GpuTensorView, HostNumericDataView,
    HostNumericTensorView, HostTensorView, ThreadProviderGuard, TransposeInfo,
};
use std::future::Future;
use std::sync::Mutex;
use std::task::{Context, Poll};

struct Owner {
    allocations: TestAllocationOwner,
    precision: ProviderPrecision,
    upload_result: Mutex<Option<GpuTensorHandle>>,
    upload_failure: Mutex<Option<&'static str>>,
    uploads: Mutex<Vec<UploadCall>>,
}
#[derive(Debug, PartialEq, Eq)]
struct UploadCall {
    shape: Vec<usize>,
    storage: GpuTensorStorage,
    payload: UploadPayload,
}
#[derive(Debug, PartialEq, Eq)]
enum UploadPayload {
    F32(Vec<u32>),
    F64(Vec<u64>),
    Other(NumericElementType),
}
#[derive(Debug)]
struct UploadFailure(&'static str);
impl std::fmt::Display for UploadFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}
impl std::error::Error for UploadFailure {}

fn owner_with_precision(precision: ProviderPrecision) -> Owner {
    Owner {
        allocations: TestAllocationOwner::new(),
        precision,
        upload_result: Mutex::new(None),
        upload_failure: Mutex::new(None),
        uploads: Mutex::new(Vec::new()),
    }
}
fn owner() -> Owner {
    owner_with_precision(ProviderPrecision::F64)
}
fn new_owner() -> &'static Owner {
    Box::leak(Box::new(owner()))
}
fn new_owner_with_precision(precision: ProviderPrecision) -> &'static Owner {
    Box::leak(Box::new(owner_with_precision(precision)))
}
impl AccelProvider for Owner {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        self.precision
    }
    fn device_info(&self) -> String {
        "fusion-output-owner".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let payload = match host.data {
            runmat_accelerate_api::HostNumericDataView::F32(data) => {
                UploadPayload::F32(data.iter().map(|value| value.to_bits()).collect())
            }
            runmat_accelerate_api::HostNumericDataView::F64(data) => {
                UploadPayload::F64(data.iter().map(|value| value.to_bits()).collect())
            }
            other => UploadPayload::Other(other.element_type()),
        };
        self.uploads.lock().unwrap().push(UploadCall {
            shape: host.shape.to_vec(),
            storage: host.storage,
            payload,
        });
        if let Some(message) = self.upload_failure.lock().unwrap().take() {
            return Err(ProviderExecutionError::from_source(UploadFailure(message)));
        }
        Ok(self
            .upload_result
            .lock()
            .unwrap()
            .take()
            .expect("configured upload"))
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
}
fn allocation(owner: &Owner) -> TestAllocation {
    owner
        .allocations
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap()
}

// The registered inner provider and this facade have the same data address
// and namespace, but distinct trait-object vtables. Output admission must
// use the registry's facade identity rather than trait-object pointer equality.
#[repr(C)]
struct OwnerFacade {
    owner: Owner,
    reported_device: u32,
}

fn new_facade(reported_device: Option<u32>) -> &'static OwnerFacade {
    let owner = owner();
    let device = reported_device.unwrap_or_else(|| owner.device_id());
    Box::leak(Box::new(OwnerFacade {
        owner,
        reported_device: device,
    }))
}

impl AccelProvider for OwnerFacade {
    fn device_id(&self) -> u32 {
        self.reported_device
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "fusion-output-facade".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn output_admission_accepts_same_facade_across_vtables_but_not_distinct_facades() {
    let facade = new_facade(None);
    let owner: &'static dyn AccelProvider = &facade.owner;
    let selected: &'static dyn AccelProvider = facade;
    assert!(std::ptr::addr_eq(owner, selected));
    assert!(!std::ptr::eq(owner, selected));
    assert!(runmat_accelerate_api::same_provider_instance(
        owner, selected
    ));
    let _selected = ThreadProviderGuard::set(Some(owner));

    let mut admission = FusionOutputAdmission::new(&[]);
    let accepted = allocation(&facade.owner);
    let output = admission
        .admit(accepted.handle, selected, &[1, 2], ProviderPrecision::F64)
        .expect("same retained facade must be admitted despite vtable difference");
    assert_eq!(accepted.probe.retirements(), 0);
    drop(output);
    assert_eq!(accepted.probe.retirements(), 1);

    let distinct = new_facade(Some(facade.device_id()));
    assert!(!runmat_accelerate_api::same_provider_instance(
        owner, distinct
    ));
    let rejected = allocation(&facade.owner);
    assert!(admission
        .admit(rejected.handle, distinct, &[1, 2], ProviderPrecision::F64,)
        .is_err());
    assert_eq!(rejected.probe.retirements(), 1);

    let wrong_namespace = new_facade(Some(facade.device_id()));
    let inner: &dyn AccelProvider = &wrong_namespace.owner;
    let outer: &dyn AccelProvider = wrong_namespace;
    assert!(std::ptr::addr_eq(inner, outer));
    assert!(!runmat_accelerate_api::same_provider_instance(inner, outer));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn explained_results_preserve_class_and_reject_physical_input_aliases() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    for element in [NumericElementType::F32, NumericElementType::F64] {
        let allocation = owner
            .allocations
            .allocate_zeroed(vec![1, 2], element, GpuTensorStorage::Real)
            .unwrap();
        let admitted = crate::fusion_exec::explained_output::admit(
            allocation.handle.clone(),
            owner,
            &[1, 2],
            element,
            &[],
        )
        .unwrap();
        assert!(admitted.shares_allocation_with(&allocation.handle));
        let alias = admitted
            .with_view(GpuTensorView::contiguous(vec![2, 1]))
            .unwrap();
        assert!(crate::fusion_exec::explained_output::admit(
            alias,
            owner,
            &[2, 1],
            element,
            &[&allocation.handle],
        )
        .is_err());
        assert_eq!(allocation.probe.retirements(), 0);
        drop(admitted);
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn explained_results_reject_wrong_class_and_retain_cleanup_failure() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let allocation = owner
        .allocations
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    allocation.probe.fail_retirement("fixture retained result");
    let error = crate::fusion_exec::explained_output::admit(
        allocation.handle,
        owner,
        &[2, 1],
        NumericElementType::F64,
        &[],
    )
    .unwrap_err();
    assert!(std::error::Error::source(&error).is_some());
    assert!(
        !error.is_unsupported(),
        "retired outputs never authorize replay"
    );
    assert_eq!(allocation.probe.retirements(), 1);
    drop(error);
    assert_eq!(allocation.probe.retirements(), 1);
}
fn facts(
    handle: &GpuTensorHandle,
) -> (runmat_accelerate_api::GpuPhysicalDescriptor, GpuTensorView) {
    (*handle.physical(), handle.view().clone())
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_and_unpolled_drop_release_actual_outputs_once_and_retain_borrowed_aliases() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    for poll in [false, true] {
        let TestAllocation {
            handle: borrowed,
            probe: borrowed_probe,
        } = allocation(owner);
        let mut admission = FusionOutputAdmission::new(&[Value::GpuTensor(borrowed.clone())]);
        // A different lawful view is still the same borrowed physical allocation.
        let alias = borrowed
            .with_view(GpuTensorView::contiguous(vec![1, 2]))
            .unwrap();
        assert!(admission
            .admit(alias, owner, &[1, 2], ProviderPrecision::F64)
            .is_err());
        let TestAllocation {
            handle: output,
            probe: output_probe,
        } = allocation(owner);
        let output = admission
            .admit(output, owner, &[1, 2], ProviderPrecision::F64)
            .unwrap();
        let duplicate = admission
            .admit(output.clone(), owner, &[1, 2], ProviderPrecision::F64)
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_batch_releases_later_outputs_without_double_retirement() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    // Malformed operation result, not fabricated/mutable physical metadata.
    for wrong_count in [false, true] {
        let mut admission = FusionOutputAdmission::new(&[]);
        let first = allocation(owner);
        let invalid = owner
            .allocations
            .allocate_zeroed(
                vec![1, 2],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let last = allocation(owner);
        assert!(admission
            .admit_batch(
                vec![
                    first.handle,
                    invalid.handle,
                    last.handle.clone(),
                    last.handle
                ],
                owner,
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn validator_drop_preserves_retained_outputs_and_does_not_own_intermediates() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let mut admission = FusionOutputAdmission::new(&[]);
    let intermediate = allocation(owner);
    let output = allocation(owner);
    let intermediate_handle = admission
        .admit(intermediate.handle, owner, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    let output_handle = admission
        .admit(output.handle, owner, &[1, 2], ProviderPrecision::F64)
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn upload_rejection_protects_originals_and_live_staging_with_immutable_facts() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let borrowed = allocation(owner);
    let mut admission = FusionOutputAdmission::new(&[Value::GpuTensor(borrowed.handle.clone())]);
    let staged = allocation(owner);
    let staged_handle = admission
        .admit(staged.handle, owner, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    for protected in [&borrowed.handle, &staged_handle] {
        let before = facts(protected);
        *owner.upload_result.lock().unwrap() = Some(protected.clone());
        let error = admission.upload(owner, &[1.0, 2.0], &[1, 2]).unwrap_err();
        assert!(runmat_accelerate_api::is_invalid_numeric_upload_result(
            &error
        ));
        assert_eq!(facts(protected), before);
        assert_eq!(borrowed.probe.retirements(), 0);
        assert_eq!(staged.probe.retirements(), 0);
    }
    drop(owner.upload_result.lock().unwrap().take());
    drop(admission);
    assert_eq!(staged.probe.retirements(), 0);
    drop(staged_handle);
    assert_eq!(staged.probe.retirements(), 1);
    drop(borrowed.handle);
    assert_eq!(borrowed.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn expired_produced_weak_views_do_not_retain_or_block_later_uploads() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let mut admission = FusionOutputAdmission::new(&[]);
    let previous = allocation(owner);
    let previous_handle = admission
        .admit(previous.handle, owner, &[1, 2], ProviderPrecision::F64)
        .unwrap();
    drop(previous_handle);
    assert_eq!(previous.probe.retirements(), 1);

    let next = allocation(owner);
    *owner.upload_result.lock().unwrap() = Some(next.handle.clone());
    let uploaded = admission.upload(owner, &[0.0, 0.0], &[1, 2]).unwrap();
    assert!(uploaded.shares_allocation_with(&next.handle));
    drop(next.handle);
    assert_eq!(next.probe.retirements(), 0);
    drop(uploaded);
    assert_eq!(next.probe.retirements(), 1);
    drop(admission);
    assert_eq!(previous.probe.retirements(), 1);
    assert_eq!(next.probe.retirements(), 1);
    drop(owner.upload_result.lock().unwrap().take());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn admission_rejects_actual_wrong_owner_class_shape_and_layout() {
    let owner = new_owner();
    let _selected = ThreadProviderGuard::set(Some(owner));
    let other_owner = TestAllocationOwner::new();
    let cases = [
        other_owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap(),
        owner
            .allocations
            .allocate_zeroed(vec![1, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap(),
        owner
            .allocations
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap(),
        owner
            .allocations
            .allocate_logical_zeroed(vec![1, 2], ProviderPrecision::F64)
            .unwrap(),
    ];
    let mut admission = FusionOutputAdmission::new(&[]);
    for allocation in cases {
        assert!(admission
            .admit(allocation.handle, owner, &[1, 2], ProviderPrecision::F64)
            .is_err());
        assert_eq!(allocation.probe.retirements(), 1);
    }
    let allocation = allocation(owner);
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
        .admit(transposed, owner, &[1, 2], ProviderPrecision::F64)
        .is_err());
    assert_eq!(allocation.probe.retirements(), 1);
}

#[path = "outputs/typed_upload_tests.rs"]
mod typed_upload_tests;
