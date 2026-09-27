use super::*;
use crate::native_device::*;
use crate::{AccelProvider, GpuTensorDescriptor, GpuTensorStorage, NumericElementType};

struct ExportProvider {
    bytes: usize,
    calls: std::sync::atomic::AtomicUsize,
}
impl ExportProvider {
    fn new(bytes: usize) -> Self {
        Self {
            bytes,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}
impl AccelProvider for ExportProvider {
    fn upload(&self, _: &crate::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused test upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> crate::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused test download") })
    }
    fn device_info(&self) -> String {
        "owning export fixture".into()
    }
    fn device_id(&self) -> u32 {
        ID.device_id()
    }
    fn native_device_api(&self) -> Option<NativeDeviceApi> {
        Some(NativeDeviceApi::Cuda)
    }
    fn export_native_device_buffer_raw(
        &self,
        _: &GpuTensorHandle,
        _: NativeDeviceAccess,
    ) -> anyhow::Result<NativeDeviceBuffer> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Opaque fixture address is never dereferenced or submitted to a driver.
        Ok(NativeDeviceBuffer {
            context: NativeDeviceContext {
                api: NativeDeviceApi::Cuda,
                provider_device_id: ID.device_id(),
                device_ordinal: 0,
                context_identity: 1,
                stream_identity: 2,
            },
            device_address: 1,
            byte_length: self.bytes,
            descriptor: GpuTensorDescriptor::numeric(
                NumericElementType::U8,
                GpuTensorStorage::Real,
            ),
        })
    }
}

#[test]
fn exported_address_retains_physical_owner_after_view_release() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let handle = GpuTensorHandle::from_allocation(
        AllocationLease::register(ID, owner.clone()).unwrap(),
        GpuTensorView::contiguous(vec![1]),
    )
    .unwrap();
    let exported = export_native_device_buffer(
        &ExportProvider::new(2),
        &handle,
        NativeDeviceAccess::ReadOnly,
    )
    .unwrap();
    assert_eq!(exported.handle().shape(), &[1]);
    assert_eq!(exported.address_view().byte_length, 2);
    assert_eq!(
        handle.release().unwrap(),
        AllocationReleaseOutcome::ReferenceReleased
    );
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
    drop(exported);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn invalid_export_extent_does_not_retire_borrowed_source() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let handle = GpuTensorHandle::from_allocation(
        AllocationLease::register(ID, owner.clone()).unwrap(),
        GpuTensorView::contiguous(vec![2]),
    )
    .unwrap();
    for bytes in [0, 1, 3] {
        assert!(export_native_device_buffer(
            &ExportProvider::new(bytes),
            &handle,
            NativeDeviceAccess::ReadWrite
        )
        .is_err());
        assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(handle.release().unwrap(), AllocationReleaseOutcome::Retired);
}

#[test]
fn raw_native_hook_is_not_called_for_unrepresentable_transpose() {
    let source = crate::test_support::handle(
        ID.device_id(),
        vec![2, 3],
        NumericElementType::F64,
        GpuTensorStorage::Real,
    );
    let view = source
        .with_view(source.view().clone().with_shape(vec![3, 2]).with_layout(
            GpuTensorLayout::Transpose(crate::TransposeInfo {
                base_rows: 2,
                base_cols: 3,
            }),
        ))
        .unwrap();
    let provider = ExportProvider::new(48);
    let error =
        export_native_device_buffer(&provider, &view, NativeDeviceAccess::ReadOnly).unwrap_err();
    assert!(crate::is_unsupported_provider_operation(&error));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(source.shape(), &[2, 3]);
}
