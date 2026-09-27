use super::*;
use crate::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Legacy {
    device: u32,
    returned: GpuTensorHandle,
    calls: AtomicUsize,
    freed: Mutex<Vec<GpuHandleIdentity>>,
}
impl Legacy {
    fn new() -> Self {
        let device = next_device_id();
        Self {
            device,
            returned: GpuTensorHandle::new(vec![1, 1], device, 1)
                .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real),
            calls: AtomicUsize::new(0),
            freed: Mutex::default(),
        }
    }
}
impl AccelProvider for Legacy {
    fn device_id(&self) -> u32 {
        self.device
    }
    fn device_info(&self) -> String {
        "numeric upload ownership fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.returned.clone())
    }
    fn upload_integer(&self, _: &HostIntegerTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.returned.clone())
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("fixture has no payload") })
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.freed
            .lock()
            .unwrap()
            .push((handle.device_id, handle.buffer_id));
        Ok(())
    }
}
fn double() -> HostNumericTensorView<'static> {
    HostNumericTensorView {
        data: HostNumericDataView::F64(&[1.0]),
        shape: &[1, 1],
        storage: GpuTensorStorage::Real,
    }
}

#[test]
fn numeric_upload_legacy_adapter_protects_valid_and_malformed_borrowed_aliases() {
    for integer in [false, true] {
        for malformed in [false, true] {
            let mut provider = Legacy::new();
            let mut view = double();
            if integer {
                view.data = HostNumericDataView::I64(&[9_007_199_254_740_993]);
                provider.returned.descriptor.element_type = Some(NumericElementType::I64);
            }
            let borrowed = provider.returned.clone();
            if malformed {
                provider.returned.shape = vec![7, 9];
                provider.returned.descriptor.storage = None;
            }
            let error = upload_numeric_protecting(&provider, &view, &[&borrowed]).unwrap_err();
            assert!(is_invalid_numeric_upload_result(&error));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert!(provider.freed.lock().unwrap().is_empty());
            assert_eq!(borrowed.shape, vec![1, 1]);
            assert_eq!(borrowed.descriptor.storage, Some(GpuTensorStorage::Real));
        }
    }
}

#[test]
fn numeric_upload_request_validation_precedes_raw_hook_and_success_is_validated() {
    let provider = Legacy::new();
    let mut invalid = double();
    invalid.shape = &[2, 2];
    assert!(provider.upload_numeric(&invalid).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        provider.upload_numeric(&double()).unwrap(),
        provider.returned
    );
    assert!(provider.freed.lock().unwrap().is_empty());
}

#[test]
fn numeric_upload_legacy_adapter_rejects_unsupported_native_storage_without_widening() {
    let provider = Legacy::new();
    for view in [
        HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        },
        HostNumericTensorView {
            data: HostNumericDataView::I64(&[9_007_199_254_740_993, 1]),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        },
    ] {
        let error = upload_numeric_protecting(&provider, &view, &[]).unwrap_err();
        assert!(error.to_string().contains("upload_numeric_raw"));
        assert!(!is_invalid_numeric_upload_result(&error));
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(provider.freed.lock().unwrap().is_empty());
}

#[test]
fn numeric_upload_fresh_rejection_uses_exact_or_invoked_owner_not_ambient() {
    let ambient = Box::leak(Box::new(Legacy::new()));
    let _selection = ThreadProviderGuard::set(Some(ambient));
    let mut invoked = Legacy::new();
    invoked.returned.shape = vec![4, 4];
    invoked.returned.device_id = next_device_id();
    let error = invoked.upload_numeric(&double()).unwrap_err();
    assert!(is_invalid_numeric_upload_result(&error));
    assert_eq!(invoked.freed.lock().unwrap().len(), 1);
    assert!(ambient.freed.lock().unwrap().is_empty());
    invoked.returned.device_id = ambient.device_id();
    assert!(invoked.upload_numeric(&double()).is_err());
    assert_eq!(invoked.freed.lock().unwrap().len(), 1);
    assert_eq!(ambient.freed.lock().unwrap().len(), 1);
}

struct Native {
    legacy: Legacy,
    observed: Mutex<Vec<(NumericElementType, GpuTensorStorage)>>,
}
impl AccelProvider for Native {
    fn upload_numeric(&self, _: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("the protected authority must bypass an overridden convenience method")
    }
    fn device_id(&self) -> u32 {
        self.legacy.device
    }
    fn device_info(&self) -> String {
        "native numeric upload fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("native transfers must not use legacy double upload")
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.legacy.download(handle)
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.legacy.free(handle)
    }
    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        if let HostNumericDataView::I64(values) = host.data {
            assert_eq!(values, &[9_007_199_254_740_993, -3]);
        }
        self.observed
            .lock()
            .unwrap()
            .push((host.element_type(), host.storage));
        Ok(
            GpuTensorHandle::new(host.shape.to_vec(), self.device_id(), 2)
                .with_numeric_descriptor(host.element_type(), host.storage),
        )
    }
}

#[test]
fn numeric_upload_protected_authority_bypasses_convenience_override_and_preserves_native_payloads()
{
    let provider = Native {
        legacy: Legacy::new(),
        observed: Mutex::default(),
    };
    for view in [
        HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        },
        HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, -2.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        },
        HostNumericTensorView {
            data: HostNumericDataView::I64(&[9_007_199_254_740_993, -3]),
            shape: &[1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        },
    ] {
        let result =
            upload_numeric_protecting(&provider, &view, &[&provider.legacy.returned]).unwrap();
        assert_eq!(result.descriptor.element_type, Some(view.element_type()));
        assert_eq!(result.descriptor.storage, Some(view.storage));
    }
    assert_eq!(provider.observed.lock().unwrap().len(), 3);
    assert!(provider.legacy.freed.lock().unwrap().is_empty());
}
