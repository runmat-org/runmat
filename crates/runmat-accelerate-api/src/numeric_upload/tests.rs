use super::*;
use crate::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct Legacy {
    device: u32,
    result: Mutex<Option<GpuTensorHandle>>,
    calls: AtomicUsize,
}
impl Legacy {
    fn returning(device: u32, handle: GpuTensorHandle) -> Self {
        Self {
            device,
            result: Mutex::new(Some(handle)),
            calls: AtomicUsize::new(0),
        }
    }
}
impl AccelProvider for Legacy {
    fn device_id(&self) -> u32 {
        self.device
    }
    fn device_info(&self) -> String {
        "owning numeric upload fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.result
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| anyhow::anyhow!("fixture consumed"))
    }
    fn upload_integer(&self, _: &HostIntegerTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.upload(&HostTensorView {
            data: &[],
            shape: &[0],
        })
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused fixture download") })
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
fn numeric_upload_protects_same_view_and_different_view_of_borrowed_allocation() {
    for different_view in [false, true] {
        let (borrowed, owner) = crate::test_support::owned(
            17,
            vec![1, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
            false,
        );
        let returned = if different_view {
            borrowed
                .with_view(GpuTensorView::contiguous(vec![1]))
                .unwrap()
        } else {
            borrowed.clone()
        };
        let provider = Legacy::returning(17, returned);
        let before = borrowed.wire_descriptor();
        let error = upload_numeric_protecting(&provider, &double(), &[&borrowed]).unwrap_err();
        assert!(is_invalid_numeric_upload_result(&error));
        assert_eq!(borrowed.wire_descriptor(), before);
        assert_eq!(owner.retired.load(Ordering::SeqCst), 0);
        drop(borrowed);
        assert_eq!(owner.retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn numeric_upload_request_validation_precedes_raw_hook_and_success_is_validated() {
    let (handle, owner) = crate::test_support::owned(
        17,
        vec![1, 1],
        NumericElementType::F64,
        GpuTensorStorage::Real,
        false,
    );
    let provider = Legacy::returning(17, handle);
    let mut invalid = double();
    invalid.shape = &[2, 2];
    assert!(provider.upload_numeric(&invalid).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let result = provider.upload_numeric(&double()).unwrap();
    assert_eq!(result.shape(), &[1, 1]);
    assert_eq!(owner.retired.load(Ordering::SeqCst), 0);
    drop(result);
    assert_eq!(owner.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn fresh_rejection_releases_captured_owner_not_invoked_or_ambient_provider() {
    for (device, shape, element, storage, logical, transpose) in [
        (
            18,
            vec![1, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
            false,
            false,
        ),
        (
            17,
            vec![2, 2],
            NumericElementType::F64,
            GpuTensorStorage::Real,
            false,
            false,
        ),
        (
            17,
            vec![1, 1],
            NumericElementType::F32,
            GpuTensorStorage::Real,
            false,
            false,
        ),
        (
            17,
            vec![1, 1],
            NumericElementType::F64,
            GpuTensorStorage::ComplexInterleaved,
            false,
            false,
        ),
        (
            17,
            vec![1, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
            true,
            false,
        ),
        (
            17,
            vec![1, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
            false,
            true,
        ),
    ] {
        let (handle, owner) = crate::test_support::owned(device, shape, element, storage, logical);
        let returned = if transpose {
            handle
                .with_view(
                    handle
                        .view()
                        .clone()
                        .with_layout(GpuTensorLayout::Transpose(TransposeInfo {
                            base_rows: 1,
                            base_cols: 1,
                        })),
                )
                .unwrap()
        } else {
            handle.clone()
        };
        drop(handle);
        let provider = Legacy::returning(17, returned);
        let error = provider.upload_numeric(&double()).unwrap_err();
        assert!(is_invalid_numeric_upload_result(&error));
        assert_eq!(owner.retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn legacy_adapter_keeps_native_wide_integer_and_rejects_unsupported_representations() {
    let handle = crate::test_support::handle(
        17,
        vec![1, 1],
        NumericElementType::I64,
        GpuTensorStorage::Real,
    );
    let provider = Legacy::returning(17, handle);
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
        let error = provider.upload_numeric(&view).unwrap_err();
        assert!(!is_invalid_numeric_upload_result(&error));
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let view = HostNumericTensorView {
        data: HostNumericDataView::I64(&[9_007_199_254_740_993]),
        shape: &[1, 1],
        storage: GpuTensorStorage::Real,
    };
    let result = provider.upload_numeric(&view).unwrap();
    assert_eq!(result.physical().element_type(), NumericElementType::I64);
}

struct Native;
impl AccelProvider for Native {
    fn device_id(&self) -> u32 {
        17
    }
    fn device_info(&self) -> String {
        "native fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("native upload must not widen")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused") })
    }
    fn upload_numeric(&self, _: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("validated free function must bypass convenience override")
    }
    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        if let HostNumericDataView::I64(values) = host.data {
            assert_eq!(values, &[9_007_199_254_740_993, -3]);
        }
        Ok(crate::test_support::handle(
            17,
            host.shape.to_vec(),
            host.element_type(),
            host.storage,
        ))
    }
}
#[test]
fn protected_authority_bypasses_convenience_override_without_widening() {
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
        let result = upload_numeric_protecting(&Native, &view, &[]).unwrap();
        assert_eq!(result.physical().element_type(), view.element_type());
        assert_eq!(result.physical().storage(), view.storage);
    }
}
