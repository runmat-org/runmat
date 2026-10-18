use super::*;
use crate::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Provider {
    element: NumericElementType,
    logical: bool,
    calls: AtomicUsize,
}
impl Provider {
    fn new(element: NumericElementType, logical: bool) -> Self {
        Self {
            element,
            logical,
            calls: AtomicUsize::new(0),
        }
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        17
    }
    fn device_info(&self) -> String {
        "logical producer fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("logical upload must not use numeric hook")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused") })
    }
    fn upload_logical(&self, _: &HostLogicalTensorView) -> NumericUploadResult<GpuTensorHandle> {
        panic!("protected admission must bypass convenience override")
    }
    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(host.data.iter().all(|&value| value == 0));
        Ok(crate::test_support::owned(
            17,
            host.shape.to_vec(),
            self.element,
            GpuTensorStorage::Real,
            self.logical,
        )
        .0)
    }
}

#[test]
fn logical_host_admission_rejects_invalid_data_before_provider() {
    let provider = Provider::new(NumericElementType::F32, true);
    for view in [
        HostLogicalTensorView {
            data: &[2],
            shape: &[1],
        },
        HostLogicalTensorView {
            data: &[0],
            shape: &[2],
        },
        HostLogicalTensorView {
            data: &[],
            shape: &[usize::MAX, 2],
        },
    ] {
        assert!(upload_logical_protecting(&provider, &view, &[]).is_err());
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn logical_upload_preserves_explicit_backend_encoding_nd_and_empty_shape() {
    for element in [
        NumericElementType::U8,
        NumericElementType::F32,
        NumericElementType::F64,
    ] {
        let provider = Provider::new(element, true);
        for view in [
            HostLogicalTensorView {
                data: &[0, 0, 0, 0],
                shape: &[1, 2, 2],
            },
            HostLogicalTensorView {
                data: &[],
                shape: &[usize::MAX, usize::MAX, 0],
            },
        ] {
            let result = upload_logical_protecting(&provider, &view, &[]).unwrap();
            assert_eq!(result.shape(), view.shape);
            assert_eq!(result.physical().element_type(), element);
            assert!(result.view().is_logical());
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn ordinary_numeric_result_is_not_relabelled_as_logical() {
    let provider = Provider::new(NumericElementType::U8, false);
    let error = upload_logical_protecting(
        &provider,
        &HostLogicalTensorView {
            data: &[0],
            shape: &[1],
        },
        &[],
    )
    .unwrap_err();
    assert!(is_invalid_numeric_upload_result(&error));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn default_logical_unsupported_is_certified_before_effects() {
    #[derive(Debug)]
    struct Admission;
    impl runmat_execution::resource::ProviderHostMetadataAdmission for Admission {
        fn claim_view(
            &self,
            _: u64,
        ) -> Result<
            Box<dyn runmat_execution::resource::RetainedHostViewGrant>,
            runmat_execution::resource::ProviderHostMetadataError,
        > {
            panic!("unsupported hook must not consume view admission")
        }
        fn claim_record(
            &self,
            _: u64,
        ) -> Result<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
            runmat_execution::resource::ProviderHostMetadataError,
        > {
            panic!("unsupported hook must not consume record admission")
        }
    }
    struct Unavailable;
    impl AccelProvider for Unavailable {
        fn device_id(&self) -> u32 {
            19
        }
        fn device_info(&self) -> String {
            "unavailable logical fixture".into()
        }
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            panic!("numeric upload is unrelated")
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { panic!("download is unrelated") })
        }
    }
    let error = upload_logical_protecting(
        &Unavailable,
        &HostLogicalTensorView {
            data: &[1],
            shape: &[1, 1],
        },
        &[],
    )
    .expect_err("default logical raw hook is unavailable");
    let NumericUploadError::Provider(error) = error else {
        panic!("expected provider error");
    };
    assert!(
        error.is_unsupported(),
        "default unsupported is certified before effects"
    );
    let error = upload_logical_protecting_admitted(
        &Unavailable,
        &HostLogicalTensorView {
            data: &[1],
            shape: &[1, 1],
        },
        &[],
        Some(&Admission),
        None,
    )
    .expect_err("missing admitted hook cannot use ordinary logical upload");
    assert!(matches!(error, NumericUploadError::Provider(error) if error.is_unsupported()));
}
