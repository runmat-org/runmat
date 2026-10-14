//! Real WGPU allocation/readback witness for the structured fallback route.
//! The forwarding fixture intentionally declines dot; it does not claim that
//! the production WGPU dot kernel lacks lazy-layout support.
#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
use runmat_accelerate::backend::wgpu::provider::WgpuProvider;
use runmat_accelerate_api::*;
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

struct FallbackProvider {
    inner: WgpuProvider,
    hooks: AtomicUsize,
    downloads: AtomicUsize,
}
impl AccelProvider for FallbackProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(view)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(input)
    }
    fn dot<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
        _: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            left.allocation_lease().validate_live()?;
            right.allocation_lease().validate_live()?;
            self.hooks.fetch_add(1, Ordering::SeqCst);
            Err(unsupported_provider_operation(
                "fixture selects logical host dot over actual WGPU storage",
            ))
        })
    }
}
fn read(provider: &WgpuProvider, input: &GpuTensorHandle) -> Vec<f64> {
    let host = futures::executor::block_on(provider.download_numeric(input)).unwrap();
    assert_eq!(host.shape, input.shape());
    match (provider.precision(), host.data) {
        (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
            values.into_iter().map(f64::from).collect()
        }
        (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
        _ => panic!("readback changed native class"),
    }
}

#[test]
#[ignore = "requires an actual WGPU adapter; run explicitly for F32 and F64"]
fn real_lazy_transpose_fallback_preserves_logical_order_class_and_intent() {
    let inner = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    match std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref() {
        Ok("f32") => assert_eq!(inner.precision(), ProviderPrecision::F32),
        Ok("f64") => assert_eq!(
            inner.precision(),
            ProviderPrecision::F64,
            "F64 device required"
        ),
        _ => {}
    }
    let provider = Box::leak(Box::new(FallbackProvider {
        inner,
        hooks: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    for explicit in [false, true] {
        let intent = if explicit {
            GpuHandleProvenance::Explicit
        } else {
            GpuHandleProvenance::Automatic
        };
        let upload = |values: &[f64], shape: &[usize]| {
            let single = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: if provider.precision() == ProviderPrecision::F32 {
                        HostNumericDataView::F32(&single)
                    } else {
                        HostNumericDataView::F64(values)
                    },
                    shape,
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
                .with_provenance(intent)
                .unwrap()
        };
        let backing = upload(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let left = backing
            .with_view(backing.view().clone().with_shape(vec![3, 2]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                }),
            ))
            .expect("actual allocation owner must admit lazy transpose");
        let right = upload(&[7., 11., 13., 17., 19., 23.], &[3, 2]);
        assert!(left.shares_allocation_with(&backing));
        assert_eq!(read(&provider.inner, &left), [1., 3., 5., 2., 4., 6.]);
        let hooks = provider.hooks.load(Ordering::SeqCst);
        let downloads = provider.downloads.load(Ordering::SeqCst);
        let output = futures::executor::block_on(runmat_runtime::call_builtin_async(
            "dot",
            &[
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ],
        ))
        .unwrap();
        let Value::GpuTensor(output) = output.value() else {
            panic!("actual native class supports restoration")
        };
        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(handle_precision(output), Some(provider.precision()));
        assert_eq!(handle_provenance(output), Some(intent));
        assert_eq!(read(&provider.inner, output), [105., 248.]);
        assert_eq!(provider.hooks.load(Ordering::SeqCst), hooks + 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), downloads + 2);
        assert!(!output.shares_allocation_with(&left));
        assert!(!output.shares_allocation_with(&right));
        left.allocation_lease().validate_live().unwrap();
        right.allocation_lease().validate_live().unwrap();
        assert_eq!(read(&provider.inner, &backing), [1., 2., 3., 4., 5., 6.]);
        assert_eq!(read(&provider.inner, &right), [7., 11., 13., 17., 19., 23.]);
    }
}
