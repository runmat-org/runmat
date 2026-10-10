use futures::FutureExt;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderPrecision, ProviderResourceEstimate, ReductionFlavor,
};
use std::sync::atomic::{AtomicUsize, Ordering};

// This deterministic provider exercises the actual VM preparation and provider
// boundary. It is not shader/hardware qualification: the hook computes a small
// independent CPU oracle after asserting the requested shader orientation.
pub(super) struct Provider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    pub(super) uploads: AtomicUsize,
    pub(super) queries: AtomicUsize,
    pub(super) reductions: AtomicUsize,
}
impl Provider {
    pub(super) fn fixture() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            reductions: AtomicUsize::new(0),
        }))
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "deterministic reduction geometry fixture".into()
    }
    fn upload(&self, value: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.inner.upload(value)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.queries.fetch_add(1, Ordering::SeqCst);
        assert_eq!(query.operation.0, "fusion.reduction");
        let bytes = query.outputs.iter().try_fold(0u64, |total, output| {
            total.checked_add(output.checked_element_count()?.checked_mul(8)?)
        });
        ProviderFeasibility::supported(ProviderResourceEstimate {
            output_bytes: bytes,
            ..Default::default()
        })
    }
    fn fused_reduction(
        &self,
        shader: &str,
        _requirements: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        _: u32,
        flavor: ReductionFlavor,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.reductions.fetch_add(1, Ordering::SeqCst);
        let row_axis = shader.contains("let row = wid.x");
        let mut operands = Vec::new();
        for input in inputs {
            // This synchronous fixture hook uses an immediately ready in-memory
            // readback. Do not enter another executor while the VM awaits it.
            operands.push(
                self.inner
                    .download(input)
                    .now_or_never()
                    .expect("in-process fixture readback must be immediately ready")?,
            );
        }
        assert!(!operands.is_empty());
        assert!(operands.len() <= 2);
        if operands.len() == 2 {
            assert!(
                shader.contains("v + v1"),
                "fixture only implements the requested addition expression"
            );
        }
        let mut data = Vec::with_capacity(num_slices);
        for slice in 0..num_slices {
            let mut sum = 0.;
            for element in 0..reduce_len {
                let index = if row_axis {
                    slice + element * num_slices
                } else {
                    element + slice * reduce_len
                };
                for operand in &operands {
                    sum += operand.data[index];
                }
            }
            data.push(if matches!(flavor, ReductionFlavor::Mean) {
                sum / reduce_len as f64
            } else {
                sum
            });
        }
        self.inner.upload(&HostTensorView {
            data: &data,
            shape: output_shape,
        })
    }

    fn supports_shader_requirements(
        &self,
        _requirements: runmat_accelerate_api::ProviderShaderRequirements,
    ) -> bool {
        // In-process fixture validates the selected source/geometry, not a GPU feature set.
        true
    }
}
