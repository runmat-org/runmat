//! Geometry-only provider; no shader execution is simulated.
use runmat_accelerate_api::*;

pub(crate) struct GeometryProvider(pub u32);
impl GeometryProvider {
    pub(crate) fn new() -> Self {
        Self(512)
    }
}
impl AccelProvider for GeometryProvider {
    fn supports_fused_elementwise_abi(&self, _: ProviderFusedElementwiseAbi) -> bool {
        true
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn elementwise_geometry(&self) -> Option<ProviderElementwiseGeometry> {
        Some(ProviderElementwiseGeometry {
            workgroup_size: self.0,
        })
    }
    fn device_info(&self) -> String {
        "nonexecuting geometry fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("geometry fixture cannot execute")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("geometry fixture cannot execute") })
    }
}
