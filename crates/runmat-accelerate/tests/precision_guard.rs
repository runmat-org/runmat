use runmat_accelerate::precision::ensure_provider_supports_dtype;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostTensorOwned, HostTensorView, ProviderPrecision,
};
use runmat_builtins::NumericDType;

struct F32TestProvider;

impl AccelProvider for F32TestProvider {
    fn device_id(&self) -> u32 {
        0
    }
    fn gather_linear(
        &self,
        _: &GpuTensorHandle,
        _: &[u32],
        _: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("gather_linear should not be invoked in this test")
    }

    fn scatter_linear(
        &self,
        _: &GpuTensorHandle,
        _: &[u32],
        _: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        unreachable!("scatter_linear should not be invoked in this test")
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("upload should not be invoked in this test")
    }

    fn download(&self, _: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
        unreachable!("download should not be invoked in this test")
    }

    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        Ok(())
    }

    fn device_info(&self) -> String {
        "f32-test-provider".to_string()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F32
    }
}

#[test]
fn f64_inputs_disallowed_on_f32_provider_without_downcast() {
    let provider = F32TestProvider;
    ensure_provider_supports_dtype(&provider, NumericDType::F32).expect("f32 should be allowed");
    let err =
        ensure_provider_supports_dtype(&provider, NumericDType::F64).expect_err("f64 must fail");
    assert!(
        err.contains("refusing implicit downcast"),
        "unexpected error message: {err}"
    );
}
