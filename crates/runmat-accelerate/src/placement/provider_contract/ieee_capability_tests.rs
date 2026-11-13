use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, GpuTensorHandle, HostTensorView, ProviderShaderRequirements,
};

struct DoubleProvider {
    ieee_words: bool,
}

impl AccelProvider for DoubleProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("capability fixture never uploads")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("capability fixture never downloads") })
    }

    fn device_info(&self) -> String {
        "double capability fixture".into()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn supports_shader_requirements(&self, required: ProviderShaderRequirements) -> bool {
        required == ProviderShaderRequirements::Basic || self.ieee_words
    }
}

fn supports(
    snapshot: &ProviderCapabilitySnapshot,
    name: &str,
    family: ProviderOperationFamily,
) -> bool {
    snapshot.supports_operation(&ProviderOperationIdentity::new(name), family)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn missing_ieee_words_changes_only_wgpu_selected_reduction_capabilities() {
    let missing = DoubleProvider { ieee_words: false };
    let available = DoubleProvider { ieee_words: true };
    let in_process = in_process_capability_snapshot(&missing);
    for invocation in [
        runmat_types::SumInvocation::Sum,
        runmat_types::SumInvocation::NanSum,
    ] {
        let name = runmat_builtins::sum_invocation_entry(invocation)
            .identity
            .name;
        assert!(supports(
            &in_process,
            name,
            ProviderOperationFamily::Reduction
        ));
    }

    let missing_wgpu = wgpu_capability_snapshot(&missing, 1 << 20);
    let available_wgpu = wgpu_capability_snapshot(&available, 1 << 20);
    assert_eq!((missing_wgpu.revision, available_wgpu.revision), (25, 24));
    for kind in [
        FusionKind::Reduction,
        FusionKind::MatmulEpilogue,
        FusionKind::PowerStepNormalize,
    ] {
        let name = fusion_operation_token(&kind);
        assert!(!supports(
            &missing_wgpu,
            name,
            ProviderOperationFamily::Fusion
        ));
        assert!(supports(
            &available_wgpu,
            name,
            ProviderOperationFamily::Fusion
        ));
    }
    for invocation in [
        runmat_types::SumInvocation::Sum,
        runmat_types::SumInvocation::NanSum,
    ] {
        let name = runmat_builtins::sum_invocation_entry(invocation)
            .identity
            .name;
        assert!(!supports(
            &missing_wgpu,
            name,
            ProviderOperationFamily::Reduction
        ));
        assert!(supports(
            &available_wgpu,
            name,
            ProviderOperationFamily::Reduction
        ));
    }
    assert!(supports(
        &missing_wgpu,
        fusion_operation_token(&FusionKind::ElementwiseChain),
        ProviderOperationFamily::Fusion,
    ));
}
