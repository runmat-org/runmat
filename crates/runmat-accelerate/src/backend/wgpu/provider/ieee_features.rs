use super::WgpuProvider;
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::ProviderShaderRequirements;

impl WgpuProvider {
    pub(super) fn ieee_words_available(&self, precision: NumericPrecision) -> bool {
        precision != NumericPrecision::F64
            || selected_ieee_features_available(self.enabled_features)
    }

    pub(super) fn require_ieee_words(&self, precision: NumericPrecision) -> anyhow::Result<()> {
        if self.ieee_words_available(precision) {
            Ok(())
        } else {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "selected f64 IEEE shader requires the controlled binary64 feature contract",
            ))
        }
    }

    pub(super) fn supports_selected_shader_requirements(
        &self,
        requirements: ProviderShaderRequirements,
    ) -> bool {
        match requirements {
            ProviderShaderRequirements::Basic => true,
            ProviderShaderRequirements::F64IeeeWords => {
                self.ieee_words_available(NumericPrecision::F64)
            }
        }
    }

    pub(super) fn require_selected_shader_requirements(
        &self,
        requirements: ProviderShaderRequirements,
    ) -> anyhow::Result<()> {
        if self.supports_selected_shader_requirements(requirements) {
            Ok(())
        } else {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "selected shader requires unavailable enabled device features",
            ))
        }
    }
}

fn selected_ieee_features_available(enabled: wgpu::Features) -> bool {
    enabled.contains(crate::backend::wgpu::shader_compilation::capabilities::REQUIRED_FEATURES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn selected_f64_ieee_words_require_the_complete_enabled_contract() {
        assert!(!selected_ieee_features_available(wgpu::Features::empty()));
        assert!(!selected_ieee_features_available(
            wgpu::Features::SHADER_F64
        ));
        assert!(!selected_ieee_features_available(
            wgpu::Features::SHADER_INT64
        ));
        assert!(!selected_ieee_features_available(
            wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64
        ));
        assert!(selected_ieee_features_available(
            crate::backend::wgpu::shader_compilation::capabilities::REQUIRED_FEATURES
        ));
    }
}
