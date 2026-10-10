//! The selected reduction source and its physical shader requirements travel
//! together from the one WGSL producer through provider admission.

use runmat_accelerate_api::{ProviderPrecision, ProviderShaderRequirements};

pub(crate) struct GeneratedReductionShader {
    source: String,
    requirements: ProviderShaderRequirements,
}

impl GeneratedReductionShader {
    pub(super) fn new(source: String, precision: ProviderPrecision) -> Self {
        let requirements = match precision {
            ProviderPrecision::F32 => ProviderShaderRequirements::Basic,
            // This producer emits IEEE classification and canonical NaN helpers
            // for every F64 reduction, independently of the selected operation.
            ProviderPrecision::F64 => ProviderShaderRequirements::F64IeeeWords,
        };
        Self {
            source,
            requirements,
        }
    }

    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn requirements(&self) -> ProviderShaderRequirements {
        self.requirements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn selected_reduction_source_retains_its_typed_physical_requirement() {
        let single = GeneratedReductionShader::new("single".into(), ProviderPrecision::F32);
        let double = GeneratedReductionShader::new("double".into(), ProviderPrecision::F64);
        assert_eq!(single.source(), "single");
        assert_eq!(single.requirements(), ProviderShaderRequirements::Basic);
        assert_eq!(double.source(), "double");
        assert_eq!(
            double.requirements(),
            ProviderShaderRequirements::F64IeeeWords
        );
    }
}
