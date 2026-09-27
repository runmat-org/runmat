use super::*;
use runmat_accelerate_api::ProviderPrecision;

/// Fresh transform temporaries only. Borrowed inputs are never adopted, even
/// when a malformed provider returns one as an output. Drop handles cancellation.
pub(super) struct Outputs {
    borrowed: Vec<GpuTensorHandle>,
    owned: Vec<GpuTensorHandle>,
}

impl Outputs {
    pub(super) fn new(inputs: &[Value]) -> Self {
        Self {
            borrowed: inputs
                .iter()
                .filter_map(|v| match v {
                    Value::GpuTensor(h) => Some(h.clone()),
                    _ => None,
                })
                .collect(),
            owned: Vec::new(),
        }
    }

    pub(super) fn protected(&self) -> Vec<&GpuTensorHandle> {
        self.borrowed.iter().chain(&self.owned).collect()
    }

    pub(super) fn adopt(
        &mut self,
        provider: &'static dyn AccelProvider,
        output: GpuTensorHandle,
        anchor: &GpuTensorHandle,
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> BuiltinResult<GpuTensorHandle> {
        let protected = self.protected();
        let contract = gpu_helpers::BinaryGpuOutputContract {
            shape: shape.to_vec(),
            storage: GpuTensorStorage::Real,
            precision: Some(precision),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        };
        let valid = !protected
            .iter()
            .any(|h| gpu_helpers::same_gpu_handle(h, &output))
            && gpu_helpers::binary_gpu_output_matches(&output, anchor, anchor, provider, &contract)
            && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok_and(|element| {
                gpu_helpers::numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
            });
        if !valid {
            gpu_helpers::free_rejected_provider_output(&output, &protected, provider);
            return Err(pol2cart_error(
                &ERROR_INTERNAL,
                "provider returned an invalid or aliased coordinate result",
            ));
        }
        self.owned.push(output.clone());
        Ok(output)
    }

    pub(super) fn publish(&mut self, handle: GpuTensorHandle) -> Value {
        self.owned
            .retain(|owned| !gpu_helpers::same_gpu_handle(owned, &handle));
        gpu_helpers::resident_gpu_value(handle)
    }
}

impl Drop for Outputs {
    fn drop(&mut self) {
        let protected: Vec<_> = self.borrowed.iter().collect();
        for handle in &self.owned {
            gpu_helpers::free_unprotected_exact_owner(handle, &protected);
        }
    }
}
