//! Comparison masks are logical producer results, not arithmetic operands.
use super::{gpu_helpers, ExtremaKind};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_value::Value;

impl ExtremaKind {
    pub(in crate::builtins::math::reduction) async fn mask_indices(
        self,
        provider: &'static dyn AccelProvider,
        mask: GpuTensorHandle,
    ) -> BuiltinResult<Value> {
        if !runmat_accelerate_api::handle_is_logical(&mask)
            || mask.physical().storage() != GpuTensorStorage::Real
            || !std::ptr::eq(self.owner(&mask)?, provider)
        {
            return Err(gpu_helpers::reject_gpu_output(
                mask,
                self.contract_error("comparison hook did not return an owning logical mask"),
            ));
        }
        // The cast is a real numerical producer. Immutable logical views must
        // never be relabeled numeric merely to satisfy scalar arithmetic.
        if let Some(numeric) = self.optional("unary_double", provider.unary_double(&mask).await)? {
            let numeric = self.validate_index_stage(provider, &mask, numeric)?;
            if let Some(scaled) =
                self.optional("scalar_mul", provider.scalar_mul(&numeric, -1.0))?
            {
                let scaled = self.validate_index_stage(provider, &mask, scaled)?;
                if let Some(indices) =
                    self.optional("scalar_add", provider.scalar_add(&scaled, 2.0))?
                {
                    let indices = self.validate_index_stage(provider, &mask, indices)?;
                    return Ok(Value::GpuTensor(indices));
                }
            }
        }
        // Only typed capability absence gets here. Keep the already computed
        // device values and use the existing host-index result representation.
        let Value::LogicalArray(host) = self.gather(&mask).await? else {
            return Err(self.contract_error("logical comparison download changed class"));
        };
        let indices = runmat_value::Tensor::new(
            host.data
                .iter()
                .map(|value| if *value != 0 { 1.0 } else { 2.0 })
                .collect(),
            host.shape,
        )
        .map_err(|error| self.contract_error(error))?;
        Ok(crate::builtins::common::tensor::tensor_into_value(indices))
    }

    fn validate_index_stage(
        self,
        provider: &'static dyn AccelProvider,
        mask: &GpuTensorHandle,
        result: GpuTensorHandle,
    ) -> BuiltinResult<GpuTensorHandle> {
        let contract = gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: Some(ProviderPrecision::F64),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
        };
        if !gpu_helpers::unary_gpu_output_matches(&result, mask, provider, contract) {
            return Err(gpu_helpers::reject_gpu_output(
                result,
                self.contract_error("index arithmetic changed owner, shape, or Double class"),
            ));
        }
        if let Err(error) = result.allocation_lease().validate_live() {
            let mut failure = self.contract_error(format!(
                "index arithmetic returned unavailable allocation: {error}"
            ));
            failure.source = Some(Box::new(error));
            return Err(gpu_helpers::reject_gpu_output(result, failure));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
