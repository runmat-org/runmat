//! Dimensional extrema publication: typed values and public Double indices.
use super::{gpu_helpers, ExtremaKind};
use crate::BuiltinResult;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderPrecision,
    ReduceDimResult,
};
use runmat_value::{Tensor, Value};

impl ExtremaKind {
    pub(super) async fn adopt_reduction_result(
        self,
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        dimension: usize,
        result: ReduceDimResult,
    ) -> BuiltinResult<(Value, Value)> {
        let ReduceDimResult {
            mut values,
            mut indices,
        } = result;
        let mut shape = input.shape().to_vec();
        // The owning host reduction preserves empty shape rather than inventing
        // a nonempty identity along a zero-length dimension.
        if !shape.contains(&0) {
            shape[dimension] = 1;
        }
        let element = if input.view().is_logical() {
            NumericElementType::F64
        } else {
            input.physical().element_type()
        };
        let value_contract = gpu_helpers::BinaryGpuOutputContract {
            shape: shape.clone(),
            storage: input.physical().storage(),
            precision: element.precision(),
            integer: element.integer_type(),
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
        };
        let index_precision = runmat_accelerate_api::handle_precision(&indices);
        let index_contract = gpu_helpers::BinaryGpuOutputContract {
            shape,
            storage: GpuTensorStorage::Real,
            precision: index_precision,
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
        };
        let valid = gpu_helpers::binary_gpu_output_matches(
            &values,
            input,
            input,
            provider,
            &value_contract,
        ) && matches!(
            index_precision,
            Some(ProviderPrecision::F32 | ProviderPrecision::F64)
        ) && gpu_helpers::binary_gpu_output_matches(
            &indices,
            input,
            input,
            provider,
            &index_contract,
        ) && (index_precision != Some(ProviderPrecision::F32)
            || input.shape()[dimension] <= (1 << 24));
        if !valid {
            let error = self.contract_error("dimensional extrema returned invalid owner, shape, value class, or index representation");
            return Err(gpu_helpers::reject_gpu_output(
                indices,
                gpu_helpers::reject_gpu_output(values, error),
            ));
        }
        let unavailable = values
            .allocation_lease()
            .validate_live()
            .err()
            .or_else(|| indices.allocation_lease().validate_live().err());
        if let Some(source) = unavailable {
            let mut error = self.contract_error(format!(
                "dimensional extrema returned unavailable allocation: {source}"
            ));
            error.source = Some(Box::new(source));
            return Err(gpu_helpers::reject_gpu_output(
                indices,
                gpu_helpers::reject_gpu_output(values, error),
            ));
        }
        if let Err(error) = gpu_helpers::propagate_output_provenance(&mut values, [input])
            .and_then(|()| gpu_helpers::propagate_output_provenance(&mut indices, [input]))
        {
            return Err(gpu_helpers::reject_gpu_output(
                indices,
                gpu_helpers::reject_gpu_output(values, error),
            ));
        }
        let indices = if index_precision == Some(ProviderPrecision::F32) {
            // Physical F32 is not public Double. This supported mixed-residency
            // result performs a real download, retaining device values and exact
            // indices (the axis bound above is the F32 integer-exact limit).
            let host = match self.gather_tensor(&indices).await {
                Ok(host) => host,
                Err(error) => {
                    return Err(gpu_helpers::reject_gpu_output(
                        indices,
                        gpu_helpers::reject_gpu_output(values, error),
                    ))
                }
            };
            let data = crate::builtins::common::tensor::tensor_into_values_f64(host);
            if data.iter().any(|index| {
                !index.is_finite()
                    || index.fract() != 0.0
                    || *index < 1.0
                    || *index > input.shape()[dimension] as f64
            }) {
                let error =
                    self.contract_error("dimensional extrema returned an invalid one-based index");
                return Err(gpu_helpers::reject_gpu_output(
                    indices,
                    gpu_helpers::reject_gpu_output(values, error),
                ));
            }
            let host = Tensor::new(data, index_contract.shape)
                .map_err(|error| self.contract_error(error))?;
            crate::builtins::common::tensor::tensor_into_value(host)
        } else {
            Value::GpuTensor(indices)
        };
        Ok((Value::GpuTensor(values), indices))
    }
}

#[cfg(test)]
mod tests;
