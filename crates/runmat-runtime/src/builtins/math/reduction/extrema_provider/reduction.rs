//! Dimensional extrema publication: typed values and public Double indices.
use super::{gpu_helpers, ExtremaDemand, ExtremaKind, ExtremaResult};
use crate::BuiltinResult;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderPrecision,
    ReduceDimResult,
};
use runmat_value::{Tensor, Value};

/// Values and one-based indices are independent products of a dimensional
/// reduction. An immutable sibling view cannot represent both roles: the
/// index would alias value storage and may be dropped while values publish.
/// This does not change the separate policy allowing an output to reuse its
/// input allocation when that is otherwise semantically valid.
pub(super) fn independent_output_allocations(
    values: &GpuTensorHandle,
    indices: &GpuTensorHandle,
) -> bool {
    !values.shares_allocation_with(indices)
}

impl ExtremaKind {
    /// A no-longer-published provider result is still an owned reference.
    /// Checked release preserves shared input leases and reports a failed last
    /// retirement instead of allowing Drop to hide it behind successful values.
    fn retire_reduction_index(self, indices: GpuTensorHandle) -> BuiltinResult<()> {
        indices.release().map(|_| ()).map_err(|source| {
            let mut error =
                self.contract_error("discarded dimensional extrema indices could not retire");
            error.source = Some(Box::new(source));
            error
        })
    }

    pub(super) async fn adopt_reduction_result(
        self,
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        dimension: usize,
        result: ReduceDimResult,
        demand: ExtremaDemand,
    ) -> BuiltinResult<ExtremaResult> {
        let ReduceDimResult {
            mut values,
            mut indices,
        } = result;
        let Some(axis_extent) = input.shape().get(dimension).copied() else {
            let error =
                self.contract_error("dimensional extrema axis is outside the admitted input rank");
            return Err(gpu_helpers::reject_gpu_output(
                indices,
                gpu_helpers::reject_gpu_output(values, error),
            ));
        };
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
        ) && independent_output_allocations(&values, &indices)
            && (index_precision != Some(ProviderPrecision::F32) || axis_extent <= (1 << 24));
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
        if demand == ExtremaDemand::ValuesOnly {
            // The provider still produced both outputs; only adaptation work
            // is omitted. A failed discarded-result retirement is terminal.
            if let Err(error) = self.retire_reduction_index(indices) {
                return Err(gpu_helpers::reject_gpu_output(values, error));
            }
            return Ok(ExtremaResult::Values(Value::GpuTensor(values)));
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
            if host.shape() != index_contract.shape
                || host.numeric_dtype() != runmat_value::NumericDType::F32
            {
                let error = self.contract_error("index download changed native class or shape");
                return Err(gpu_helpers::reject_gpu_output(
                    indices,
                    gpu_helpers::reject_gpu_output(values, error),
                ));
            }
            let data = crate::builtins::common::tensor::tensor_into_values_f64(host);
            if data.iter().any(|index| {
                !index.is_finite()
                    || index.fract() != 0.0
                    || *index < 1.0
                    || *index > axis_extent as f64
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
            if let Err(error) = self.retire_reduction_index(indices) {
                return Err(gpu_helpers::reject_gpu_output(values, error));
            }
            crate::builtins::common::tensor::tensor_into_value(host)
        } else {
            Value::GpuTensor(indices)
        };
        Ok(ExtremaResult::Pair {
            values: Value::GpuTensor(values),
            indices,
        })
    }
}

#[cfg(test)]
mod tests;
