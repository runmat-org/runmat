//! Provider value execution, separate from exact origin selection and demand.
use super::{
    output::{like_contract, pair_contract},
    ExtremaKind,
};
use crate::{
    builtins::common::{broadcast, gpu_helpers},
    BuiltinResult,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType};
use runmat_value::Value;

impl ExtremaKind {
    pub(in crate::builtins::math::reduction) async fn pair_values(
        self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        let Some(provider) = self.pair_owner(left, right)? else {
            return Ok(None);
        };
        // The real provider hook cannot select complex values by the host
        // magnitude/phase order, even if it returns a plausible descriptor.
        if left.physical().storage() == GpuTensorStorage::ComplexInterleaved
            || right.physical().storage() == GpuTensorStorage::ComplexInterleaved
        {
            return Ok(None);
        }
        let Some((shape, repeats_left, repeats_right)) =
            broadcast_repeats(left.shape(), right.shape())
                .map_err(|error| self.contract_error(error))?
        else {
            return Ok(None);
        };
        let Some(contract) = pair_contract(left, right, shape.clone()) else {
            return Ok(None);
        };
        if left.shape() != right.shape() {
            if let Some(output) = self.resident_double_scalar(provider, left, right).await? {
                return self
                    .admit_output(provider, left, right, &contract, output)
                    .map(Some);
            }
        }
        let left_was_expanded = repeats_left.iter().any(|repeat| *repeat != 1);
        let right_was_expanded = repeats_right.iter().any(|repeat| *repeat != 1);
        let expanded_left = if left_was_expanded {
            let output = match provider.repmat(left, &repeats_left) {
                Ok(Some(output)) => output,
                Ok(None) => return Ok(None),
                Err(source) => {
                    return Err(crate::runtime_error::with_provider_execution_source(
                        self.contract_error(format!("provider repmat failed: {source}")),
                        source,
                    ))
                }
            };
            if !gpu_helpers::replication_output_matches(left, &output, provider, &shape) {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    self.contract_error("invalid left expansion layout or class"),
                ));
            }
            self.admit_output(
                provider,
                left,
                right,
                &like_contract(left, shape.clone()),
                output,
            )?
        } else {
            left.clone()
        };
        let expanded_right = if right_was_expanded {
            let repeat = provider.repmat(right, &repeats_right);
            let output = match repeat {
                Ok(Some(output)) => output,
                Ok(None) if !left_was_expanded => return Ok(None),
                Ok(None) => {
                    return Err(crate::runtime_error::with_provider_execution_source(
                        self.contract_error(
                            "required right expansion unavailable after left expansion",
                        ),
                        runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                            "required repmat unavailable"
                        ))
                        .after_effect(),
                    ))
                }
                Err(source) => {
                    return Err(crate::runtime_error::with_provider_execution_source(
                        self.contract_error(format!("provider repmat failed: {source}")),
                        if left_was_expanded {
                            source.after_effect()
                        } else {
                            source
                        },
                    ))
                }
            };
            if !gpu_helpers::replication_output_matches(right, &output, provider, &shape)
                || (left_was_expanded && output.shares_allocation_with(&expanded_left))
            {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    self.contract_error("invalid right expansion layout, class or alias"),
                ));
            }
            self.admit_output(
                provider,
                right,
                left,
                &like_contract(right, shape.clone()),
                output,
            )?
        } else {
            right.clone()
        };
        let result = match self {
            Self::Min => provider.elem_min(&expanded_left, &expanded_right).await,
            Self::Max => provider.elem_max(&expanded_left, &expanded_right).await,
        };
        let result = if left_was_expanded || right_was_expanded {
            self.optional_after_effect("elementwise extrema", result)?
        } else {
            self.optional("elementwise extrema", result)?
        };
        if let Some(output) = result {
            return self
                .admit_output(provider, left, right, &contract, output)
                .map(Some);
        }
        if left.shape() == right.shape() {
            if let Some(output) = self.resident_double_scalar(provider, left, right).await? {
                return self
                    .admit_output(provider, left, right, &contract, output)
                    .map(Some);
            }
        }
        Ok(None)
    }

    // Preserve max's supported resident-Double-scalar route without converting
    // Single or exact integer scalars through the f64 hook ABI.
    async fn resident_double_scalar(
        self,
        provider: &'static dyn AccelProvider,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        if !matches!(self, Self::Max) {
            return Ok(None);
        }
        let is_double_scalar = |input: &GpuTensorHandle| {
            !input.view().is_logical()
                && input.physical().storage() == GpuTensorStorage::Real
                && input.physical().element_type() == NumericElementType::F64
                && input.shape().iter().all(|extent| *extent == 1)
        };
        let (tensor, scalar, scalar_left) = if is_double_scalar(right) {
            (left, right, false)
        } else if is_double_scalar(left) {
            (right, left, true)
        } else {
            return Ok(None);
        };
        let host = self.gather_tensor(scalar).await?;
        let number = crate::builtins::common::tensor::tensor_value_f64(&host, 0);
        self.scalar_values_with_owner(
            provider,
            tensor,
            &Value::Num(number),
            number,
            scalar_left,
            Some(scalar),
        )
        .await
    }

    pub(in crate::builtins::math::reduction) async fn scalar_values(
        self,
        input: &GpuTensorHandle,
        other: &Value,
        scalar: f64,
        scalar_left: bool,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        let provider = self.compute_owner(input)?;
        self.scalar_values_with_owner(provider, input, other, scalar, scalar_left, None)
            .await
    }

    async fn scalar_values_with_owner(
        self,
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        other: &Value,
        scalar: f64,
        scalar_left: bool,
        protected_scalar: Option<&GpuTensorHandle>,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        if !super::scalar_hook_preserves_class(input, other) {
            return Ok(None);
        }
        let contract = like_contract(input, input.shape().to_vec());
        let result = if matches!(self, Self::Max) {
            let fill = provider.fill_like(input, scalar).map_err(|source| {
                crate::runtime_error::with_provider_execution_source(
                    self.contract_error(format!("provider fill_like failed: {source}")),
                    source.with_unproven_effects(),
                )
            })?;
            if let Some(fill) = fill {
                // A final selector result may lawfully retain an input, but
                // its constant operand must be a distinct produced allocation.
                let fill_contract = gpu_helpers::BinaryGpuOutputContract {
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
                    ..like_contract(input, input.shape().to_vec())
                };
                let protected_other = protected_scalar.unwrap_or(input);
                let fill =
                    self.admit_output(provider, input, protected_other, &fill_contract, fill)?;
                let result = if scalar_left {
                    provider.elem_max(&fill, input).await
                } else {
                    provider.elem_max(input, &fill).await
                };
                match self.optional_after_effect("elem_max", result)? {
                    Some(output) => Some(output),
                    None => self.optional("scalar_max", provider.scalar_max(input, scalar))?,
                }
            } else {
                self.optional("scalar_max", provider.scalar_max(input, scalar))?
            }
        } else {
            self.optional("scalar_min", provider.scalar_min(input, scalar))?
        };
        result
            .map(|output| self.admit_output(provider, input, input, &contract, output))
            .transpose()
    }
}

type BroadcastExpansion = (Vec<usize>, Vec<usize>, Vec<usize>);

fn broadcast_repeats(
    left: &[usize],
    right: &[usize],
) -> Result<Option<BroadcastExpansion>, &'static str> {
    let Ok(shape) = broadcast::broadcast_shapes("extrema", left, right) else {
        return Ok(None);
    };
    if !shape.contains(&0)
        && shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .is_none()
    {
        return Err("broadcast extrema output element count overflows addressable storage");
    }
    let aligned_left = broadcast::align_shape(left, shape.len());
    let aligned_right = broadcast::align_shape(right, shape.len());
    let repeats = |input: &[usize]| {
        input
            .iter()
            .zip(&shape)
            .map(|(input, output)| if input == output { 1 } else { *output })
            .collect()
    };
    Ok(Some((
        shape.clone(),
        repeats(&aligned_left),
        repeats(&aligned_right),
    )))
}

#[cfg(test)]
mod tests;
