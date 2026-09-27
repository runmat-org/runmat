//! Provider value execution, separate from exact origin selection and demand.
use super::{
    output::{like_contract, pair_contract},
    ExtremaKind,
};
use crate::{builtins::common::broadcast, BuiltinResult};
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
        let expanded_left = if repeats_left.iter().any(|repeat| *repeat != 1) {
            let Some(output) = self.optional("repmat", provider.repmat(left, &repeats_left))?
            else {
                return Ok(None);
            };
            self.admit_output(
                provider,
                left,
                left,
                &like_contract(left, shape.clone()),
                output,
            )?
        } else {
            left.clone()
        };
        let expanded_right = if repeats_right.iter().any(|repeat| *repeat != 1) {
            let Some(output) = self.optional("repmat", provider.repmat(right, &repeats_right))?
            else {
                return Ok(None);
            };
            self.admit_output(
                provider,
                right,
                right,
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
        if let Some(output) = self.optional("elementwise extrema", result)? {
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
        self.scalar_values_with_owner(provider, tensor, &Value::Num(number), number, scalar_left)
            .await
    }

    pub(in crate::builtins::math::reduction) async fn scalar_values(
        self,
        input: &GpuTensorHandle,
        other: &Value,
        scalar: f64,
        scalar_left: bool,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        let provider = self.owner(input)?;
        self.scalar_values_with_owner(provider, input, other, scalar, scalar_left)
            .await
    }

    async fn scalar_values_with_owner(
        self,
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        other: &Value,
        scalar: f64,
        scalar_left: bool,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        if !super::scalar_hook_preserves_class(input, other) {
            return Ok(None);
        }
        let contract = like_contract(input, input.shape().to_vec());
        let result = if matches!(self, Self::Max) {
            if let Some(fill) = self.optional("fill_like", provider.fill_like(input, scalar))? {
                let fill = self.admit_output(provider, input, input, &contract, fill)?;
                let result = if scalar_left {
                    provider.elem_max(&fill, input).await
                } else {
                    provider.elem_max(input, &fill).await
                };
                self.optional("elem_max", result)?
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
