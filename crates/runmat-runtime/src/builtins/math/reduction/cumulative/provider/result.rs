use crate::{builtins::common::gpu_helpers, BuiltinResult};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderExtremaDemand, ProviderPrecision,
};
use runmat_builtins::CumulativeKind;
pub(in crate::builtins::math::reduction) enum ResultValue {
    Values(GpuTensorHandle),
    Pair {
        values: GpuTensorHandle,
        indices: GpuTensorHandle,
    },
}
impl ResultValue {
    pub(in crate::builtins::math::reduction) fn reject(
        self,
        error: crate::RuntimeError,
    ) -> crate::RuntimeError {
        match self {
            Self::Values(value) => gpu_helpers::reject_gpu_output(value, error),
            Self::Pair { values, indices } => {
                gpu_helpers::reject_gpu_outputs([values, indices], error)
            }
        }
    }
}
pub(super) fn validate(
    kind: CumulativeKind,
    input: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    mut result: ResultValue,
    demand: ProviderExtremaDemand,
) -> BuiltinResult<ResultValue> {
    let values = match &result {
        ResultValue::Values(value) | ResultValue::Pair { values: value, .. } => value,
    };
    let class = gpu_helpers::UnaryGpuOutputContract {
        storage: runmat_accelerate_api::handle_storage(input),
        precision: runmat_accelerate_api::handle_precision(input),
        integer: runmat_accelerate_api::handle_integer_type(input),
        logical: false,
        alias: if input.shape().contains(&0) {
            gpu_helpers::GpuOutputAliasPolicy::AllowInput
        } else {
            gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation
        },
    };
    let mut valid = gpu_helpers::unary_gpu_output_matches(values, input, provider, class);
    if let ResultValue::Pair { values, indices } = &result {
        let precision = runmat_accelerate_api::handle_precision(indices);
        valid &= gpu_helpers::unary_gpu_output_matches(
            indices,
            input,
            provider,
            gpu_helpers::UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: if demand == ProviderExtremaDemand::ValuesAndIndices {
                    Some(ProviderPrecision::F64)
                } else {
                    precision
                },
                integer: None,
                logical: false,
                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        ) && !indices.shares_allocation_with(values)
            && matches!(
                precision,
                Some(ProviderPrecision::F32 | ProviderPrecision::F64)
            );
    }
    if !valid {
        return Err(result.reject(super::super::failure(
            kind,
            "provider returned malformed, aliased, foreign, or wrong-class scan output",
        )));
    }
    let provenance = match &mut result {
        ResultValue::Values(value) => gpu_helpers::propagate_output_provenance(value, [input]),
        ResultValue::Pair { values, indices } => {
            gpu_helpers::propagate_output_provenance(values, [input])
                .and_then(|()| gpu_helpers::propagate_output_provenance(indices, [input]))
        }
    };
    if let Err(error) = provenance {
        return Err(result.reject(error));
    }
    Ok(result)
}
