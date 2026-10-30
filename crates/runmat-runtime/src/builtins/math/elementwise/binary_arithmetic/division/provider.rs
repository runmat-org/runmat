use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::gpu_helpers;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    common_owner, compatible_inputs, float_handle_supported, valid_real_binary_output,
    FloatingPairAdmission,
};
use crate::BuiltinResult;

use super::{DivisionContext, PublicOperandOrder};

mod scalar;
pub(super) use scalar::{try_resident_denominator, try_resident_numerator};

pub(super) async fn try_pair(
    context: DivisionContext,
    numerator: &GpuTensorHandle,
    denominator: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::require_resident_compute_owners(&[numerator, denominator], false, || {
        context.provider_contract_error("resident inputs require one exact owning provider")
    })?;
    let Some(provider) = common_owner(numerator, denominator) else {
        return Ok(None);
    };
    if float_handle_supported(numerator)
        && float_handle_supported(denominator)
        && runmat_accelerate_api::handle_precision(numerator)
            != runmat_accelerate_api::handle_precision(denominator)
    {
        use runmat_types::OperatorKind;
        // Preserve the division domain error before an unsupported provider
        // route can decline. Admission's optional shape is not this boundary.
        BroadcastPlan::new(numerator.shape(), denominator.shape())
            .map_err(|detail| context.error_with_detail(context.size_mismatch, detail))?;
        let (operator, left, right) = match context.public_order {
            PublicOperandOrder::Numerical => {
                (OperatorKind::ElementwiseDivide, numerator, denominator)
            }
            PublicOperandOrder::Reversed => {
                (OperatorKind::ElementwiseLeftDivide, denominator, numerator)
            }
        };
        let Some(admission) =
            FloatingPairAdmission::admit(context.identity, operator, left, right, provider)
        else {
            return Ok(None);
        };
        // The public ldivide operands are reversed above for its contract;
        // elem_div still receives the original numerical numerator first.
        let output = divide(provider, numerator, denominator)
            .await
            .map_err(|error| context.provider_error(error))?;
        return if admission.accepts(&output, left, right, provider) {
            Ok(Some(output))
        } else {
            Err(gpu_helpers::reject_gpu_output(
                output,
                context.provider_contract_error(
                    "provider returned invalid mixed floating division metadata",
                ),
            ))
        };
    }
    if !compatible_inputs(numerator, denominator) {
        return Ok(None);
    }
    let expected_shape = BroadcastPlan::new(numerator.shape(), denominator.shape())
        .map_err(|detail| context.error_with_detail(context.size_mismatch, detail))?
        .output_shape()
        .to_vec();
    let output = match divide(provider, numerator, denominator).await {
        Ok(output) => output,
        Err(error) if error.is_unsupported() => return Ok(None),
        Err(error) => return Err(context.provider_error(error)),
    };
    if valid_real_binary_output(
        &output,
        numerator,
        Some(denominator),
        provider,
        &expected_shape,
    ) {
        Ok(Some(output))
    } else {
        Err(gpu_helpers::reject_gpu_output(
            output,
            context.provider_contract_error(
                "provider returned invalid element-wise division metadata",
            ),
        ))
    }
}

async fn divide(
    provider: &dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
    let _left = gpu_helpers::ResidencyRestoreGuard::new(left);
    let _right = gpu_helpers::ResidencyRestoreGuard::new(right);
    provider.elem_div(left, right).await
}
