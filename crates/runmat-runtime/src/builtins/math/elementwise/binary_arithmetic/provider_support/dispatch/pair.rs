use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Value;

use crate::BuiltinResult;

mod complex;
mod expansion;
use complex::try_complex_pair;
use expansion::{expand_pair, expanded_inputs_are_valid};

#[cfg(test)]
mod complex_tests;
#[cfg(test)]
mod owned_tests;

use super::super::operation::ArithmeticPairRoute;
use super::super::scalar::is_scalar_shape;
use super::super::{broadcast_repetitions, device_real_scalar, ArithmeticProviderOperation};
use super::common::{
    accept_output, common_owner, float_handle_supported, provider_contract_error, provider_error,
    provider_error_after_effect,
};

pub(in crate::builtins::math::elementwise::binary_arithmetic) async fn try_pair(
    operation: ArithmeticProviderOperation,
    identity: BuiltinCatalogIdentity,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    crate::builtins::common::gpu_helpers::require_resident_compute_owners(
        &[left, right],
        false,
        || {
            provider_contract_error(
                operation,
                "resident inputs require one exact owning provider",
            )
        },
    )?;
    let Some(provider) = common_owner(left, right) else {
        return Ok(None);
    };
    if float_handle_supported(left)
        && float_handle_supported(right)
        && runmat_accelerate_api::handle_precision(left)
            != runmat_accelerate_api::handle_precision(right)
    {
        use runmat_types::OperatorKind;
        let operator = match operation {
            ArithmeticProviderOperation::Add => OperatorKind::Add,
            ArithmeticProviderOperation::Subtract => OperatorKind::Subtract,
            ArithmeticProviderOperation::Multiply => OperatorKind::ElementwiseMultiply,
        };
        let Some(admission) =
            super::super::FloatingPairAdmission::admit(identity, operator, left, right, provider)
        else {
            return Ok(None);
        };
        let output = operation
            .apply(provider, left, right)
            .await
            .map_err(|error| provider_error(operation, error))?;
        return if admission.accepts(&output, left, right, provider) {
            super::super::resident_output_from_sources(output, [left, right]).map(Some)
        } else {
            Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                output,
                provider_contract_error(
                    operation,
                    "provider returned invalid mixed floating metadata",
                ),
            ))
        };
    }
    let Some(route) = operation.pair_route(left, right) else {
        return Ok(None);
    };
    if let ArithmeticPairRoute::Complex(precision) = route {
        return try_complex_pair(operation, precision, provider, left, right).await;
    }
    if left.shape() == right.shape() {
        match operation.apply(provider, left, right).await {
            Ok(output) => {
                return accept_output(operation, output, left, Some(right), provider, left.shape())
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => return Err(provider_error(operation, error)),
        }
    }
    if left.shape() != right.shape() {
        if let Some(output) = try_broadcast(operation, provider, left, right).await? {
            return Ok(Some(output));
        }
    }
    try_device_scalar(operation, identity, left, right).await
}

async fn try_broadcast(
    operation: ArithmeticProviderOperation,
    provider: &dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let Some((output_shape, left_repetitions, right_repetitions)) =
        broadcast_repetitions(left.shape(), right.shape())
    else {
        return Ok(None);
    };
    let Some(expanded) = expand_pair(
        operation,
        provider,
        left,
        right,
        &left_repetitions,
        &right_repetitions,
    )?
    else {
        return Ok(None);
    };
    if !expanded_inputs_are_valid(&expanded, &output_shape) {
        return Err(provider_contract_error(
            operation,
            "provider returned invalid singleton-expanded input metadata",
        ));
    }
    let result = match operation
        .apply(provider, &expanded.left, &expanded.right)
        .await
    {
        Ok(output)
            if output.shares_allocation_with(left) || output.shares_allocation_with(right) =>
        {
            Err(provider_contract_error(
                operation,
                "provider result aliases a borrowed original operand",
            ))
        }
        Ok(output) => accept_output(
            operation,
            output,
            &expanded.left,
            Some(&expanded.right),
            provider,
            &output_shape,
        ),
        Err(error) => Err(provider_error_after_effect(operation, error)),
    };
    result
}

async fn try_device_scalar(
    operation: ArithmeticProviderOperation,
    identity: BuiltinCatalogIdentity,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    let Some(provider) = common_owner(left, right) else {
        return Ok(None);
    };
    if !float_handle_supported(left) || !float_handle_supported(right) {
        return Ok(None);
    }
    let (result, class_source, other_source) = if is_scalar_shape(left.shape()) {
        let Some(scalar) = device_real_scalar(identity, left).await? else {
            return Ok(None);
        };
        (operation.scalar_left(provider, right, scalar), right, left)
    } else if is_scalar_shape(right.shape()) {
        let Some(scalar) = device_real_scalar(identity, right).await? else {
            return Ok(None);
        };
        (operation.scalar_right(provider, left, scalar), left, right)
    } else {
        return Ok(None);
    };
    match result {
        Ok(output) => accept_output(
            operation,
            output,
            class_source,
            Some(other_source),
            provider,
            class_source.shape(),
        ),
        Err(error) => Err(provider_error_after_effect(operation, error)),
    }
}

#[cfg(test)]
#[path = "pair/scalar_dispatch_shape_tests.rs"]
mod scalar_dispatch_shape_tests;
