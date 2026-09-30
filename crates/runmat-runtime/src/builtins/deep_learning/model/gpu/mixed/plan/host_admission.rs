//! Pure complete-route admission for a proposed host upload. No allocation,
//! gather, provider dispatch, timing, or placement selection occurs here.

use super::*;
use crate::builtins::deep_learning::model::admission::{forward_data, resident_input_handle};
use crate::builtins::deep_learning::model::plan::host::HostNetworkPlan;
use runmat_accelerate_api::{
    AccelProvider, DeepLearningForwardCapability, NumericElementType, ProviderPrecision,
};

pub(in crate::builtins::deep_learning::model::gpu::mixed) fn admit_host<'h, 'n>(
    host: &'h HostNetworkPlan<'n>,
    target_axes: &AxisPlan,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<Result<AdmittedHostRoute<'h, 'n>, GpuCapabilityReason>>> {
    let network = host.admitted_network();
    let function = host.function();
    if !lowering::applicable(host, RoutePurpose::HostPlacement) {
        return Ok(None);
    }
    if resident_input_handle(host.admitted_input()).is_some() {
        return Ok(Some(Err(GpuCapabilityReason::InputStorage)));
    }
    let Some(plan) = lowering::lower_for_host_placement(network, host, target_axes, function)?
    else {
        return Ok(None);
    };
    let plan = match plan {
        Ok(plan) => plan,
        Err(reason) => return Ok(Some(Err(reason))),
    };
    let Some(route) = super::super::physical::from_plan(host, &plan, function)? else {
        return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)));
    };
    let precision = match route.input.element_type() {
        NumericElementType::F32 => ProviderPrecision::F32,
        NumericElementType::F64 => ProviderPrecision::F64,
        _ => return Ok(Some(Err(GpuCapabilityReason::InputPrecision))),
    };
    // Authenticate every live learnable consumed by the lowered route before
    // an upload. No explicit resident operand may fall through a host gather.
    for row in host.parameter_views() {
        let Some(Value::GpuTensor(handle)) = row.value.map(forward_data) else {
            continue;
        };
        if !plan
            .consumed_resident_parameters
            .iter()
            .any(|used| used.shares_allocation_with(handle))
        {
            continue;
        }
        let invalid_owner = || {
            super::super::super::admission::invalid_owner(
                function,
                super::super::super::admission::GpuOperand::Parameter(row.key.role),
            )
        };
        let owner = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
            .ok_or_else(|| invalid_owner())?;
        if !runmat_accelerate_api::same_provider_instance(provider, owner)
            || owner.device_id() != handle.device_id()
        {
            return Err(invalid_owner());
        }
        if runmat_accelerate_api::handle_precision(handle) != Some(precision) {
            if runmat_accelerate_api::handle_is_explicit(handle) {
                return Err(unsupported_error(
                    function,
                    format!(
                        "{function}: explicit resident {} precision differs from input",
                        row.key.role.name()
                    ),
                ));
            }
            return Ok(Some(Err(GpuCapabilityReason::ParameterPrecision(
                row.key.role,
            ))));
        }
    }
    let capability = provider
        .deep_learning_forward_route_capability(&route)
        .map_err(|error| {
            provider_failure(function, "physical host network admission", error.into())
        })?;
    Ok(Some(match capability {
        Some(DeepLearningForwardCapability::Ready(_)) => Ok(AdmittedHostRoute {
            host,
            plan,
            provider,
            route_facts: route.timing_route_digest(),
        }),
        Some(DeepLearningForwardCapability::ResourceLimit) => {
            Err(GpuCapabilityReason::ResourceLimit)
        }
        Some(DeepLearningForwardCapability::UnsupportedOperation) | None => {
            Err(GpuCapabilityReason::ProviderOperation)
        }
    }))
}
