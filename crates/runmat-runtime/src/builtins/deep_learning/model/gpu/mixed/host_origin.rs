//! Host-origin half of the same admitted mixed stage sequence. Placement owns
//! the decision to call `execute_host_route`; this module never uploads to
//! probe capability or invents a cost for the route.

use super::*;
use crate::builtins::common::gpu_helpers::{
    exact_provider_for_handle, upload_tensor_protecting_typed,
};
use crate::builtins::deep_learning::model::execution::host::network::HostValue;
use runmat_accelerate_api::{
    same_provider_instance, DeepLearningForwardCapability, DeepLearningForwardSource,
};

pub(in crate::builtins::deep_learning) fn admit_host_route<'h, 'n>(
    host: &'h HostNetworkPlan<'n>,
    target_axes: &AxisPlan,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<Option<Result<plan::AdmittedHostRoute<'h, 'n>, GpuCapabilityReason>>> {
    plan::admit_host(host, target_axes, provider)
}

/// Failure provenance for a selected route. Host preparation and cancellation
/// must not become negative evidence about the provider's execution.
#[derive(Debug)]
pub(in crate::builtins::deep_learning) enum HostRouteFailure {
    Preparation(crate::RuntimeError),
    Operational(crate::RuntimeError),
    Abandoned(crate::RuntimeError),
}

impl HostRouteFailure {
    pub(in crate::builtins::deep_learning) fn is_operational(&self) -> bool {
        matches!(self, Self::Operational(_))
    }

    pub(in crate::builtins::deep_learning) fn into_error(self) -> crate::RuntimeError {
        match self {
            Self::Preparation(error) | Self::Operational(error) | Self::Abandoned(error) => error,
        }
    }
}

/// The caller has selected this exact admitted provider route. All created
/// handles remain lease-owned: failed uploads, later provider failures and
/// cancellation drop only this call's references, never borrowed learnables.
pub(in crate::builtins::deep_learning) async fn execute_host_route_selected(
    admitted: plan::AdmittedHostRoute<'_, '_>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<GpuTensorHandle, HostRouteFailure> {
    let host = admitted.host;
    let network = host.admitted_network();
    let function = host.function();
    execution::admit_cancellation(runtime, function).map_err(HostRouteFailure::Abandoned)?;
    let HostValue::Real(input) = host
        .prepare_input(host.admitted_input())
        .await
        .map_err(HostRouteFailure::Preparation)?
    else {
        return Err(HostRouteFailure::Preparation(unsupported_error(
            function,
            format!("{function}: complex network input has no real provider route"),
        )));
    };
    execution::admit_cancellation(runtime, function).map_err(HostRouteFailure::Abandoned)?;
    let protected = admitted
        .plan
        .consumed_resident_parameters
        .iter()
        .collect::<Vec<_>>();
    let uploaded =
        upload_tensor_protecting_typed(admitted.provider, &input, &protected).map_err(|error| {
            HostRouteFailure::Operational(provider_failure(
                function,
                "network input upload",
                ProviderExecutionError::from_source(error),
            ))
        })?;
    let owner = exact_provider_for_handle(&uploaded).ok_or_else(|| {
        HostRouteFailure::Operational(admission::invalid_owner(
            function,
            admission::GpuOperand::Input,
        ))
    })?;
    if !same_provider_instance(owner, admitted.provider) {
        return Err(HostRouteFailure::Operational(admission::invalid_owner(
            function,
            admission::GpuOperand::Input,
        )));
    }
    execution::admit_cancellation(runtime, function).map_err(HostRouteFailure::Abandoned)?;

    // A live resident query catches a changed provider/lease after the pure
    // host-source query. Any change is an operational error, never a reason
    // to run the CPU path after upload or dispatch.
    let Some(mut physical) = physical::from_plan(host, &admitted.plan, function)
        .map_err(HostRouteFailure::Preparation)?
    else {
        return Err(HostRouteFailure::Operational(
            execution::capability_changed(function, GpuCapabilityReason::ProviderOperation),
        ));
    };
    physical.input = DeepLearningForwardSource::Resident(&uploaded);
    match admitted
        .provider
        .deep_learning_forward_route_capability(&physical)
        .map_err(|error| {
            HostRouteFailure::Operational(provider_failure(
                function,
                "live network route admission",
                error.into(),
            ))
        })? {
        Some(DeepLearningForwardCapability::Ready(_)) => {}
        Some(DeepLearningForwardCapability::ResourceLimit) => {
            return Err(HostRouteFailure::Operational(
                execution::capability_changed(function, GpuCapabilityReason::ResourceLimit),
            ));
        }
        Some(DeepLearningForwardCapability::UnsupportedOperation) | None => {
            return Err(HostRouteFailure::Operational(
                execution::capability_changed(function, GpuCapabilityReason::ProviderOperation),
            ));
        }
    }
    execution::admit_cancellation(runtime, function).map_err(HostRouteFailure::Abandoned)?;
    execution::execute(network, &uploaded, admitted.plan, false, runtime, function)
        .await
        .map_err(|failure| match failure {
            super::super::stage_failure::StageExecutionFailure::Abandoned(error) => {
                HostRouteFailure::Abandoned(error)
            }
            super::super::stage_failure::StageExecutionFailure::Operational(error) => {
                HostRouteFailure::Operational(error)
            }
        })
        .map(|execution| execution.output)
}
