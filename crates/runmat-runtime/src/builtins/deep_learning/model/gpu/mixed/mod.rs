//! Typed whole-route plan for composed resident forward stages.

mod execution;
mod parameters;
mod plan;
#[cfg(test)]
mod tests;

use super::*;
use crate::builtins::deep_learning::model::plan::host::HostNetworkPlan;

pub(in crate::builtins::deep_learning) async fn forward(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    input: &GpuTensorHandle,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<GpuTensorHandle, GpuCapabilityReason>>> {
    match plan::admit(network, host, input, target_axes, function)? {
        None => Ok(None),
        Some(Err(reason)) => Ok(Some(Err(reason))),
        Some(Ok(plan)) => Ok(Some(Ok(
            execution::execute(network, input, plan, function).await?
        ))),
    }
}
