//! Typed whole-route plan for composed resident forward stages.

mod execution;
mod parameters;
mod physical;
mod plan;
#[cfg(test)]
mod tests;

use super::*;
use crate::builtins::deep_learning::model::admission::resident_input_handle;
use crate::builtins::deep_learning::model::plan::host::HostNetworkPlan;

pub(in crate::builtins::deep_learning) async fn forward(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<GpuTensorHandle, GpuCapabilityReason>>> {
    match plan::admit(network, host, target_axes, function)? {
        None => Ok(None),
        Some(Err(reason)) => Ok(Some(Err(reason))),
        Some(Ok(plan)) => {
            let input = resident_input_handle(host.admitted_input())
                .expect("resident route admitted the bound network input");
            Ok(Some(Ok(
                execution::execute(network, input, plan, function).await?
            )))
        }
    }
}
