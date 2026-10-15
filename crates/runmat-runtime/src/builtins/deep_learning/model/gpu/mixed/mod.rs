//! Typed whole-route plan for composed resident forward stages.

mod backward;
mod execution;
mod host_origin;
mod parameters;
mod physical;
mod plan;
mod trace;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod host_origin_tests;
#[cfg(test)]
mod resident_layer_tests;

use super::*;
use crate::builtins::deep_learning::model::admission::resident_input_handle;
use crate::builtins::deep_learning::model::plan::host::HostNetworkPlan;
pub(in crate::builtins::deep_learning) use backward::reverse as reverse_resident;
pub(in crate::builtins::deep_learning::model) use execution::admit_cancellation;
pub(in crate::builtins::deep_learning) use execution::capability_changed;
use runmat_accelerate_api::{ProviderActivationAdjoint, ProviderResidentAdjointStage};
pub(in crate::builtins::deep_learning) use trace::{ResidentExecution, ResidentTrace};

pub(in crate::builtins::deep_learning) use host_origin::execute_host_route_selected;

/// Capture is admitted before the first provider effect. An unsupported
/// derivative stage is never allowed to run a partial forward pass.
pub(in crate::builtins::deep_learning::model) async fn forward_with_trace(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    capture: bool,
    runtime: Option<&crate::context::RuntimeContext>,
    function: &'static str,
) -> BuiltinResult<Option<Result<ResidentExecution, GpuCapabilityReason>>> {
    let admitted = if capture {
        plan::admit_for_tape(network, host, target_axes, function)?
    } else {
        plan::admit(network, host, target_axes, function)?
    };
    match admitted {
        None => Ok(None),
        Some(Err(reason)) => Ok(Some(Err(reason))),
        Some(Ok(plan)) => {
            let input = resident_input_handle(host.admitted_input())
                .expect("resident route admitted the bound network input");
            if capture {
                let provider =
                    crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
                        .ok_or_else(|| {
                            admission::invalid_owner(function, admission::GpuOperand::Input)
                        })?;
                let precision =
                    runmat_accelerate_api::handle_precision(input).ok_or_else(|| {
                        deep_learning_error(
                            function,
                            "resident tape input has no floating precision",
                        )
                    })?;
                let mut reverse_stages = Vec::new();
                for stage in &plan.stages {
                    match stage {
                        plan::Stage::Materialize | plan::Stage::Transpose { .. } => {}
                        plan::Stage::Dense { .. } => {
                            reverse_stages.push(ProviderResidentAdjointStage::Dense)
                        }
                        plan::Stage::Conv { .. } => {
                            reverse_stages.push(ProviderResidentAdjointStage::Conv1D)
                        }
                        plan::Stage::Relu => {
                            reverse_stages.push(ProviderResidentAdjointStage::Activation(
                                ProviderActivationAdjoint::Relu,
                            ))
                        }
                        plan::Stage::Elu { alpha } => {
                            reverse_stages.push(ProviderResidentAdjointStage::Activation(
                                ProviderActivationAdjoint::Elu { alpha: *alpha },
                            ))
                        }
                        plan::Stage::Softmax => {
                            reverse_stages.push(ProviderResidentAdjointStage::SoftmaxRows)
                        }
                        plan::Stage::Normalize { .. } => {
                            reverse_stages.push(ProviderResidentAdjointStage::LayerNorm)
                        }
                        plan::Stage::Pool { .. } => {
                            reverse_stages.push(ProviderResidentAdjointStage::Pool1D)
                        }
                    }
                }
                if (reverse_stages.iter().any(|stage| {
                    matches!(
                        stage,
                        ProviderResidentAdjointStage::Dense
                            | ProviderResidentAdjointStage::SoftmaxRows
                    )
                }) && !provider.real_sum_native_dim_supported(precision))
                    || !provider.resident_network_adjoint_supported(&reverse_stages, precision)
                {
                    return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)));
                }
            }
            let execution = execution::execute(network, input, plan, capture, runtime, function)
                .await
                .map_err(super::stage_failure::StageExecutionFailure::into_error)?;
            execution::admit_cancellation(runtime, function)?;
            Ok(Some(Ok(execution)))
        }
    }
}

pub(in crate::builtins::deep_learning) use host_origin::admit_host_route;
pub(in crate::builtins::deep_learning) use plan::AdmittedHostRoute;
