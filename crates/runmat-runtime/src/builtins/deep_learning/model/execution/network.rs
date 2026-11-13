//! One admitted execution route for public network callers.
//!
//! Whole-network geometry, option, parameter, state, and input admission is
//! complete before provider access or host materialization. Callers adapt
//! wrappers, requested outputs, and public compatibility policy separately.

use runmat_value::Value;

use super::{layout, placement, request::*};
use crate::builtins::deep_learning::model::{
    admission::{forward_data, resident_input_handle},
    execution::host::network::HostValue,
    gpu::{self, evaluate_network_gpu},
    plan::host::HostNetworkPlan,
};
use crate::builtins::deep_learning::{autodiff, deep_learning_error, unsupported_error};
use crate::BuiltinResult;

pub(in crate::builtins::deep_learning) async fn execute_network(
    request: NetworkExecutionRequest<'_>,
) -> BuiltinResult<NetworkExecutionOutcome> {
    let function = request.caller.function();
    let plan = HostNetworkPlan::admit(request.network, request.input, function)?;
    let output_layout = request.output_layout.resolve(
        plan.input_axes().shape().len(),
        plan.output_axes().shape().len(),
    );
    let output_axes = layout::target_axes(plan.output_axes(), output_layout, function)?;
    if request.caller == NetworkCaller::Forward
        && (crate::builtins::common::validation::value_has_native_integer_class(forward_data(
            request.input,
        )) || plan.has_integer_parameter())
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::FORWARD_INTEGER_EXTENSION,
            function,
        )?;
    }
    if let Some(runtime) = request.runtime.filter(|runtime| runtime.is_cancelled()) {
        return Err(crate::runtime_error::build_runtime_error(format!(
            "{function}: execution was cancelled"
        ))
        .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
        .build());
    }
    let active_tape = autodiff::tape_is_active();
    gpu::preflight_network_compute_owners(&plan, resident_input_handle(request.input), function)?;
    if let Some(handle) = resident_input_handle(request.input) {
        let resident_tape_admission = if active_tape {
            Some(autodiff::admit_resident_tape_parents(
                request.network,
                request.input,
                handle,
                function,
            )?)
        } else {
            None
        };
        if let Some(mixed) = gpu::mixed::forward_with_trace(
            request.network,
            &plan,
            &output_axes,
            active_tape,
            request.runtime,
            function,
        )
        .await?
        {
            match mixed {
                Ok(execution) => {
                    let trace = match (execution.trace, resident_tape_admission) {
                        (Some(trace), Some(admission)) => {
                            Some(NetworkTrace::Resident { trace, admission })
                        }
                        (None, None) => None,
                        _ => {
                            return Err(deep_learning_error(
                                function,
                                "resident execution did not preserve its admitted tape",
                            ));
                        }
                    };
                    return Ok(NetworkExecutionOutcome {
                        data: crate::builtins::common::gpu_helpers::resident_gpu_value(
                            execution.output,
                        )?,
                        output_axes,
                        state_updates: Vec::new(),
                        trace,
                        runtime: request.runtime.cloned(),
                        placement: None,
                    });
                }
                Err(reason) if active_tape => {
                    return Err(unsupported_error(
                        function,
                        format!("{function}: resident network adjoint is unavailable: {reason}"),
                    ));
                }
                Err(reason) if runmat_accelerate_api::handle_is_explicit(handle) => {
                    return Err(unsupported_error(
                        function,
                        format!("{function}: {reason} for an explicit gpuArray input"),
                    ));
                }
                Err(_) => {}
            }
        }
        if active_tape {
            return Err(unsupported_error(
                function,
                format!("{function}: resident network adjoint is unavailable for this route"),
            ));
        }
        if let Some(normalization) = plan
            .standalone_normalization()
            .map_err(|error| deep_learning_error(function, format!("{function}: {error}")))?
        {
            let check = || gpu::mixed::admit_cancellation(request.runtime, function);
            match gpu::normalize_network_gpu(
                request.network,
                handle,
                &normalization,
                &output_axes,
                &check,
                function,
            )
            .await
            .map_err(gpu::stage_failure::StageExecutionFailure::into_error)?
            {
                Ok(output) => {
                    return Ok(NetworkExecutionOutcome {
                        data: crate::builtins::common::gpu_helpers::resident_gpu_value(output)?,
                        output_axes,
                        state_updates: Vec::new(),
                        trace: None,
                        runtime: request.runtime.cloned(),
                        placement: None,
                    })
                }
                Err(reason) if runmat_accelerate_api::handle_is_explicit(handle) => {
                    return Err(unsupported_error(
                        function,
                        format!("{function}: {reason} for an explicit gpuArray input"),
                    ));
                }
                Err(_) => {}
            }
        }
        let Some(Value::Cell(layers)) = request.network.properties().get("Layers") else {
            unreachable!("whole-network admission validated Layers")
        };
        let route = gpu::admit_gpu_forward(
            layers.data(),
            handle,
            plan.input_axes(),
            plan.output_axes(),
            &output_axes,
            function,
        )?;
        if let gpu::GpuForwardAdmission::Ready(layout) = &route {
            if let Some(output) = evaluate_network_gpu(
                request.network,
                handle,
                *layout,
                output_layout,
                plan.input_axes(),
                plan.output_axes(),
                &output_axes,
                request.runtime,
                function,
            )
            .await?
            {
                if output.shape() != output_axes.shape() {
                    return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                        output,
                        crate::build_runtime_error(format!(
                            "{function}: provider output shape differs from admitted output axes"
                        ))
                        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build(),
                    ));
                }
                return Ok(NetworkExecutionOutcome {
                    data: crate::builtins::common::gpu_helpers::resident_gpu_value(output)?,
                    output_axes,
                    state_updates: Vec::new(),
                    trace: None,
                    runtime: request.runtime.cloned(),
                    placement: None,
                });
            }
        }
        if runmat_accelerate_api::handle_is_explicit(handle) {
            let detail = match route {
                gpu::GpuForwardAdmission::UnsupportedCapability(reason) => reason,
                gpu::GpuForwardAdmission::Ready(_) => gpu::GpuCapabilityReason::ProviderOperation,
            };
            return Err(unsupported_error(
                function,
                format!("{function}: {detail} for an explicit gpuArray input"),
            ));
        }
    }

    // A host-origin provider route is eligible only after the same semantic
    // admission as the CPU route. The shared placement service selects one
    // executable route; this call never runs either route as a timing probe.
    let selected = placement::select(&request, &plan, &output_axes, active_tape)?;
    let pending = match selected {
        Some(placement::SelectedNetworkRoute::Provider {
            route,
            mut observation,
            present_resident,
        }) => {
            let route_device = route.provider().device_id();
            let output = match gpu::execute_host_route_selected(route, request.runtime).await {
                Ok(output) => output,
                Err(failure) => {
                    if failure.is_operational() {
                        if let Some(observation) = observation.take() {
                            observation.operational_failure();
                        }
                    }
                    return Err(failure.into_error());
                }
            };
            let expected_element = match plan.execution_precision() {
                runmat_value::NumericDType::F32 => runmat_accelerate_api::NumericElementType::F32,
                runmat_value::NumericDType::F64 => runmat_accelerate_api::NumericElementType::F64,
                _ => {
                    return Err(deep_learning_error(
                        function,
                        "admitted network has no provider floating output class",
                    ));
                }
            };
            if output.shape() != output_axes.shape()
                || output.physical().element_type() != expected_element
                || output.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || output.device_id() != route_device
            {
                if let Some(observation) = observation.take() {
                    observation.operational_failure();
                }
                return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                    output,
                    crate::build_runtime_error(format!(
                        "{function}: provider output differs from admitted shape, class or owner"
                    ))
                    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
                ));
            }
            if present_resident {
                let data = match crate::builtins::common::gpu_helpers::resident_gpu_value(output) {
                    Ok(data) => data,
                    Err(error) => {
                        if let Some(observation) = observation.take() {
                            observation.operational_failure();
                        }
                        return Err(error);
                    }
                };
                return Ok(NetworkExecutionOutcome {
                    data,
                    output_axes,
                    state_updates: Vec::new(),
                    trace: None,
                    runtime: request.runtime.cloned(),
                    placement: observation,
                });
            }
            let tensor =
                match crate::builtins::common::gpu_helpers::gather_tensor_async(&output).await {
                    Ok(tensor) => tensor,
                    Err(error) => {
                        if placement::is_provider_download_failure(&error) {
                            if let Some(observation) = observation.take() {
                                observation.operational_failure();
                            }
                        }
                        return Err(error);
                    }
                };
            if tensor.shape() != output_axes.shape()
                || tensor.numeric_dtype() != plan.execution_precision()
            {
                if let Some(observation) = observation.take() {
                    observation.operational_failure();
                }
                return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                    output,
                    crate::build_runtime_error(format!(
                        "{function}: downloaded provider output differs from admitted network"
                    ))
                    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
                ));
            }
            return Ok(NetworkExecutionOutcome {
                data: Value::Tensor(tensor),
                output_axes,
                state_updates: Vec::new(),
                trace: None,
                runtime: request.runtime.cloned(),
                placement: observation,
            });
        }
        Some(placement::SelectedNetworkRoute::Host(observation)) => Some(observation),
        None => None,
    };

    let prepared = plan.prepare().await?;
    let host_input = prepared.prepare_input(request.input).await?;
    let result = prepared.execute(host_input, active_tape)?;
    let value = layout::present_host(result.value, &result.axes, &output_axes, function)?;
    let data = match value {
        HostValue::Real(tensor) => Value::Tensor(tensor),
        HostValue::Complex(tensor) => Value::ComplexTensor(tensor),
    };
    Ok(NetworkExecutionOutcome {
        data,
        output_axes,
        state_updates: result.next_state,
        // The immutable trace remains in the executor's axes. `forward`
        // preserves those axes; `predict` currently returns raw scores and
        // does not publish an autodiff node for a reordered result.
        trace: result.trace.map(NetworkTrace::Host),
        runtime: request.runtime.cloned(),
        placement: pending,
    })
}

#[cfg(test)]
mod tests;
