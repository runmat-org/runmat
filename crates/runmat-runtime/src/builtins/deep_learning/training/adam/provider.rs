//! Resident Adam provider preparation and result ownership.
use super::*;

pub(super) struct AdamUpdateGpuEval {
    parameters: GpuTensorHandle,
    average_grad: GpuTensorHandle,
    average_sq_grad: GpuTensorHandle,
}

impl AdamUpdateGpuEval {
    fn from_provider(result: ProviderAdamUpdateResult) -> Self {
        Self {
            parameters: result.parameters,
            average_grad: result.average_grad,
            average_sq_grad: result.average_sq_grad,
        }
    }

    pub(super) fn output(self) -> BuiltinResult<Value> {
        match crate::output_count::current_output_count() {
            None => gpu_helpers::resident_gpu_value(self.parameters),
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![gpu_helpers::resident_gpu_value(
                self.parameters,
            )?])),
            Some(2) => Ok(Value::OutputList(vec![
                gpu_helpers::resident_gpu_value(self.parameters)?,
                gpu_helpers::resident_gpu_value(self.average_grad)?,
            ])),
            Some(3) => Ok(Value::OutputList(vec![
                gpu_helpers::resident_gpu_value(self.parameters)?,
                gpu_helpers::resident_gpu_value(self.average_grad)?,
                gpu_helpers::resident_gpu_value(self.average_sq_grad)?,
            ])),
            Some(requested) => Err(deep_learning_error(
                "adamupdate",
                format!("adamupdate: requested {requested} outputs, but at most 3 are supported"),
            )),
        }
    }
}

pub(super) fn try_adamupdate_gpu(args: &[Value]) -> BuiltinResult<Option<AdamUpdateGpuEval>> {
    if !(5..=9).contains(&args.len()) {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: expected 5 to 9 positional arguments",
        ));
    }

    let Value::GpuTensor(parameters) = &args[0] else {
        return Ok(None);
    };
    let Value::GpuTensor(gradient) = &args[1] else {
        return Ok(None);
    };
    if args[4..]
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }
    let average_grad = match gpu_optimizer_state(&args[2], "averageGrad")? {
        GpuOptimizerState::Resident(handle) => Some(handle),
        GpuOptimizerState::Empty => None,
        GpuOptimizerState::HostFallback => return Ok(None),
    };
    let average_sq_grad = match gpu_optimizer_state(&args[3], "averageSqGrad")? {
        GpuOptimizerState::Resident(handle) => Some(handle),
        GpuOptimizerState::Empty => None,
        GpuOptimizerState::HostFallback => return Ok(None),
    };

    if runmat_accelerate_api::handle_storage(parameters) == GpuTensorStorage::ComplexInterleaved
        || runmat_accelerate_api::handle_storage(gradient) == GpuTensorStorage::ComplexInterleaved
        || average_grad.is_some_and(|handle| {
            runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
        })
        || average_sq_grad.is_some_and(|handle| {
            runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
        })
    {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: complex gpuArray optimizer tensors are not supported",
        ));
    }

    if parameters.shape() != gradient.shape()
        || average_grad.is_some_and(|handle| handle.shape() != parameters.shape())
        || average_sq_grad.is_some_and(|handle| handle.shape() != parameters.shape())
    {
        return Err(deep_learning_error(
            "adamupdate",
            format!(
                "adamupdate: optimizer tensors must match parameter shape {:?}",
                parameters.shape()
            ),
        ));
    }

    let Some(provider) = gpu_helpers::exact_provider_for_handle(parameters) else {
        return Ok(None);
    };
    let Some(gradient_provider) = gpu_helpers::exact_provider_for_handle(gradient) else {
        return Ok(None);
    };
    if !runmat_accelerate_api::same_provider_instance(gradient_provider, provider) {
        return Ok(None);
    }
    if average_grad.is_some_and(|handle| {
        gpu_helpers::exact_provider_for_handle(handle).is_none_or(|state_provider| {
            !runmat_accelerate_api::same_provider_instance(state_provider, provider)
        })
    }) || average_sq_grad.is_some_and(|handle| {
        gpu_helpers::exact_provider_for_handle(handle).is_none_or(|state_provider| {
            !runmat_accelerate_api::same_provider_instance(state_provider, provider)
        })
    }) {
        return Ok(None);
    }

    let precision = runmat_accelerate_api::handle_precision(parameters);
    if precision.is_none()
        || [Some(gradient), average_grad, average_sq_grad]
            .into_iter()
            .flatten()
            .any(|handle| {
                runmat_accelerate_api::handle_is_logical(handle)
                    || runmat_accelerate_api::handle_precision(handle) != precision
            })
        || runmat_accelerate_api::handle_is_logical(parameters)
    {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: resident optimizer tensors must have matching real single or double storage",
        ));
    }

    let iteration = positive_iteration(&args[4])?;
    let request = ProviderAdamUpdateRequest {
        parameters,
        gradient,
        average_grad,
        average_sq_grad,
        iteration,
        learn_rate: optional_positive_scalar(args, 5, 0.001, "learnRate")?,
        gradient_decay_factor: optional_unit_scalar(args, 6, 0.9, "gradDecay")?,
        squared_gradient_decay_factor: optional_unit_scalar(args, 7, 0.999, "sqGradDecay")?,
        epsilon: optional_positive_scalar(args, 8, 1.0e-8, "epsilon")?,
    };

    match provider.adam_update(&request) {
        Ok(result) => {
            validate_adam_gpu_result(&result, &request, provider)?;
            Ok(Some(AdamUpdateGpuEval::from_provider(result)))
        }
        Err(err) if runmat_accelerate_api::is_unsupported_provider_operation(&err) => Ok(None),
        Err(err) => Err(adam_provider_error(err)),
    }
}

pub(super) fn adam_provider_error(error: anyhow::Error) -> crate::RuntimeError {
    let source = ProviderExecutionError::from(error);
    crate::runtime_error::with_provider_execution_source(
        deep_learning_error("adamupdate", format!("adamupdate: {source}")),
        source,
    )
}

/// Validate the complete provider result before any view is published into
/// runtime residency. The owned result is dropped as a unit on failure, so
/// physical leases—not a second ad hoc release protocol—perform cleanup.
pub(super) fn validate_adam_gpu_result(
    result: &ProviderAdamUpdateResult,
    request: &ProviderAdamUpdateRequest<'_>,
    provider: &dyn AccelProvider,
) -> BuiltinResult<()> {
    let outputs = [
        &result.parameters,
        &result.average_grad,
        &result.average_sq_grad,
    ];
    let inputs = [
        Some(request.parameters),
        Some(request.gradient),
        request.average_grad,
        request.average_sq_grad,
    ];
    let expected_precision = runmat_accelerate_api::handle_precision(request.parameters);
    for output in outputs {
        if output.shape() != request.parameters.shape()
            || runmat_accelerate_api::handle_storage(output) != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(output)
            || runmat_accelerate_api::handle_precision(output) != expected_precision
            || gpu_helpers::exact_provider_for_handle(output)
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: provider returned an optimizer tensor with the wrong owner, shape, or numeric class",
            ));
        }
        output
            .allocation_lease()
            .validate_live()
            .map_err(|source| {
                crate::build_runtime_error("adamupdate: provider returned a retired allocation")
                    .with_builtin("adamupdate")
                    .with_identifier("RunMat:deepLearning:InvalidInput")
                    .with_source(source)
                    .build()
            })?;
        if inputs
            .iter()
            .flatten()
            .any(|input| output.allocation_identity() == input.allocation_identity())
        {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: provider returned an alias of a borrowed optimizer input",
            ));
        }
    }
    if outputs[0].allocation_identity() == outputs[1].allocation_identity()
        || outputs[0].allocation_identity() == outputs[2].allocation_identity()
        || outputs[1].allocation_identity() == outputs[2].allocation_identity()
    {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: provider returned aliased optimizer outputs",
        ));
    }
    Ok(())
}

enum GpuOptimizerState<'a> {
    Resident(&'a GpuTensorHandle),
    Empty,
    HostFallback,
}

fn gpu_optimizer_state<'a>(
    value: &'a Value,
    label: &'static str,
) -> BuiltinResult<GpuOptimizerState<'a>> {
    match value {
        Value::GpuTensor(handle) => Ok(GpuOptimizerState::Resident(handle)),
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0 => {
            Ok(GpuOptimizerState::Empty)
        }
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(GpuOptimizerState::HostFallback),
        other => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be a gpuArray or empty numeric array for resident gpuArray updates, got {other:?}"),
        )),
    }
}
