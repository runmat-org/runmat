//! Resident Adam provider preparation and result ownership.
use super::*;

pub(super) struct AdamUpdateGpuEval {
    parameters: GpuTensorHandle,
    average_grad: GpuTensorHandle,
    average_sq_grad: GpuTensorHandle,
}

impl AdamUpdateGpuEval {
    fn from_provider(
        mut result: ProviderAdamUpdateResult,
        request: &ProviderAdamUpdateRequest<'_>,
    ) -> BuiltinResult<Self> {
        let inputs = [
            Some(request.parameters),
            Some(request.gradient),
            request.average_grad,
            request.average_sq_grad,
        ];
        let adoption = (|| {
            for output in [
                &mut result.parameters,
                &mut result.average_grad,
                &mut result.average_sq_grad,
            ] {
                gpu_helpers::propagate_output_provenance(output, inputs.iter().copied().flatten())?;
                output
                    .allocation_lease()
                    .validate_live()
                    .map_err(|source| {
                        crate::build_runtime_error("adamupdate: cannot adopt provider output")
                            .with_builtin("adamupdate")
                            .with_source(source)
                            .build()
                    })?;
                runmat_accelerate_api::mark_residency(output);
            }
            Ok::<(), crate::RuntimeError>(())
        })();
        if let Err(error) = adoption {
            return Err(reject_adam_result(result, error));
        }
        Ok(Self {
            parameters: result.parameters,
            average_grad: result.average_grad,
            average_sq_grad: result.average_sq_grad,
        })
    }

    pub(super) fn output(self) -> BuiltinResult<Value> {
        let count = crate::output_count::current_output_count();
        let requested = count.unwrap_or(1);
        if requested > 3 {
            return Err(gpu_helpers::reject_gpu_outputs(
                [self.parameters, self.average_grad, self.average_sq_grad],
                deep_learning_error(
                    "adamupdate",
                    format!(
                        "adamupdate: requested {requested} outputs, but at most 3 are supported"
                    ),
                ),
            ));
        }
        let mut remaining = [self.parameters, self.average_grad, self.average_sq_grad]
            .into_iter()
            .enumerate();
        if count.is_none() {
            let (_, parameters) = remaining.next().expect("adam result has parameters");
            while let Some((_, unused)) = remaining.next() {
                if let Err(source) = unused.release() {
                    return Err(gpu_helpers::reject_gpu_outputs(
                        std::iter::once(parameters).chain(remaining.map(|(_, handle)| handle)),
                        crate::build_runtime_error(
                            "adamupdate: unused provider output retirement failed",
                        )
                        .with_builtin("adamupdate")
                        .with_source(source)
                        .build(),
                    ));
                }
            }
            return Ok(Value::GpuTensor(parameters));
        }
        let mut outputs = Vec::with_capacity(requested);
        while let Some((index, handle)) = remaining.next() {
            if index < requested {
                outputs.push(Value::GpuTensor(handle));
            } else if let Err(source) = handle.release() {
                let error = crate::build_runtime_error(
                    "adamupdate: unused provider output retirement failed",
                )
                .with_builtin("adamupdate")
                .with_source(source)
                .build();
                return Err(gpu_helpers::reject_gpu_outputs(
                    outputs
                        .into_iter()
                        .filter_map(|value| match value {
                            Value::GpuTensor(handle) => Some(handle),
                            _ => None,
                        })
                        .chain(remaining.map(|(_, handle)| handle)),
                    error,
                ));
            }
        }
        Ok(Value::output_list(outputs))
    }
}

pub(super) fn try_adamupdate_gpu(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<AdamUpdateGpuEval>> {
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
    if args
        .range(4..)
        .expect("Adam scalar-control suffix")
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

    gpu_helpers::require_resident_compute_owners(
        &[
            parameters,
            gradient,
            average_grad.unwrap_or(parameters),
            average_sq_grad.unwrap_or(parameters),
        ],
        false,
        || {
            deep_learning_error(
                "adamupdate",
                "adamupdate: no exact provider owns the optimizer tensors",
            )
        },
    )?;
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
            if let Err(error) = validate_adam_gpu_result(&result, &request, provider) {
                return Err(reject_adam_result(result, error));
            }
            Ok(Some(AdamUpdateGpuEval::from_provider(result, &request)?))
        }
        Err(err)
            if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(&err) =>
        {
            Ok(None)
        }
        Err(err) => Err(adam_provider_error(err)),
    }
}

fn reject_adam_result(
    result: ProviderAdamUpdateResult,
    error: crate::RuntimeError,
) -> crate::RuntimeError {
    gpu_helpers::reject_gpu_outputs(
        [
            result.parameters,
            result.average_grad,
            result.average_sq_grad,
        ],
        error,
    )
}

pub(super) fn adam_provider_error(error: anyhow::Error) -> crate::RuntimeError {
    let source = ProviderExecutionError::from(error);
    crate::runtime_error::with_provider_execution_source(
        deep_learning_error("adamupdate", format!("adamupdate: {source}")),
        source,
    )
}

/// Validate the complete provider result before any view is published into
/// runtime residency. The caller explicitly retires the complete owned cohort
/// on failure, including any cleanup errors.
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
