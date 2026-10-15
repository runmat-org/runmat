use super::{
    admit_provider_availability, provider, reduction_arithmetic, AdmittedBindings,
    FusionCpuObservation, FusionExecutionOutcome, FusionExecutionRequest,
};
use anyhow::anyhow;
use runmat_accelerate_api::ProviderExecutionResult as ExecutionResult;
use runmat_value::Value;

mod execute;
mod host_readback;
mod input_preparation;
mod legacy;
mod select;
mod validation;
pub use host_readback::CpuReadbackFailure;
pub(crate) use host_readback::CpuReadbackPlan;
use legacy::execute_reduction_with_shape_core;

#[cfg(test)]
mod tests;

pub struct TypedReductionVmOutput {
    pub value: Value,
    pub completion: Option<super::FusionProviderCompletion>,
}

pub fn execute_reduction(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    workgroup_size: u32,
) -> ExecutionResult<Value> {
    match execute_reduction_with_shape_placed(
        request,
        reduce_len,
        num_slices,
        &[num_slices],
        workgroup_size,
    )? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime").into())
        }
    }
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub fn execute_reduction_with_shape(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
) -> ExecutionResult<Value> {
    match execute_reduction_with_shape_placed(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
    )? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime").into())
        }
    }
}

pub fn execute_reduction_with_shape_placed(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    // Extrema have no legacy raw-hook flavor. Admit a projected producer
    // before that compatibility-only flavor check so sync callers can use the
    // same checked recipe as async callers without manufacturing a Sum lane.
    if let (Some(bindings), Some(owner)) = (AdmittedBindings::new(&request), provider()) {
        if legacy::selected_legacy_flavor(&request, &bindings).is_none()
            && reduction_arithmetic::requires_selected_arithmetic(&request, &bindings)?
        {
            if let Some(arithmetic) =
                reduction_arithmetic::admitted_static(&request, &bindings, owner.precision())?
            {
                if arithmetic.requires_typed() {
                    return execute_projected_sync(
                        request,
                        reduce_len,
                        num_slices,
                        output_shape,
                        workgroup_size,
                    );
                }
            }
        }
    }
    execute_reduction_with_shape_core(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        None,
    )
}

/// The VM route can await selected operation-owned arithmetic/domain proof
/// before the legacy reduction emitter reaches any provider effect.
pub async fn execute_reduction_with_shape_placed_async(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    match execute_reduction_with_shape_placed_async_inner(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        None,
        false,
    )
    .await?
    {
        FusionExecutionOutcome::Executed(result) => {
            Ok(FusionExecutionOutcome::Executed(result.value))
        }
        FusionExecutionOutcome::SharedRuntime(observation) => {
            Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    }
}

pub async fn execute_reduction_with_shape_placed_async_vm(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
    right: runmat_runtime::context::VmFusionCompletionRight,
) -> ExecutionResult<FusionExecutionOutcome<TypedReductionVmOutput>> {
    execute_reduction_with_shape_placed_async_inner(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        Some(right),
        false,
    )
    .await
}

async fn execute_reduction_with_shape_placed_async_inner(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
    mut right: Option<runmat_runtime::context::VmFusionCompletionRight>,
    synchronous: bool,
) -> ExecutionResult<FusionExecutionOutcome<TypedReductionVmOutput>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    let selected_provider = match admit_provider_availability(&request, provider())? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    let selected_identity = bindings.identity(None)?;
    let arithmetic = if synchronous {
        reduction_arithmetic::admitted_static(&request, &bindings, selected_provider.precision())?
    } else {
        reduction_arithmetic::admitted(&request, &bindings, selected_provider.precision()).await?
    };
    let Some(arithmetic) = arithmetic else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    let current = AdmittedBindings::new(&request)
        .ok_or_else(|| anyhow!("selected reduction binding changed during admission"))?;
    if current.identity(None)? != selected_identity
        || !provider().is_some_and(|current_provider| {
            runmat_accelerate_api::same_provider_instance(current_provider, selected_provider)
        })
    {
        return Err(anyhow!("selected reduction authority changed during admission").into());
    }
    validation::request(&request, reduce_len, num_slices, output_shape)?;
    let program = if synchronous {
        super::reduction_program::resolve_reduction_program_static(
            &request,
            selected_provider.precision(),
        )
        .await?
    } else {
        super::resolve_reduction_program(&request, selected_provider.precision()).await?
    };
    if let Some(program) = program {
        let current = AdmittedBindings::new(&request)
            .ok_or_else(|| anyhow!("selected reduction binding changed during typed admission"))?;
        if current.identity(None)? != selected_identity
            || !provider().is_some_and(|current_provider| {
                runmat_accelerate_api::same_provider_instance(current_provider, selected_provider)
            })
        {
            return Err(
                anyhow!("selected reduction authority changed during typed admission").into(),
            );
        }
        let preflight = input_preparation::preflight_sources(selected_provider, &program)?;
        let reduce_elements = u32::try_from(reduce_len)
            .map_err(|_| anyhow!("typed reduction length exceeds provider range"))?;
        let slices = u32::try_from(num_slices)
            .map_err(|_| anyhow!("typed reduction slice count exceeds provider range"))?;
        let config = match selected_provider.reduction_recipe_config(reduce_elements, slices) {
            Ok(config) => Some(config),
            Err(error) if error.is_unsupported() => None,
            Err(error) => return Err(error),
        };
        if let Some(config) = config {
            let recipe = crate::fusion::reduction_recipe::build(&program, config)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            if recipe.geometry().reduce_elements == reduce_elements
                && recipe.geometry().slices == slices
                && recipe.physical().descriptor().output_shape == output_shape
            {
                let inventory_selection =
                    runmat_accelerate_api::reduction::ReductionSelection::new(
                        recipe.physical(),
                        runmat_accelerate_api::reduction::ReductionStrategy::SinglePass,
                    )
                    .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
                let Some(input_plan) =
                    input_preparation::plan(preflight, &recipe, &inventory_selection)?
                else {
                    return Ok(FusionExecutionOutcome::SharedRuntime(
                        FusionCpuObservation::new(None),
                    ));
                };
                // The synchronous adapter polls this shared route exactly
                // once. Only borrow/upload preparation is non-suspending;
                // asynchronous cast owners stay on the async entrypoint.
                if synchronous && !input_plan.casts.is_empty() {
                    return Ok(FusionExecutionOutcome::SharedRuntime(
                        FusionCpuObservation::new(None),
                    ));
                }
                let strategies =
                    match selected_provider.reduction_recipe_candidates(recipe.physical()) {
                        Ok(strategies) => strategies,
                        Err(error) if error.is_unsupported() => Vec::new(),
                        Err(error) => return Err(error),
                    };
                let mut selections = strategies
                    .into_iter()
                    .map(|strategy| {
                        runmat_accelerate_api::reduction::ReductionSelection::new(
                            recipe.physical(),
                            strategy,
                        )
                        .map_err(runmat_accelerate_api::ProviderExecutionError::from)
                    })
                    .collect::<ExecutionResult<Vec<_>>>()?;
                if !selections.is_empty() {
                    match select::choose(
                        &request,
                        selected_provider,
                        &program,
                        &selections,
                        &input_plan,
                        right.as_ref(),
                    )? {
                        select::Choice::Shared {
                            observation,
                            readback,
                        } => {
                            let observation = match right.take() {
                                Some(right) => observation.with_vm_host_route_readback(
                                    right,
                                    request.runtime.as_ref(),
                                    program.host_bindings(),
                                    readback,
                                ),
                                None => observation,
                            };
                            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
                        }
                        select::Choice::Provider { index, ticket } => {
                            let selection = selections.swap_remove(index);
                            let value = execute::selected(
                                &request,
                                selected_provider,
                                &program,
                                &recipe,
                                execute::SelectedExecution {
                                    selection,
                                    plan: input_plan,
                                    ticket,
                                    right,
                                },
                                selected_identity,
                            )
                            .await?;
                            return Ok(FusionExecutionOutcome::Executed(value));
                        }
                    }
                }
                return Ok(FusionExecutionOutcome::SharedRuntime(
                    FusionCpuObservation::new(None),
                ));
            } else {
                return Err(
                    anyhow!("typed reduction geometry differs from selected invocation").into(),
                );
            }
        }
    }
    // The legacy ABI and producer emission own only the established matrix
    // axis geometry. A typed recipe can index an exact N-D axis, but failure to
    // select that recipe must return to ordinary runtime execution rather than
    // reinterpret an N-D buffer as contiguous legacy rows or columns.
    if output_shape.len() > 2 || arithmetic.requires_typed() {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let legacy = execute_reduction_with_shape_core(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        Some(&arithmetic),
    )?;
    Ok(match legacy {
        FusionExecutionOutcome::Executed(value) => {
            FusionExecutionOutcome::Executed(TypedReductionVmOutput {
                value,
                completion: None,
            })
        }
        FusionExecutionOutcome::SharedRuntime(observation) => {
            FusionExecutionOutcome::SharedRuntime(observation)
        }
    })
}

/// Drive the effect-free static admission and non-suspending typed execution
/// once, never blocking a runtime or replaying a partially executed future.
pub(super) fn execute_projected_sync(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    use futures::FutureExt;
    let result = execute_reduction_with_shape_placed_async_inner(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        None,
        true,
    )
    .now_or_never()
    .ok_or_else(|| {
        runmat_accelerate_api::ProviderExecutionError::from(anyhow!(
            "non-suspending projected reduction unexpectedly suspended"
        ))
        .after_effect()
    })??;
    Ok(match result {
        FusionExecutionOutcome::Executed(output) => FusionExecutionOutcome::Executed(output.value),
        FusionExecutionOutcome::SharedRuntime(observation) => {
            FusionExecutionOutcome::SharedRuntime(observation)
        }
    })
}
