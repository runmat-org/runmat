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
    // Sync and async callers share one selected mathematical contract owner.
    // Static admission never blocks to invent missing source/domain proof.
    execute_projected_sync(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
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
    let mut right = Some(right);
    execute_reduction_with_shape_placed_async_inner(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        Some(super::VmFusionExecutionRight::Completion(&mut right)),
        false,
    )
    .await
}

/// A sealed VM window may decline before effects without acquiring FullCall
/// completion or host-readback authority.
pub async fn execute_reduction_with_shape_placed_async_vm_decline(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
    decline: runmat_runtime::context::VmFusionDeclineRight,
) -> ExecutionResult<FusionExecutionOutcome<TypedReductionVmOutput>> {
    execute_reduction_with_shape_placed_async_inner(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        Some(super::VmFusionExecutionRight::Decline(&decline)),
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
    mut authority: Option<super::VmFusionExecutionRight<'_>>,
    synchronous: bool,
) -> ExecutionResult<FusionExecutionOutcome<TypedReductionVmOutput>> {
    let host_runtime = request.runtime.clone();
    let host_control =
        runmat_runtime::context::host_allocation::ProviderHostControlAdmission::capture(
            host_runtime.as_ref(),
        )
        .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
    let host_metadata = host_control.as_ref().map(|admission| {
        admission as &dyn runmat_execution::resource::ProviderHostMetadataAdmission
    });
    let binding_bytes = super::binding_storage_bytes(request.plan.operations.len())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| anyhow!("reduction binding capture size overflow"))?;
    let _entry_binding_claim = crate::wgsl::host_buffers::claim_bytes(
        host_metadata,
        usize::try_from(binding_bytes)
            .map_err(|_| anyhow!("reduction binding size exceeds usize"))?,
    )
    .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
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
    {
        let current = AdmittedBindings::new(&request)
            .ok_or_else(|| anyhow!("selected reduction binding changed during admission"))?;
        if current.identity(None)? != selected_identity
            || !provider().is_some_and(|current_provider| {
                runmat_accelerate_api::same_provider_instance(current_provider, selected_provider)
            })
        {
            return Err(anyhow!("selected reduction authority changed during admission").into());
        }
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
    let Some(program) = program else {
        // Missing or provider-inexpressible mathematics is not permission to
        // forget its accumulator/class contract and try a raw legacy kernel.
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    {
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
            let recipe = crate::fusion::reduction_recipe::build(&program, config, host_metadata)
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
                let strategies = match selected_provider
                    .reduction_recipe_candidates(recipe.physical(), host_metadata)
                {
                    Ok(strategies) => strategies,
                    Err(error) if error.is_unsupported() => [None, None],
                    Err(error) => return Err(error),
                };
                let selection_bytes = std::alloc::Layout::array::<
                    runmat_accelerate_api::reduction::ReductionSelection<'_>,
                >(2)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?
                .size();
                let _selections_claim =
                    crate::wgsl::host_buffers::claim_bytes(host_metadata, selection_bytes)
                        .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
                let mut selections = runmat_value::exact_capacity_vec(2)
                    .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
                for strategy in strategies.into_iter().flatten() {
                    selections.push(
                        runmat_accelerate_api::reduction::ReductionSelection::new(
                            recipe.physical(),
                            strategy,
                        )
                        .map_err(runmat_accelerate_api::ProviderExecutionError::from)?,
                    );
                }
                if !selections.is_empty() {
                    match select::choose(
                        &request,
                        selected_provider,
                        &program,
                        &selections,
                        &input_plan,
                        authority
                            .as_ref()
                            .and_then(super::VmFusionExecutionRight::as_selection),
                        host_metadata,
                    )? {
                        select::Choice::Shared {
                            observation,
                            readback,
                        } => {
                            let observation = match authority
                                .take()
                                .and_then(super::VmFusionExecutionRight::into_completion)
                            {
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
                                    right: authority
                                        .and_then(super::VmFusionExecutionRight::into_completion),
                                },
                                selected_identity,
                                host_metadata,
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
    if output_shape.len() > 2 {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
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
    // Deferred legacy admission may suspend or probe. Authenticate the same
    // selected binding/provider again before this fallback reaches effects.
    let current = AdmittedBindings::new(&request)
        .ok_or_else(|| anyhow!("selected reduction binding changed during legacy admission"))?;
    if current.identity(None)? != selected_identity
        || !provider().is_some_and(|current_provider| {
            runmat_accelerate_api::same_provider_instance(current_provider, selected_provider)
        })
    {
        return Err(anyhow!("selected reduction authority changed during legacy admission").into());
    }
    if arithmetic.requires_typed() {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let legacy = execute_reduction_with_shape_core(
        &request,
        legacy::LegacyReductionProofs {
            mathematical: &program,
            producer: Some(&arithmetic),
            owner: selected_provider,
        },
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
        authority
            .as_ref()
            .and_then(super::VmFusionExecutionRight::as_selection)
            .and_then(super::VmFusionSelectionRight::decline),
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
