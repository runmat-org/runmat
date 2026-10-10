use super::{super::*, input_preparation::TypedInputPlan};
use crate::fusion::reduction_recipe::ReductionRecipe;
use runmat_accelerate_api::observation::observe_provider_future;
use runmat_accelerate_api::reduction::{ReductionSelection, ReductionSubmission};
use runmat_accelerate_api::{AccelProvider, ProviderExecutionError};
use runmat_runtime::context::VmFusionCompletionRight;

pub(super) struct SelectedExecution<'recipe, 'inputs> {
    pub selection: ReductionSelection<'recipe>,
    pub plan: TypedInputPlan<'inputs>,
    pub ticket: Box<FusionPlacementSelection>,
    pub right: Option<VmFusionCompletionRight>,
}

pub(super) async fn selected(
    request: &FusionExecutionRequest<'_>,
    selected_provider: &'static dyn AccelProvider,
    program: &ResolvedReductionProgram<'_>,
    recipe: &ReductionRecipe,
    physical: SelectedExecution<'_, '_>,
    selected_identity: runmat_execution::Digest,
) -> ExecutionResult<TypedReductionVmOutput> {
    let SelectedExecution {
        selection,
        plan,
        ticket,
        right,
    } = physical;
    if !std::ptr::eq(recipe.physical(), selection.recipe()) {
        return Err(anyhow!("typed reduction selection changed before execution").into());
    }
    let prepared_effects = plan.has_physical_effects();
    let current = AdmittedBindings::new(request)
        .ok_or_else(|| anyhow!("selected reduction binding changed before preparation"))?;
    if current.identity(None)? != selected_identity
        || !provider().is_some_and(|current_provider| {
            runmat_accelerate_api::same_provider_instance(current_provider, selected_provider)
        })
    {
        return Err(anyhow!("selected reduction authority changed before preparation").into());
    }
    let recorder = ticket.recorder();
    observe_provider_future(recorder, async move {
        let prepared = plan.prepare().await?;
        let current = AdmittedBindings::new(request).ok_or_else(|| {
            phase(
                ProviderExecutionError::from(anyhow!(
                    "selected reduction binding changed after preparation"
                )),
                prepared_effects,
            )
        })?;
        if current
            .identity(None)
            .map_err(|error| phase(error.into(), prepared_effects))?
            != selected_identity
            || !provider().is_some_and(|current_provider| {
                runmat_accelerate_api::same_provider_instance(current_provider, selected_provider)
            })
        {
            return Err(phase(
                ProviderExecutionError::from(anyhow!(
                    "selected reduction authority changed after preparation"
                )),
                prepared_effects,
            ));
        }
        let precision = selection.recipe().descriptor().output_precision;
        let output_shape = selection.recipe().descriptor().output_shape.clone();
        let submitted_strategy = selection.strategy();
        let submitted_work = selection.work();
        let submission =
            ReductionSubmission::new(selection, &prepared, selected_provider.device_id())
                .map_err(|error| phase(error, prepared_effects))?;
        let output = selected_provider
            .submit_reduction_recipe(&submission)
            .map_err(|error| phase(error, prepared_effects))?;
        let originals = program
            .data_inputs()
            .iter()
            .filter_map(|input| match input.value {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            });
        let mut admission =
            outputs::FusionOutputAdmission::with_borrowed_handles(originals.chain(prepared.iter()));
        let mut output = admission
            .admit(
                output,
                selected_provider,
                output_shape.as_slice(),
                precision,
            )
            .map_err(|error| phase(error, true))?;
        super::super::mark_fusion_output_from_values(
            &mut output,
            program.data_inputs().iter().map(|input| input.value),
        )
        .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        ticket
            .observer
            .typed_reduction_submitted(submitted_strategy, submitted_work);
        let completion = if let Some(right) = right {
            (*ticket).into_vm_completion(right)
        } else {
            (*ticket).submitted();
            None
        };
        Ok(TypedReductionVmOutput {
            value: Value::GpuTensor(output),
            completion,
        })
    })
    .await
}

fn phase(error: ProviderExecutionError, after_preparation: bool) -> ProviderExecutionError {
    if after_preparation {
        error.after_effect()
    } else {
        error
    }
}
