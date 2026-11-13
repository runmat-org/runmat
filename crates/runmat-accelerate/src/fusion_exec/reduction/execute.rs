use super::{super::*, input_preparation::TypedInputPlan};
use crate::fusion::reduction_recipe::ReductionRecipe;
use runmat_accelerate_api::observation::observe_provider_future;
use runmat_accelerate_api::reduction::{ReductionSelection, ReductionSubmission};
use runmat_accelerate_api::{AccelProvider, ProviderExecutionError};
use runmat_runtime::context::host_allocation::{
    provider_registration_host_quote, HostAllocationError, HostAllocationPlan,
    HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_runtime::context::VmFusionCompletionRight;

/// The provider-specific permit owns actual registration rights; the original
/// Runtime receipt keeps preparation/headroom alive until that Box is gone.
struct OutputRegistration {
    permit: Option<Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>>,
    _receipt: Option<HostAllocationReceipt>,
}
impl OutputRegistration {
    fn prepare(
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
        provider: &dyn AccelProvider,
        rank: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Self> {
        if HostAllocationReceipt::policy(runtime).map_err(ProviderExecutionError::from_source)?
            == HostAllocationPolicy::Unbounded
        {
            return Ok(Self {
                permit: None,
                _receipt: None,
            });
        }
        let runtime = runtime.ok_or_else(|| {
            ProviderExecutionError::from_source(HostAllocationError::MissingManagedOwner)
        })?;
        let quote = provider_registration_host_quote(provider, rank, 1, 0)
            .map_err(ProviderExecutionError::from_source)?
            .ok_or_else(|| {
                ProviderExecutionError::from(runmat_accelerate_api::unsupported_provider_operation(
                    "selected reduction provider has no finite output registration owner",
                ))
            })?;
        let plan = HostAllocationPlan::new(quote.preparation_bytes, quote.retained_bytes)
            .map_err(ProviderExecutionError::from_source)?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(ProviderExecutionError::from_source)?
            .ok_or_else(|| {
                ProviderExecutionError::from_source(HostAllocationError::MissingManagedOwner)
            })?;
        let retained = receipt
            .retained_view_permit(runtime)
            .map_err(ProviderExecutionError::from_source)?;
        let permit = provider.prepare_output_host_metadata(Box::new(retained), 1)?;
        Ok(Self {
            permit: Some(permit),
            _receipt: Some(receipt),
        })
    }
    fn permit(&self) -> Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission> {
        self.permit.as_deref()
    }
}

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
    host_metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
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
    program.require_current_controls(request)?;
    let current = AdmittedBindings::new(request)
        .ok_or_else(|| anyhow!("selected reduction binding changed before preparation"))?;
    if current.identity(None)? != selected_identity
        || !provider().is_some_and(|current_provider| {
            runmat_accelerate_api::same_provider_instance(current_provider, selected_provider)
        })
    {
        return Err(anyhow!("selected reduction authority changed before preparation").into());
    }
    // Funding and exact provider-table capacity precede even input staging.
    // This is not the separate non-view control metadata port.
    let output_registration = OutputRegistration::prepare(
        request.runtime.as_ref(),
        selected_provider,
        recipe.physical().descriptor().output_shape.len(),
    )?;
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
        let output_shape = &recipe.physical().descriptor().output_shape;
        let submitted_strategy = selection.strategy();
        let submitted_work = selection.work();
        let submission = ReductionSubmission::new(
            selection,
            &prepared,
            selected_provider.device_id(),
            host_metadata,
            output_registration.permit(),
        )
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
