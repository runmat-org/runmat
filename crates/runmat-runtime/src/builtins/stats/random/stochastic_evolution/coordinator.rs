//! Join semantic preparation, whole-call placement and ordered execution.
//! Every offered route is physically prepared before source effects.
use super::execution::execute_selected_host_source;
use super::host_readback::StochasticHostReadback;
use super::placement::{
    select_source, ProviderCandidate, StochasticPlacementSelection, StochasticRoute,
};
use super::planning::{prepare_provider, StochasticProviderPlan};
use super::provider::{
    InitialProviderSource, SelectedStochasticProvider, StochasticNoiseUploadPlan,
    StochasticSelectedProviderRoute,
};
use super::resources::{
    host_resources, provider_resources, StochasticHostResources, StochasticProviderResources,
};
use super::source_eligibility::{
    StochasticCheckpoint, StochasticPreparedSource, StochasticSourceExecution,
};
use crate::builtins::common::random::{self, gpu_like::host_random_mode};
use crate::context::host_presentation::HostSinkReceipt;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};
use runmat_execution::ResourceByteBound;
use runmat_value::Value;

/// The executor owns publication and its resource evidence. Runtime must not
/// guess how many workspace bindings, roots or shape views a callback keeps.
pub struct StochasticSourcePublication<'a> {
    /// Bound supplied by the actual publication owner, whose funding must
    /// cover retained metadata too. Unavailable evidence is never zero demand.
    pub host_workspace: ResourceByteBound,
    pub checkpoint: &'a mut dyn FnMut(StochasticCheckpoint) -> BuiltinResult<()>,
}

struct PreparedProviderRoute {
    plan: StochasticProviderPlan,
    initial: InitialProviderSource,
    noise_upload: StochasticNoiseUploadPlan,
    resources: StochasticProviderResources,
}

fn prepare_provider_route(
    context: &RuntimeContext,
    prepared: &StochasticPreparedSource,
    host: &StochasticHostResources,
    publication: ResourceByteBound,
) -> BuiltinResult<Option<PreparedProviderRoute>> {
    let Some(plan) = prepare_provider(context, prepared)? else {
        return Ok(None);
    };
    let Some(initial) =
        InitialProviderSource::prepare(context, plan.provider, prepared, &plan.selected)?
    else {
        return Ok(None);
    };
    let Some(noise_upload) =
        StochasticNoiseUploadPlan::prepare(context, plan.provider, &plan.selected)?
    else {
        return Ok(None);
    };
    let Some(resources) = provider_resources(
        plan.provider,
        &plan.selected,
        &initial,
        &noise_upload,
        host,
        publication,
        1_000,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(PreparedProviderRoute {
        plan,
        initial,
        noise_upload,
        resources,
    }))
}

/// A decline is possible only before the first source checkpoint, draw or
/// device effect. A selected executor owns every later error and cancellation.
pub async fn execute_stochastic_source(
    context: &RuntimeContext,
    prepared: StochasticPreparedSource,
    publication: StochasticSourcePublication<'_>,
) -> BuiltinResult<StochasticSourceExecution> {
    StochasticPreparedSource::checkpoint_receipt(prepared.receipt(), context)?;
    let mode = host_random_mode(Some(context));
    let algorithm = random::normal_source_algorithm(context)?;
    let host = host_resources(prepared.pattern(), mode, publication.host_workspace)?;
    let mut readback = StochasticHostReadback::prepare(context, &prepared)?;
    let provider = prepare_provider_route(context, &prepared, &host, publication.host_workspace)?;
    let offered_provider = provider.as_ref().map(|route| ProviderCandidate {
        plan: &route.plan,
        resources: &route.resources,
        noise_host_bytes: host.noise_output_bytes,
    });
    let offered_host = match prepared.state() {
        Value::GpuTensor(_) if readback.is_none() => None,
        _ => Some(&host),
    };
    let Some(selection) = select_source(
        context,
        &prepared,
        offered_host,
        readback.as_ref(),
        offered_provider,
        algorithm,
    )?
    else {
        return Ok(StochasticSourceExecution::Declined);
    };
    let mut publish = |checkpoint| (publication.checkpoint)(checkpoint);
    match selection.route {
        StochasticRoute::Host => {
            drop(provider);
            if let Some(readback) = readback.as_mut() {
                if !readback.admit_selected_native(context)? {
                    return Ok(StochasticSourceExecution::Declined);
                }
            }
            let completion = selection.selected.map(|selected| match selected {
                StochasticPlacementSelection::Portable(selected) => {
                    HostSinkReceipt::new(context.clone(), selected)
                }
                StochasticPlacementSelection::Managed(selected) => {
                    HostSinkReceipt::new_managed(context.clone(), selected)
                }
            });
            execute_selected_host_source(
                context,
                prepared,
                readback,
                mode,
                algorithm,
                completion,
                &mut publish,
            )
            .await
        }
        StochasticRoute::Provider => {
            drop(readback);
            let (Some(provider), Some(selected)) = (provider, selection.selected) else {
                return Err(build_runtime_error(
                    "selected stochastic provider lost its prepared route",
                )
                .with_identifier("RunMat:StochasticSource:InvalidSelection")
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .build());
            };
            SelectedStochasticProvider::new(
                context,
                StochasticSelectedProviderRoute {
                    plan: provider.plan,
                    initial: provider.initial,
                    noise_upload: provider.noise_upload,
                    resources: provider.resources,
                    selected,
                    mode,
                    algorithm,
                },
            )?
            .execute(context, prepared, &mut publish)
            .await
        }
    }
}
