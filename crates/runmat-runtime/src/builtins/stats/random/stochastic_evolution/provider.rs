//! Selected provider source phase. Runtime owns every ordered normal draw and
//! supplies bounded borrowed noise tiles; the provider owns only arithmetic.

use super::placement::StochasticPlacementSelection;
use super::planning::StochasticProviderPlan;
use super::resources::StochasticProviderResources;
use super::source_eligibility::StochasticPreparedSource;
use super::source_eligibility::{StochasticCheckpoint, StochasticSourceExecution};
use crate::builtins::array::creation::randn::draw_source_normal;
use crate::builtins::common::random::{self, gpu_like::ExactHostRandomMode, RngAlgorithm};
use crate::context::host_allocation::{
    provider_registration_host_quote, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt, ProviderRegistrationHostQuote,
};
use crate::context::host_presentation::HostPresentationReceipt;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::observation::{
    observe_provider_resource_future, with_provider_observation, ProviderObservationFailure,
    ProviderObservationSupport, ProviderOperationRecorder, ProviderResourcePhasePlan,
    ProviderResourceScope,
};
use runmat_accelerate_api::stochastic_evolution::{
    ProviderStochasticEvolutionPending, ProviderStochasticEvolutionProgress,
    ProviderStochasticEvolutionSelection, ProviderStochasticEvolutionTileRequest,
};
use runmat_accelerate_api::{
    class_preserving_upload_query, upload_numeric_protecting_admitted, AccelProvider,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    ProviderElementType, ProviderFeasibility, ProviderFeasibilityQuery, ProviderLayout,
    ProviderNumericUploadPhysicalEvidence, ProviderQueryMetadataFootprint, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};
use runmat_execution::resource::{ProviderHostMetadataAdmission, RetainedHostAllocationClaim};
use runmat_execution::ResourceByteBound;
use runmat_types::NumericClass;
use runmat_value::{HostNumericBuffer, NumericDType, Value};
use std::sync::Arc;

fn source_contract_error() -> RuntimeError {
    build_runtime_error("selected stochastic noise changed representation")
        .with_identifier("RunMat:StochasticSource:InvalidNoise")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn upload_plan_error(detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("selected stochastic noise upload: {detail}"))
        .with_identifier("RunMat:StochasticSource:InvalidUploadPlan")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn upload_admission_error(context: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    build_runtime_error(format!(
        "selected stochastic noise upload admission: {cause}"
    ))
    .with_identifier(format!(
        "{}:{}",
        context.error_namespace(),
        cause.runtime_identifier_suffix()
    ))
    .with_source(cause)
    .with_gpu_gather_retry(GpuGatherRetry::Never)
    .build()
}

mod execution;
mod noise;
mod upload;

use noise::RetainedSourceNoise;
use upload::upload_selected_phase;
pub(super) use upload::{
    InitialProviderSource, StochasticNoiseUploadPhase, StochasticNoiseUploadPlan,
};

/// A complete provider route already selected by the canonical whole-work
/// placement owner. This holds the sole feedback receipt, planning/query
/// allocations, exact upload evidence, metadata claims and physical bounds
/// through all source checkpoints and submitted tiles.
pub(super) struct SelectedStochasticProvider {
    plan: StochasticProviderPlan,
    initial: Option<InitialProviderSource>,
    noise_upload: StochasticNoiseUploadPlan,
    resources: StochasticProviderResources,
    host_receipt: Option<HostAllocationReceipt>,
    metadata: Option<Box<dyn ProviderHostMetadataAdmission>>,
    completion_control: Option<Arc<dyn RetainedHostAllocationClaim>>,
    completion: Option<HostPresentationReceipt>,
    mode: ExactHostRandomMode,
    algorithm: RngAlgorithm,
}

/// All immutable, already selected authorities for one provider source call.
/// The placement owner constructs this once; neither construction nor
/// execution reselects a route or recreates physical upload evidence.
pub(super) struct StochasticSelectedProviderRoute {
    pub(super) plan: StochasticProviderPlan,
    pub(super) initial: InitialProviderSource,
    pub(super) noise_upload: StochasticNoiseUploadPlan,
    pub(super) resources: StochasticProviderResources,
    pub(super) selected: StochasticPlacementSelection,
    pub(super) mode: ExactHostRandomMode,
    pub(super) algorithm: RngAlgorithm,
}

impl SelectedStochasticProvider {
    pub(super) fn new(
        context: &RuntimeContext,
        route: StochasticSelectedProviderRoute,
    ) -> BuiltinResult<Self> {
        let StochasticSelectedProviderRoute {
            plan,
            initial,
            noise_upload,
            resources,
            selected,
            mode,
            algorithm,
        } = route;
        if !plan.selected.proposal.validate_for(&plan.selected.plan) {
            return Err(upload_plan_error(
                "provider selection changed before receipt issuance",
            ));
        }
        let host_receipt = HostAllocationReceipt::admit(Some(context), resources.host_plan)
            .map_err(|cause| upload_admission_error(context, cause))?;
        let metadata = if let Some(receipt) = host_receipt.as_ref() {
            let permit = receipt
                .retained_view_permit(context)
                .map_err(|cause| upload_admission_error(context, cause))?;
            Some(
                plan.provider
                    .prepare_reusable_output_host_metadata(
                        Box::new(permit),
                        resources.metadata_registration_count,
                    )
                    .map_err(|cause| {
                        provider_failure("stochastic output metadata preparation failed", cause)
                    })?,
            )
        } else {
            None
        };
        let completion_control = if let Some(receipt) = host_receipt.as_ref() {
            Some(
                receipt
                    .claim_retained_metadata(
                        context,
                        plan.selected.proposal.completion_control_host_bytes,
                    )
                    .map_err(|cause| upload_admission_error(context, cause))?,
            )
        } else {
            None
        };
        let publication_metadata = if let Some(receipt) = host_receipt.as_ref() {
            Some(
                receipt
                    .claim_retained_metadata(context, resources.publication_metadata_payload_bytes)
                    .map_err(|cause| upload_admission_error(context, cause))?,
            )
        } else {
            None
        };
        let ProviderObservationSupport::Observed(mut observation) =
            plan.provider.begin_operation_observation(&plan.query)
        else {
            // The whole-call provider route is already selected. A changed
            // observation capability is a failed selection, not a license to
            // run a second route or re-draw the caller's random stream.
            return Err(upload_plan_error(
                "selected provider observation is no longer available",
            ));
        };
        // This exact selected plan binds initial/noise transfers to the SAME
        // whole-call receipt and internally requires the exact provider step.
        // Transfer fences never confirm arithmetic.
        observation.require_stochastic_evolution_source_transfers(
            plan.provider.device_id(),
            &plan.selected,
        );
        if !observation
            .recorder()
            .expects_stochastic_evolution_selection(&plan.selected)
        {
            return Err(upload_plan_error(
                "selected source transfer observation was not issued",
            ));
        }
        let completion = match selected {
            StochasticPlacementSelection::Portable(selected) => {
                HostPresentationReceipt::new(context.clone(), selected, observation)
                    .with_publication_metadata_claim(publication_metadata)
            }
            StochasticPlacementSelection::Managed(selected) => {
                let claim = publication_metadata.ok_or_else(|| {
                    upload_plan_error("managed source feedback has no admitted metadata owner")
                })?;
                HostPresentationReceipt::new_managed(context.clone(), selected, observation, claim)
                    .map_err(upload_plan_error)?
            }
        };
        Ok(Self {
            plan,
            initial: Some(initial),
            noise_upload,
            resources,
            host_receipt,
            metadata,
            completion_control,
            completion: Some(completion),
            mode,
            algorithm,
        })
    }

    fn recorder(&self) -> ProviderOperationRecorder {
        self.completion
            .as_ref()
            .expect("selected route holds feedback")
            .recorder()
    }

    async fn finish(&mut self, published: Result<bool, ProviderObservationFailure>) {
        if let Some(completion) = self.completion.take() {
            completion.finish_selected(published).await;
        }
    }
}

fn provider_failure(
    detail: &'static str,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        build_runtime_error(detail)
            .with_identifier("RunMat:StochasticSource:ProviderFailed")
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build(),
        source,
    )
}
