use super::{super::*, input_preparation::TypedInputPlan};
use crate::placement::{
    plan_local_multi, LocalMultiPlacementRequest, LocalPhysicalCandidate, LocalPlacementOutcome,
    PlacementPolicy, TotalCandidateObservationContract,
};
use runmat_accelerate_api::reduction::ReductionSelection;
use runmat_accelerate_api::{
    AccelProvider, ProviderOperationFamily, ProviderPreparationProgram,
    ProviderTypedReductionWorkload, ProviderWorkload,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    EstimateConfidence, EstimateSource, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    ExecutionCostComponents, ExecutionCostEstimate,
};
use runmat_runtime::context::VmFusionCompletionRight;

pub(super) enum Choice {
    Provider {
        index: usize,
        ticket: Box<FusionPlacementSelection>,
    },
    Shared(FusionCpuObservation),
}

pub(super) fn choose(
    request: &FusionExecutionRequest<'_>,
    provider: &'static dyn AccelProvider,
    program: &ResolvedReductionProgram<'_>,
    selections: &[ReductionSelection<'_>],
    input_plan: &TypedInputPlan<'_>,
    vm_right: Option<&VmFusionCompletionRight>,
) -> ExecutionResult<Choice> {
    let data_inputs: Vec<_> = program
        .data_inputs()
        .iter()
        .map(|input| input.value)
        .collect();
    let operation = crate::placement::fusion_operation_token(&request.plan.group.kind);
    let source_inputs: Vec<_> = input_plan
        .inventory
        .entries()
        .iter()
        .map(|entry| entry.source.clone())
        .collect();
    let prepared_inputs: Vec<_> = input_plan
        .inventory
        .entries()
        .iter()
        .map(|entry| entry.prepared.clone())
        .collect();
    let resources = crate::placement::typed_reduction_input_resources(
        provider.device_id(),
        &input_plan.inventory,
    )
    .ok_or_else(|| anyhow!("typed reduction input resource projection overflow"))?;
    let execution_units = program.producer_work_units().max(1);
    let cost = |execution_ns| ExecutionCostEstimate {
        components: ExecutionCostComponents {
            execution_ns,
            ..ExecutionCostComponents::default()
        },
        scratch_bytes: Some(0),
        confidence: EstimateConfidence::Low,
        source: EstimateSource::StaticPrior,
    };
    let cpu = ExecutionCandidateDescriptor {
        identity: format!("cpu.shared.{operation}.typed"),
        region: None,
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        cost: cost(execution_units.saturating_mul(100).max(100)),
        output_residency: CandidateOutputResidency::Host,
        presentation: runmat_execution::CandidateOutputPresentation::Direct,
        guards: Vec::new(),
    };
    let physical = selections
        .iter()
        .map(|selection| {
            let typed = ProviderTypedReductionWorkload::from_selection(selection)
                .with_input_preparation(input_plan.inventory.clone())
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            Ok(LocalPhysicalCandidate {
                physical_identity: Some(selection.identity()),
                provider_kind: ExecutionCandidateKind::ProviderFusion,
                data_inputs: &data_inputs,
                source_inputs: Some(source_inputs.clone()),
                prepared_inputs: Some(prepared_inputs.clone()),
                input_adaptation: Some(resources.route),
                outputs: vec![typed.output.clone()],
                workload: ProviderWorkload {
                    scalar_work_units: Some(typed.work.producer_scalar_work_units),
                    typed_reduction: Some(typed),
                    ..Default::default()
                },
                preparation: CandidatePreparationState::Cold,
                preparation_program: Some(ProviderPreparationProgram::TypedReduction { selection }),
                shader_requirements: None,
                fallback: cost(execution_units.saturating_mul(10).max(100)),
                required_upload_bytes: Some(resources.required_upload_bytes),
                required_download_bytes: 0,
                downstream_materialization: false,
            })
        })
        .collect::<ExecutionResult<Vec<_>>>()?;
    let outcome = plan_local_multi(
        provider,
        LocalMultiPlacementRequest {
            operation,
            semantic_identity: Some(program.identity()),
            observation_contract: if vm_right.is_some() {
                TotalCandidateObservationContract::FullCall
            } else {
                TotalCandidateObservationContract::PreparationBodyCleanup
            },
            prepared_authority: None,
            vm_right,
            family: ProviderOperationFamily::Fusion,
            cpu,
            physical,
            compare_profitability: request.placement.is_some(),
            runtime: request.runtime.as_ref(),
        },
        PlacementPolicy::default(),
    )
    .map_err(|error| runmat_accelerate_api::ProviderExecutionError::from_source(*error))?;
    let observer = FusionPlacementObserver::new(request.placement);
    match outcome {
        LocalPlacementOutcome::ProviderRejected { code, feedback, .. } => {
            observer.provider_rejected(code);
            Ok(Choice::Shared(FusionCpuObservation::new(feedback)))
        }
        LocalPlacementOutcome::Selected {
            selected,
            provider_cost,
            feedback,
            provider_query,
            provider_observation,
            provider_index: Some(index),
            ..
        } if selected.kind.is_provider() => {
            observer.provider_candidate();
            observer.selected(
                PlacementVariant::ProviderFusion,
                "typed_reduction_complete_cost",
                provider_cost.checked_total_ns(),
            );
            let mut ticket = if vm_right.is_some() {
                FusionPlacementSelection::new_vm(
                    observer,
                    feedback.map(|feedback| *feedback),
                    provider,
                    provider_observation.then_some(provider_query.as_ref()),
                )
            } else {
                FusionPlacementSelection::new(
                    observer,
                    feedback.map(|feedback| *feedback),
                    provider,
                    provider_observation.then_some(provider_query.as_ref()),
                )
            };
            if vm_right.is_some() {
                ticket.require_vm_provider_execution(
                    provider.device_id(),
                    provider_query.operation.clone(),
                );
            }
            Ok(Choice::Provider {
                index,
                ticket: Box::new(ticket),
            })
        }
        LocalPlacementOutcome::Selected {
            selected,
            cpu_cost,
            feedback,
            provider_index: None,
            ..
        } if !selected.kind.is_provider() => {
            observer.provider_candidate();
            observer.selected(
                PlacementVariant::SharedRuntime,
                "typed_reduction_complete_cost",
                cpu_cost.checked_total_ns(),
            );
            Ok(Choice::Shared(FusionCpuObservation::new(feedback)))
        }
        LocalPlacementOutcome::Selected { .. } => Err(anyhow!(
            "typed reduction placement returned an inconsistent physical ordinal"
        )
        .into()),
    }
}
