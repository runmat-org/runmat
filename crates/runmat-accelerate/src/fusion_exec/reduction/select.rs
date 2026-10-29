use super::{
    super::*,
    host_readback::{self, CpuHostAdmission, CpuReadbackPlan},
    input_preparation::TypedInputPlan,
};
use crate::placement::{
    plan_local_multi, LocalCpuAdmission, LocalMultiPlacementRequest, LocalPhysicalCandidate,
    LocalPlacementOutcome, LocalPublicationSemantics, PlacementPolicy,
    TotalCandidateObservationContract,
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

pub(super) enum Choice {
    Provider {
        index: usize,
        ticket: Box<FusionPlacementSelection>,
    },
    Shared {
        observation: FusionCpuObservation,
        readback: Option<CpuReadbackPlan>,
    },
}

pub(super) fn choose(
    request: &FusionExecutionRequest<'_>,
    provider: &'static dyn AccelProvider,
    program: &ResolvedReductionProgram<'_>,
    selections: &[ReductionSelection<'_>],
    input_plan: &TypedInputPlan<'_>,
    vm_selection: Option<super::super::VmFusionSelectionRight<'_>>,
    host_metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> ExecutionResult<Choice> {
    let vm_right = vm_selection.and_then(super::super::VmFusionSelectionRight::completion);
    let vm_decline = vm_selection.and_then(super::super::VmFusionSelectionRight::decline);
    let vm_decline = vm_decline.filter(|right| {
        request.runtime.as_ref().is_some_and(|runtime| {
            right.is_current(runtime)
                && right.matches_group(
                    request.plan.group.id,
                    request.plan.group.span.start,
                    request.plan.group.span.end,
                )
        })
    });
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
    let cost = |execution_ns| ExecutionCostEstimate {
        components: ExecutionCostComponents {
            execution_ns,
            ..ExecutionCostComponents::default()
        },
        scratch_bytes: Some(0),
        confidence: EstimateConfidence::Low,
        source: EstimateSource::StaticPrior,
    };
    let host = if vm_right.is_some() {
        host_readback::admit(request, program, provider)
    } else if data_inputs
        .iter()
        .all(|value| !runmat_runtime::dispatcher::value_contains_gpu(value))
    {
        Some(CpuHostAdmission::Direct)
    } else {
        None
    };
    let host_execution_ns = program
        .host_total_work_units()
        .and_then(|units| units.checked_mul(100));
    // An executable ordinary route without a price remains ordinary execution,
    // not a fabricated placement candidate or a provider feasibility failure.
    if host.is_some() && host_execution_ns.is_none() {
        return Ok(Choice::Shared {
            observation: FusionCpuObservation::new(None),
            readback: selected_readback(host),
        });
    }
    let mut cpu = ExecutionCandidateDescriptor {
        identity: format!("cpu.shared.{operation}.typed"),
        region: None,
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        // With no executable host route, LocalCpuAdmission::Unavailable keeps
        // this structural descriptor out of the numeric graph and feedback.
        cost: cost(host_execution_ns.unwrap_or(0)),
        output_residency: CandidateOutputResidency::Host,
        presentation: runmat_execution::CandidateOutputPresentation::Direct,
        guards: Vec::new(),
    };
    let cpu_admission = match &host {
        Some(CpuHostAdmission::Direct) => LocalCpuAdmission::Legacy,
        Some(CpuHostAdmission::Readback(admission)) => {
            cpu.presentation = admission.presentation.clone();
            cpu.cost.scratch_bytes = Some(admission.scratch_bytes);
            LocalCpuAdmission::Exact(Box::new(admission.resources))
        }
        None => LocalCpuAdmission::Unavailable,
    };
    let physical = selections
        .iter()
        .map(|selection| {
            let execution_units = selection
                .work()
                .checked_total_scalar_work_units()
                .ok_or_else(|| anyhow!("selected reduction total work overflow"))?;
            let execution_ns = execution_units
                .checked_mul(10)
                .ok_or_else(|| anyhow!("selected reduction execution prior overflow"))?;
            let typed = ProviderTypedReductionWorkload::from_selection(selection)
                .with_input_preparation(input_plan.inventory.clone())
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            Ok(LocalPhysicalCandidate {
                authority: None,
                physical_identity: Some(selection.identity()),
                provider_kind: ExecutionCandidateKind::ProviderFusion,
                data_inputs: &data_inputs,
                source_inputs: Some(source_inputs.clone()),
                prepared_inputs: Some(prepared_inputs.clone()),
                input_adaptation: Some(resources.route),
                outputs: vec![typed.output.clone()],
                workload: ProviderWorkload {
                    scalar_work_units: Some(execution_units),
                    typed_reduction: Some(typed),
                    ..Default::default()
                },
                preparation: CandidatePreparationState::Cold,
                preparation_program: Some(ProviderPreparationProgram::TypedReduction {
                    selection,
                    host_metadata,
                }),
                shader_requirements: None,
                fallback: runmat_execution::CandidateCostEvidence::Estimated(cost(execution_ns)),
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
            vm_decline,
            source_decline: crate::placement::LocalSourceDecline::None,
            family: ProviderOperationFamily::Fusion,
            cpu: cpu.into(),
            cpu_admission,
            publication: LocalPublicationSemantics::Atomic,
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
            if host.is_none() && vm_decline.is_none() {
                return Err(anyhow!(
                    "typed provider rejected and no executable host reduction route exists"
                )
                .into());
            }
            Ok(Choice::Shared {
                observation: FusionCpuObservation::new(if host.is_some() {
                    feedback
                } else {
                    None
                }),
                readback: selected_readback(host),
            })
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
                provider_cost.and_then(|cost| cost.checked_total_ns()),
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
                cpu_cost.estimate().and_then(|cost| cost.checked_total_ns()),
            );
            if host.is_none() {
                return Err(anyhow!(
                    "placement selected a host reduction without an executable host route"
                )
                .into());
            }
            Ok(Choice::Shared {
                observation: FusionCpuObservation::new(feedback),
                readback: selected_readback(host),
            })
        }
        LocalPlacementOutcome::Selected { .. } => Err(anyhow!(
            "typed reduction placement returned an inconsistent physical ordinal"
        )
        .into()),
    }
}

fn selected_readback(host: Option<CpuHostAdmission>) -> Option<CpuReadbackPlan> {
    match host {
        Some(CpuHostAdmission::Readback(admission)) => Some(admission.plan),
        _ => None,
    }
}
