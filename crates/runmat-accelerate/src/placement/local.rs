use runmat_accelerate_api::{
    AccelProvider, ProviderFeasibilityQuery, ProviderOperationFamily, ProviderRejectionCode,
    ProviderRepresentation, ProviderWorkload,
};
use runmat_execution::{
    CandidatePreparationState, Digest, ExecutionCandidateKind, ExecutionCostEstimate,
    PlacementRevision, PlacementSignature,
};
use runmat_value::Value;

use super::PlacementPolicy;
mod cpu;
mod multi;
pub(crate) use cpu::LocalCpuCandidate;
pub(crate) use multi::{
    plan_local_multi, plan_local_multi_bounded, LocalCpuAdmission, LocalMultiPlacementRequest,
    LocalPhysicalCandidate, LocalProviderAdmission, LocalPublicationSemantics, LocalSourceDecline,
    PreparedPhysicalAuthority,
};
mod rejected;
mod session_facts;
mod signature;
use rejected::{rejected_cpu_feedback, RejectedCpuFeedbackRequest};
use session_facts::session_facts;
fn plan_error(error: crate::placement::PlacementPlanError) -> Box<runmat_runtime::RuntimeError> {
    Box::new(
        runmat_runtime::build_runtime_error(error.message.clone())
            .with_identifier(error.code)
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .with_source(error)
            .build(),
    )
}
#[cfg(test)]
mod resource_tests;

pub(crate) struct LocalPlacementRequest<'a> {
    pub(crate) compound: Option<&'a runmat_runtime::context::preparation::AdmittedBuiltinCompound>,
    pub(crate) operation: &'a str,
    /// Optional deterministic identity of the selected typed program. This
    /// augments, rather than replaces, argument/resource/preparation facts.
    pub(crate) semantic_identity: Option<Digest>,
    pub(crate) observation_contract: TotalCandidateObservationContract,
    pub(crate) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    /// The selected source implementation may handle a typed provider
    /// feasibility refusal without manufacturing a CPU placement candidate.
    pub(crate) source_decline: LocalSourceDecline,
    pub(crate) family: ProviderOperationFamily,
    pub(crate) provider_kind: ExecutionCandidateKind,
    pub(crate) inputs: &'a [&'a Value],
    pub(crate) outputs: Vec<ProviderRepresentation>,
    pub(crate) workload: ProviderWorkload,
    pub(crate) preparation: CandidatePreparationState,
    pub(crate) preparation_program: Option<runmat_accelerate_api::ProviderPreparationProgram<'a>>,
    /// Requirement sealed by the selected shader producer. This is checked
    /// through the same rejection/feedback path as provider feasibility.
    pub(crate) shader_requirements: Option<runmat_accelerate_api::ProviderShaderRequirements>,
    pub(crate) cpu: LocalCpuCandidate,
    /// Explicit physical execution evidence without transfer components, or
    /// typed absence. Provider-owned quotes take precedence over this evidence.
    pub(crate) provider_fallback: runmat_execution::CandidateCostEvidence,
    /// Physical transfer bytes when an admitted preparation changes the input
    /// representation. None preserves ordinary residency-based accounting.
    pub(crate) required_upload_bytes: Option<u64>,
    pub(crate) required_download_bytes: u64,
    pub(crate) downstream_materialization: bool,
    /// Direct provider APIs preserve caller intent after feasibility succeeds;
    /// correlated runtime requests compare against the CPU candidate.
    pub(crate) compare_profitability: bool,
    /// Explicit session authority for guarded cached/adaptive planning. Direct
    /// provider APIs and legacy call sites may omit it and use deterministic
    /// local selection.
    pub(crate) runtime: Option<&'a runmat_runtime::context::RuntimeContext>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TotalCandidateObservationContract {
    PreparationBodyCleanup,
    FullCall,
    NativeFullCall,
}

fn observation_revision(
    runtime: &runmat_runtime::context::RuntimeContext,
    contract: TotalCandidateObservationContract,
    authority: Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    vm_right: Option<&runmat_runtime::context::VmFusionCompletionRight>,
    native_right: Option<
        &runmat_runtime::numeric_region::service::NativeNumericRegionSelectionAuthority,
    >,
    provider_digest: Digest,
    policy: PlacementPolicy,
) -> Option<PlacementRevision> {
    match (contract, authority) {
        (TotalCandidateObservationContract::NativeFullCall, None)
            if native_right.is_some_and(|right| right.is_current(runtime)) =>
        {
            Some(super::revision::local(
                runtime.program_revision().cloned(),
                provider_digest,
                policy,
            ))
        }
        (TotalCandidateObservationContract::PreparationBodyCleanup, None) => Some(
            super::revision::local(runtime.program_revision().cloned(), provider_digest, policy),
        ),
        (TotalCandidateObservationContract::FullCall, None) if vm_right.is_some() => {
            let right = vm_right?;
            // Recheck the installed VM generation without retaining a lease
            // across planning; publication consumes the selected right.
            session_revision_only(right.planning_revision(runtime)?)
        }
        (TotalCandidateObservationContract::FullCall, Some(authority)) => {
            let revision = runtime
                .service_ports()
                .placement()?
                .whole_work_runtime_binding_session_revision(
                    runtime.program_revision().cloned(),
                    authority.binding(),
                    authority.generation(),
                )?;
            session_revision_only(revision)
        }
        _ => None,
    }
}

fn session_revision_only(revision: PlacementRevision) -> Option<PlacementRevision> {
    matches!(
        revision.authority,
        runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
    )
    .then_some(revision)
}

/// The selected provider capability snapshot is an invalidation namespace,
/// not a replacement for the VM/prepared binding execution authority.
fn session_provider_provenance(revision: &PlacementRevision, provider: Digest) -> Option<Digest> {
    matches!(
        revision.authority,
        runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
    )
    .then_some(provider)
}

fn observation_semantic_identity(
    contract: TotalCandidateObservationContract,
    existing: Option<Digest>,
    provider: Digest,
) -> Option<Digest> {
    match contract {
        TotalCandidateObservationContract::PreparationBodyCleanup => existing,
        TotalCandidateObservationContract::FullCall
        | TotalCandidateObservationContract::NativeFullCall => Some(
            Digest::sha256_serialized(&(provider, existing, b"prepared-provider-authority-v3"))
                .expect("typed provider authority serializes"),
        ),
    }
}

#[derive(Clone, Debug)]
pub(crate) enum LocalPlacementOutcome {
    Selected {
        selected: Box<runmat_execution::SelectedExecutionCandidate>,
        cpu_cost: runmat_execution::CandidateCostEvidence,
        provider_cost: Option<ExecutionCostEstimate>,
        provider_rejection: Option<ProviderRejectionCode>,
        decision_kind: runmat_execution::WholeWorkDecisionKind,
        predicted_total_ns: Option<u64>,
        feedback: Option<Box<LocalPlacementFeedback>>,
        /// Exact compact query already validated by this placement authority.
        provider_query: Box<ProviderFeasibilityQuery>,
        provider_observation: bool,
        /// Original physical-record ordinal, if the provider won. The exact
        /// selected query above belongs to this same record.
        provider_index: Option<usize>,
    },
    ProviderRejected {
        code: ProviderRejectionCode,
        cpu_cost: runmat_execution::CandidateCostEvidence,
        feedback: Option<Box<LocalPlacementFeedback>>,
    },
}

#[derive(Clone)]
pub(crate) struct LocalPlacementFeedback {
    pub(crate) runtime: runmat_runtime::context::RuntimeContext,
    pub(crate) signature: PlacementSignature,
    pub(crate) candidate: String,
    pub(crate) kind: ExecutionCandidateKind,
    pub(crate) authority: LocalPlacementFeedbackAuthority,
}

#[derive(Clone)]
pub(crate) enum LocalPlacementFeedbackAuthority {
    Graph(std::sync::Arc<dyn runmat_runtime::context::RuntimeGraphObservationPermit>),
    /// Shared one-shot selection: the probe is Send, not Sync. The mutex
    /// transports that same lease into deferred completion without retaining
    /// a RuntimeContext/service or cloning its authority.
    WholeWork(
        std::sync::Arc<
            std::sync::Mutex<Option<runmat_runtime::context::RuntimeWholeWorkSelection>>,
        >,
    ),
    /// One finite managed decision and its retained in-flight observation
    /// lease. Deferred completion consumes the same selection exactly once.
    Managed(
        std::sync::Arc<
            std::sync::Mutex<Option<runmat_runtime::context::RuntimeManagedWholeWorkSelection>>,
        >,
    ),
}

impl LocalPlacementFeedbackAuthority {
    pub(super) fn authorizes(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
    ) -> bool {
        match self {
            Self::Graph(permit) => permit.authorizes(signature, candidate, kind),
            Self::WholeWork(slot) => {
                // No permit callback, service call, or probe destruction may
                // run under this transport lock. Reentry sees the selection
                // temporarily unavailable and therefore fails closed.
                let selection = match slot.lock() {
                    Ok(mut slot) => slot.take(),
                    Err(_) => None,
                };
                let Some(selection) = selection else {
                    return false;
                };
                let current = selection.observation_permit.is_some()
                    && selection.observation_is_current()
                    && selection.decision.signature == *signature
                    && selection.decision.selected.candidate == candidate
                    && selection.decision.selected.kind == kind;
                let restored = match slot.lock() {
                    Ok(mut slot) => {
                        *slot = Some(selection);
                        true
                    }
                    Err(_) => false,
                };
                current && restored
            }
            Self::Managed(slot) => {
                let selection = match slot.lock() {
                    Ok(mut slot) => slot.take(),
                    Err(_) => None,
                };
                let Some(selection) = selection else {
                    return false;
                };
                let current = selection.observation_is_current()
                    && signature.permit_identity().ok() == Some(selection.identity)
                    && runmat_execution::Digest::parse_canonical(candidate)
                        == Some(selection.selected.candidate)
                    && selection.selected.kind == kind;
                let restored = match slot.lock() {
                    Ok(mut slot) => {
                        *slot = Some(selection);
                        true
                    }
                    Err(_) => false,
                };
                current && restored
            }
        }
    }

    pub(super) fn delivery(
        &self,
        signature: &PlacementSignature,
        candidate: &str,
        kind: ExecutionCandidateKind,
        elapsed_ns: u64,
        succeeded: bool,
    ) -> Option<runmat_runtime::context::RuntimePlacementFeedbackDelivery> {
        use runmat_runtime::context::RuntimePlacementFeedbackDelivery;
        if !self.authorizes(signature, candidate, kind) {
            return None;
        }
        let outcome = if succeeded {
            runmat_execution::WholeWorkObservationOutcome::Success
        } else {
            runmat_execution::WholeWorkObservationOutcome::OperationalFailure
        };
        match self {
            Self::Graph(permit) => Some(RuntimePlacementFeedbackDelivery::Graph {
                feedback: runmat_execution::PlacementFeedback {
                    signature: signature.clone(),
                    candidate: candidate.to_owned(),
                    kind,
                    total_elapsed_ns: elapsed_ns,
                    succeeded,
                },
                permit: std::sync::Arc::clone(permit),
            }),
            Self::WholeWork(selection) => {
                let selected = selection.lock().ok()?.take()?;
                Some(RuntimePlacementFeedbackDelivery::WholeWork {
                    selected,
                    elapsed_ns,
                    outcome,
                })
            }
            Self::Managed(slot) => {
                let selected = slot.lock().ok()?.take()?;
                Some(RuntimePlacementFeedbackDelivery::ManagedWholeWork {
                    selected,
                    elapsed_ns,
                    outcome,
                })
            }
        }
    }
}

impl std::fmt::Debug for LocalPlacementFeedback {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalPlacementFeedback")
            .field("signature", &self.signature)
            .field("candidate", &self.candidate)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl LocalPlacementFeedback {
    pub(crate) fn is_current(&self) -> bool {
        self.authority
            .authorizes(&self.signature, &self.candidate, self.kind)
    }

    pub(crate) fn observe(self, total_elapsed_ns: u64, succeeded: bool) {
        if !self.is_current() {
            return;
        }
        let Some(service) = self.runtime.service_ports().placement() else {
            return;
        };
        let feedback = runmat_execution::PlacementFeedback {
            signature: self.signature,
            candidate: self.candidate,
            kind: self.kind,
            total_elapsed_ns,
            succeeded,
        };
        match &self.authority {
            LocalPlacementFeedbackAuthority::Graph(permit) => {
                let _ = service.observe_selected_graph(permit.as_ref(), feedback);
            }
            LocalPlacementFeedbackAuthority::WholeWork(_) => {
                if let Some(delivery) = self.authority.delivery(
                    &feedback.signature,
                    &feedback.candidate,
                    feedback.kind,
                    feedback.total_elapsed_ns,
                    feedback.succeeded,
                ) {
                    let _ = service.observe_deferred_feedback(delivery);
                }
            }
            LocalPlacementFeedbackAuthority::Managed(slot) => {
                let selected = slot.lock().ok().and_then(|mut slot| slot.take());
                if let Some(selected) = selected {
                    let outcome = if succeeded {
                        runmat_execution::WholeWorkObservationOutcome::Success
                    } else {
                        runmat_execution::WholeWorkObservationOutcome::OperationalFailure
                    };
                    let _ =
                        service.observe_managed_whole_work(&selected, total_elapsed_ns, outcome);
                }
            }
        }
    }
}

/// Exercise the production owner with ordinary host admission in fixtures.
#[cfg(test)]
pub(crate) fn plan_local(
    provider: &dyn AccelProvider,
    request: LocalPlacementRequest<'_>,
    policy: PlacementPolicy,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    plan_local_vm_cpu_with_provider(
        provider,
        request,
        policy,
        None,
        None,
        multi::LocalCpuAdmission::Legacy,
        LocalProviderAdmission::Available,
    )
}

pub(crate) fn plan_local_vm_cpu_with_provider(
    provider: &dyn AccelProvider,
    request: LocalPlacementRequest<'_>,
    policy: PlacementPolicy,
    vm_right: Option<&runmat_runtime::context::VmFusionCompletionRight>,
    vm_decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
    cpu_admission: multi::LocalCpuAdmission,
    implementation: LocalProviderAdmission,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let LocalPlacementRequest {
        compound,
        operation,
        semantic_identity,
        observation_contract,
        prepared_authority,
        source_decline,
        family,
        provider_kind,
        inputs,
        outputs,
        workload,
        preparation,
        preparation_program,
        shader_requirements,
        cpu,
        provider_fallback,
        required_upload_bytes,
        required_download_bytes,
        downstream_materialization,
        compare_profitability,
        runtime,
    } = request;
    let publication =
        if cpu.route.kind == runmat_execution::ExecutionCandidateKind::OriginalNativeRegion {
            multi::LocalPublicationSemantics::SourceOrderedNative
        } else {
            multi::LocalPublicationSemantics::Atomic
        };
    let outcome = multi::plan_local_multi_with_provider(
        provider,
        multi::LocalMultiPlacementRequest {
            operation,
            semantic_identity,
            observation_contract,
            prepared_authority,
            vm_right,
            vm_decline,
            source_decline,
            family,
            cpu,
            cpu_admission,
            publication,
            physical: vec![multi::LocalPhysicalCandidate {
                authority: compound.map(multi::PreparedPhysicalAuthority::Compound),
                physical_identity: compound.map(|plan| plan.identity()),
                provider_kind,
                data_inputs: inputs,
                source_inputs: compound.map(|plan| plan.source_inputs()),
                prepared_inputs: compound.map(|plan| plan.source_inputs()),
                input_adaptation: None,
                outputs: compound.map_or(outputs, |plan| vec![plan.query_output()]),
                workload,
                preparation,
                preparation_program,
                shader_requirements,
                fallback: provider_fallback,
                required_upload_bytes,
                required_download_bytes,
                downstream_materialization,
            }],
            compare_profitability,
            runtime,
        },
        policy,
        implementation,
    )?;
    if let LocalPlacementOutcome::Selected { provider_index, .. } = &outcome {
        debug_assert!(provider_index.is_none_or(|index| index == 0));
    }
    Ok(outcome)
}
pub(crate) fn transfer_prior_ns(bytes: u64) -> u64 {
    if bytes == 0 {
        0
    } else {
        20_000_u64.saturating_add(bytes.saturating_add(7) / 8)
    }
}
