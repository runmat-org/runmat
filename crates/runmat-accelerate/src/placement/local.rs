use runmat_accelerate_api::{
    AccelProvider, ProviderFeasibilityQuery, ProviderOperationFamily, ProviderRejectionCode,
    ProviderRepresentation, ProviderWorkload,
};
use runmat_execution::{
    CandidatePreparationState, Digest, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    ExecutionCostEstimate, PlacementRevision, PlacementSignature,
};
use runmat_value::Value;

use super::PlacementPolicy;
mod multi;
pub(crate) use multi::{plan_local_multi, LocalMultiPlacementRequest, LocalPhysicalCandidate};
mod rejected;
mod session_facts;
mod signature;
use rejected::{rejected_cpu_feedback, RejectedCpuFeedbackRequest};
use session_facts::session_facts;
#[cfg(test)]
mod resource_tests;

pub(crate) struct LocalPlacementRequest<'a> {
    pub(crate) operation: &'a str,
    /// Optional deterministic identity of the selected typed program. This
    /// augments, rather than replaces, argument/resource/preparation facts.
    pub(crate) semantic_identity: Option<Digest>,
    pub(crate) observation_contract: TotalCandidateObservationContract,
    pub(crate) prepared_authority:
        Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
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
    pub(crate) cpu: ExecutionCandidateDescriptor,
    /// Provider execution prior without transfer components. It is used only
    /// when the provider has no trustworthy estimate of its own.
    pub(crate) provider_fallback: ExecutionCostEstimate,
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
}

fn observation_revision(
    runtime: &runmat_runtime::context::RuntimeContext,
    contract: TotalCandidateObservationContract,
    authority: Option<runmat_runtime::context::preparation::PreparedBindingAuthority>,
    vm_right: Option<&runmat_runtime::context::VmFusionCompletionRight>,
    provider_digest: Digest,
    policy: PlacementPolicy,
) -> Option<PlacementRevision> {
    match (contract, authority) {
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
        TotalCandidateObservationContract::FullCall => Some(
            Digest::sha256_serialized(&(provider, existing, b"prepared-provider-authority-v3"))
                .expect("typed provider authority serializes"),
        ),
    }
}

#[derive(Clone, Debug)]
pub(crate) enum LocalPlacementOutcome {
    Selected {
        selected: Box<ExecutionCandidateDescriptor>,
        cpu_cost: ExecutionCostEstimate,
        provider_cost: ExecutionCostEstimate,
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
        cpu_cost: ExecutionCostEstimate,
        feedback: Option<Box<LocalPlacementFeedback>>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct LocalPlacementFeedback {
    pub(crate) runtime: runmat_runtime::context::RuntimeContext,
    pub(crate) signature: PlacementSignature,
    pub(crate) candidate: String,
    pub(crate) kind: ExecutionCandidateKind,
    pub(crate) permit:
        Option<std::sync::Arc<dyn runmat_runtime::context::RuntimeGraphObservationPermit>>,
}

impl LocalPlacementFeedback {
    pub(crate) fn is_current(&self) -> bool {
        self.permit
            .as_ref()
            .is_some_and(|permit| permit.authorizes(&self.signature, &self.candidate, self.kind))
    }

    pub(crate) fn observe(self, total_elapsed_ns: u64, succeeded: bool) {
        let Some(permit) = self.permit else {
            return;
        };
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
        let _ = service.observe_selected_graph(permit.as_ref(), feedback);
    }
}

/// Normalize provider feasibility and complete transfer costs before comparing
/// profitability. This is the one local authority shared by legacy automatic
/// offload and correlated fusion execution.
pub(crate) fn plan_local(
    provider: &dyn AccelProvider,
    request: LocalPlacementRequest<'_>,
    policy: PlacementPolicy,
) -> Result<LocalPlacementOutcome, Box<runmat_runtime::RuntimeError>> {
    let LocalPlacementRequest {
        operation,
        semantic_identity,
        observation_contract,
        prepared_authority,
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
    let outcome = multi::plan_local_multi(
        provider,
        multi::LocalMultiPlacementRequest {
            operation,
            semantic_identity,
            observation_contract,
            prepared_authority,
            vm_right: None,
            family,
            cpu,
            physical: vec![multi::LocalPhysicalCandidate {
                physical_identity: None,
                provider_kind,
                data_inputs: inputs,
                source_inputs: None,
                prepared_inputs: None,
                input_adaptation: None,
                outputs,
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
