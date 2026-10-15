//! One host-route carrier; numerical graph descriptors require real estimates.
use runmat_execution::{
    CandidateCostEvidence, CandidateExecutionLocation, CandidateOutputPresentation,
    CandidateOutputResidency, CandidatePreparationState, CandidateResourceDemand, Digest,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, ResourceBoundUnavailable,
    ResourceByteBound, WholeWorkCandidate,
};
use runmat_types::{RegionGuardId, RegionId};

pub(crate) struct LocalCpuCandidate {
    pub(super) identity: String,
    pub(super) region: Option<RegionId>,
    pub(super) guards: Vec<RegionGuardId>,
    pub(super) route: WholeWorkCandidate,
}

impl LocalCpuCandidate {
    pub(crate) fn with_presentation(mut self, presentation: CandidateOutputPresentation) -> Self {
        self.route.presentation = presentation;
        self
    }
    pub(crate) fn host(identity: String, cost: CandidateCostEvidence) -> Self {
        let guards = Vec::<RegionGuardId>::new();
        let route_facts =
            Digest::sha256_serialized(&(&identity, &guards)).expect("host route facts serialize");
        Self {
            identity,
            region: None,
            guards,
            route: WholeWorkCandidate {
                kind: ExecutionCandidateKind::SharedRuntime,
                execution_location: CandidateExecutionLocation::Host,
                preparation: CandidatePreparationState::Ready,
                output_residency: CandidateOutputResidency::Host,
                presentation: CandidateOutputPresentation::Direct,
                route_facts,
                cost,
                resources: legacy_resources(cost),
                transactional_results: true,
            },
        }
    }

    pub(super) fn into_priced(self) -> Option<ExecutionCandidateDescriptor> {
        Some(ExecutionCandidateDescriptor {
            identity: self.identity,
            region: self.region,
            guards: self.guards,
            kind: self.route.kind,
            execution_location: self.route.execution_location,
            preparation: self.route.preparation,
            output_residency: self.route.output_residency,
            presentation: self.route.presentation,
            cost: self.route.cost.estimate()?,
        })
    }
}

impl From<ExecutionCandidateDescriptor> for LocalCpuCandidate {
    fn from(cpu: ExecutionCandidateDescriptor) -> Self {
        let cost = CandidateCostEvidence::Estimated(cpu.cost);
        Self {
            route: WholeWorkCandidate {
                kind: cpu.kind,
                execution_location: cpu.execution_location,
                preparation: cpu.preparation,
                output_residency: cpu.output_residency,
                presentation: cpu.presentation,
                route_facts: Digest::sha256_serialized(&(&cpu.identity, &cpu.guards))
                    .expect("host route facts serialize"),
                cost,
                resources: legacy_resources(cost),
                transactional_results: true,
            },
            identity: cpu.identity,
            region: cpu.region,
            guards: cpu.guards,
        }
    }
}

fn legacy_resources(cost: CandidateCostEvidence) -> CandidateResourceDemand {
    CandidateResourceDemand {
        cpu_millicores: 1_000,
        retained_bytes: ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoRetainedValueBound,
        ),
        scratch_bytes: cost.estimate().and_then(|cost| cost.scratch_bytes).map_or(
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound),
            Into::into,
        ),
        queue_slots: 0,
        provider_host_workspace: None,
        provider_input_stages: None,
        peak_live: None,
    }
}
