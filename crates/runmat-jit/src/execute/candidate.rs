use runmat_execution::{
    CandidateOutputResidency, CandidatePreparationState, EstimateConfidence, EstimateSource,
    ExecutionCandidateDescriptor, ExecutionCandidateKind, ExecutionCostComponents,
    ExecutionCostEstimate,
};
use runmat_types::{RegionContract, RegionId};

use super::GenericExecutor;

impl GenericExecutor {
    pub fn cpu_candidate(
        &self,
        region: RegionId,
        expected_invocations: u64,
        observed_execution_ns: Option<u64>,
    ) -> Option<ExecutionCandidateDescriptor> {
        let contract = self
            .regions
            .iter()
            .find(|candidate| candidate.id == region)?;
        let region_count = u64::try_from(self.regions.len()).ok()?.max(1);
        let compile_share = self.compile_duration_ns / region_count;
        let amortized_compile = compile_share / expected_invocations.max(1);
        let live_values = contract
            .live_in
            .len()
            .saturating_add(contract.live_out.len());
        let static_execution = 10_000_u64.saturating_add(
            u64::try_from(live_values)
                .unwrap_or(u64::MAX)
                .saturating_mul(1_000),
        );
        let (execution_ns, confidence, source) = match observed_execution_ns {
            Some(observed) => (
                observed,
                EstimateConfidence::Medium,
                EstimateSource::Observation,
            ),
            None => (
                static_execution,
                EstimateConfidence::Prior,
                EstimateSource::Compiler,
            ),
        };
        let mut guards = contract
            .guards
            .iter()
            .map(|guard| guard.id)
            .collect::<Vec<_>>();
        guards.sort_unstable();
        Some(ExecutionCandidateDescriptor {
            identity: format!("cpu.generic.f{}.r{}", region.function.0, region.ordinal),
            region: Some(region),
            kind: ExecutionCandidateKind::GenericNativeCpu,
            preparation: CandidatePreparationState::Warm,
            cost: ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    compile_or_prepare_ns: amortized_compile,
                    execution_ns,
                    ..ExecutionCostComponents::default()
                },
                scratch_bytes: 0,
                confidence,
                source,
            },
            output_residency: CandidateOutputResidency::Host,
            guards,
        })
    }

    pub fn region_contracts(&self) -> &[RegionContract] {
        &self.regions
    }

    pub fn compile_duration_ns(&self) -> u64 {
        self.compile_duration_ns
    }
}
