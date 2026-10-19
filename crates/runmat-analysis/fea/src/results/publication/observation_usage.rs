//! Required terminal observation tasks come from publication artifacts, not supplied usage.
use crate::execution::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationAssemblyRequest,
    ObservationHistoryCheckpointRequest,
};
use crate::planning::resource::{
    FeaResourceDemand, FeaResourcePhase, FeaResourcePlan, ResourceUsageEvidence,
};
use crate::results::observation::{ObservationPublicationCatalog, ObservationPublicationEntry};
use runmat_execution::Digest;

mod partition;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Task {
    Partition(Digest),
    Reduction(Digest),
    Checkpoint(Digest),
    Finalization(Digest),
    Assembly(Digest),
}

fn task(plan: &FeaResourcePlan) -> Option<Task> {
    match plan.input.phase {
        FeaResourcePhase::StructuralObservationPartition(request) => {
            Some(Task::Partition(request.digest()))
        }
        FeaResourcePhase::StructuralObservationReduction(request) => {
            Some(Task::Reduction(request.digest()))
        }
        FeaResourcePhase::StructuralObservationHistoryCheckpoint(request) => {
            Some(Task::Checkpoint(request.digest()))
        }
        FeaResourcePhase::ObservationHistoryFinalization(request) => {
            Some(Task::Finalization(request.digest()))
        }
        FeaResourcePhase::StructuralObservationAssembly(request) => {
            Some(Task::Assembly(request.digest()))
        }
        _ => None,
    }
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("observation publication accounting: {error}"))
}

/// Exact temporary pointer-index storage; sorting is in-place and histories are streamed.
pub(crate) fn observation_usage_index_bytes(plan_count: usize) -> Option<u64> {
    u64::try_from(plan_count)
        .ok()?
        .checked_mul(std::mem::size_of::<&FeaResourcePlan>() as u64)
}

/// Requires canonical partition, reduction, checkpoint, finalization and assembly accounting.
/// Catalog source validation authenticates the producing policy and selected field blocks.
/// This does not independently attest numerical execution or completeness of failed retries.
pub fn validate_observation_publication_usage(
    observations: &ObservationPublicationCatalog,
    usage: &ResourceUsageEvidence,
) -> AnalysisExecutionResult<()> {
    observations.validate()?;
    usage.validate_identity().map_err(invalid)?;
    let mut plans = Vec::new();
    plans
        .try_reserve_exact(usage.plans.len())
        .map_err(invalid)?;
    plans.extend(usage.plans.iter().filter(|plan| task(plan).is_some()));
    plans.sort_unstable_by_key(|plan| task(plan));
    let require = |expected: Task,
                   accepts: &dyn Fn(&FeaResourceDemand) -> bool|
     -> AnalysisExecutionResult<()> {
        let start = plans.partition_point(|plan| task(plan) < Some(expected));
        let end = plans.partition_point(|plan| task(plan) <= Some(expected));
        if start == end {
            return Err(invalid(format!("missing admitted task {expected:?}")));
        }
        // Usage validation proves exact plan/attempt coverage. Preserve retries and reject
        // any conflicting demand for this task even beside one correct plan.
        if plans[start..end]
            .iter()
            .any(|plan| !accepts(&plan.input.demand))
        {
            return Err(invalid("task demand differs from publication sources"));
        }
        Ok(())
    };
    partition::validate_partition_tasks(observations, &require)?;
    for entry in observations.entries() {
        match entry {
            ObservationPublicationEntry::Sample(sample) => {
                let request = ObservationAssemblyRequest::published_identity(
                    &sample.request,
                    &sample.manifest,
                )?;
                require(Task::Assembly(request.digest()), &|demand| {
                    matches!(demand, FeaResourceDemand::StructuralObservationAssembly(value)
                        if value.topology_digest == sample.manifest.topology_digest
                            && value.result_count.0 == sample.manifest.results.len() as u64
                            && value.evidence_count.0 == sample.manifest.results.len() as u64)
                })?;
            }
            ObservationPublicationEntry::History(history) => {
                require(
                    Task::Finalization(history.request.identity().digest()),
                    &|demand| {
                        matches!(demand, FeaResourceDemand::ObservationHistoryFinalization(value)
                        if value.request == history.request.identity()
                            && value.topology_digest == history.history.topology_digest
                            && value.retained_sample_count.0 == history.history.samples.len() as u64)
                    },
                )?;
                let transitions = history
                    .checkpoint
                    .canonical_prefix_identities()
                    .map_err(invalid)?;
                for (transition, sample) in transitions.zip(&history.history.samples) {
                    let (predecessor, ordinal, _) = transition.map_err(invalid)?;
                    let request = ObservationHistoryCheckpointRequest::published_sample_identity(
                        history.request.plan.identity,
                        history.request.producer,
                        &history.request.observation_id,
                        predecessor,
                        ordinal,
                        sample,
                        history.checkpoint.partition_evidence[ordinal as usize].identity,
                    )?;
                    require(Task::Checkpoint(request.digest()), &|demand| {
                        matches!(demand, FeaResourceDemand::StructuralObservationHistoryCheckpoint(value)
                            if value.topology_digest == history.history.topology_digest
                                && value.predecessor_present == predecessor.is_some()
                                && value.predecessor_sample_count.0 == ordinal
                                && value.retained_sample_count.0 == ordinal + 1)
                    })?;
                }
            }
        }
    }
    Ok(())
}
