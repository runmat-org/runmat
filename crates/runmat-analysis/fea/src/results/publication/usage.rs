//! Source-preserving terminal accounting, shared by workers, bridge and durable runtime.
use super::PublishedAnalysisResult;
use crate::execution::{AnalysisExecutionError, AnalysisExecutionResult};
use crate::planning::resource::{
    record_resource_graph_usage, FeaResourceDemand, FeaResourcePhase, ResourcePlanningError,
    ResourceUsageEvidence,
};

fn resource_error(error: ResourcePlanningError) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}

/// Requires exactly the measured publication attempt in addition to every baseline attempt.
/// The immutable receipt names the baseline; terminal accounting cannot be embedded in that
/// receipt without introducing a content-addressing cycle.
pub fn validate_result_publication_resource_usage(
    publication: &PublishedAnalysisResult,
    baseline: &ResourceUsageEvidence,
    terminal: &ResourceUsageEvidence,
) -> AnalysisExecutionResult<()> {
    publication.validate()?;
    let primary = baseline.primary_plan().map_err(resource_error)?;
    terminal.validate_identity().map_err(resource_error)?;
    if publication.source_resource_usage != baseline.identity()
        || publication.source_resource_plan != baseline.primary_plan
        || publication.source_result.digest() != baseline.result_digest
        || terminal.primary_plan != baseline.primary_plan
        || terminal.workload_digest != baseline.workload_digest
        || terminal.result_digest != baseline.result_digest
        || !baseline.overruns.is_empty()
        || !terminal.overruns.is_empty()
        || terminal.attempts.len() != baseline.attempts.len() + 1
    {
        return Err(AnalysisExecutionError::Identity(
            "terminal publication usage differs from its baseline",
        ));
    }
    // Identity validation guarantees unique, sorted attempt IDs. Merge the inventories
    // in linear time rather than repeatedly scanning a potentially large task graph.
    let mut originals = baseline.attempts.iter().peekable();
    let mut added = None;
    for attempt in &terminal.attempts {
        if originals
            .peek()
            .is_some_and(|original| original.attempt_id == attempt.attempt_id)
        {
            if originals.next() != Some(attempt) {
                return Err(AnalysisExecutionError::Identity(
                    "terminal publication usage rewrites a baseline attempt",
                ));
            }
        } else if added.replace(attempt).is_some() {
            return Err(AnalysisExecutionError::Identity(
                "terminal publication usage adds more than one attempt",
            ));
        }
    }
    if originals.next().is_some() {
        return Err(AnalysisExecutionError::Identity(
            "terminal publication usage omits a baseline attempt",
        ));
    }
    let added = added.ok_or(AnalysisExecutionError::Identity(
        "terminal usage omits publication attempt",
    ))?;
    let plan = terminal
        .plans
        .iter()
        .find(|plan| plan.identity == added.plan)
        .ok_or(AnalysisExecutionError::Identity(
            "terminal usage omits publication admission",
        ))?;
    let FeaResourceDemand::ResultPublication(demand) = &plan.input.demand else {
        return Err(AnalysisExecutionError::Identity(
            "terminal attempt is not result publication",
        ));
    };
    if plan.input.phase != FeaResourcePhase::ResultPublication(publication.request)
        || demand.request != publication.request
        || demand.source_result != publication.source_result
        || demand.source_accuracy != publication.source_accuracy
        || demand.source_usage != baseline.identity()
        || baseline
            .plans
            .iter()
            .any(|existing| existing.identity == plan.identity)
    {
        return Err(AnalysisExecutionError::Identity(
            "terminal publication admission names different sources",
        ));
    }
    let mut plans = baseline
        .plans
        .iter()
        .filter(|plan| plan.identity != primary.identity)
        .cloned()
        .collect::<Vec<_>>();
    plans.push(plan.clone());
    let mut attempts = baseline.attempts.clone();
    attempts.push(added.clone());
    let wall = baseline
        .actual
        .scheduler
        .wall_millis
        .checked_add(added.resources.usage.wall_millis)
        .ok_or_else(|| {
            AnalysisExecutionError::Invalid("terminal publication wall time overflow".into())
        })?;
    let expected = record_resource_graph_usage(
        primary,
        plans,
        baseline.workload_digest,
        baseline.result_digest,
        attempts,
        wall,
    )
    .map_err(resource_error)?;
    if &expected != terminal {
        return Err(AnalysisExecutionError::Identity(
            "terminal usage is not the exact publication extension",
        ));
    }
    Ok(())
}
