//! Source-preserving terminal accounting, shared by workers, bridge and durable runtime.
use super::PublishedAnalysisResult;
use crate::execution::{
    AccuracyEvaluationRequestIdentity, AnalysisExecutionError, AnalysisExecutionResult,
    SolverResultIdentity,
};
use crate::planning::resource::{
    record_resource_graph_usage, FeaResourceDemand, FeaResourcePhase, ResourcePlanningError,
    ResourceUsageEvidence,
};

fn resource_error(error: ResourcePlanningError) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}

/// The accuracy artifact/request, not the supplied usage inventory, identifies required work.
/// Multiple attempts are allowed: retry accounting must not be collapsed to one attempt.
/// This proves an admitted scheduler attempt, not success attestation or completeness of all
/// earlier solve/observation/state work; those require their own independently bound lineage.
pub(crate) fn validate_accuracy_resource_usage(
    request: AccuracyEvaluationRequestIdentity,
    source_result: SolverResultIdentity,
    usage: &ResourceUsageEvidence,
) -> AnalysisExecutionResult<()> {
    usage.validate_identity().map_err(resource_error)?;
    if usage.result_digest != source_result.digest() {
        return Err(AnalysisExecutionError::Identity(
            "accuracy accounting names a different solver result",
        ));
    }
    let mut accounted = false;
    for plan in &usage.plans {
        if plan.input.phase != FeaResourcePhase::AccuracyEvaluation(request) {
            continue;
        }
        let FeaResourceDemand::AccuracyEvaluation(demand) = &plan.input.demand else {
            return Err(AnalysisExecutionError::Identity(
                "accuracy accounting demand differs",
            ));
        };
        if demand.request != request || demand.source_result != source_result {
            return Err(AnalysisExecutionError::Identity(
                "accuracy accounting sources differ",
            ));
        }
        // validate_identity requires the plan catalog to exactly cover measured attempts.
        // Thus this plan already has an attempt; do not rescan the entire attempt graph.
        accounted = true;
    }
    if accounted {
        return Ok(());
    }
    Err(AnalysisExecutionError::Identity(
        "publication omits its source-bound accuracy scheduler attempt",
    ))
}

/// Validates a cold terminal extension using its retained scheduler interval.
/// This checks internal consistency, not an independently observed elapsed time.
pub fn validate_retained_result_publication_resource_usage(
    publication: &PublishedAnalysisResult,
    baseline: &ResourceUsageEvidence,
    terminal: &ResourceUsageEvidence,
) -> AnalysisExecutionResult<()> {
    let interval = terminal
        .actual
        .scheduler
        .wall_millis
        .checked_sub(baseline.actual.scheduler.wall_millis)
        .filter(|interval| *interval > 0)
        .ok_or(AnalysisExecutionError::Identity(
            "publication omits its retained execution interval",
        ))?;
    validate_result_publication_resource_usage(publication, baseline, terminal, interval)
}

/// Requires every original publication attempt in addition to every baseline attempt.
/// The immutable receipt names the baseline; terminal accounting cannot be embedded in that
/// receipt without introducing a content-addressing cycle.
pub fn validate_result_publication_resource_usage(
    publication: &PublishedAnalysisResult,
    baseline: &ResourceUsageEvidence,
    terminal: &ResourceUsageEvidence,
    publication_wall_millis: u64,
) -> AnalysisExecutionResult<()> {
    publication.validate()?;
    validate_accuracy_resource_usage(
        publication.accuracy_request,
        publication.source_result,
        baseline,
    )?;
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
        || terminal.attempts.len() <= baseline.attempts.len()
        || publication_wall_millis == 0
    {
        return Err(AnalysisExecutionError::Identity(
            "terminal publication usage differs from its baseline",
        ));
    }
    // Identity validation guarantees unique, sorted attempt IDs. Merge the inventories
    // in linear time rather than repeatedly scanning a potentially large task graph.
    let mut originals = baseline.attempts.iter().peekable();
    let mut added = Vec::new();
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
        } else {
            added.push(attempt);
        }
    }
    if originals.next().is_some() {
        return Err(AnalysisExecutionError::Identity(
            "terminal publication usage omits a baseline attempt",
        ));
    }
    let first = added.first().ok_or(AnalysisExecutionError::Identity(
        "terminal usage omits publication attempts",
    ))?;
    if added.iter().any(|attempt| attempt.plan != first.plan) {
        return Err(AnalysisExecutionError::Identity(
            "publication retries name distinct plans",
        ));
    }
    let plan = terminal
        .plans
        .iter()
        .find(|plan| plan.identity == first.plan)
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
    attempts.extend(added.into_iter().cloned());
    let wall = baseline
        .actual
        .scheduler
        .wall_millis
        .checked_add(publication_wall_millis)
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
