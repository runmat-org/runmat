use runmat_analysis_fea::results::accuracy::{
    AccuracyArtifact, AccuracyStatus, PublicationDecision,
};

use crate::analysis::{QualityGate, QualityReason, QualityReasonCode, RunStatus};

pub(super) fn validate_publication_chain(
    workload_digest: runmat_execution::Digest,
    expected_primary_phase: &runmat_analysis_fea::planning::resource::FeaResourcePhase,
    accuracy: &AccuracyArtifact,
    usage: &runmat_analysis_fea::planning::resource::ResourceUsageEvidence,
    publication: &runmat_analysis_fea::results::publication::PublishedAnalysisResult,
) -> Result<(), String> {
    usage
        .validate_identity()
        .map_err(|error| error.to_string())?;
    let primary_plan = usage
        .plans
        .iter()
        .find(|plan| plan.identity == usage.primary_plan)
        .ok_or_else(|| "resource evidence omits its primary plan".to_string())?;
    publication
        .validate_sources(accuracy, primary_plan, usage)
        .map_err(|error| error.to_string())?;
    if usage.workload_digest != workload_digest {
        return Err("resource evidence names another analysis workload".into());
    }
    if &primary_plan.input.phase != expected_primary_phase {
        return Err("resource evidence names another terminal analysis phase".into());
    }
    Ok(())
}

pub(super) struct CompletedAccuracy {
    pub run_status: RunStatus,
    pub numerically_acceptable: bool,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub quality_reasons: Vec<QualityReason>,
}

pub(super) fn classify_accuracy(artifact: &AccuracyArtifact) -> CompletedAccuracy {
    let status = artifact.evaluation().report().status;
    let publication_allowed = artifact.gate().decision == PublicationDecision::Allow;
    let numerically_acceptable = status == AccuracyStatus::Satisfied;
    let result_quality = match (status, publication_allowed) {
        (AccuracyStatus::Satisfied, true) => QualityGate::Pass,
        (AccuracyStatus::Indeterminate | AccuracyStatus::NotSatisfied, true) => QualityGate::Warn,
        _ => QualityGate::Fail,
    };
    let quality_reasons = match status {
        AccuracyStatus::Satisfied => Vec::new(),
        AccuracyStatus::Indeterminate => vec![QualityReason {
            code: QualityReasonCode::AccuracyEvidenceIndeterminate,
            detail: criterion_detail(
                "mandatory accuracy evidence is indeterminate",
                artifact.gate().governing_criterion_id.as_deref(),
            ),
        }],
        AccuracyStatus::NotSatisfied => vec![QualityReason {
            code: QualityReasonCode::AccuracyLimitExceeded,
            detail: criterion_detail(
                "a mandatory accuracy limit was exceeded",
                artifact.gate().governing_criterion_id.as_deref(),
            ),
        }],
    };
    CompletedAccuracy {
        run_status: super::super::super::result_policy::classify_run_status(
            result_quality,
            numerically_acceptable,
        ),
        numerically_acceptable,
        solver_convergence: QualityGate::Pass,
        result_quality,
        quality_reasons,
    }
}

fn criterion_detail(summary: &str, criterion: Option<&str>) -> String {
    match criterion {
        Some(criterion) => format!("{summary}: {criterion}"),
        None => summary.to_string(),
    }
}
