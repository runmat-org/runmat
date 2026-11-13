use crate::results::{
    field::RecoveredFieldPartitionSet,
    observation::{
        ObservationEventAttribution, ObservationPartitionContribution, ObservationPartitionPlan,
        SolverEventSource,
    },
};

use super::{AnalysisExecutionError, AnalysisExecutionResult, ObservationEvaluationRequest};

pub(super) fn validate_event_attribution(
    source: &ObservationEvaluationRequest,
    partition_plan: &ObservationPartitionPlan,
    source_field: &RecoveredFieldPartitionSet,
    attribution: Option<ObservationEventAttribution>,
) -> AnalysisExecutionResult<()> {
    let observation = source
        .plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == partition_plan.observation_id)
        .ok_or(AnalysisExecutionError::Identity(
            "observation partition event is absent from its source plan",
        ))?;
    match (&observation.spec.sampling, attribution) {
        (runmat_analysis_core::SamplingPolicy::Event(expected), Some(attribution)) => {
            attribution
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            if attribution.event() != *expected || attribution.sample() != source_field.sample {
                return Err(AnalysisExecutionError::Identity(
                    "observation partition event differs from its plan or source sample",
                ));
            }
            validate_completed_result(
                source,
                attribution,
                "completed solver event names a different source result",
            )
        }
        (runmat_analysis_core::SamplingPolicy::Event(_), None) => {
            Err(AnalysisExecutionError::Invalid(
                "event observation partition requires source attribution".into(),
            ))
        }
        (_, Some(_)) => Err(AnalysisExecutionError::Invalid(
            "non-event observation partition cannot carry event attribution".into(),
        )),
        (_, None) => Ok(()),
    }
}

pub(super) fn validate_contribution_attribution(
    source: &ObservationEvaluationRequest,
    partition_plan: &ObservationPartitionPlan,
    contributions: &[ObservationPartitionContribution],
) -> AnalysisExecutionResult<()> {
    let expected = contributions
        .first()
        .and_then(|contribution| contribution.event_attribution);
    if contributions
        .iter()
        .any(|contribution| contribution.event_attribution != expected)
    {
        return Err(AnalysisExecutionError::Identity(
            "observation reduction contributions name different event occurrences",
        ));
    }
    let observation = source
        .plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == partition_plan.observation_id)
        .ok_or(AnalysisExecutionError::Identity(
            "observation reduction is absent from its source plan",
        ))?;
    match (&observation.spec.sampling, expected) {
        (runmat_analysis_core::SamplingPolicy::Event(event), Some(attribution))
            if attribution.event() == *event =>
        {
            validate_completed_result(
                source,
                attribution,
                "observation reduction event names a different source result",
            )
        }
        (runmat_analysis_core::SamplingPolicy::Event(_), _) => {
            Err(AnalysisExecutionError::Identity(
                "observation reduction lacks the planned event attribution",
            ))
        }
        (_, Some(_)) => Err(AnalysisExecutionError::Identity(
            "non-event observation reduction carries event attribution",
        )),
        (_, None) => Ok(()),
    }
}

fn validate_completed_result(
    source: &ObservationEvaluationRequest,
    attribution: ObservationEventAttribution,
    mismatch_message: &'static str,
) -> AnalysisExecutionResult<()> {
    if let ObservationEventAttribution::Solver(
        crate::results::observation::SolverEventAttribution {
            source: SolverEventSource::CompletedResult { result },
            ..
        },
    ) = attribution
    {
        if source.selection.source().solver_result() != Some(result) {
            return Err(AnalysisExecutionError::Identity(mismatch_message));
        }
    }
    Ok(())
}
