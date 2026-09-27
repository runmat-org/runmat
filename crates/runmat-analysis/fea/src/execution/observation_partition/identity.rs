//! Canonical request identities shared by payload constructors and metadata publication gates.
use super::*;

pub(super) fn evaluation_identity(
    producer: StepIdentity,
    source: ObservationEvaluationRequestIdentity,
    plan: ObservationPartitionPlanIdentity,
    partition: ObservationPartitionIdentity,
    fields: RecoveredFieldPartitionSetIdentity,
    event: Option<ObservationEventAttribution>,
) -> AnalysisExecutionResult<ObservationPartitionEvaluationRequestIdentity> {
    ObservationPartitionEvaluationRequestIdentity::derive(
        "observation-partition-evaluation-request",
        &(producer, source, plan, partition, fields, event),
    )
}

pub(super) fn reduction_identity(
    producer: StepIdentity,
    source: ObservationEvaluationRequestIdentity,
    plan: ObservationPartitionPlanIdentity,
    contributions: &[ObservationPartitionContributionIdentity],
) -> AnalysisExecutionResult<ObservationPartitionReductionRequestIdentity> {
    ObservationPartitionReductionRequestIdentity::derive(
        "observation-partition-reduction-request",
        &(producer, source, plan, contributions),
    )
}

impl ObservationPartitionEvaluationRequest {
    /// Reconstruct the canonical task identity without loading numerical field blocks.
    /// The caller must authenticate the selected block-set identity and event attribution
    /// against the authoritative source catalog; an opaque identity cannot prove coverage.
    pub(crate) fn published_identity(
        source: &ObservationEvaluationRequest,
        plan: &ObservationPartitionPlan,
        partition: &ObservationPartition,
        fields: RecoveredFieldPartitionSetIdentity,
        event: Option<ObservationEventAttribution>,
    ) -> AnalysisExecutionResult<ObservationPartitionEvaluationRequestIdentity> {
        source.validate()?;
        plan.validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if plan.partitions.get(partition.ordinal as usize) != Some(partition) {
            return Err(AnalysisExecutionError::Identity(
                "published observation partition is noncanonical",
            ));
        }
        if let Some(event) = event {
            event
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        }
        evaluation_identity(
            source.producer,
            source.identity(),
            plan.identity,
            partition.identity,
            fields,
            event,
        )
    }
}

impl ObservationPartitionReductionRequest {
    /// Contributions must come in canonical partition order from validated full evidence.
    /// Identities alone establish neither numerical payload integrity nor partition ownership.
    pub(crate) fn published_identity(
        source: &ObservationEvaluationRequest,
        plan: &ObservationPartitionPlan,
        contributions: &[ObservationPartitionContributionIdentity],
    ) -> AnalysisExecutionResult<ObservationPartitionReductionRequestIdentity> {
        source.validate()?;
        plan.validate_for(&source.plan)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let mut ordered = Vec::new();
        ordered
            .try_reserve_exact(contributions.len())
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        ordered.extend_from_slice(contributions);
        ordered.sort_unstable();
        if contributions.len() != plan.partitions.len()
            || contributions.is_empty()
            || ordered.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(AnalysisExecutionError::Identity(
                "published observation reduction contribution coverage differs",
            ));
        }
        reduction_identity(
            source.producer,
            source.identity(),
            plan.identity,
            contributions,
        )
    }
}
