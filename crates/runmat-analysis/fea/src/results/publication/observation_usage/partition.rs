//! Derive required numerical observation tasks from source-bound publication metadata.
use super::*;
use crate::execution::{
    ObservationPartitionEvaluationRequest, ObservationPartitionReductionRequest,
};

pub(super) fn validate_partition_tasks(
    observations: &ObservationPublicationCatalog,
    require: &impl Fn(Task, &dyn Fn(&FeaResourceDemand) -> bool) -> AnalysisExecutionResult<()>,
) -> AnalysisExecutionResult<()> {
    observations.visit_execution(|source, partition_plan, evidence, selections| {
        let field = source
            .selection
            .fields()
            .iter()
            .find(|entry| entry.field == evidence.source_field)
            .ok_or_else(|| invalid("execution source field is absent"))?;
        let components = u64::try_from(
            field
                .definition
                .storage_component_count()
                .map_err(invalid)?,
        )
        .map_err(invalid)?;
        if selections.len() != partition_plan.partitions.len() {
            return Err(invalid("execution source groups omit canonical partitions"));
        }
        // One source and partition-plan validation serves every partition and the reduction.
        crate::execution::validate_published_observation_source(source, partition_plan)?;
        for (partition, selected) in partition_plan.partitions.iter().zip(selections) {
            let request = ObservationPartitionEvaluationRequest::published_identity_validated(
                source,
                partition_plan,
                partition,
                selected.identity,
                evidence.event_attribution,
            )?;
            let entities = selected
                .descriptors
                .iter()
                .try_fold(0_u64, |total, descriptor| {
                    total
                        .checked_add(u64::try_from(descriptor.entity_count).map_err(invalid)?)
                        .ok_or_else(|| invalid("execution source entity count overflow"))
                })?;
            let values = entities
                .checked_mul(components)
                .ok_or_else(|| invalid("execution source value count overflow"))?;
            require(Task::Partition(request.digest()), &|demand| {
                matches!(demand, FeaResourceDemand::StructuralObservationPartition(value)
                    if value.topology_digest == source.plan.topology_digest
                        && value.partition_entity_count.0 == partition.target.entity_count() as u64
                        && value.source_partition_count.0 == selected.descriptors.len() as u64
                        && value.source_entity_count.0 == entities
                        && value.source_value_count.0 == values)
            })?;
        }
        let request = ObservationPartitionReductionRequest::published_identity_validated(
            source,
            partition_plan,
            &evidence.contributions,
        )?;
        require(Task::Reduction(request.digest()), &|demand| {
            matches!(demand, FeaResourceDemand::StructuralObservationReduction(value)
                if value.topology_digest == source.plan.topology_digest
                    && value.contribution_count.0 == evidence.contributions.len() as u64)
        })
    })
}
