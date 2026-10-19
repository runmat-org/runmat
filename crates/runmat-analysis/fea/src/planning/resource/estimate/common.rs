//! Shared resource-plan revision, output-set, worker and accelerator preconditions.
use super::*;

pub(super) fn validate_common(
    input: &FeaResourcePlanningInput,
) -> Result<(), ResourcePlanningError> {
    if input.estimation_model_revision != RESOURCE_ESTIMATION_MODEL_REVISION {
        return Err(ResourcePlanningError::Invalid(
            "unsupported resource estimation model revision".into(),
        ));
    }
    if input.worker.cpu_millicores == 0 || input.worker.maximum_wall_milliseconds == 0 {
        return Err(ResourcePlanningError::Invalid(
            "worker CPU and wall-time bounds must be nonzero".into(),
        ));
    }
    if input.demand.governs_prepared_outputs()
        != input
            .output_set
            .as_ref()
            .is_some_and(|facts| facts.encoded_bytes_upper_bound.0 > 0)
    {
        return Err(ResourcePlanningError::Invalid(
            "independently prepared outputs require one bounded output set".into(),
        ));
    }
    if input.worker.maximum_wall_milliseconds
        > input.accuracy_ceiling.maximum_wall_time_milliseconds
    {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "wall time milliseconds",
            required: input.worker.maximum_wall_milliseconds,
            maximum: input.accuracy_ceiling.maximum_wall_time_milliseconds,
        });
    }
    let mut accelerator_classes = std::collections::BTreeSet::new();
    for accelerator in &input.worker.accelerators {
        if accelerator.class.as_str().trim().is_empty()
            || accelerator.count == 0
            || accelerator.memory_bytes_each.0 == 0
            || !accelerator_classes.insert(accelerator.class.as_str())
        {
            return Err(ResourcePlanningError::Invalid(
                "accelerator class, count, and memory must be nonzero and classes must be unique"
                    .into(),
            ));
        }
    }
    Ok(())
}
