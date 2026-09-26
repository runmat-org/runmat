use super::*;

pub(super) fn validate_accuracy_demand(
    input: &FeaResourcePlanningInput,
    demand: &AccuracyResourceDemand,
    observations: &ObservationResourceFacts,
) -> Result<(), ResourcePlanningError> {
    let refinements = u64::try_from(input.accuracy_ceiling.maximum_refinement_cycles)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy refinement ceiling"))?;
    let tightenings = u64::try_from(input.accuracy_ceiling.maximum_solver_tightenings)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy tightening ceiling"))?;
    let samples = u64::try_from(input.accuracy_ceiling.maximum_samples)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy sample ceiling"))?;
    if demand.refinement_cycles.0 > refinements {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "refinement cycles",
            required: demand.refinement_cycles.0,
            maximum: refinements,
        });
    }
    if demand.solver_tightenings.0 > tightenings {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "solver tightenings",
            required: demand.solver_tightenings.0,
            maximum: tightenings,
        });
    }
    let required_samples = demand
        .samples
        .0
        .max(observations.maximum_samples_per_observation.0);
    if required_samples > samples {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "samples",
            required: required_samples,
            maximum: samples,
        });
    }
    Ok(())
}
