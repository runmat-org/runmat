//! Complete acoustic fan-in uses the shared governed host-phase estimator.
use super::{
    build_resource_plan, AcousticFrequencyAssemblyResourceDemand, FeaResourceDemand,
    FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput, PreparedOutputSetResourceFacts,
    ResourcePlanningError, RESOURCE_ESTIMATION_MODEL_REVISION,
};

pub fn build_acoustic_frequency_assembly_resource_plan(
    demand: AcousticFrequencyAssemblyResourceDemand,
    output_set: PreparedOutputSetResourceFacts,
    authored: runmat_analysis_core::AccuracyResourceCeiling,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let generic = super::post_process::post_process_ceiling();
    let ceiling = runmat_analysis_core::AccuracyResourceCeiling {
        maximum_refinement_cycles: authored
            .maximum_refinement_cycles
            .min(generic.maximum_refinement_cycles),
        maximum_solver_tightenings: authored
            .maximum_solver_tightenings
            .min(generic.maximum_solver_tightenings),
        maximum_samples: authored.maximum_samples.min(generic.maximum_samples),
        maximum_wall_time_milliseconds: authored
            .maximum_wall_time_milliseconds
            .min(generic.maximum_wall_time_milliseconds),
        // The tighter of two optional ceilings; `None` (no cap) never tightens.
        maximum_resident_bytes: match (authored.resident_limit(), generic.resident_limit()) {
            (Some(authored), Some(generic)) => Some(authored.min(generic)),
            (limit, None) | (None, limit) => limit,
        },
    };
    if demand.entry_count.0 > ceiling.maximum_samples as u64 {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "frequency samples",
            required: demand.entry_count.0,
            maximum: ceiling.maximum_samples as u64,
        });
    }
    let mut worker = super::post_process::post_process_worker();
    worker.maximum_wall_milliseconds = ceiling.maximum_wall_time_milliseconds;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::AcousticFrequencyAssembly(demand.workload),
        demand: FeaResourceDemand::AcousticFrequencyAssembly(demand),
        output_set: Some(output_set),
        worker,
        accuracy_ceiling: ceiling,
    })
}
