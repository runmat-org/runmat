//! Sequential original compiler phase inside the same structural task reservation.
use super::*;

pub(super) fn original_compiler_peak(
    input: &FeaResourcePlanningInput,
) -> Result<u64, ResourcePlanningError> {
    let peak = match &input.demand {
        FeaResourceDemand::StructuralLinearPipeline(demand) => {
            demand.constraint_compilation_peak_bytes.0
        }
        FeaResourceDemand::StructuralNonlinearPipeline(demand) => {
            demand.constraint_compilation_peak_bytes.0
        }
        FeaResourceDemand::StructuralNonlinearRecovery(demand) => {
            demand.constraint_compilation_peak_bytes.0
        }
        FeaResourceDemand::StructuralModalExtraction(demand) => {
            demand.constraint_compilation_peak_bytes.0
        }
        FeaResourceDemand::StructuralFrequencyEntry(demand)
        | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
            demand.constraint_compilation_peak_bytes.0
        }
        _ => return Ok(0),
    };
    if peak == 0 {
        return Err(ResourcePlanningError::Invalid(
            "structural compiler phase has no construction reservation".into(),
        ));
    }
    Ok(peak)
}
