//! Original projection admission precedes owned observation topology construction.
use super::*;

pub fn resolve_observation_plan(
    step: &ProcedureStep,
    projection: &AnalysisProjection,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    validate_request_counts(step, limits)?;
    let (topology, remaining) = ObservationTopology::from_projection_admitted(
        projection,
        limits.maximum_peak_resident_bytes,
    )?;
    let topology = if ObservationTopology::needs_faces(fields) {
        topology.with_projection_faces()?
    } else {
        topology
    };
    let limits = ObservationLimits {
        maximum_peak_resident_bytes: remaining,
        ..limits
    };
    resolve_observation_plan_for_topology(step, &topology, None, None, fields, limits)
}
