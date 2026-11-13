//! Inline observation assembly for direct thermal callers; portable tasks publish primary fields.
use runmat_analysis_core::ProcedureStep;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    contracts::{FeaRunError, FeaThermalRunResult},
    pipeline::observation_history::{
        evaluate_field_histories, HistoryFieldSeries, HistoryTopology,
    },
    topology::AnalysisProjection,
};

pub(super) fn evaluate(
    result: &mut FeaThermalRunResult,
    step: &ProcedureStep,
    mesh: &SolverMeshArtifact,
) -> Result<(), FeaRunError> {
    if step.observations.iter().any(|request| {
        matches!(
            request,
            runmat_analysis_core::ObservationRequest::History(_)
        )
    }) {
        let projection = AnalysisProjection::derive(mesh)
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let coordinates = result
            .time_points_s
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::TimeSeconds)
            .collect::<Vec<_>>();
        let definitions = crate::physics::thermal::history_fields::definitions()
            .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        let series = definitions
            .iter()
            .zip([
                result.temperature_snapshots.as_slice(),
                result.temperature_gradient_snapshots.as_slice(),
                result.heat_flux_snapshots.as_slice(),
                result.energy_storage_rate_snapshots.as_slice(),
            ])
            .map(|(definition, snapshots)| HistoryFieldSeries {
                quantity: definition.quantity(),
                source_field_id: definition.field_id(),
                location: definition.location(),
                entity_domain: definition.entity_domain(),
                value_shape: definition.value_shape().clone(),
                numeric_representation: definition.numeric_representation(),
                snapshots,
            })
            .collect::<Vec<_>>();
        let evaluated_histories = evaluate_field_histories(
            step,
            HistoryTopology::Solid(&projection),
            &coordinates,
            None,
            &series,
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
        result.run.observation_histories = evaluated_histories.histories;
        result.run.series_reductions = evaluated_histories.series_reductions;
    }
    Ok(())
}
