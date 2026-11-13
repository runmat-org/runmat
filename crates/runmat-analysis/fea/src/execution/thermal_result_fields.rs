//! Bind thermal numerical content to the shared field identity contract, one field at a time.
use super::{AnalysisExecutionError, AnalysisExecutionResult, FeaThermalRunResult};
use crate::results::field::{RecoveredField, RecoveredFieldIdentity};

/// Called only after complete-result validation proves aligned history lengths and shapes.
/// Order is temperature, gradient, flux, storage-rate density, matching the typed catalog, then
/// the second-side interface temperature when the study has a finite-conductance interface.
pub(super) fn retained_field_identities(
    result: &FeaThermalRunResult,
) -> AnalysisExecutionResult<Vec<Vec<RecoveredFieldIdentity>>> {
    let invalid = |error: crate::results::field::RecoveredFieldError| {
        AnalysisExecutionError::Invalid(format!("thermal retained field: {error}"))
    };
    let histories = [
        &result.temperature_snapshots,
        &result.temperature_gradient_snapshots,
        &result.heat_flux_snapshots,
        &result.energy_storage_rate_snapshots,
        &result.interface_temperature_snapshots,
    ];
    let count = if result.interface_node_ids.is_empty() {
        4
    } else {
        5
    };
    result
        .time_points_s
        .iter()
        .enumerate()
        .map(|(ordinal, time)| {
            let identify = |index: usize| {
                let storage = &histories[index][ordinal];
                // Retain the canonical snapshot identifier and explicit physical sample time.
                let definition =
                    crate::physics::thermal::history_fields::snapshot_definition(index, ordinal)
                        .map_err(invalid)?;
                if definition.field_id() != storage.field_id {
                    return Err(AnalysisExecutionError::Invalid(
                        "thermal retained field identifier differs from its history".into(),
                    ));
                }
                RecoveredField::identity_for(
                    &definition,
                    crate::physics::thermal::history_fields::sample(
                        result.time_points_s.len(),
                        *time,
                    ),
                    storage,
                )
                .map_err(invalid)
            };
            (0..count).map(identify).collect()
        })
        .collect()
}
