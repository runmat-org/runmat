//! Bind thermal numerical content to the shared field identity contract, one field at a time.
use super::{AnalysisExecutionError, AnalysisExecutionResult, FeaThermalRunResult};
use crate::results::field::{
    RecoveredField, RecoveredFieldDefinition, RecoveredFieldIdentity, RecoveredFieldSample,
};

/// Called only after complete-result validation proves aligned history lengths and shapes.
/// Array order is temperature, gradient, flux, storage-rate density, matching the typed catalog.
pub(super) fn retained_field_identities(
    result: &FeaThermalRunResult,
) -> AnalysisExecutionResult<Vec<[RecoveredFieldIdentity; 4]>> {
    let invalid = |error: crate::results::field::RecoveredFieldError| {
        AnalysisExecutionError::Invalid(format!("thermal retained field: {error}"))
    };
    let definitions = crate::physics::thermal::history_fields::definitions().map_err(invalid)?;
    let histories = [
        &result.temperature_snapshots,
        &result.temperature_gradient_snapshots,
        &result.heat_flux_snapshots,
        &result.energy_storage_rate_snapshots,
    ];
    result
        .time_points_s
        .iter()
        .enumerate()
        .map(|(ordinal, time)| {
            let identify = |index: usize| {
                let template = &definitions[index];
                let storage = &histories[index][ordinal];
                // Retain the canonical snapshot identifier and explicit physical sample time.
                let definition = RecoveredFieldDefinition::new_derived(
                    &storage.field_id,
                    template.quantity(),
                    template.location(),
                    template.entity_domain(),
                    template.value_shape().clone(),
                    template.numeric_representation(),
                    template.derivation(),
                )
                .map_err(invalid)?;
                RecoveredField::identity_for(
                    &definition,
                    RecoveredFieldSample::TimeSeconds(*time),
                    storage,
                )
                .map_err(invalid)
            };
            Ok([identify(0)?, identify(1)?, identify(2)?, identify(3)?])
        })
        .collect()
}
