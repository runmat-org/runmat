use runmat_analysis_core::AnalysisField;
use runmat_canonical_codec::SerializedCanonicalProfile;
use serde::Serialize;

use crate::execution::{
    AnalysisExecutionError, AnalysisExecutionResult, SemanticStatePublicationRequest,
    ThermalSolverResultManifest,
};

struct RepeatedTemperatures(usize);

impl Serialize for RepeatedTemperatures {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::repeat_n(0.0_f64, self.0))
    }
}

#[derive(Serialize)]
struct ThermalPayload<'a, T: Serialize> {
    kind: &'static str,
    state: ThermalState<'a, T>,
}

#[derive(Serialize)]
struct ThermalState<'a, T: Serialize> {
    temperature: T,
    ordered_node_ids: &'a [u64],
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    reference_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    sample_time_s: runmat_types::physical::Time,
    maximum_relative_linear_residual: f64,
    #[serde(skip_serializing_if = "<[u64]>::is_empty")]
    interface_node_ids: &'a [u64],
    #[serde(skip_serializing_if = "Option::is_none")]
    interface_temperature: Option<T>,
}

/// Profile only the state encoding/materialization, without reading numerical partitions.
pub(super) fn profile(
    request: &SemanticStatePublicationRequest,
    manifest: &ThermalSolverResultManifest,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    manifest.validate_publication_request(request)?;
    let definition = &manifest.terminal_temperature()?.definition;
    let invalid = |error: crate::results::field::RecoveredFieldError| {
        AnalysisExecutionError::Invalid(format!("thermal state encoding profile: {error}"))
    };
    let entity_count = manifest.ordered_node_ids.len();
    let count = entity_count
        .checked_mul(definition.storage_component_count().map_err(invalid)?)
        .ok_or_else(|| {
            AnalysisExecutionError::Invalid("thermal state encoding profile size overflow".into())
        })?;
    let shape = definition.storage_shape(entity_count).map_err(invalid)?;
    // The shared profiler charges every f64 at its maximum encoded width. These placeholders
    // are lazy serialization input, not a fabricated published state or a numerical buffer.
    let values = RepeatedTemperatures(count);
    // The published state carries the terminal snapshot's storage layout, step ordinal included.
    let terminal_step = manifest.fields.len().saturating_sub(1);
    let storage_layout = crate::physics::thermal::history_fields::storage_layout;
    let layout = storage_layout(definition, terminal_step);
    let interface_fields = manifest.terminal_state_fields()?;
    let interface_layout =
        interface_fields.get(1).map(|entry| storage_layout(&entry.definition, terminal_step));
    let temperature =
        AnalysisField::host_f64_serialization(definition.field_id(), &layout, &shape, &values);
    let interface_values = RepeatedTemperatures(manifest.interface_node_ids.len());
    let interface_shape = [manifest.interface_node_ids.len()];
    let interface_temperature = interface_fields.get(1).zip(interface_layout.as_ref()).map(|(entry, layout)| {
        AnalysisField::host_f64_serialization(
            entry.definition.field_id(),
            layout,
            &interface_shape,
            &interface_values,
        )
    });
    let payload = ThermalPayload {
        kind: "thermal",
        state: ThermalState {
            temperature,
            ordered_node_ids: &manifest.ordered_node_ids,
            reference_temperature_k: runmat_types::physical::Temperature::new(
                manifest.reference_temperature_k.canonical_value(),
            ),
            sample_time_s: runmat_types::physical::Time::new(
                *manifest.time_points_s.last().ok_or_else(|| {
                    AnalysisExecutionError::Invalid(
                        "thermal state profile missing terminal time".into(),
                    )
                })?,
            ),
            maximum_relative_linear_residual: manifest
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            interface_node_ids: &manifest.interface_node_ids,
            interface_temperature,
        },
    };
    super::profile_state_payload(request, &payload)
}

#[cfg(test)]
mod tests;
