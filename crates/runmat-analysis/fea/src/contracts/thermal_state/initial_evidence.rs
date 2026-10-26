//! One borrowed wire schema for realized and projected initial-state evidence.
use super::ThermalInitialStateEvidence;
use runmat_canonical_codec::{
    profile_serialized_shape, CanonicalCodecError, CanonicalLimits, SerializedCanonicalProfile,
};
use serde::{Serialize, Serializer};

#[derive(Serialize)]
#[serde(rename = "ThermalInitialStateEvidence")]
struct EvidenceWire<'a, I: Serialize + ?Sized> {
    procedure_step_id: &'a str,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    reference_temperature_k: runmat_types::physical::Temperature,
    applied_initial_condition_ids: &'a I,
    authored_node_count: usize,
    prescribed_node_count: usize,
    authored_and_prescribed_node_count: usize,
    reference_node_count: usize,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    minimum_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    maximum_temperature_k: runmat_types::physical::Temperature,
}

impl Serialize for ThermalInitialStateEvidence {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        EvidenceWire {
            procedure_step_id: &self.procedure_step_id,
            reference_temperature_k: runmat_types::physical::Temperature::new(
                self.reference_temperature_k.canonical_value(),
            ),
            applied_initial_condition_ids: &self.applied_initial_condition_ids,
            authored_node_count: self.authored_node_count,
            prescribed_node_count: self.prescribed_node_count,
            authored_and_prescribed_node_count: self.authored_and_prescribed_node_count,
            reference_node_count: self.reference_node_count,
            minimum_temperature_k: runmat_types::physical::Temperature::new(
                self.minimum_temperature_k.canonical_value(),
            ),
            maximum_temperature_k: runmat_types::physical::Temperature::new(
                self.maximum_temperature_k.canonical_value(),
            ),
        }
        .serialize(serializer)
    }
}

impl ThermalInitialStateEvidence {
    /// Counts and finite floating-point values use the shared profiler's maximum scalar width.
    /// These placeholders describe serialization shape only, never a realized physical state.
    pub(crate) fn projected_profile<I: Serialize + ?Sized>(
        procedure_step_id: &str,
        applied_initial_condition_ids: &I,
        limits: CanonicalLimits,
    ) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
        // Stand-in counts bound every future count, so integers reserve their full range.
        profile_serialized_shape(
            &EvidenceWire {
                procedure_step_id,
                reference_temperature_k: runmat_types::physical::Temperature::new(1.0),
                applied_initial_condition_ids,
                authored_node_count: 0,
                prescribed_node_count: 0,
                authored_and_prescribed_node_count: 0,
                reference_node_count: 0,
                minimum_temperature_k: runmat_types::physical::Temperature::new(1.0),
                maximum_temperature_k: runmat_types::physical::Temperature::new(1.0),
            },
            limits,
        )
    }
}
