//! Shared numerical-identity tuple for retained content and preassembly shape projection.
use super::*;
use crate::{
    admission::AnalysisRunIdentity,
    execution::{StepIdentity, ThermalNumericalIntentIdentity},
};
use serde::Serialize;

// Tuple order is part of numerical identity. Generic borrowed/repeated inventories preserve that
// same wire authority without building hypothetical histories for resource admission.
#[derive(Serialize)]
pub(in crate::execution) struct IdentitySerialization<'a, T, F, R, E, N, L, I, H, J>(
    pub ThermalNumericalIntentIdentity,
    pub &'a AnalysisRunIdentity,
    pub T,
    pub F,
    pub R,
    pub f64,
    pub E,
    pub f64,
    pub Digest,
    pub StepIdentity,
    pub N,
    pub L,
    pub I,
    pub H,
    pub J,
);

impl ThermalResultMetadata<'_> {
    pub(in crate::execution) fn identity_serialization<'a, F: Serialize + 'a>(
        &'a self,
        fields: F,
    ) -> impl Serialize + 'a {
        IdentitySerialization(
            self.numerical_intent,
            &self.run.identity,
            self.time_points_s,
            fields,
            self.residual_norms,
            self.residual_target,
            self.energy_balances,
            self.reference_temperature_k,
            self.topology_digest,
            self.producer,
            self.ordered_node_ids,
            self.ordered_element_ids,
            self.initial_state,
            self.nonlinear_evidence,
            self.interface_node_ids,
        )
    }

    pub(in crate::execution) fn identity<F: Serialize>(
        &self,
        fields: F,
    ) -> AnalysisExecutionResult<SolverResultIdentity> {
        // Presentation diagnostics and execution telemetry are not numerical identity inputs.
        let content = encode_contract(
            PREFIX,
            "thermal-solver-result-accepted-intent-field-content",
            &self.identity_serialization(fields),
            LIMITS,
        )?;
        SolverResultIdentity::from_step_result(self.producer, Digest::sha256(content))
    }
}

pub(in crate::execution) struct ManifestFieldIdentities<'a>(
    pub &'a [Vec<crate::results::field::RecoveredFieldArtifactEntry>],
);
impl Serialize for ManifestFieldIdentities<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for entries in self.0 {
            sequence.serialize_element(
                &entries.iter().map(|entry| entry.field).collect::<Vec<_>>(),
            )?;
        }
        sequence.end()
    }
}
