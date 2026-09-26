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
pub(in crate::execution) struct IdentitySerialization<'a, T, F, R, E, N, L, I>(
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
);

impl ThermalResultMetadata<'_> {
    pub(in crate::execution) fn identity_serialization<'a>(
        &'a self,
        fields: &'a [[crate::results::field::RecoveredFieldIdentity; 4]],
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
        )
    }

    pub(in crate::execution) fn identity(
        &self,
        fields: &[[crate::results::field::RecoveredFieldIdentity; 4]],
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
